"""User-defined selector strategies (Phase 2).

A user strategy is a JSON document with a flat list of comparison rules
that get evaluated against per-symbol indicator metrics. The vocabulary is
fixed (METRIC_REGISTRY + OPERATORS), so we never have to `exec()` user
input — the LLM-compiled NL-prompt path produces the same JSON.

Schema (validated by `validate()`):

    {
      "label": "我的突破策略",
      "description": "",                       # optional
      "match_mode": "AND" | "OR_AT_LEAST",     # default AND
      "min_match_rules": 1,                    # only used when OR_AT_LEAST
      "rules": [
        {"metric": "vol_ratio_ma5", "op": ">=", "value": 1.5},
        {"metric": "price_position_60d", "op": "between", "value": [60, 95]},
        {"metric": "close_above_ma20", "op": "==", "value": true}
      ]
    }
"""
from __future__ import annotations

import math
from typing import Any, Callable

import pandas as pd


# ── Metric vocabulary ─────────────────────────────────────────────────

MetricFn = Callable[[pd.DataFrame, dict[str, float]], float | int | bool | None]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _ma(hist: pd.DataFrame, n: int) -> float:
    if hist is None or len(hist) < n or "收盘" not in hist.columns:
        return 0.0
    return float(hist["收盘"].tail(n).mean() or 0.0)


def _vol_ratio_ma(hist: pd.DataFrame, n: int, vol_today: float) -> float:
    if hist is None or len(hist) < n + 1 or "成交量" not in hist.columns:
        return 0.0
    prior = hist["成交量"].iloc[-(n + 1):-1]
    mean = float(prior.mean() or 0.0)
    return vol_today / mean if mean > 0 else 0.0


def _price_position(hist: pd.DataFrame, price: float, n: int) -> float:
    if hist is None or len(hist) < n or "收盘" not in hist.columns:
        return 0.0
    window = hist["收盘"].tail(n)
    lo = float(window.min())
    hi = float(window.max())
    if hi <= lo:
        return 50.0
    return (price - lo) / (hi - lo) * 100.0


def _pct_chg_5d(hist: pd.DataFrame, price: float) -> float:
    if hist is None or len(hist) < 6 or "收盘" not in hist.columns:
        return 0.0
    base = float(hist["收盘"].iloc[-6] or 0.0)
    return (price - base) / base * 100.0 if base > 0 else 0.0


def _consecutive_up_days(hist: pd.DataFrame, pct_chg_today: float) -> int:
    if hist is None or "涨跌幅" not in hist.columns:
        return 1 if pct_chg_today > 0 else 0
    count = 1 if pct_chg_today > 0 else 0
    if count == 0:
        return 0
    for v in hist["涨跌幅"].iloc[::-1]:
        if _safe_float(v) > 0:
            count += 1
        else:
            break
    return count


def _breakout_n_day_high(hist: pd.DataFrame, price: float, n: int) -> bool:
    if hist is None or len(hist) < n or "收盘" not in hist.columns:
        return False
    prior_max = float(hist["收盘"].tail(n).max() or 0.0)
    return prior_max > 0 and price > prior_max


def _rsi_14(hist: pd.DataFrame) -> float:
    if hist is None or len(hist) < 15 or "收盘" not in hist.columns:
        return 50.0
    closes = hist["收盘"].astype(float).tail(15).reset_index(drop=True)
    diffs = closes.diff().dropna()
    gains = diffs.clip(lower=0)
    losses = (-diffs).clip(lower=0)
    avg_gain = float(gains.mean() or 0.0)
    avg_loss = float(losses.mean() or 0.0)
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_metrics(
    hist: pd.DataFrame,
    price: float,
    vol_today: float,
    amount_today: float,
    pct_chg_today: float,
) -> dict[str, float | int | bool]:
    """Compute all supported metrics once per symbol. Returns a dict whose
    keys are exactly METRIC_REGISTRY's keys."""
    ma5 = _ma(hist, 5)
    ma10 = _ma(hist, 10)
    ma20 = _ma(hist, 20)
    ma60 = _ma(hist, 60)

    return {
        "vol_ratio_ma5": _vol_ratio_ma(hist, 5, vol_today),
        "vol_ratio_ma20": _vol_ratio_ma(hist, 20, vol_today),
        "price_position_60d": _price_position(hist, price, 60),
        "price_position_120d": _price_position(hist, price, 120),
        "pct_chg_today": float(pct_chg_today),
        "pct_chg_5d": _pct_chg_5d(hist, price),
        "amount_today_yi": amount_today / 1e8,
        "close_above_ma5": price > ma5 if ma5 > 0 else False,
        "close_above_ma10": price > ma10 if ma10 > 0 else False,
        "close_above_ma20": price > ma20 if ma20 > 0 else False,
        "close_above_ma60": price > ma60 if ma60 > 0 else False,
        "ma5_above_ma20": ma5 > ma20 if ma5 > 0 and ma20 > 0 else False,
        "consecutive_up_days": _consecutive_up_days(hist, pct_chg_today),
        "breakout_60d_high": _breakout_n_day_high(hist, price, 60),
        "rsi_14": _rsi_14(hist),
    }


# Metric -> (type label, human description, example value)
METRIC_REGISTRY: dict[str, dict[str, Any]] = {
    "vol_ratio_ma5":       {"type": "float", "desc_en": "Today volume ÷ avg of last 5 days",   "desc_zh": "今日成交量÷5日均量",     "example": 1.5},
    "vol_ratio_ma20":      {"type": "float", "desc_en": "Today volume ÷ avg of last 20 days",  "desc_zh": "今日成交量÷20日均量",    "example": 2.0},
    "price_position_60d":  {"type": "pct",   "desc_en": "Position in 60-day high-low range",   "desc_zh": "60日高低位置百分比",     "example": 70.0},
    "price_position_120d": {"type": "pct",   "desc_en": "Position in 120-day high-low range",  "desc_zh": "120日高低位置百分比",    "example": 50.0},
    "pct_chg_today":       {"type": "pct",   "desc_en": "Today's % change",                    "desc_zh": "当日涨跌幅%",            "example": 3.0},
    "pct_chg_5d":          {"type": "pct",   "desc_en": "5-day cumulative % change",           "desc_zh": "5日累计涨跌幅%",         "example": 8.0},
    "amount_today_yi":     {"type": "float", "desc_en": "Today's turnover in 亿元",            "desc_zh": "今日成交额(亿元)",       "example": 5.0},
    "close_above_ma5":     {"type": "bool",  "desc_en": "Close > MA5",                         "desc_zh": "收盘价>5日均线",         "example": True},
    "close_above_ma10":    {"type": "bool",  "desc_en": "Close > MA10",                        "desc_zh": "收盘价>10日均线",        "example": True},
    "close_above_ma20":    {"type": "bool",  "desc_en": "Close > MA20",                        "desc_zh": "收盘价>20日均线",        "example": True},
    "close_above_ma60":    {"type": "bool",  "desc_en": "Close > MA60",                        "desc_zh": "收盘价>60日均线",        "example": True},
    "ma5_above_ma20":      {"type": "bool",  "desc_en": "MA5 > MA20 (short-term golden align)","desc_zh": "5日均线>20日均线",       "example": True},
    "consecutive_up_days": {"type": "int",   "desc_en": "Consecutive positive close days",     "desc_zh": "连续上涨天数",           "example": 3},
    "breakout_60d_high":   {"type": "bool",  "desc_en": "Close above prior 60-day high",       "desc_zh": "突破60日新高",           "example": True},
    "rsi_14":              {"type": "float", "desc_en": "RSI(14)",                             "desc_zh": "RSI(14)",                "example": 65.0},
}


# ── Operators ──────────────────────────────────────────────────────────

OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a >  b,
    "<":  lambda a, b: a <  b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "between": lambda a, b: b[0] <= a <= b[1],
}

VALID_MATCH_MODES = {"AND", "OR_AT_LEAST"}


# ── Validation ─────────────────────────────────────────────────────────

def validate(doc: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors. Empty list = OK."""
    errors: list[str] = []

    if not isinstance(doc, dict):
        return ["Strategy must be a JSON object."]

    label = doc.get("label", "").strip() if isinstance(doc.get("label"), str) else ""
    if not label:
        errors.append("`label` is required.")
    elif len(label) > 60:
        errors.append("`label` must be at most 60 characters.")

    desc = doc.get("description", "")
    if not isinstance(desc, str):
        errors.append("`description` must be a string.")
    elif len(desc) > 500:
        errors.append("`description` must be at most 500 characters.")

    match_mode = doc.get("match_mode", "AND")
    if match_mode not in VALID_MATCH_MODES:
        errors.append(f"`match_mode` must be one of {sorted(VALID_MATCH_MODES)}.")

    rules = doc.get("rules")
    if not isinstance(rules, list) or not rules:
        errors.append("`rules` must be a non-empty list.")
        return errors
    if len(rules) > 20:
        errors.append("`rules` must contain at most 20 entries.")

    if match_mode == "OR_AT_LEAST":
        min_n = doc.get("min_match_rules", 1)
        if not isinstance(min_n, int) or min_n < 1 or min_n > len(rules):
            errors.append("`min_match_rules` must be an integer between 1 and len(rules).")

    for i, rule in enumerate(rules):
        prefix = f"rules[{i}]"
        if not isinstance(rule, dict):
            errors.append(f"{prefix} must be an object.")
            continue
        metric = rule.get("metric")
        op = rule.get("op")
        value = rule.get("value")

        meta = METRIC_REGISTRY.get(metric)
        if meta is None:
            errors.append(f"{prefix}.metric `{metric}` is not a supported metric.")
            continue
        if op not in OPERATORS:
            errors.append(f"{prefix}.op `{op}` is not a supported operator.")
            continue

        mtype = meta["type"]
        if op == "between":
            if not (isinstance(value, list) and len(value) == 2
                    and all(isinstance(x, (int, float)) for x in value)
                    and value[0] <= value[1]):
                errors.append(f"{prefix}.value must be [lo, hi] (numbers, lo<=hi) when op='between'.")
            if mtype == "bool":
                errors.append(f"{prefix}: op 'between' is not valid for boolean metric '{metric}'.")
        else:
            if mtype == "bool":
                if not isinstance(value, bool):
                    errors.append(f"{prefix}.value must be true/false for boolean metric '{metric}'.")
                elif op not in ("==", "!="):
                    errors.append(f"{prefix}: only '==' or '!=' allowed for boolean metric '{metric}'.")
            elif mtype == "int":
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(f"{prefix}.value must be an integer for metric '{metric}'.")
            else:  # float, pct
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(f"{prefix}.value must be a number for metric '{metric}'.")

    return errors


# ── Evaluation ─────────────────────────────────────────────────────────

def evaluate(
    doc: dict[str, Any],
    metrics: dict[str, Any],
    *,
    strategy_label: str | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate a strategy against pre-computed metrics. Returns
    (matched, reasons). Caller must have validated `doc` already."""
    rules = doc.get("rules", [])
    match_mode = doc.get("match_mode", "AND")
    min_n = doc.get("min_match_rules", 1) if match_mode == "OR_AT_LEAST" else len(rules)

    hits: list[str] = []
    for rule in rules:
        metric = rule["metric"]
        op = rule["op"]
        value = rule["value"]
        actual = metrics.get(metric)
        if actual is None:
            continue
        try:
            ok = OPERATORS[op](actual, value)
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            meta = METRIC_REGISTRY.get(metric, {})
            zh = meta.get("desc_zh", metric)
            if op == "between":
                hits.append(f"{zh}∈[{value[0]},{value[1]}] (实际 {_fmt(actual)})")
            else:
                hits.append(f"{zh} {op} {_fmt(value)} (实际 {_fmt(actual)})")

    matched = len(hits) >= min_n
    if not matched:
        return False, []

    label_part = f"【{strategy_label}】" if strategy_label else "【自定义】"
    summary = label_part + "、".join(hits)
    return True, [summary]


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, list):
        return f"[{', '.join(_fmt(x) for x in v)}]"
    return str(v)


# ── Helpers for API surface ───────────────────────────────────────────

def metric_catalog_for_ui() -> list[dict[str, Any]]:
    """Return metric metadata formatted for the UI dropdown."""
    return [
        {"key": key, **{k: v for k, v in meta.items()}}
        for key, meta in METRIC_REGISTRY.items()
    ]


def operator_catalog_for_ui() -> list[str]:
    return list(OPERATORS.keys())
