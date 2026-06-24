"""
AI Stock Selector service — Phase 1.

Ports the 天量战法 scanner (originally `stock_selector.py`) into QuantMind's
watchlist UX. Provides:

* `list_strategies()` — built-in selector strategies, for UI population
* `submit_job(...)` — kicks off an async scan, returns job_id
* `get_job(job_id)` — polled by UI for status / result

A "selector strategy" is conceptually different from `strategy_catalog.STRATEGIES`:
strategies in that catalog operate on **one symbol's history + Kronos forecast**
to produce backtest signals. Selector strategies operate on the **whole market
universe** to surface candidate symbols for the watchlist.

The patterns are based on the 天量战法 教学:
  1. 天量连板  — 天量涨停板，判断是否为板块龙一，决定首封/回封/低吸
  2. 天量反包  — 天量断板后次日弱转强，打板或分时均线低吸
  3. 天量不跌  — 天量后横盘2天小阴小阳，尾盘低吸
  4. 天量回踩  — 天量后回踩到大阳线一半位置，潜伏
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config import ToolkitConfig
import user_strategy as _us

logger = logging.getLogger(__name__)


# ── Tunables ───────────────────────────────────────────────────────────
SKY_VOL_WINDOW = 120         # bars used for "近半年最大量" comparison
SKY_VOL_RATIO = 0.95         # today's vol must be >= ratio * window max
BOARD_LIMIT = 4              # 连板最高关注到 4 板
POSITION_HIGH_PCT = 60.0     # > this % above MA60 ⇒ "高位" (excluded unless 连板龙头)

# Cloud / HF Spaces tend to be rate-limited by East-Money / Sina, so we
# default to a much lower concurrency and add per-request retries when we
# detect we're running on Hugging Face Spaces.
_ON_HF = bool(os.environ.get("SPACE_ID") or os.environ.get("SPACE_HOST"))
DEFAULT_MAX_WORKERS = 3 if _ON_HF else 6
RETRY_ATTEMPTS = 4 if _ON_HF else 2
RETRY_BASE_DELAY = 0.4       # seconds; exponential backoff with jitter

DEFAULT_TOP_ACTIVE_A = 400
DEFAULT_TOP_ACTIVE_HK = 150
DEFAULT_HISTORY_DAYS = 210


# ── Browser-like default headers for AKShare's underlying requests calls ─
# East-Money and Sina close the connection (RemoteDisconnected) on requests
# that look automated, especially from foreign cloud IPs. We monkey-patch
# requests' default headers ONCE at import time so AKShare gets a realistic
# User-Agent and Referer without having to pass them at every call site.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://quote.eastmoney.com/",
    "Connection": "keep-alive",
}


def _patch_requests_defaults_once() -> None:
    """Inject browser headers into all `requests` calls that don't override them."""
    try:
        import requests
    except ImportError:
        return
    if getattr(requests, "_qm_selector_patched", False):
        return

    _orig_request = requests.Session.request

    def _patched_request(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        headers = dict(_BROWSER_HEADERS)
        if kwargs.get("headers"):
            headers.update(kwargs["headers"])
        kwargs["headers"] = headers
        # Be defensive about hangs from servers that accept the connection
        # then never reply.
        kwargs.setdefault("timeout", 15)
        return _orig_request(self, method, url, **kwargs)

    requests.Session.request = _patched_request  # type: ignore[assignment]
    requests._qm_selector_patched = True  # type: ignore[attr-defined]
    logger.info("Patched requests.Session.request with browser headers + 15s default timeout")


_patch_requests_defaults_once()


def _retry(fn: Callable[[], Any], *, attempts: int = RETRY_ATTEMPTS,
           base_delay: float = RETRY_BASE_DELAY) -> Any:
    """Retry a callable on transient network errors with exponential backoff."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if i == attempts - 1:
                break
            sleep = base_delay * (2 ** i) + random.uniform(0, base_delay)
            time.sleep(sleep)
    if last_exc is not None:
        raise last_exc


# ── Strategy definitions (UI metadata) ─────────────────────────────────

@dataclass(frozen=True, slots=True)
class SelectorStrategy:
    key: str
    label_en: str
    label_zh: str
    description_en: str
    description_zh: str
    markets: tuple[str, ...] = ("A", "HK")


SELECTOR_STRATEGIES: dict[str, SelectorStrategy] = {
    "tianliang_lianban": SelectorStrategy(
        key="tianliang_lianban",
        label_en="Sky Volume + Consecutive Limit-Up",
        label_zh="天量连板",
        description_en="Today closes at limit-up on the largest volume of the past 120 bars, with recent consecutive limit-ups.",
        description_zh="今日涨停 + 近120日最大量，且近5日有连板。",
    ),
    "tianliang_fanbao": SelectorStrategy(
        key="tianliang_fanbao",
        label_en="Sky Volume Reversal",
        label_zh="天量反包",
        description_en="Day-before limit-up, yesterday broke the streak, today turns strong on sky volume.",
        description_zh="前天涨停 → 昨天断板 → 今天天量转强。",
    ),
    "tianliang_buzhi": SelectorStrategy(
        key="tianliang_buzhi",
        label_en="Sky Volume Hold (Watch)",
        label_zh="天量不跌(观察)",
        description_en="Sky-volume up day at moderate position; needs 2-day follow-up of tight consolidation.",
        description_zh="天量涨停/大涨，位置不高(MA60+35%以内)，需跟踪2天小阴小阳。",
    ),
    "tianliang_huicai": SelectorStrategy(
        key="tianliang_huicai",
        label_en="Sky Volume Pullback (Watch)",
        label_zh="天量回踩(观察)",
        description_en="Big up day on sky volume without limit-up; wait for pullback to mid-candle.",
        description_zh="天量大涨但没涨停，等回踩到大阳线一半位置。",
    ),
}


def list_strategies() -> list[dict[str, Any]]:
    return [
        {
            "key": s.key,
            "label_en": s.label_en,
            "label_zh": s.label_zh,
            "description_en": s.description_en,
            "description_zh": s.description_zh,
            "markets": list(s.markets),
        }
        for s in SELECTOR_STRATEGIES.values()
    ]


# ── Helpers ────────────────────────────────────────────────────────────

def _is_trading_day(date: datetime) -> bool:
    return date.weekday() < 5


def _recent_trading_date() -> str:
    d = datetime.now()
    while not _is_trading_day(d):
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def _start_date(history_days: int = DEFAULT_HISTORY_DAYS) -> str:
    return (datetime.now() - timedelta(days=history_days)).strftime("%Y%m%d")


def _stock_board(code: str) -> str:
    if code.startswith("688"):
        return "科创板"
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith(("8", "4")):
        return "北交所"
    return "主板"


def _limit_up_pct(code: str) -> float:
    board = _stock_board(code)
    if board in ("科创板", "创业板"):
        return 20.0
    if board == "北交所":
        return 30.0
    return 10.0


def _classify(code: str, pct_chg: float) -> str:
    limit = _limit_up_pct(code)
    if pct_chg >= limit - 0.5:
        return "涨停"
    if pct_chg <= -limit + 0.5:
        return "跌停"
    if pct_chg >= 5.0:
        return "大涨"
    return "正常"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except (TypeError, ValueError):
        return default


# ── AKShare wrappers (lazy import; surfaces clean errors) ──────────────

def _ak():
    try:
        import akshare as ak
        return ak
    except ImportError as exc:
        raise RuntimeError("akshare is not installed in this environment.") from exc


def _normalize_hist_sina(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Sina's stock_zh_a_daily returns: date, open, high, low, close, volume,
    (amount). Translate to the EM Chinese-column schema the analyzer expects
    and clip to the [start, end] window (Sina returns full history)."""
    if df is None or df.empty:
        return df
    df = df.reset_index()
    rename = {
        "date": "日期", "open": "开盘", "close": "收盘",
        "high": "最高", "low": "最低", "volume": "成交量",
        "amount": "成交额",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "成交额" not in df.columns and {"成交量", "收盘"}.issubset(df.columns):
        # Sina volume is in 股 (shares); amount = volume * close.
        df["成交额"] = (
            pd.to_numeric(df["成交量"], errors="coerce").fillna(0)
            * pd.to_numeric(df["收盘"], errors="coerce").fillna(0)
        )
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
        s = pd.to_datetime(start).strftime("%Y-%m-%d")
        e = pd.to_datetime(end).strftime("%Y-%m-%d")
        df = df[(df["日期"] >= s) & (df["日期"] <= e)]
    return df


def _normalize_hist_tx(df: pd.DataFrame) -> pd.DataFrame:
    """Tencent's stock_zh_a_hist_tx returns columns: date, open, close, high,
    low, amount(volume in 手). Translate to the EM Chinese-column schema the
    analyzer expects."""
    if df is None or df.empty:
        return df
    rename = {
        "date": "日期", "open": "开盘", "close": "收盘",
        "high": "最高", "low": "最低", "amount": "成交量",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "成交额" not in df.columns and {"成交量", "收盘"}.issubset(df.columns):
        # rough proxy: amount ≈ volume(手)*100 * close
        df["成交额"] = (
            pd.to_numeric(df["成交量"], errors="coerce").fillna(0) * 100
            * pd.to_numeric(df["收盘"], errors="coerce").fillna(0)
        )
    return df


def _fetch_hist_a(code: str, start: str, end: str) -> tuple[str, pd.DataFrame | None]:
    """Per-symbol daily history with Sina-first fallback chain.

    Order matches the rest of the app (Run Analysis / AI Analysis), which all
    use Sina endpoints successfully on Hugging Face Spaces:

    1. Sina  : ak.stock_zh_a_daily      (host: hq.sinajs.cn)
    2. EM    : ak.stock_zh_a_hist        (host: push2.eastmoney.com — IP-blocked on HF)
    3. Tencent: ak.stock_zh_a_hist_tx    (host: web.ifzq.gtimg.cn)
    """
    ak = _ak()
    sina_sym = ("sh" if code.startswith(("6", "9")) else "sz") + code

    # 1) Sina (same endpoint family as Run Analysis / AI Analysis)
    try:
        hist = _retry(lambda: ak.stock_zh_a_daily(symbol=sina_sym, adjust="qfq"))
        normalized = _normalize_hist_sina(hist, start, end)
        if normalized is not None and not normalized.empty:
            return code, normalized
    except Exception as exc:  # noqa: BLE001
        logger.debug("Sina history fetch failed for %s: %s", code, exc)

    # 2) East-Money (works locally; blocked on HF datacenter IPs)
    try:
        hist = _retry(lambda: ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start, end_date=end, adjust="qfq",
        ))
        if hist is not None and not hist.empty:
            return code, hist
    except Exception as exc:  # noqa: BLE001
        logger.debug("EM history fetch failed for %s: %s", code, exc)

    # 3) Tencent (last-resort; coarser amount proxy)
    try:
        hist_tx = _retry(lambda: ak.stock_zh_a_hist_tx(
            symbol=sina_sym,
            start_date=start, end_date=end, adjust="qfq",
        ))
        return code, _normalize_hist_tx(hist_tx)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Tencent history fallback failed for %s: %s", code, exc)

    return code, None


def _fetch_hist_hk(code: str, start: str, end: str) -> tuple[str, pd.DataFrame | None]:
    try:
        ak = _ak()
        hist = _retry(lambda: ak.stock_hk_hist(
            symbol=code, period="daily",
            start_date=start, end_date=end, adjust="qfq",
        ))
        return code, hist
    except Exception as exc:  # noqa: BLE001
        logger.debug("history HK fetch failed for %s: %s", code, exc)
        return code, None


# ── Per-symbol scoring ─────────────────────────────────────────────────

def _analyze_a(code: str, name: str, price: float, vol_today: float,
               amount_today: float, pct_chg: float, hist: pd.DataFrame,
               wanted: set[str],
               user_strategies: list[dict[str, Any]] | None = None,
               ) -> dict[str, Any] | None:
    """Return scan result if any wanted built-in strategy or user strategy
    matches, else None.

    Built-in (天量战法) strategies require a sky-volume precondition. User
    strategies evaluate independently of that precondition — their rules
    fully determine match.
    """
    if hist is None or len(hist) < 60 or "成交量" not in hist.columns:
        return None

    matched: list[str] = []
    reasons: list[str] = []

    # ── User strategies first (no preconditions; rules-only) ────────
    user_strategies = user_strategies or []
    if user_strategies:
        metrics = _us.compute_metrics(hist, price, vol_today, amount_today, pct_chg)
        for strat in user_strategies:
            try:
                ok, hits = _us.evaluate(strat, metrics, strategy_label=strat.get("label"))
            except Exception as exc:  # noqa: BLE001
                logger.debug("user strategy %s eval failed on %s: %s",
                             strat.get("id"), code, exc)
                continue
            if ok:
                matched.append(f"user:{strat['id']}")
                reasons.extend(hits)

    # Compute baseline indicators always (needed for response payload too)
    ma60 = float(hist["收盘"].tail(60).mean())
    price_position = ((price - ma60) / ma60 * 100.0) if ma60 > 0 else 0.0
    limit = _limit_up_pct(code)
    limit_type = _classify(code, pct_chg)

    # consecutive limit-up board count looking back from today
    recent_5 = hist.tail(5)
    board_count = 0
    for i in range(len(recent_5) - 1, -1, -1):
        if _safe_float(recent_5.iloc[i].get("涨跌幅", 0)) >= limit - 1.0:
            board_count += 1
        else:
            break

    # ── Built-in (天量战法) branch: sky-volume precondition ─────────
    wanted_builtin = wanted & set(SELECTOR_STRATEGIES.keys())
    if wanted_builtin and ma60 > 0:
        window = hist["成交量"].tail(SKY_VOL_WINDOW + 1)
        max_vol = float(window.iloc[:-1].max() or 0)
        sky_volume = max_vol > 0 and vol_today >= max_vol * SKY_VOL_RATIO

        # High-position filter: exclude unless 连板龙头 (≥2 consecutive limits)
        position_block = price_position > POSITION_HIGH_PCT and board_count < 2

        if sky_volume and not position_block:
            yesterday_pct = _safe_float(hist.iloc[-2].get("涨跌幅", 0)) if len(hist) >= 2 else 0
            day_before_pct = _safe_float(hist.iloc[-3].get("涨跌幅", 0)) if len(hist) >= 3 else 0
            yesterday_limit = yesterday_pct >= limit - 1.0
            day_before_limit = day_before_pct >= limit - 1.0
            is_board = board_count >= 1

            if ("tianliang_lianban" in wanted
                    and limit_type == "涨停" and 1 <= board_count <= BOARD_LIMIT):
                matched.append("tianliang_lianban")
                reasons.append(
                    f"【天量连板·{board_count}板】确认是否为板块龙一：龙一可在2-5%区间低吸，"
                    f"或等换手充分后打回封板；非龙一仅打换手回封板。"
                    f"板块：{_stock_board(code)}。"
                )

            if ("tianliang_fanbao" in wanted
                    and day_before_limit and not yesterday_limit and pct_chg >= 0
                    and board_count <= BOARD_LIMIT):
                matched.append("tianliang_fanbao")
                reasons.append(
                    "【天量反包】前天涨停断板，今日转强：分时黄线>0轴可在2-5%低吸，>5%等打板，"
                    "回封板积极参与。"
                )

            if ("tianliang_buzhi" in wanted
                    and limit_type in ("涨停", "大涨")
                    and board_count <= 2 and price_position < 35.0):
                matched.append("tianliang_buzhi")
                reasons.append(
                    "【天量不跌·观察】纳入观察池，需跟踪2天小阴小阳(振幅<3%)不跌破，"
                    "第3天尾盘可介入(尾盘买入法)。"
                )

            if ("tianliang_huicai" in wanted
                    and limit_type == "大涨" and not is_board and price_position < 30.0):
                matched.append("tianliang_huicai")
                reasons.append(
                    "【天量回踩·观察】等待回调到大阳线一半位置再考虑底仓，"
                    "适合潜伏；爆发力弱于其他模式。"
                )

    if not matched:
        return None

    return {
        "symbol": code,
        "name": name,
        "market": "A",
        "price": round(price, 3),
        "pct_chg": round(pct_chg, 2),
        "amount_yi": round(amount_today / 1e8, 2),
        "trend": limit_type,
        "position_pct": round(price_position, 1),
        "board_count": board_count,
        "strategies": matched,
        "reasons": reasons,
    }


def _analyze_hk(code: str, name: str, price: float, vol_today: float,
                amount_today: float, pct_chg: float, hist: pd.DataFrame,
                wanted: set[str]) -> dict[str, Any] | None:
    if hist is None or len(hist) < 60 or "成交量" not in hist.columns:
        return None

    window = hist["成交量"].tail(SKY_VOL_WINDOW + 1)
    max_vol = float(window.iloc[:-1].max() or 0)
    if max_vol <= 0 or vol_today < max_vol * SKY_VOL_RATIO:
        return None

    ma60 = float(hist["收盘"].tail(60).mean())
    if ma60 <= 0:
        return None
    price_position = (price - ma60) / ma60 * 100.0
    if price_position > 50.0:
        return None

    matched: list[str] = []
    reasons: list[str] = []

    # HK has no symmetric limit; map onto fanbao/lianban as a "strong momentum" proxy.
    if ("tianliang_lianban" in wanted or "tianliang_fanbao" in wanted) \
            and pct_chg >= 8.0 and price_position < 40.0:
        # report under whichever of the two the user asked for first
        chosen = "tianliang_lianban" if "tianliang_lianban" in wanted else "tianliang_fanbao"
        matched.append(chosen)
        reasons.append(
            f"【港股天量信号】涨幅 {pct_chg:+.1f}%，港股无涨跌停限制，"
            f"波动更大；确认板块龙头地位后再操作，仓位控制更严格。"
        )

    if "tianliang_buzhi" in wanted and 0 <= pct_chg <= 5 and price_position < 30:
        matched.append("tianliang_buzhi")
        reasons.append("【港股天量不跌】跟踪2天小阴小阳不跌后可尾盘介入。")

    if not matched:
        return None

    return {
        "symbol": code,
        "name": name,
        "market": "HK",
        "price": round(price, 3),
        "pct_chg": round(pct_chg, 2),
        "amount_yi": round(amount_today / 1e8, 2),
        "trend": "大涨" if pct_chg >= 8 else "正常",
        "position_pct": round(price_position, 1),
        "board_count": 0,
        "strategies": matched,
        "reasons": reasons,
    }


def _get_a_spot() -> pd.DataFrame:
    """Fetch A-share spot snapshot, falling back across providers when the
    HF datacenter IP is blocked by one of them. Sina-first to match the rest
    of the app (Run Analysis / AI Analysis), which all use Sina successfully
    on Hugging Face Spaces.

    Returns a DataFrame with EM-style Chinese columns: 代码 名称 最新价
    成交量 成交额 涨跌幅
    """
    ak = _ak()
    errors: list[str] = []

    # 1) Sina aggregate (host: hq.sinajs.cn — same family as AI Analysis)
    try:
        df = _retry(lambda: ak.stock_zh_a_spot())
        if df is not None and not df.empty:
            if "代码" in df.columns:
                df["代码"] = df["代码"].astype(str).str.replace(
                    r"^(sh|sz|bj)", "", regex=True
                )
            logger.info("A spot via stock_zh_a_spot (Sina): %d rows", len(df))
            return df
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sina_aggregate: {exc}")
        logger.warning("stock_zh_a_spot (Sina) failed: %s", exc)

    # 2) East-Money aggregate (works locally; blocked on HF datacenter IPs)
    try:
        df = _retry(lambda: ak.stock_zh_a_spot_em())
        if df is not None and not df.empty:
            logger.info("A spot via stock_zh_a_spot_em: %d rows", len(df))
            return df
    except Exception as exc:  # noqa: BLE001
        errors.append(f"em_aggregate: {exc}")
        logger.warning("stock_zh_a_spot_em failed: %s", exc)

    # 3) East-Money per-board (last resort)
    try:
        parts = []
        for fn_name in ("stock_sh_a_spot_em", "stock_sz_a_spot_em"):
            try:
                fn = getattr(ak, fn_name)
                parts.append(_retry(fn))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{fn_name}: {exc}")
        parts = [p for p in parts if p is not None and not p.empty]
        if parts:
            df = pd.concat(parts, ignore_index=True)
            logger.info("A spot via per-board EM: %d rows", len(df))
            return df
    except Exception as exc:  # noqa: BLE001
        errors.append(f"em_per_board: {exc}")

    raise RuntimeError(
        "All A-share spot data sources failed. Tried: "
        + " | ".join(errors[:5])
    )


# ── Universe scan ──────────────────────────────────────────────────────

def _scan_market(market: str, wanted: set[str], top_n: int, max_workers: int,
                 progress: Callable[[float, str], None],
                 user_strategies: list[dict[str, Any]] | None = None,
                 ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ak = _ak()
    start = _start_date()
    end = _recent_trading_date()

    progress(0.05, f"Loading {market} spot snapshot…")
    if market == "A":
        df = _get_a_spot()
        df = df[~df["名称"].astype(str).str.contains("ST|退|N |C ", na=False)]
        df = df[~df["代码"].astype(str).str.startswith(("8", "4"))]  # exclude BJ
        fetch = _fetch_hist_a
        analyze = _analyze_a
    elif market == "HK":
        df = _retry(lambda: ak.stock_hk_spot_em())
        df = df[~df["名称"].astype(str).str.contains("退", na=False)]
        fetch = _fetch_hist_hk
        analyze = _analyze_hk
    else:
        raise ValueError(f"Unknown market: {market}")

    if "成交额" not in df.columns:
        # Sina sometimes uses different column names; fallback to volume
        sort_col = "成交量" if "成交量" in df.columns else df.columns[0]
    else:
        sort_col = "成交额"
    df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce").fillna(0)
    df = df.sort_values(sort_col, ascending=False).head(top_n).copy()

    todo: list[tuple[str, str, float, float, float, float]] = []
    for _, row in df.iterrows():
        code = str(row["代码"])
        name = str(row["名称"])
        price = _safe_float(row.get("最新价"))
        vol = _safe_float(row.get("成交量"))
        amt = _safe_float(row.get("成交额"))
        pct = _safe_float(row.get("涨跌幅"))
        if price > 0 and vol > 0 and amt > 0:
            todo.append((code, name, price, vol, amt, pct))

    total = len(todo)
    if total == 0:
        return [], {"market": market, "scanned": 0, "fetched_ok": 0, "matched": 0}

    progress(0.10, f"Fetching history for {total} {market} symbols…")
    hist_map: dict[str, pd.DataFrame | None] = {}
    done = 0
    fetched_ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch, s[0], start, end): s for s in todo}
        for fut in as_completed(futures):
            code, hist = fut.result()
            hist_map[code] = hist
            done += 1
            if hist is not None:
                fetched_ok += 1
            if done % 25 == 0 or done == total:
                # 10..70% of total progress allocated to history fetch
                progress(0.10 + 0.60 * done / total,
                         f"Fetched {done}/{total} {market} histories ({fetched_ok} ok)…")

    fail_rate = 1.0 - (fetched_ok / total)
    if fail_rate >= 0.5:
        logger.warning(
            "Selector %s scan: %d/%d history fetches failed (%.0f%%). "
            "Likely upstream rate-limit / IP block.",
            market, total - fetched_ok, total, fail_rate * 100,
        )

    progress(0.75, f"Analyzing {market} candidates…")
    results: list[dict[str, Any]] = []
    for code, name, price, vol, amt, pct in todo:
        hist = hist_map.get(code)
        if hist is None:
            continue
        # Only A-market evaluates user strategies (Phase 2 scope = A-share)
        if market == "A":
            res = analyze(code, name, price, vol, amt, pct, hist, wanted,
                          user_strategies=user_strategies)
        else:
            res = analyze(code, name, price, vol, amt, pct, hist, wanted)
        if res:
            results.append(res)
    progress(0.95, f"{market} scan found {len(results)} candidates.")
    stats = {
        "market": market,
        "scanned": total,
        "fetched_ok": fetched_ok,
        "matched": len(results),
    }
    return results, stats


# ── Async job registry ─────────────────────────────────────────────────

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_MAX_KEEP = 50  # cap in-memory history


def _set_job(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id, {})
        job.update(fields)
        _JOBS[job_id] = job


def _get_job(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        return dict(_JOBS[job_id]) if job_id in _JOBS else None


def _gc_jobs() -> None:
    with _JOBS_LOCK:
        if len(_JOBS) <= _MAX_KEEP:
            return
        # Keep newest by started_at
        ordered = sorted(_JOBS.items(),
                         key=lambda kv: kv[1].get("started_at", 0),
                         reverse=True)
        keep = dict(ordered[:_MAX_KEEP])
        _JOBS.clear()
        _JOBS.update(keep)


def _save_run(config: ToolkitConfig, payload: dict[str, Any]) -> str | None:
    try:
        out_dir = Path(config.output_dir) / "selector_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = out_dir / f"{ts}.json"
        fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        return str(fp)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist selector run")
        return None


def _run_job(job_id: str, config: ToolkitConfig, markets: list[str],
             strategies: list[str], top_n_a: int, top_n_hk: int,
             max_workers: int, user_id: int = 1) -> None:
    started_at = time.time()
    _set_job(job_id, status="running", progress=0.0, message="Starting…",
             started_at=started_at)

    # Split built-in strategy keys from user-strategy refs ("user:<id>")
    builtin_keys: set[str] = set()
    user_strategy_ids: list[str] = []
    for s in strategies:
        if s.startswith("user:"):
            user_strategy_ids.append(s[5:])
        elif s in SELECTOR_STRATEGIES:
            builtin_keys.add(s)

    # Load user strategy docs from storage (if any)
    user_strategies_docs: list[dict[str, Any]] = []
    if user_strategy_ids:
        try:
            from trade_storage import get_user_strategy, list_user_strategies  # local import to avoid cycles
            if user_strategy_ids == ["*"]:
                user_strategies_docs = [
                    s for s in list_user_strategies(config, user_id=user_id, enabled_only=True)
                ]
            else:
                for sid in user_strategy_ids:
                    s = get_user_strategy(config, sid, user_id=user_id)
                    if s and s.get("enabled", True):
                        user_strategies_docs.append(s)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load user strategies for job %s", job_id)

    def progress(p: float, msg: str) -> None:
        _set_job(job_id, progress=round(p, 3), message=msg)

    try:
        results: list[dict[str, Any]] = []
        scan_stats: list[dict[str, int]] = []
        n_markets = len(markets)
        for idx, market in enumerate(markets):
            offset = idx / max(n_markets, 1)
            span = 1.0 / max(n_markets, 1)

            def sub_progress(p: float, msg: str, _o=offset, _s=span) -> None:
                progress(_o + _s * p, msg)

            top_n = top_n_a if market == "A" else top_n_hk
            picks, stats = _scan_market(
                market, builtin_keys, top_n, max_workers, sub_progress,
                user_strategies=user_strategies_docs if market == "A" else None,
            )
            results.extend(picks)
            scan_stats.append(stats)

        # Stable sort: limit-up first, then larger turnover
        results.sort(
            key=lambda r: (0 if r["trend"] == "涨停" else 1, -r["amount_yi"])
        )
        finished_at = time.time()

        # Aggregate fetch health to surface upstream rate-limiting
        total_scanned = sum(s["scanned"] for s in scan_stats)
        total_ok = sum(s["fetched_ok"] for s in scan_stats)
        fetch_success_rate = (total_ok / total_scanned) if total_scanned else 1.0

        warnings: list[str] = []
        if total_scanned > 0 and fetch_success_rate < 0.5:
            warnings.append(
                f"Only {total_ok}/{total_scanned} history fetches succeeded "
                f"({fetch_success_rate:.0%}). Upstream data source likely "
                f"rate-limited this server's IP. Picks may be incomplete."
            )

        payload = {
            "job_id": job_id,
            "markets": markets,
            "strategies": strategies,
            "top_n_a": top_n_a,
            "top_n_hk": top_n_hk,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": round(finished_at - started_at, 2),
            "trade_date": _recent_trading_date(),
            "scan_stats": scan_stats,
            "fetch_success_rate": round(fetch_success_rate, 3),
            "warnings": warnings,
            "picks": results,
        }
        saved = _save_run(config, payload)
        _set_job(job_id, status="done", progress=1.0,
                 message=f"Found {len(results)} picks.",
                 finished_at=finished_at, result=payload, saved_path=saved)
        _gc_jobs()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Selector job %s failed", job_id)
        _set_job(job_id, status="failed", finished_at=time.time(),
                 error=str(exc), traceback=traceback.format_exc())


# ── Public API ─────────────────────────────────────────────────────────

def submit_job(config: ToolkitConfig, *, markets: list[str],
               strategies: list[str], top_n_a: int = DEFAULT_TOP_ACTIVE_A,
               top_n_hk: int = DEFAULT_TOP_ACTIVE_HK,
               max_workers: int = DEFAULT_MAX_WORKERS,
               user_id: int = 1) -> str:
    markets = [m for m in markets if m in ("A", "HK")]
    if not markets:
        raise ValueError("At least one market required (A or HK).")
    # Accept built-in strategy keys ("tianliang_*") and user refs ("user:<id>")
    cleaned: list[str] = []
    for s in strategies:
        if s in SELECTOR_STRATEGIES:
            cleaned.append(s)
        elif s.startswith("user:") and len(s) > 5:
            cleaned.append(s)
    strategies = cleaned
    if not strategies:
        raise ValueError("At least one strategy required.")
    top_n_a = max(20, min(int(top_n_a), 1000))
    top_n_hk = max(20, min(int(top_n_hk), 500))
    max_workers = max(1, min(int(max_workers), 16))

    job_id = uuid.uuid4().hex[:10]
    _set_job(job_id, status="queued", progress=0.0,
             message="Queued", started_at=None, finished_at=None,
             markets=markets, strategies=strategies)

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, config, markets, strategies, top_n_a, top_n_hk,
              max_workers, user_id),
        daemon=True, name=f"selector-{job_id}",
    )
    thread.start()
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    return _get_job(job_id)
