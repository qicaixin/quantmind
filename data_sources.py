from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Literal

import pandas as pd

from config import ToolkitConfig

DataSource = Literal["qlib", "akshare", "auto"]
_QLIB_READY = False


def normalize_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if cleaned.startswith(("SH", "SZ")):
        return cleaned
    if cleaned.isdigit() and len(cleaned) == 6:
        return f"SH{cleaned}" if cleaned.startswith(("5", "6", "9")) else f"SZ{cleaned}"
    raise ValueError(f"Unsupported symbol format: {symbol}")


def akshare_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    return normalized[:2].lower() + normalized[2:]


def qlib_available(config: ToolkitConfig) -> bool:
    return importlib.util.find_spec("qlib") is not None and config.qlib_data_path.exists()


def init_qlib(config: ToolkitConfig) -> None:
    global _QLIB_READY
    if _QLIB_READY:
        return
    import qlib
    from qlib.config import REG_CN

    qlib.init(provider_uri=str(config.qlib_data_path), region=REG_CN)
    _QLIB_READY = True


def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    expected = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after load: {missing}")

    result = df[expected].copy()
    result["timestamps"] = pd.to_datetime(result["timestamps"])
    for col in expected[1:]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = result.dropna().sort_values("timestamps").reset_index(drop=True)
    return result


def load_from_qlib(config: ToolkitConfig, symbol: str, start: str, end: str) -> pd.DataFrame:
    init_qlib(config)
    from qlib.data import D

    instrument = normalize_symbol(symbol)
    raw = D.features(
        instruments=[instrument],
        fields=["$open", "$high", "$low", "$close", "$volume"],
        start_time=start,
        end_time=end,
    )
    if raw.empty:
        raise ValueError(f"No Qlib data found for {instrument}.")

    df = raw.reset_index().rename(
        columns={
            "datetime": "timestamps",
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
        }
    )
    df["amount"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0 * df["volume"]
    return standardize_frame(df)


def _is_etf_code(code: str) -> bool:
    """Return True for fund/ETF codes (SH: 5xxxxx, SZ: 15xxxx/16xxxx)."""
    return code.startswith("5") or code.startswith("15") or code.startswith("16")


def load_from_akshare(symbol: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ["NO_PROXY"] = "*"

    normalized = normalize_symbol(symbol)
    code = normalized[2:]
    ak_sym = akshare_symbol(symbol)  # e.g. sh515030, sh601169

    if _is_etf_code(code):
        # ETF/fund: use Sina fund history API (stock_zh_a_daily doesn't support ETFs)
        raw = ak.fund_etf_hist_sina(symbol=ak_sym)
        df = raw.rename(columns={"date": "timestamps"})
    else:
        raw = ak.stock_zh_a_daily(symbol=ak_sym, adjust="qfq")
        df = raw.reset_index().rename(columns={"date": "timestamps"})

    df = df.rename(columns={
        "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume", "amount": "amount",
    })
    if "amount" not in df.columns:
        df["amount"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0 * df["volume"]

    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df[(df["timestamps"] >= pd.Timestamp(start)) & (df["timestamps"] <= pd.Timestamp(end))]
    return standardize_frame(df)


def load_history(
    config: ToolkitConfig,
    symbol: str,
    start: str,
    end: str,
    source: DataSource = "auto",
) -> tuple[pd.DataFrame, str]:
    if source in ("qlib", "auto") and qlib_available(config):
        try:
            return load_from_qlib(config, symbol, start, end), "qlib"
        except Exception:
            if source == "qlib":
                raise

    return load_from_akshare(symbol, start, end), "akshare"


_name_cache: dict[str, str] = {}


def get_stock_name(symbol: str) -> str:
    """Return the short Chinese name of an A-share. Falls back to the symbol code on error."""
    try:
        normalized = normalize_symbol(symbol)
    except ValueError:
        return symbol
    if normalized in _name_cache:
        return _name_cache[normalized]
    try:
        import akshare as ak
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ["NO_PROXY"] = "*"
        code = normalized[2:]
        info = ak.stock_individual_info_em(symbol=code)
        if "item" in info.columns and "value" in info.columns:
            for key in ("股票简称", "名称", "简称"):
                rows = info[info["item"] == key]
                if not rows.empty:
                    name = str(rows.iloc[0]["value"])
                    _name_cache[normalized] = name
                    return name
    except Exception:
        pass
    _name_cache[normalized] = normalized
    return normalized


def get_last_close(symbol: str, config: ToolkitConfig) -> float | None:
    """Return the most recent closing price. Returns None on failure."""
    from datetime import date, timedelta
    try:
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=30)).isoformat()
        df, _ = load_history(config, normalize_symbol(symbol), start, end)
        if not df.empty:
            return float(df.iloc[-1]["close"])
    except Exception:
        pass
    return None


_quote_cache: dict[str, dict] = {}
_QUOTE_TTL = 15.0  # seconds


def _get_sina_prefix(code: str) -> str:
    """Map 6-digit A-share code to Sina exchange prefix (sh/sz/bj)."""
    if code.startswith(("5", "6", "9")):
        return "sh"
    if code.startswith(("8", "4")):
        return "bj"
    return "sz"


def _sina_daily(ak_module, sina_sym: str, code: str,
                start_date: str, end_date: str, adjust: str = "qfq"):
    """Fetch daily OHLCV; uses fund_etf_hist_sina for ETF/fund codes."""
    if _is_etf_code(code):
        df = ak_module.fund_etf_hist_sina(symbol=sina_sym)
        if df is None or df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
        return df[mask].reset_index(drop=True)
    return ak_module.stock_zh_a_daily(symbol=sina_sym, start_date=start_date,
                                      end_date=end_date, adjust=adjust)


def get_realtime_quote(symbol: str) -> dict | None:
    """Return intraday real-time quote using Sina minute data (fast & reliable).
    Falls back to latest daily data when market is closed."""
    import time
    from datetime import date, timedelta
    try:
        normalized = normalize_symbol(symbol)
    except ValueError:
        return None

    cached = _quote_cache.get(normalized)
    if cached and time.time() - cached.get("_ts", 0) < _QUOTE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    code = normalized[2:]
    try:
        import akshare as ak
        import pandas as pd
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ["NO_PROXY"] = "*"

        prefix = _get_sina_prefix(code)
        sina_sym = f"{prefix}{code}"

        # Get prev close from daily data
        start_d = (date.today() - timedelta(days=10)).strftime("%Y%m%d")
        end_d = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
        df_daily = _sina_daily(ak, sina_sym, code, start_d, end_d)
        prev_close = float(df_daily.iloc[-1]["close"]) if df_daily is not None and not df_daily.empty else None

        # Get intraday minute data
        df_min = ak.stock_zh_a_minute(symbol=sina_sym, period="1", adjust="qfq")

        today_str = date.today().isoformat()[:10]
        if df_min is not None and not df_min.empty:
            df_min["volume"] = pd.to_numeric(df_min["volume"], errors="coerce")
            df_min["amount"] = pd.to_numeric(df_min["amount"], errors="coerce")
            today_rows = df_min[df_min["day"].astype(str).str.startswith(today_str)].copy()
        else:
            today_rows = pd.DataFrame()

        if not today_rows.empty:
            # Compute price from amount/volume since OHLC may be NaN for current day
            today_rows["price_calc"] = today_rows["amount"] / today_rows["volume"].replace(0, float("nan"))
            last_row = today_rows.iloc[-1]
            last_price = round(float(today_rows["price_calc"].dropna().iloc[-1]), 3) if not today_rows["price_calc"].dropna().empty else None
            open_price = round(float(today_rows["price_calc"].dropna().iloc[0]), 3) if last_price else None
            high_price = round(float(today_rows["price_calc"].max()), 3) if last_price else None
            low_price = round(float(today_rows["price_calc"].min()), 3) if last_price else None
            total_vol = float(today_rows["volume"].sum())
            total_amt = float(today_rows["amount"].sum())
            last_time = str(last_row["day"])
            is_realtime = True
        else:
            # Market closed — use latest daily row
            df_all = _sina_daily(ak, sina_sym, code, start_d, date.today().strftime("%Y%m%d"))
            if df_all is None or df_all.empty:
                return None
            row = df_all.iloc[-1]
            last_price = float(row["close"])
            open_price = float(row["open"])
            high_price = float(row["high"])
            low_price = float(row["low"])
            total_vol = float(row["volume"])
            total_amt = float(row.get("amount", 0))
            last_time = str(row["date"])
            if prev_close is None:
                prev_close = float(df_all.iloc[-2]["close"]) if len(df_all) >= 2 else None
            is_realtime = False

        change_amount = round(last_price - prev_close, 3) if last_price and prev_close else None
        change_pct = round(change_amount / prev_close * 100, 2) if change_amount and prev_close else None

        result = {
            "symbol": normalized,
            "name": get_stock_name(normalized),
            "date": last_time,
            "last_price": last_price,
            "change_pct": change_pct,
            "change_amount": change_amount,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "prev_close": prev_close,
            "volume": total_vol,
            "amount": total_amt,
            "is_realtime": is_realtime,
        }
        _quote_cache[normalized] = {**result, "_ts": time.time()}
        return result
    except Exception as e:
        return None


_kline_cache: dict[str, dict] = {}
_KLINE_TTL = 30.0


def get_kline_data(symbol: str, period: str = "daily", days: int = 180) -> dict | None:
    """Return OHLCV series for candlestick chart.

    period: 'daily'|'weekly'|'monthly'|'5min'|'15min'|'30min'|'60min'
    days: calendar days to look back (for daily/weekly/monthly)
    """
    import time
    from datetime import date, timedelta

    try:
        normalized = normalize_symbol(symbol)
    except ValueError:
        return None

    cache_key = f"{normalized}:{period}:{days}"
    cached = _kline_cache.get(cache_key)
    if cached and time.time() - cached.get("_ts", 0) < _KLINE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    code = normalized[2:]

    try:
        import akshare as ak
        import pandas as pd
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ["NO_PROXY"] = "*"

        prefix = _get_sina_prefix(code)
        sina_sym = f"{prefix}{code}"
        today = date.today()

        if period in ("5min", "15min", "30min", "60min"):
            min_period = period.replace("min", "")
            df = ak.stock_zh_a_minute(symbol=sina_sym, period=min_period, adjust="qfq")
            if df is None or df.empty:
                return None
            df = df.copy()
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
            # For today's intraday: OHLC may be NaN, compute from amount/volume
            for col in ("open", "high", "low", "close"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            mask_nan = df["close"].isna()
            if mask_nan.any():
                price = df["amount"] / df["volume"].replace(0, float("nan"))
                df.loc[mask_nan, "open"] = price[mask_nan]
                df.loc[mask_nan, "high"] = price[mask_nan]
                df.loc[mask_nan, "low"] = price[mask_nan]
                df.loc[mask_nan, "close"] = price[mask_nan]
            df = df.dropna(subset=["close"])
            # Keep last 2 trading days for intraday
            df["day"] = df["day"].astype(str)
            records = [
                {
                    "t": row["day"],
                    "o": round(float(row["open"]), 3),
                    "h": round(float(row["high"]), 3),
                    "l": round(float(row["low"]), 3),
                    "c": round(float(row["close"]), 3),
                    "v": int(row["volume"]),
                }
                for _, row in df.iterrows()
            ]
        else:
            # Daily / weekly / monthly
            start_d = (today - timedelta(days=days)).strftime("%Y%m%d")
            end_d = today.strftime("%Y%m%d")
            df = _sina_daily(ak, sina_sym, code, start_d, end_d)
            if df is None or df.empty:
                return None
            if period == "weekly":
                df["date"] = pd.to_datetime(df["date"])
                df = df.resample("W-FRI", on="date").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                ).dropna().reset_index()
            elif period == "monthly":
                df["date"] = pd.to_datetime(df["date"])
                df = df.resample("ME", on="date").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                ).dropna().reset_index()
            records = [
                {
                    "t": str(row["date"])[:10],
                    "o": round(float(row["open"]), 3),
                    "h": round(float(row["high"]), 3),
                    "l": round(float(row["low"]), 3),
                    "c": round(float(row["close"]), 3),
                    "v": int(row["volume"]),
                }
                for _, row in df.iterrows()
            ]

        result = {"symbol": normalized, "period": period, "bars": records}
        _kline_cache[cache_key] = {**result, "_ts": time.time()}
        return result
    except Exception:
        return None


_t0_cache: dict[str, dict] = {}
_T0_TTL = 30.0


def get_t0_indicators(symbol: str, force: bool = False) -> dict | None:
    """Compute intraday T+0 technical indicators from minute data.

    Returns VWAP, MA5/10/20, RSI(14), MACD(12,26,9), Bollinger(20,2),
    volume trend, and a composite BUY/HOLD/SELL signal.
    """
    import time
    from datetime import date, timedelta

    try:
        normalized = normalize_symbol(symbol)
    except ValueError:
        return None

    if force:
        _t0_cache.pop(normalized, None)
    cached = _t0_cache.get(normalized)
    if cached and time.time() - cached.get("_ts", 0) < _T0_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    code = normalized[2:]
    try:
        import akshare as ak
        import numpy as np
        os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
        os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
        os.environ["NO_PROXY"] = "*"

        prefix = _get_sina_prefix(code)
        sina_sym = f"{prefix}{code}"
        today = date.today().isoformat()[:10]

        # Get 1-min data (last ~2 trading days for MACD warmup)
        df = ak.stock_zh_a_minute(symbol=sina_sym, period="1", adjust="qfq")
        if df is None or df.empty:
            return None

        df = df.copy()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill today's NaN OHLC with amount/volume
        mask = df["close"].isna()
        if mask.any():
            price = df["amount"] / df["volume"].replace(0, float("nan"))
            for col in ("open", "high", "low", "close"):
                df.loc[mask, col] = price[mask]

        df = df.dropna(subset=["close"]).reset_index(drop=True)
        day_str = df["day"].astype(str).str[:10]
        today_df = df[day_str == today].copy()
        session_date = today
        is_current_session = True

        if today_df.empty:
            available_sessions = sorted(d for d in day_str.dropna().unique() if d)
            if not available_sessions:
                return None
            session_date = available_sessions[-1]
            is_current_session = False
            today_df = df[day_str == session_date].copy()

        if today_df.empty:
            return None

        closes = today_df["close"].values.astype(float)
        vols   = today_df["volume"].values.astype(float)
        amts   = today_df["amount"].values.astype(float)

        def r2(x):
            return round(float(x), 3) if x is not None and x == x else None

        # VWAP
        vwap = float(amts.sum() / vols.sum()) if vols.sum() > 0 else closes[-1]
        last_price = closes[-1]
        vwap_dev = round((last_price - vwap) / vwap * 100, 2) if vwap else None

        # Intraday MAs
        def sma(arr, n):
            return float(arr[-n:].mean()) if len(arr) >= n else None

        ma5  = sma(closes, 5)
        ma10 = sma(closes, 10)
        ma20 = sma(closes, 20)

        # RSI(14)
        def rsi(arr, n=14):
            if len(arr) < n + 1:
                return None
            deltas = np.diff(arr[-(n+1):])
            gains  = np.where(deltas > 0, deltas, 0).mean()
            losses = np.where(deltas < 0, -deltas, 0).mean()
            return round(100 - 100 / (1 + gains / losses), 1) if losses > 0 else 100.0

        rsi_val = rsi(closes)

        # MACD(12, 26, 9) on all-day closes for warmup
        all_closes = df["close"].values.astype(float)

        def ema(arr, n):
            k = 2 / (n + 1)
            result = np.zeros_like(arr)
            result[0] = arr[0]
            for i in range(1, len(arr)):
                result[i] = arr[i] * k + result[i - 1] * (1 - k)
            return result

        if len(all_closes) >= 26:
            ema12 = ema(all_closes, 12)
            ema26 = ema(all_closes, 26)
            dif   = ema12 - ema26
            dea   = ema(dif, 9)
            hist  = (dif - dea) * 2
            macd_val  = r2(dif[-1])
            macd_sig  = r2(dea[-1])
            macd_hist = r2(hist[-1])
        else:
            macd_val = macd_sig = macd_hist = None

        # Bollinger Bands(20, 2)
        def bollinger(arr, n=20, k=2):
            if len(arr) < n:
                return None, None, None
            mid = arr[-n:].mean()
            std = arr[-n:].std()
            return round(mid + k * std, 3), round(mid, 3), round(mid - k * std, 3)

        bb_upper, bb_mid, bb_lower = bollinger(closes)

        # Volume trend: last 5 vs prev 5
        vol_trend = "—"
        if len(vols) >= 10:
            last5 = vols[-5:].mean()
            prev5 = vols[-10:-5].mean()
            ratio = last5 / prev5 if prev5 > 0 else 1
            vol_trend = f"放量 {ratio:.1f}x" if ratio > 1.2 else ("缩量 {:.1f}x".format(ratio) if ratio < 0.8 else "平稳")

        # Vol ratio vs previous days (5-day avg)
        past_df = df[day_str < session_date]
        vol_ratio = "—"
        if not past_df.empty:
            daily_vols = past_df.groupby(past_df["day"].astype(str).str[:10])["volume"].sum()
            avg5 = float(daily_vols.tail(5).mean()) if len(daily_vols) >= 1 else None
            today_vol = float(vols.sum())
            # Scale by fraction of day elapsed (~4.5h trading day = 270 min)
            elapsed_min = len(today_df)
            projected = today_vol / elapsed_min * 270 if elapsed_min > 0 else today_vol
            if avg5 and avg5 > 0:
                vol_ratio = round(projected / avg5, 2)

        # Prev close & open
        prev_df = df[day_str < session_date]
        prev_close = r2(prev_df["close"].iloc[-1]) if not prev_df.empty else None
        open_price = r2(today_df["close"].iloc[0])
        high_today = r2(today_df["close"].max())
        low_today  = r2(today_df["close"].min())

        # Composite signal
        score = 0
        reasons = []
        if vwap_dev is not None:
            if vwap_dev < -0.5:
                score += 1; reasons.append("价格低于VWAP")
            elif vwap_dev > 0.5:
                score -= 1; reasons.append("价格高于VWAP")
        if rsi_val is not None:
            if rsi_val < 35:
                score += 2; reasons.append(f"RSI超卖({rsi_val})")
            elif rsi_val < 45:
                score += 1; reasons.append(f"RSI偏低({rsi_val})")
            elif rsi_val > 65:
                score -= 2; reasons.append(f"RSI超买({rsi_val})")
            elif rsi_val > 55:
                score -= 1; reasons.append(f"RSI偏高({rsi_val})")
        if macd_hist is not None:
            if macd_hist > 0:
                score += 1; reasons.append("MACD柱>0")
            else:
                score -= 1; reasons.append("MACD柱<0")
        if bb_lower is not None and last_price < bb_lower:
            score += 1; reasons.append("触及布林下轨")
        if bb_upper is not None and last_price > bb_upper:
            score -= 1; reasons.append("触及布林上轨")
        if not is_current_session:
            reasons.append(f"使用最近交易日数据({session_date})")

        if score >= 2:
            signal = "🟢 买入 (T+0低吸)"
        elif score <= -2:
            signal = "🔴 卖出 (T+0高抛)"
        else:
            signal = "🟡 观望"

        result = {
            "symbol": normalized,
            "session_date": session_date,
            "is_current_session": is_current_session,
            "last_time": str(today_df["day"].iloc[-1]),
            "last_price": r2(last_price),
            "prev_close": prev_close,
            "open_price": open_price,
            "high": high_today,
            "low": low_today,
            "vwap": r2(vwap),
            "vwap_dev_pct": vwap_dev,
            "ma5": r2(ma5), "ma10": r2(ma10), "ma20": r2(ma20),
            "rsi": rsi_val,
            "macd": macd_val, "macd_signal": macd_sig, "macd_hist": macd_hist,
            "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
            "vol_trend": vol_trend,
            "vol_ratio": vol_ratio,
            "signal": signal,
            "signal_reasons": reasons,
        }
        _t0_cache[normalized] = {**result, "_ts": time.time()}
        return result
    except Exception:
        return None



def future_trading_days(
    config: ToolkitConfig,
    last_timestamp: pd.Timestamp,
    periods: int,
) -> pd.Series:
    if qlib_available(config):
        try:
            init_qlib(config)
            from qlib.data import D

            cal = pd.DatetimeIndex(D.calendar())
            future = cal[cal > last_timestamp][:periods]
            if len(future) == periods:
                return pd.Series(future)
        except Exception:
            pass

    return pd.Series(pd.bdate_range(last_timestamp + pd.Timedelta(days=1), periods=periods))
