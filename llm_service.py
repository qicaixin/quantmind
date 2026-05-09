"""LLM service — calls any OpenAI-compatible chat API for T+0 analysis."""
from __future__ import annotations

import json
import os
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "data" / "llm_config.json"
_DEFAULT_CONFIG = {
    "provider": "Ollama",
    "base_url": "http://localhost:11434",
    "api_key": "",
    "model": "qwen2.5:7b",
    "max_tokens": 800,
    "temperature": 0.3,
}

# Capture system proxy at import time (before akshare can clear env vars)
_SYSTEM_PROXY = {
    k: v for k, v in os.environ.items()
    if k.lower() in ("http_proxy", "https_proxy", "all_proxy")
}


def _is_local_url(url: str) -> bool:
    return any(h in url for h in ("localhost", "127.0.0.1", "::1", "0.0.0.0"))


def _get_proxy_url() -> str | None:
    """Read Windows IE/WinINet proxy setting from registry."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        )
        enabled = winreg.QueryValueEx(key, "ProxyEnable")[0]
        server  = winreg.QueryValueEx(key, "ProxyServer")[0]
        if enabled and server:
            return server if "://" in server else f"http://{server}"
    except Exception:
        pass
    # Fall back to env vars captured before akshare clears them
    for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        v = _SYSTEM_PROXY.get(k)
        if v:
            return v
    return None


def load_llm_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return {**_DEFAULT_CONFIG, **json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return dict(_DEFAULT_CONFIG)


def save_llm_config(config: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged = {**_DEFAULT_CONFIG, **config}
    _CONFIG_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def call_llm(system_prompt: str, user_prompt: str, *, config_override: dict | None = None) -> str:
    """Call the configured LLM and return the assistant message text.
    
    If config_override is provided, it is merged on top of the global config
    (per-user settings take precedence).
    """
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    config = load_llm_config()
    if config_override:
        # Only merge non-empty values from user config
        for k, v in config_override.items():
            if v not in (None, ""):
                config[k] = v
    api_key     = config.get("api_key", "").strip()
    base_url    = config.get("base_url", "").rstrip("/")
    model       = config.get("model", "qwen2.5:7b")
    max_tokens  = int(config.get("max_tokens", 800))
    temperature = float(config.get("temperature", 0.3))

    if not base_url:
        return "⚠️ 未配置 API 地址，请点击右上角「⚙ 设置」填写。"

    is_local = _is_local_url(base_url)
    if not is_local and not api_key:
        return "⚠️ 外部 API 需要填写 API Key，请点击右上角「⚙ 设置」。"

    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    # Proxy config: local URLs bypass proxy; external use system proxy
    if is_local:
        proxies = {"http": None, "https": None}
        verify  = True
    else:
        proxy_url = _get_proxy_url()
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else {}
        verify  = False  # Corporate MITM proxy may use its own CA

    try:
        resp = requests.post(url, headers=headers, json=payload,
                             proxies=proxies, verify=verify, timeout=60)
        resp.encoding = "utf-8"
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        if resp.status_code == 401:
            return "❌ API Key 无效或已过期，请在设置中更新。"
        if resp.status_code == 404:
            return f"❌ 模型 '{model}' 未找到，请检查设置中的模型名称。"
        return f"❌ LLM API 错误 {resp.status_code}：{resp.text[:300]}"

    except requests.exceptions.ConnectionError as e:
        err = str(e)
        if is_local:
            return (
                "❌ 无法连接本地 Ollama。\n\n"
                "请确认 Ollama 已启动（系统托盘图标）：\n"
                f"  终端运行：ollama serve\n"
                f"  拉取模型：ollama pull {model}"
            )
        return (
            "❌ 无法连接 API（可能被代理屏蔽）。\n\n"
            "建议：\n"
            "• 改用本地 Ollama（无需外网）：https://ollama.com\n"
            "• 或在设置中尝试 OpenAI / 其他可访问的服务"
        )
    except requests.exceptions.Timeout:
        return "❌ 请求超时（60s），请检查网络或 API 服务状态。"
    except Exception as e:
        return f"❌ 请求失败：{e}"


_T0_SYSTEM = """你是一位专业的 A 股量化交易员，擅长日内做 T（T+0 高抛低吸）。
用户会提供某只股票的分钟级行情数据和技术指标，请给出简明的做T建议。
如果数据不是当日交易数据，需要明确提示只能作为最近交易日复盘参考。
输出格式要求（Markdown）：
1. **当前趋势**：（多/空/震荡，一句话说明理由）
2. **做T建议**：（具体操作：低吸价位区间 / 高抛价位区间 / 暂不操作）
3. **关键价位**：（支撑位、压力位）
4. **风险提示**：（简短1-2条）
回答简洁，不超过250字。"""


def analyze_t0(symbol: str, indicators: dict, *, llm_config: dict | None = None) -> str:
    ind = indicators
    user_prompt = f"""
股票：{symbol}
数据日期：{ind.get('session_date', '—')}　是否当日数据：{'是' if ind.get('is_current_session') is not False else '否'}
数据时间：{ind.get('last_time', '—')}
当前价：{ind.get('last_price', '—')}　前收：{ind.get('prev_close', '—')}
该日开：{ind.get('open_price', '—')}　该日高：{ind.get('high', '—')}　该日低：{ind.get('low', '—')}
VWAP：{ind.get('vwap', '—')}　价格/VWAP偏差：{ind.get('vwap_dev_pct', '—')}%
MA5：{ind.get('ma5', '—')}　MA10：{ind.get('ma10', '—')}　MA20：{ind.get('ma20', '—')}
RSI(14)：{ind.get('rsi', '—')}
MACD：{ind.get('macd', '—')}　Signal：{ind.get('macd_signal', '—')}　Histogram：{ind.get('macd_hist', '—')}
布林上轨：{ind.get('bb_upper', '—')}　中轨：{ind.get('bb_mid', '—')}　下轨：{ind.get('bb_lower', '—')}
成交量趋势（近5分钟 vs 前5分钟）：{ind.get('vol_trend', '—')}
该日量比（该日量/5日均量）：{ind.get('vol_ratio', '—')}
综合信号：{ind.get('signal', '—')}
"""
    return call_llm(_T0_SYSTEM, user_prompt.strip(), config_override=llm_config)

