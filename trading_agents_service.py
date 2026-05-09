"""TradingAgents integration service — async background jobs with SQLite persistence."""
from __future__ import annotations

import os
import threading
import uuid
from contextlib import contextmanager
from datetime import date
from typing import TYPE_CHECKING

from config import ToolkitConfig
from llm_service import (
    _get_proxy_url,
    api_base_from_endpoint_url,
    chat_completions_url,
    endpoint_url_candidates,
    load_llm_config,
    resolve_llm_model,
)

if TYPE_CHECKING:
    pass

_lock = threading.Lock()
# Cache TradingAgentsGraph instances keyed by (provider, model) so we reuse them
_ta_instances: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------

@contextmanager
def _proxy_env():
    """Temporarily restore system proxy env vars for the duration of a block.

    akshare clears HTTP_PROXY / HTTPS_PROXY on import.  This context manager
    re-injects them so yfinance (used by TradingAgents) can reach Yahoo Finance
    through a corporate proxy.
    """
    proxy_url = _get_proxy_url()
    if not proxy_url:
        yield
        return

    original = {
        k: os.environ.get(k)
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    }
    try:
        os.environ["HTTP_PROXY"]  = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        os.environ["http_proxy"]  = proxy_url
        os.environ["https_proxy"] = proxy_url
        yield
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _check_llm_reachable(llm_cfg: dict) -> str | None:
    """Return an error string if the configured LLM is not reachable/usable, else None."""
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    provider = llm_cfg.get("provider", "Ollama").lower()
    base_url = llm_cfg.get("base_url", "").rstrip("/") or "https://api.openai.com"
    api_key  = llm_cfg.get("api_key", "").strip()
    model    = resolve_llm_model(llm_cfg)

    is_local = any(h in base_url for h in ("localhost", "127.0.0.1", "::1"))

    if is_local or provider == "ollama":
        url = base_url or "http://localhost:11434"
        try:
            requests.get(url, timeout=4, proxies={"http": None, "https": None})
            return None
        except requests.exceptions.ConnectionError:
            return (
                f"Ollama is not running at {url}. "
                "Please start it: ollama serve"
            )
        except requests.exceptions.Timeout:
            return f"Ollama did not respond at {url} (timeout 4s)."

    # For remote APIs: check key present then do a quick models/chat probe
    if not api_key:
        return (
            f"No API key configured for '{llm_cfg.get('provider')}'. "
            "Add one in ⚙ Settings."
        )

    # Quick probe: POST a minimal chat request (1 token max)
    proxy_url = _get_proxy_url()
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else {}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if not model:
        return "Could not auto-detect an LLM model. Please fill the model name in ⚙ Settings."

    llm_cfg["model"] = model
    probe_body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    try:
        last_probe_url = chat_completions_url(base_url)
        resp = None
        for probe_url in endpoint_url_candidates(base_url, "chat/completions"):
            last_probe_url = probe_url
            resp = requests.post(probe_url, json=probe_body, headers=headers,
                                 proxies=proxies, verify=False, timeout=20)
            if resp.status_code != 404:
                break
        if resp is None:
            return "LLM pre-flight failed: request was not sent."
        if resp.status_code in (200, 201):
            llm_cfg["base_url"] = api_base_from_endpoint_url(last_probe_url, "chat/completions")
            return None
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        error = data.get("error")
        if isinstance(error, dict):
            err_msg = str(error.get("message") or resp.text[:200])
            code = str(error.get("code") or "")
        elif isinstance(error, str):
            err_msg = error
            code = ""
        else:
            err_msg = resp.text[:200]
            code = ""
        if resp.status_code == 429 or code == "insufficient_quota":
            return f"OpenAI quota exceeded — please check your billing at platform.openai.com."
        if resp.status_code == 401:
            return f"API key rejected (401). Please update it in ⚙ Settings."
        return f"LLM API error {resp.status_code}: {err_msg}"
    except requests.exceptions.Timeout:
        return f"LLM API timed out at {last_probe_url}. Check network/proxy."
    except requests.exceptions.ConnectionError as e:
        return f"Cannot reach LLM API at {base_url}. Check URL in ⚙ Settings."
    except Exception as e:
        return f"LLM pre-flight failed: {e}"


# ---------------------------------------------------------------------------
# Ticker mapping: bare A-stock code → yfinance format
# ---------------------------------------------------------------------------

def to_yfinance_ticker(symbol: str) -> str:
    """Map an A-stock code to yfinance ticker format.

    Examples:
        601169  → 601169.SS   (Shanghai)
        000651  → 000651.SZ   (Shenzhen)
        SH601169 → 601169.SS
        SZ000651 → 000651.SZ
    """
    code = symbol.upper().strip()
    if code.startswith("SH"):
        return code[2:] + ".SS"
    if code.startswith("SZ"):
        return code[2:] + ".SZ"
    # Already formatted
    if code.endswith(".SS") or code.endswith(".SZ"):
        return code
    # Auto-detect by first digit
    if len(code) == 6 and code.isdigit():
        if code[0] == "6":
            return code + ".SS"
        if code[0] in ("0", "1", "2", "3"):
            return code + ".SZ"
        # 4xxxxx, 5xxxxx, 8xxxxx, 9xxxxx → treat as SH
        return code + ".SS"
    # Non-Chinese ticker (e.g. AAPL) — pass through unchanged
    return symbol


# ---------------------------------------------------------------------------
# Config mapping: Kronos llm_config.json → TradingAgents config dict
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "ollama":    "ollama",
    "openai":    "openai",
    "deepseek":  "openai",   # OpenAI-compatible endpoint
    "qwen":      "openai",   # DashScope OpenAI-compatible
    "azure":     "azure",
    "anthropic": "anthropic",
    "google":    "google",
    "custom":    "openai",
}

_DEFAULT_BACKENDS = {
    "deepseek": "https://api.deepseek.com/v1",
    "qwen":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


def _patch_tradingagents_custom_openai_chat_completions() -> None:
    """Force TradingAgents custom OpenAI-compatible URLs to use chat completions.

    The upstream TradingAgents OpenAI client enables OpenAI's Responses API for
    provider="openai". Custom gateways often expose only /chat/completions, so
    route non-openai.com backends through ChatOpenAI with use_responses_api=False.
    """
    try:
        import tradingagents.graph.trading_graph as trading_graph
        from langchain_openai import ChatOpenAI
        from tradingagents.llm_clients.base_client import normalize_content
    except Exception:
        return

    if getattr(trading_graph, "_quantmind_chat_completions_patch", False):
        return

    original_create_llm_client = trading_graph.create_llm_client

    class NormalizedChatCompletionsOpenAI(ChatOpenAI):
        def invoke(self, input, config=None, **kwargs):
            return normalize_content(super().invoke(input, config, **kwargs))

    class CustomOpenAICompatibleClient:
        def __init__(self, model: str, base_url: str, **kwargs):
            self.model = model
            self.base_url = base_url
            self.kwargs = kwargs

        def get_llm(self):
            llm_kwargs = {
                "model": self.model,
                "base_url": self.base_url,
                "use_responses_api": False,
            }
            api_key = self.kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if api_key:
                llm_kwargs["api_key"] = api_key
            for key in ("timeout", "max_retries", "callbacks", "http_client", "http_async_client"):
                if key in self.kwargs:
                    llm_kwargs[key] = self.kwargs[key]
            return NormalizedChatCompletionsOpenAI(**llm_kwargs)

    def create_llm_client(provider: str, model: str, base_url: str | None = None, **kwargs):
        if (
            provider.lower() == "openai"
            and base_url
            and "api.openai.com" not in base_url.lower()
        ):
            return CustomOpenAICompatibleClient(model, base_url, **kwargs)
        return original_create_llm_client(provider, model, base_url, **kwargs)

    trading_graph.create_llm_client = create_llm_client
    trading_graph._quantmind_chat_completions_patch = True


def _build_ta_config(llm_cfg: dict, *, lang: str = "en") -> dict:
    """Map Kronos llm_config.json settings to a TradingAgents config dict."""
    from tradingagents.default_config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG.copy()
    cfg["data_vendors"] = DEFAULT_CONFIG["data_vendors"].copy()

    provider_raw = llm_cfg.get("provider", "Ollama").lower()
    ta_provider  = _PROVIDER_MAP.get(provider_raw, "openai")
    model        = str(llm_cfg.get("model") or "").strip()
    base_url     = llm_cfg.get("base_url", "").rstrip("/")
    api_key      = llm_cfg.get("api_key", "").strip()
    if not model:
        model = resolve_llm_model(llm_cfg) or ("qwen2.5:7b" if ta_provider == "ollama" else "gpt-4o-mini")

    cfg["llm_provider"]    = ta_provider
    cfg["deep_think_llm"]  = model
    cfg["quick_think_llm"] = model
    cfg["max_debate_rounds"]       = 1
    cfg["max_risk_discuss_rounds"] = 1
    cfg["output_language"] = "Chinese" if lang == "zh" else "English"

    # Set backend URL
    if ta_provider == "ollama":
        cfg["backend_url"] = base_url or "http://localhost:11434"
    elif base_url and "openai.com" not in base_url:
        cfg["backend_url"] = base_url
    elif provider_raw in _DEFAULT_BACKENDS:
        cfg["backend_url"] = _DEFAULT_BACKENDS[provider_raw]

    # Inject API key into environment so LangChain picks it up
    if api_key:
        env_map = {
            "openai":    "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google":    "GOOGLE_API_KEY",
            "azure":     "AZURE_OPENAI_API_KEY",
        }
        env_key = env_map.get(ta_provider)
        if env_key:
            os.environ[env_key] = api_key
        # Also set a generic key for OpenAI-compatible endpoints
        if ta_provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key

    return cfg


def _instance_key(cfg: dict) -> str:
    return f"{cfg['llm_provider']}:{cfg['deep_think_llm']}:{cfg.get('backend_url','')}:{cfg.get('output_language','English')}"


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_job(job_id: str, symbol: str, trade_date: str, config: ToolkitConfig, *, lang: str = "en", user_id: int = 1) -> None:
    """Execute a TradingAgents analysis in a background thread."""
    from trade_storage import update_ta_job

    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        _patch_tradingagents_custom_openai_chat_completions()
    except ImportError as exc:
        update_ta_job(config, job_id, status="failed",
                      error="TradingAgents library is not installed. "
                            "Install it with: pip install tradingagents")
        return

    try:
        update_ta_job(config, job_id, status="running")

        llm_cfg = load_llm_config()

        # Merge per-user LLM overrides
        from trade_storage import get_user_llm_config
        user_llm = get_user_llm_config(config, user_id)
        for k, v in user_llm.items():
            if k == "model" or v not in (None, ""):
                llm_cfg[k] = v

        # Pre-flight: verify LLM is reachable before spinning up the whole graph
        llm_err = _check_llm_reachable(llm_cfg)
        if llm_err:
            update_ta_job(config, job_id, status="failed", error=llm_err)
            return

        ta_cfg = _build_ta_config(llm_cfg, lang=lang)
        ticker = to_yfinance_ticker(symbol)

        with _lock:
            key = _instance_key(ta_cfg)
            if key not in _ta_instances:
                _ta_instances[key] = TradingAgentsGraph(debug=False, config=ta_cfg)
            ta: TradingAgentsGraph = _ta_instances[key]

        # Restore proxy env vars so yfinance can reach Yahoo Finance
        # (akshare clears HTTP_PROXY/HTTPS_PROXY on import)
        with _proxy_env():
            final_state, decision = ta.propagate(ticker, trade_date)

        reports = {
            "market_report":        final_state.get("market_report", ""),
            "sentiment_report":     final_state.get("sentiment_report", ""),
            "news_report":          final_state.get("news_report", ""),
            "fundamentals_report":  final_state.get("fundamentals_report", ""),
            "investment_plan":      final_state.get("investment_plan", ""),
            "trader_plan":          final_state.get("trader_investment_plan", ""),
            "final_decision":       final_state.get("final_trade_decision", ""),
            "debate_judge":         (final_state.get("investment_debate_state") or {}).get("judge_decision", ""),
            "risk_judge":           (final_state.get("risk_debate_state") or {}).get("judge_decision", ""),
        }

        update_ta_job(config, job_id, status="done", decision=decision, reports=reports)

    except Exception as exc:
        from trade_storage import update_ta_job as _utj
        _utj(config, job_id, status="failed", error=str(exc))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def submit_job(config: ToolkitConfig, symbol: str, trade_date: str | None = None, *, lang: str = "en", user_id: int = 1) -> str:
    """Fire a TradingAgents analysis job in the background. Returns job_id."""
    from trade_storage import save_ta_job

    if not trade_date:
        trade_date = date.today().isoformat()

    job_id = uuid.uuid4().hex[:8]
    save_ta_job(config, job_id, symbol, trade_date, user_id=user_id)

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, symbol, trade_date, config),
        kwargs={"lang": lang, "user_id": user_id},
        daemon=True,
        name=f"ta-{job_id}",
    )
    thread.start()
    return job_id


def get_job(config: ToolkitConfig, job_id: str) -> dict | None:
    """Load a TA job's status and result from SQLite."""
    from trade_storage import load_ta_job
    return load_ta_job(config, job_id)


def get_latest(config: ToolkitConfig, symbol: str, *, user_id: int = 1) -> dict | None:
    """Return the most recent completed TA analysis for a symbol."""
    from trade_storage import get_latest_ta_analysis
    return get_latest_ta_analysis(config, symbol, user_id=user_id)


def reflect(config: ToolkitConfig, realized_pnl_pct: float) -> None:
    """Notify all cached TradingAgentsGraph instances of a trade outcome."""
    with _lock:
        for instance in _ta_instances.values():
            if hasattr(instance, "reflect_and_remember"):
                try:
                    instance.reflect_and_remember(realized_pnl_pct)
                except Exception:
                    pass


def build_consensus(kronos_action: str, ta_decision: str) -> dict:
    """Merge a Kronos BUY/SELL/HOLD signal with a TradingAgents 5-tier decision.

    Returns a dict with: signal, label, confidence, description.
    """
    ta = ta_decision.upper().strip()
    kr = kronos_action.upper().strip()

    # Normalise TA to 3 tiers
    if ta in ("BUY", "OVERWEIGHT"):
        ta3 = "BUY"
    elif ta in ("SELL", "UNDERWEIGHT"):
        ta3 = "SELL"
    else:
        ta3 = "HOLD"

    if kr == ta3 == "BUY":
        return {"signal": "STRONG_BUY",  "label": "✅ Strong BUY",  "confidence": "high",
                "description": "Both Kronos quant forecast and AI agents agree: buy."}
    if kr == ta3 == "SELL":
        return {"signal": "STRONG_SELL", "label": "🔴 Strong SELL", "confidence": "high",
                "description": "Both Kronos and AI agents agree: sell / exit."}
    if kr == ta3 == "HOLD":
        return {"signal": "HOLD",        "label": "⏸ Hold",         "confidence": "medium",
                "description": "Both models suggest holding current position."}
    if kr == "BUY" and ta3 == "HOLD":
        return {"signal": "LEAN_BUY",    "label": "📈 Lean BUY",    "confidence": "low",
                "description": "Kronos forecasts upside; AI agents are neutral — proceed cautiously."}
    if kr == "SELL" and ta3 == "HOLD":
        return {"signal": "LEAN_SELL",   "label": "📉 Lean SELL",   "confidence": "low",
                "description": "Kronos signals exit; AI agents are neutral — consider reducing."}
    if kr == "HOLD" and ta3 == "BUY":
        return {"signal": "LEAN_BUY",    "label": "📈 Lean BUY",    "confidence": "low",
                "description": "AI agents are bullish; Kronos is neutral — watch for entry."}
    if kr == "HOLD" and ta3 == "SELL":
        return {"signal": "LEAN_SELL",   "label": "📉 Lean SELL",   "confidence": "low",
                "description": "AI agents flag risk; Kronos is neutral — reduce exposure."}
    # Outright conflict
    return {"signal": "CONFLICT",        "label": "⚠️ Conflict",    "confidence": "none",
            "description": f"Kronos says {kr} but AI agents say {ta_decision} — do not trade blindly."}
