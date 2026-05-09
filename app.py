from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user

from analysis_service import run_backtest_analysis, run_prediction_analysis
from config import ToolkitConfig
from data_sources import get_kline_data, get_last_close, get_realtime_quote, get_stock_name, get_t0_indicators, normalize_symbol
from llm_service import analyze_t0
from strategy_catalog import strategy_options
import trade_storage
import trading_agents_service as ta_service
from trading_service import (
    execute_paper_trade,
    export_manual_live_order,
    get_paper_portfolio_summary,
    load_live_state,
    load_paper_state,
    sync_live_portfolio,
)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.json.ensure_ascii = False          # Flask ≥2.2 — emit raw UTF-8 in JSON
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "quantmind-default-secret-key-change-in-prod")
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True


@app.after_request
def _set_utf8(response):
    """Ensure every JSON response explicitly declares charset=utf-8."""
    ct = response.content_type or ""
    if "application/json" in ct and "charset" not in ct:
        response.content_type = "application/json; charset=utf-8"
    return response

# ── Flask-Login setup ──────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id: int, username: str, email: str):
        self.id = id
        self.username = username
        self.email = email


@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    config = ToolkitConfig()
    trade_storage.ensure_storage(config)
    row = trade_storage.get_user_by_id(config, int(user_id))
    if row is None:
        return None
    return User(id=row["id"], username=row["username"], email=row["email"])


def _uid() -> int:
    """Shorthand to get the current logged-in user's id."""
    return int(current_user.id)


def default_form_data() -> dict:
    return {
        "symbol": "601169",
        "label": "601169",
        "label_auto": True,
        "start": "2020-01-01",
        "end": "2026-12-31",
        "lookback": 400,
        "pred_len": 20,
        "backtest_pred_len": 5,
        "rebalance_step": 20,
        "signal_threshold": 0.0,
        "source": "auto",
        "device": "cpu",
        "strategy": ToolkitConfig().recommended_strategy,
        "live_current_shares": 0,
        "live_avg_price": 0.0,
        "live_available_cash": ToolkitConfig().default_paper_cash,
    }


def parse_form(payload: dict) -> dict:
    defaults = default_form_data()
    data = defaults | payload
    data["symbol"] = str(data["symbol"]).strip()
    data["label"] = str(data["label"]).strip()
    data["label_auto"] = str(data.get("label_auto", defaults["label_auto"])).lower() in ("1", "true", "yes", "on")
    data["lookback"] = int(data["lookback"])
    data["pred_len"] = int(data["pred_len"])
    data["backtest_pred_len"] = int(data["backtest_pred_len"])
    data["rebalance_step"] = int(data["rebalance_step"])
    data["signal_threshold"] = float(data["signal_threshold"])
    data["live_current_shares"] = int(data["live_current_shares"])
    data["live_avg_price"] = float(data["live_avg_price"])
    data["live_available_cash"] = float(data["live_available_cash"])
    if data["label_auto"] or not data["label"]:
        data["label"] = data["symbol"]
    return data


def relative_output_path(path: Path, config: ToolkitConfig, user_id: int = 1) -> str:
    out_dir = config.user_output_dir(user_id)
    try:
        return str(path.relative_to(out_dir)).replace("\\", "/")
    except ValueError:
        # Fallback for legacy paths under global output_dir
        return str(path.relative_to(config.output_dir)).replace("\\", "/")


def execute_analysis(form_data: dict, *, user_id: int = 1) -> dict:
    config = ToolkitConfig()
    config.ensure_directories()

    prediction = run_prediction_analysis(
        config=config,
        symbol=form_data["symbol"],
        label=form_data["label"],
        start=form_data["start"],
        end=form_data["end"],
        lookback=form_data["lookback"],
        pred_len=form_data["pred_len"],
        source=form_data["source"],
        device=form_data["device"],
        strategy_key=form_data["strategy"],
        signal_threshold=form_data["signal_threshold"],
        user_id=user_id,
    )
    backtest = run_backtest_analysis(
        config=config,
        symbol=form_data["symbol"],
        start=form_data["start"],
        end=form_data["end"],
        lookback=min(form_data["lookback"], 240),
        pred_len=form_data["backtest_pred_len"],
        rebalance_step=form_data["rebalance_step"],
        signal_threshold=form_data["signal_threshold"],
        source=form_data["source"],
        device=form_data["device"],
        strategy_key=form_data["strategy"],
        user_id=user_id,
    )
    market_price = prediction["summary"]["last_close"]
    paper_portfolio = get_paper_portfolio_summary(
        config=config,
        symbol=prediction["summary"]["symbol"],
        market_price=market_price,
        user_id=user_id,
    )
    live_portfolio = sync_live_portfolio(
        config=config,
        symbol=prediction["summary"]["symbol"],
        current_shares=form_data["live_current_shares"],
        avg_price=form_data["live_avg_price"],
        available_cash=form_data["live_available_cash"],
        market_price=market_price,
        user_id=user_id,
    )

    return {
        "form": form_data,
        "prediction": {
            "summary": prediction["summary"],
            "recommendation": prediction["recommendation"],
            "forecast_image": relative_output_path(prediction["forecast_path"], config, user_id),
            "summary_file": relative_output_path(prediction["summary_path"], config, user_id),
            "prediction_file": relative_output_path(prediction["prediction_path"], config, user_id),
            "history_file": relative_output_path(prediction["history_path"], config, user_id),
        },
        "backtest": {
            "summary": backtest["summary"],
            "summary_file": relative_output_path(backtest["summary_path"], config, user_id),
            "trades_file": relative_output_path(backtest["trades_path"], config, user_id),
            "daily_file": relative_output_path(backtest["daily_path"], config, user_id),
        },
        "paper_portfolio": {
            "summary": paper_portfolio["portfolio"],
            "state_file": relative_output_path(Path(paper_portfolio["paper_state_file"]), config, user_id),
            "database_file": relative_output_path(Path(paper_portfolio["database_file"]), config, user_id),
        },
        "live_portfolio": {
            "summary": live_portfolio["portfolio"],
            "state_file": relative_output_path(Path(live_portfolio["live_state_file"]), config, user_id),
            "database_file": relative_output_path(Path(live_portfolio["database_file"]), config, user_id),
        },
    }


def execute_and_store_analysis(form_data: dict, *, user_id: int = 1) -> dict:
    config = ToolkitConfig()
    trade_storage.ensure_storage(config)
    result = execute_analysis(form_data, user_id=user_id)
    return trade_storage.create_analysis_run(config, user_id, form_data, result)


def load_stored_analysis(analysis_id: str, *, user_id: int) -> tuple[dict, dict]:
    config = ToolkitConfig()
    trade_storage.ensure_storage(config)
    run = trade_storage.get_analysis_run(config, user_id, analysis_id)
    if run is None:
        raise ValueError("Analysis run not found. Please run analysis again.")
    return run["form"], run["result"]


def resolve_analysis_for_action(payload: dict, *, user_id: int) -> tuple[dict, dict]:
    analysis_id = str(payload.get("analysis_id", "")).strip()
    if analysis_id:
        return load_stored_analysis(analysis_id, user_id=user_id)
    form_data = parse_form(payload)
    return form_data, execute_and_store_analysis(form_data, user_id=user_id)


def _best_execution_price(symbol: str, fallback_close: float) -> tuple[float, str]:
    """Return (price, source_label). Prefers real-time quote during market hours."""
    try:
        quote = get_realtime_quote(symbol)
        if quote and quote.get("is_realtime") and quote.get("last_price"):
            return float(quote["last_price"]), "realtime"
    except Exception:
        pass
    return float(fallback_close), "last_close"


def execute_paper_trade_action(form_data: dict, analysis_result: dict, *, user_id: int = 1) -> dict:
    config = ToolkitConfig()
    recommendation = analysis_result["prediction"]["recommendation"]
    symbol = analysis_result["prediction"]["summary"]["symbol"]
    fallback = analysis_result["prediction"]["summary"]["last_close"]
    execution_price, price_source = _best_execution_price(symbol, fallback)
    result = execute_paper_trade(
        config=config,
        symbol=symbol,
        recommendation=recommendation,
        execution_price=execution_price,
        user_id=user_id,
    )
    result["trade"]["price_source"] = "🔴 实时价" if price_source == "realtime" else "📅 收盘价"

    # Reflection learning: when a SELL is executed, feed the realized return to TA agents
    if result["trade"].get("status") == "sold":
        avg_buy = result["trade"].get("avg_buy_price", 0.0)
        sell_price = result["trade"].get("execution_price", 0.0)
        if avg_buy and avg_buy > 0:
            pnl_pct = round((sell_price / avg_buy - 1.0) * 100, 4)
            try:
                ta_service.reflect(config, pnl_pct)
            except Exception:
                pass

    return result


def execute_live_export_action(form_data: dict, analysis_result: dict, *, user_id: int = 1) -> dict:
    config = ToolkitConfig()
    recommendation = analysis_result["prediction"]["recommendation"]
    symbol = analysis_result["prediction"]["summary"]["symbol"]
    fallback = analysis_result["prediction"]["summary"]["last_close"]
    execution_price, price_source = _best_execution_price(symbol, fallback)
    export = export_manual_live_order(
        config=config,
        symbol=analysis_result["prediction"]["summary"]["symbol"],
        recommendation=recommendation,
        execution_price=execution_price,
        current_shares=form_data["live_current_shares"],
        avg_price=form_data["live_avg_price"],
        available_cash=form_data["live_available_cash"],
        user_id=user_id,
    )
    export["order"]["price_source"] = "🔴 实时价" if price_source == "realtime" else "📅 收盘价"
    export["order_file_relative"] = relative_output_path(Path(export["order_file"]), config, user_id)
    export["database_file_relative"] = relative_output_path(Path(export["database_file"]), config, user_id)
    return export


# ── Auth routes ────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        identifier = request.form.get("identifier", request.form.get("email", "")).strip()
        password = request.form.get("password", "")
        config = ToolkitConfig()
        trade_storage.ensure_storage(config)
        user_row = trade_storage.authenticate_user(config, identifier, password)
        if user_row:
            user = User(id=user_row["id"], username=user_row["username"], email=user_row["email"])
            login_user(user)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("index"))
        error = "Invalid username/email or password"
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not username or not email or not password:
            error = "All fields are required"
        elif len(password) < 6:
            error = "Password must be at least 6 characters"
        else:
            config = ToolkitConfig()
            trade_storage.ensure_storage(config)
            try:
                user_row = trade_storage.create_user(config, username, email, password)
                user = User(id=user_row["id"], username=user_row["username"], email=user_row["email"])
                login_user(user)
                return redirect(url_for("index"))
            except ValueError as exc:
                error = str(exc)
    return render_template("register.html", error=error)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    form_data = default_form_data()
    result = None
    error = None
    trade_result = None
    uid = _uid()
    if request.method == "POST":
        try:
            action = request.form.get("action", "analyze")
            analysis_id = request.form.get("analysis_id", "").strip()
            if analysis_id:
                form_data, result = load_stored_analysis(analysis_id, user_id=uid)
            else:
                form_data = parse_form(request.form.to_dict())
                result = execute_and_store_analysis(form_data, user_id=uid)
            if action == "paper_trade":
                trade_result = {"paper": execute_paper_trade_action(form_data, result, user_id=uid)}
            elif action == "live_export":
                trade_result = {"live": execute_live_export_action(form_data, result, user_id=uid)}
        except Exception as exc:
            error = str(exc)
    return render_template(
        "index.html",
        form=form_data,
        result=result,
        error=error,
        trade_result=trade_result,
        strategy_options=strategy_options(),
        now_date=__import__("datetime").date.today().isoformat(),
    )


@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze_api():
    payload = request.get_json(silent=True) or {}
    try:
        form_data = parse_form(payload)
        result = execute_and_store_analysis(form_data, user_id=_uid())
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/analysis-latest/<symbol>")
@login_required
def analysis_latest_api(symbol: str):
    try:
        sym = normalize_symbol(symbol.strip())
        config = ToolkitConfig()
        trade_storage.ensure_storage(config)
        run = trade_storage.get_latest_analysis_run(config, _uid(), sym)
        if run is None:
            return jsonify({"error": "No saved analysis found"}), 404
        return jsonify(run)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/paper-trade", methods=["POST"])
@login_required
def paper_trade_api():
    payload = request.get_json(silent=True) or {}
    try:
        uid = _uid()
        form_data, result = resolve_analysis_for_action(payload, user_id=uid)
        trade_result = execute_paper_trade_action(form_data, result, user_id=uid)
        return jsonify({"analysis": result, "paper_trade": trade_result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/live-export", methods=["POST"])
@login_required
def live_export_api():
    payload = request.get_json(silent=True) or {}
    try:
        uid = _uid()
        form_data, result = resolve_analysis_for_action(payload, user_id=uid)
        trade_result = execute_live_export_action(form_data, result, user_id=uid)
        return jsonify({"analysis": result, "live_export": trade_result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/outputs/<path:filename>")
@login_required
def outputs(filename: str):
    config = ToolkitConfig()
    config.ensure_directories()
    out_dir = config.user_output_dir(_uid())
    return send_from_directory(out_dir, filename)


@app.route("/api/stock-name/<symbol>")
@login_required
def stock_name_api(symbol: str):
    try:
        sym = normalize_symbol(symbol.strip())
        name = get_stock_name(sym)
        return jsonify({"symbol": sym, "name": name})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/realtime-quote/<symbol>")
@login_required
def realtime_quote_api(symbol: str):
    try:
        sym = normalize_symbol(symbol.strip())
        quote = get_realtime_quote(sym)
        if quote is None:
            return jsonify({"error": "Quote not available"}), 404
        return jsonify(quote)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/kline/<symbol>")
@login_required
def kline_api(symbol: str):
    """Return OHLCV data for candlestick chart.
    ?period=daily|weekly|monthly|5min|15min|30min|60min
    ?days=N  (number of calendar days to look back, default 180 for daily)
    """
    try:
        sym = normalize_symbol(symbol.strip())
        period = request.args.get("period", "daily")
        days = int(request.args.get("days", 180))
        data = get_kline_data(sym, period=period, days=days)
        if data is None:
            return jsonify({"error": "Data not available"}), 404
        return jsonify(data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/llm-config", methods=["GET", "POST"])
@login_required
def llm_config_api():
    config = ToolkitConfig()
    trade_storage.ensure_storage(config)
    uid = _uid()
    if request.method == "POST":
        try:
            cfg = request.get_json(silent=True) or {}
            trade_storage.save_user_llm_config(config, uid, cfg)
            return jsonify({"ok": True})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
    # Merge global defaults with per-user overrides
    from llm_service import load_llm_config as _load_global
    merged = _load_global()
    user_cfg = trade_storage.get_user_llm_config(config, uid)
    for k, v in user_cfg.items():
        if v not in (None, ""):
            merged[k] = v
    merged.pop("api_key", None)  # never send key back to browser
    merged["has_api_key"] = bool(user_cfg.get("api_key", "").strip())
    return jsonify(merged)


@app.route("/api/account", methods=["GET", "POST"])
@login_required
def account_api():
    config = ToolkitConfig()
    trade_storage.ensure_storage(config)
    uid = _uid()
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        try:
            updated = trade_storage.update_user_credentials(
                config=config,
                user_id=uid,
                username=str(payload.get("username", "")).strip(),
                current_password=str(payload.get("current_password", "")),
                new_password=str(payload.get("new_password", "")),
            )
            current_user.username = updated["username"]
            return jsonify({"ok": True, "user": updated})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    row = trade_storage.get_user_by_id(config, uid)
    if row is None:
        return jsonify({"error": "User not found"}), 404
    return jsonify(row)


# ── Broker config & order execution ────────────────────────────────────

@app.route("/api/broker-config", methods=["GET", "POST"])
@login_required
def broker_config_api():
    config = ToolkitConfig()
    trade_storage.ensure_storage(config)
    uid = _uid()
    if request.method == "POST":
        try:
            cfg = request.get_json(silent=True) or {}
            trade_storage.save_user_broker_config(config, uid, cfg)
            return jsonify({"ok": True})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
    broker_cfg = trade_storage.get_user_broker_config(config, uid)
    return jsonify(broker_cfg)


@app.route("/api/broker-order", methods=["POST"])
@login_required
def broker_order_api():
    """Execute a real broker order via MiniQMT."""
    from broker_service import BrokerConfig, execute_order, OrderSide, OrderType

    config = ToolkitConfig()
    trade_storage.ensure_storage(config)
    uid = _uid()
    payload = request.get_json(silent=True) or {}

    broker_dict = trade_storage.get_user_broker_config(config, uid)
    broker_cfg = BrokerConfig.from_dict(broker_dict)

    try:
        symbol = payload.get("symbol", "").strip()
        side = OrderSide(payload.get("side", "BUY").upper())
        volume = int(payload.get("volume", 0))
        price = float(payload.get("price", 0))
        order_type = OrderType(payload.get("order_type", "LIMIT").upper())

        # Get today's order count for risk check
        orders_today = trade_storage.get_broker_orders(config, uid, limit=200)
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_count = sum(1 for o in orders_today if o["created_at"].startswith(today_str))

        result = execute_order(
            broker_cfg, symbol, side, volume, price, order_type,
            daily_order_count=daily_count,
        )

        # Record order in DB
        trade_storage.record_broker_order(config, uid, result)
        return jsonify(result)

    except Exception as exc:
        return jsonify({"order_id": None, "status": "FAILED", "message": str(exc)}), 400


@app.route("/api/broker-signal", methods=["POST"])
@login_required
def broker_signal_api():
    """Execute a QuantMind signal (BUY/SELL/HOLD) as a real broker order."""
    from broker_service import BrokerConfig, execute_signal

    config = ToolkitConfig()
    trade_storage.ensure_storage(config)
    uid = _uid()
    payload = request.get_json(silent=True) or {}

    broker_dict = trade_storage.get_user_broker_config(config, uid)
    broker_cfg = BrokerConfig.from_dict(broker_dict)

    try:
        symbol = payload.get("symbol", "").strip()
        action = payload.get("action", "HOLD").upper()
        price = float(payload.get("price", 0))
        position_shares = int(payload.get("position_shares", 0))
        available_cash = float(payload.get("available_cash", 0))

        orders_today = trade_storage.get_broker_orders(config, uid, limit=200)
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_count = sum(1 for o in orders_today if o["created_at"].startswith(today_str))

        result = execute_signal(
            broker_cfg, symbol, action, price,
            position_shares=position_shares,
            available_cash=available_cash,
            daily_order_count=daily_count,
        )

        if result.get("order_id"):
            trade_storage.record_broker_order(config, uid, result)
        return jsonify(result)

    except Exception as exc:
        return jsonify({"order_id": None, "status": "FAILED", "message": str(exc)}), 400


@app.route("/api/broker-orders")
@login_required
def broker_orders_api():
    """Return recent broker orders for the current user."""
    config = ToolkitConfig()
    trade_storage.ensure_storage(config)
    orders = trade_storage.get_broker_orders(config, _uid())
    return jsonify({"orders": orders})


@app.route("/api/broker-account")
@login_required
def broker_account_api():
    """Query broker account asset and positions."""
    from broker_service import BrokerConfig, query_account_asset, query_positions

    config = ToolkitConfig()
    broker_dict = trade_storage.get_user_broker_config(config, _uid())
    broker_cfg = BrokerConfig.from_dict(broker_dict)

    asset = query_account_asset(broker_cfg)
    positions = query_positions(broker_cfg)
    return jsonify({"asset": asset, "positions": positions})


@app.route("/api/t0-indicators/<symbol>")
@login_required
def t0_indicators_api(symbol: str):
    try:
        sym = normalize_symbol(symbol.strip())
        force = request.args.get("force", "0") == "1"
        data = get_t0_indicators(sym, force=force)
        if data is None:
            return jsonify({"error": "Indicator data unavailable — market may be closed"}), 404
        return jsonify(data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/t0-analysis", methods=["POST"])
@login_required
def t0_analysis_api():
    payload = request.get_json(silent=True) or {}
    try:
        sym = normalize_symbol(payload.get("symbol", "").strip())
        indicators = get_t0_indicators(sym)
        if indicators is None:
            return jsonify({"error": "No intraday data available"}), 404
        config = ToolkitConfig()
        user_llm = trade_storage.get_user_llm_config(config, _uid())
        text = analyze_t0(sym, indicators, llm_config=user_llm or None)
        return jsonify({"symbol": sym, "indicators": indicators, "analysis": text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/portfolio")
@login_required
def portfolio():
    return render_template("portfolio.html")


@app.route("/api/portfolio-data")
@login_required
def portfolio_data_api():
    config = ToolkitConfig()
    config.ensure_directories()
    uid = _uid()
    paper_state = load_paper_state(config, user_id=uid)
    live_state = load_live_state(config, user_id=uid)

    all_symbols: set[str] = set()
    for sym, pos in paper_state.get("positions", {}).items():
        if pos.get("shares", 0) > 0:
            all_symbols.add(sym)
    for sym, pos in live_state.get("positions", {}).items():
        if pos.get("shares", 0) > 0:
            all_symbols.add(sym)

    prices: dict[str, float] = {}
    for sym in all_symbols:
        price = get_last_close(sym, config)
        if price is not None:
            prices[sym] = price

    def build_positions(state: dict) -> dict:
        result: dict = {}
        for sym, pos in state.get("positions", {}).items():
            shares = int(pos.get("shares", 0))
            avg_price = float(pos.get("avg_price", 0.0))
            last_price = prices.get(sym, avg_price)
            market_value = round(shares * last_price, 2)
            cost_basis = round(shares * avg_price, 2)
            unrealized_pnl = round(market_value - cost_basis, 2)
            unrealized_pct = round((last_price / avg_price - 1.0) * 100, 2) if avg_price > 0 and shares > 0 else 0.0
            result[sym] = {
                "name": get_stock_name(sym),
                "shares": shares,
                "avg_price": avg_price,
                "last_price": last_price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pct": unrealized_pct,
            }
        return result

    paper_positions = build_positions(paper_state)
    live_positions = build_positions(live_state)

    paper_initial = float(paper_state.get("initial_equity", paper_state.get("initial_cash", 0.0)))
    paper_cash = round(float(paper_state.get("cash", 0.0)), 2)
    paper_total = round(paper_cash + sum(p["market_value"] for p in paper_positions.values()), 2)
    paper_return_pct = round((paper_total / paper_initial - 1.0) * 100, 2) if paper_initial > 0 else 0.0

    live_initial = float(live_state.get("initial_equity", 0.0))
    live_cash = round(float(live_state.get("cash", 0.0)), 2)
    live_total = round(live_cash + sum(p["market_value"] for p in live_positions.values()), 2)
    live_return_pct = round((live_total / live_initial - 1.0) * 100, 2) if live_initial > 0 else 0.0

    paper_curve = []
    if paper_initial > 0:
        paper_curve.append({"t": "Initial", "equity": paper_initial, "status": "initial", "action": None})
    for trade in paper_state.get("trade_history", []):
        equity = round(
            float(trade["cash_after"]) + int(trade["position_after"]["shares"]) * float(trade["execution_price"]),
            2,
        )
        paper_curve.append({
            "t": trade["timestamp"],
            "equity": equity,
            "status": trade["status"],
            "action": trade["action"],
            "symbol": trade["symbol"],
            "price": trade["execution_price"],
            "shares": trade["shares_delta"],
        })

    live_curve = [
        {
            "t": s["timestamp"],
            "equity": s["total_equity"],
            "status": "sync",
            "symbol": s["symbol"],
            "shares": s["shares"],
            "price": s["market_price"],
        }
        for s in live_state.get("sync_history", [])
    ]

    return jsonify({
        "paper": {
            "initial_equity": paper_initial,
            "cash": paper_cash,
            "realized_pnl": round(float(paper_state.get("realized_pnl", 0.0)), 2),
            "total_equity": paper_total,
            "total_return_pct": paper_return_pct,
            "positions": paper_positions,
            "equity_curve": paper_curve,
        },
        "live": {
            "initial_equity": live_initial,
            "cash": live_cash,
            "realized_pnl": round(float(live_state.get("realized_pnl", 0.0)), 2),
            "total_equity": live_total,
            "total_return_pct": live_return_pct,
            "positions": live_positions,
            "equity_curve": live_curve,
        },
    })


@app.route("/api/ta-analysis", methods=["POST"])
@login_required
def ta_analysis_api():
    """Fire a TradingAgents multi-agent analysis job. Returns job_id immediately."""
    payload = request.get_json(silent=True) or {}
    try:
        raw_symbol = payload.get("symbol", "").strip()
        if not raw_symbol:
            return jsonify({"error": "symbol is required"}), 400
        symbol = normalize_symbol(raw_symbol)
        trade_date = payload.get("date") or ""
        lang = payload.get("lang", "en")  # "en" or "zh"
        config = ToolkitConfig()
        config.ensure_directories()

        # Quick pre-flight LLM check using per-user config
        from llm_service import load_llm_config as _load_global
        user_llm = trade_storage.get_user_llm_config(config, _uid())
        merged_llm = _load_global()
        for k, v in user_llm.items():
            if v not in (None, ""):
                merged_llm[k] = v
        llm_err = ta_service._check_llm_reachable(merged_llm)
        if llm_err:
            return jsonify({"error": llm_err}), 400

        job_id = ta_service.submit_job(config, symbol, trade_date or None, lang=lang, user_id=_uid())
        return jsonify({"job_id": job_id, "symbol": symbol, "status": "pending"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/ta-status/<job_id>")
@login_required
def ta_status_api(job_id: str):
    """Poll the status and result of a TradingAgents job."""
    config = ToolkitConfig()
    job = ta_service.get_job(config, job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/ta-latest/<symbol>")
@login_required
def ta_latest_api(symbol: str):
    """Return the most recent completed TradingAgents analysis for a symbol."""
    try:
        sym = normalize_symbol(symbol.strip())
        config = ToolkitConfig()
        result = ta_service.get_latest(config, sym, user_id=_uid())
        if result is None:
            return jsonify({"error": "No completed analysis found"}), 404
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/consensus/<symbol>")
@login_required
def consensus_api(symbol: str):
    """Merge the latest Kronos recommendation with the latest TA decision."""
    try:
        sym = normalize_symbol(symbol.strip())
        config = ToolkitConfig()
        uid = _uid()
        ta_result = ta_service.get_latest(config, sym, user_id=uid)
        if ta_result is None:
            return jsonify({"error": "No TA analysis found — run agent analysis first"}), 404

        # Prefer an explicit analysis run so consensus is tied to the result the user reviewed.
        kronos_action = "HOLD"
        analysis_id = request.args.get("analysis_id", "").strip()
        if analysis_id:
            run = trade_storage.get_analysis_run(config, uid, analysis_id)
            if run is None:
                return jsonify({"error": "Analysis run not found"}), 404
            kronos_action = run["result"]["prediction"]["recommendation"].get("action", "HOLD")
        else:
            out_dir = config.user_output_dir(uid)
            summary_path = out_dir / f"{sym}_summary.json"
            if summary_path.exists():
                import json as _json
                try:
                    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
                    kronos_action = summary.get("recommended_action", "HOLD")
                except Exception:
                    pass
        consensus = ta_service.build_consensus(kronos_action, ta_result["decision"] or "HOLD")
        return jsonify({
            "symbol":        sym,
            "kronos_action": kronos_action,
            "ta_decision":   ta_result["decision"],
            "ta_trade_date": ta_result["trade_date"],
            "consensus":     consensus,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ── Watchlist API (SQLite-backed, per-user) ────────────────────────────

@app.route("/api/watchlist", methods=["GET"])
@login_required
def watchlist_get():
    config = ToolkitConfig()
    trade_storage.ensure_storage(config)
    items = trade_storage.get_watchlist(config, user_id=_uid())
    return jsonify({"items": items})


@app.route("/api/watchlist", methods=["POST"])
@login_required
def watchlist_add():
    payload = request.get_json(silent=True) or {}
    raw_symbol = payload.get("symbol", "").strip()
    if not raw_symbol:
        return jsonify({"error": "symbol required"}), 400
    try:
        sym = normalize_symbol(raw_symbol)
        name = get_stock_name(sym)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    config = ToolkitConfig()
    trade_storage.ensure_storage(config)
    trade_storage.add_watchlist_item(config, sym, name, user_id=_uid())
    items = trade_storage.get_watchlist(config, user_id=_uid())
    return jsonify({"items": items})


@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
@login_required
def watchlist_remove(symbol: str):
    try:
        sym = normalize_symbol(symbol.strip())
    except Exception:
        sym = symbol.strip()

    config = ToolkitConfig()
    trade_storage.ensure_storage(config)
    trade_storage.remove_watchlist_item(config, sym, user_id=_uid())
    items = trade_storage.get_watchlist(config, user_id=_uid())
    return jsonify({"items": items})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7080, debug=False)
