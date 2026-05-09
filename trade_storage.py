from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path

from werkzeug.security import generate_password_hash, check_password_hash

from config import ToolkitConfig

DEFAULT_USER_ID = 1
DEFAULT_USER_EMAIL = "admin@quantmind.local"
DEFAULT_USER_PASSWORD = "admin"


def _connect(config: ToolkitConfig) -> sqlite3.Connection:
    config.ensure_directories()
    conn = sqlite3.connect(config.trading_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_storage(config: ToolkitConfig) -> None:
    with closing(_connect(config)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                llm_config_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portfolio_state (
                user_id INTEGER NOT NULL DEFAULT 1,
                mode TEXT NOT NULL,
                initial_cash REAL NOT NULL,
                initial_equity REAL NOT NULL,
                cash REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, mode),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS positions (
                user_id INTEGER NOT NULL DEFAULT 1,
                mode TEXT NOT NULL,
                symbol TEXT NOT NULL,
                shares INTEGER NOT NULL,
                avg_price REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, mode, symbol),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL DEFAULT 1,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                execution_price REAL NOT NULL,
                shares_delta INTEGER NOT NULL,
                cash_after REAL NOT NULL,
                position_shares INTEGER NOT NULL,
                position_avg_price REAL NOT NULL,
                status TEXT NOT NULL,
                strategy TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS live_syncs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL DEFAULT 1,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                shares INTEGER NOT NULL,
                avg_price REAL NOT NULL,
                cash REAL NOT NULL,
                market_price REAL NOT NULL,
                total_equity REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS live_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                execution_price_reference REAL NOT NULL,
                current_shares INTEGER NOT NULL,
                current_avg_price REAL NOT NULL,
                target_shares INTEGER NOT NULL,
                order_shares_delta INTEGER NOT NULL,
                available_cash REAL NOT NULL,
                note TEXT NOT NULL,
                rationale_json TEXT NOT NULL,
                order_file TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS ta_analyses (
                job_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL DEFAULT 1,
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                decision TEXT,
                reports_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS analysis_runs (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL DEFAULT 1,
                symbol TEXT NOT NULL,
                form_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS watchlist_items (
                user_id INTEGER NOT NULL DEFAULT 1,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                added_at TEXT NOT NULL,
                PRIMARY KEY (user_id, symbol),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS broker_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL DEFAULT 1,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                volume INTEGER NOT NULL,
                price REAL NOT NULL,
                order_type TEXT NOT NULL DEFAULT 'LIMIT',
                status TEXT NOT NULL DEFAULT 'PENDING',
                broker_order_id TEXT,
                message TEXT,
                details_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )
        conn.commit()

    _ensure_default_user(config)
    _migrate_schema(config)
    _migrate_legacy_json(config, "paper")
    _migrate_legacy_json(config, "live")
    _migrate_legacy_watchlist(config)


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

def _ensure_default_user(config: ToolkitConfig) -> None:
    with closing(_connect(config)) as conn:
        row = conn.execute("SELECT 1 FROM users WHERE id = ?", (DEFAULT_USER_ID,)).fetchone()
        if row:
            return
        conn.execute(
            "INSERT INTO users (id, email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                DEFAULT_USER_ID,
                DEFAULT_USER_EMAIL,
                generate_password_hash(DEFAULT_USER_PASSWORD),
                "Admin",
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()


def create_user(config: ToolkitConfig, username: str, email: str, password: str) -> dict:
    username = username.strip()
    email = email.strip()
    with closing(_connect(config)) as conn:
        existing = conn.execute(
            "SELECT email, display_name FROM users WHERE lower(email) = lower(?) OR lower(display_name) = lower(?)",
            (email, username),
        ).fetchone()
        if existing:
            if existing["email"].lower() == email.lower():
                raise ValueError("Email is already registered")
            raise ValueError("Username is already taken")

        try:
            cursor = conn.execute(
                "INSERT INTO users (email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
                (email, generate_password_hash(password), username, datetime.now().isoformat(timespec="seconds")),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Email is already registered") from exc
        conn.commit()
        return {"id": cursor.lastrowid, "username": username, "email": email}


def authenticate_user(config: ToolkitConfig, identifier: str, password: str) -> dict | None:
    identifier = identifier.strip()
    with closing(_connect(config)) as conn:
        rows = conn.execute(
            """
            SELECT id, email, password_hash, display_name
            FROM users
            WHERE lower(email) = lower(?) OR lower(display_name) = lower(?)
            ORDER BY lower(email) = lower(?) DESC, id ASC
            """,
            (identifier, identifier, identifier),
        ).fetchall()
        for row in rows:
            if check_password_hash(row["password_hash"], password):
                return {"id": row["id"], "email": row["email"], "username": row["display_name"]}
        return None


def get_user_by_id(config: ToolkitConfig, user_id: int) -> dict | None:
    with closing(_connect(config)) as conn:
        row = conn.execute(
            "SELECT id, email, display_name FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        return {"id": row["id"], "email": row["email"], "username": row["display_name"]}


def update_user_credentials(
    config: ToolkitConfig,
    user_id: int,
    username: str,
    current_password: str = "",
    new_password: str = "",
) -> dict:
    username = username.strip()
    if not username:
        raise ValueError("Username is required")
    if new_password and len(new_password) < 6:
        raise ValueError("Password must be at least 6 characters")

    with closing(_connect(config)) as conn:
        row = conn.execute(
            "SELECT id, email, password_hash FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            raise ValueError("User not found")
        if new_password and not check_password_hash(row["password_hash"], current_password):
            raise ValueError("Current password is incorrect")
        existing = conn.execute(
            "SELECT 1 FROM users WHERE lower(display_name) = lower(?) AND id <> ?",
            (username, user_id),
        ).fetchone()
        if existing:
            raise ValueError("Username is already taken")

        if new_password:
            conn.execute(
                "UPDATE users SET display_name = ?, password_hash = ? WHERE id = ?",
                (username, generate_password_hash(new_password), user_id),
            )
        else:
            conn.execute(
                "UPDATE users SET display_name = ? WHERE id = ?",
                (username, user_id),
            )
        conn.commit()
        return {"id": row["id"], "email": row["email"], "username": username}


def get_user_llm_config(config: ToolkitConfig, user_id: int) -> dict:
    """Return the per-user LLM config as a dict (empty dict if not set)."""
    with closing(_connect(config)) as conn:
        row = conn.execute(
            "SELECT llm_config_json FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row and row["llm_config_json"]:
            try:
                return json.loads(row["llm_config_json"])
            except (json.JSONDecodeError, TypeError):
                pass
    return {}


def save_user_llm_config(config: ToolkitConfig, user_id: int, llm_cfg: dict) -> None:
    """Persist per-user LLM config."""
    with closing(_connect(config)) as conn:
        conn.execute(
            "UPDATE users SET llm_config_json = ? WHERE id = ?",
            (json.dumps(llm_cfg, ensure_ascii=False), user_id),
        )
        conn.commit()


def get_user_broker_config(config: ToolkitConfig, user_id: int) -> dict:
    """Return the per-user broker config as a dict (empty dict if not set)."""
    with closing(_connect(config)) as conn:
        row = conn.execute(
            "SELECT broker_config_json FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row and row["broker_config_json"]:
            try:
                return json.loads(row["broker_config_json"])
            except (json.JSONDecodeError, TypeError):
                pass
    return {}


def save_user_broker_config(config: ToolkitConfig, user_id: int, broker_cfg: dict) -> None:
    """Persist per-user broker config."""
    with closing(_connect(config)) as conn:
        conn.execute(
            "UPDATE users SET broker_config_json = ? WHERE id = ?",
            (json.dumps(broker_cfg, ensure_ascii=False), user_id),
        )
        conn.commit()


def create_analysis_run(config: ToolkitConfig, user_id: int, form_data: dict, result: dict) -> dict:
    analysis_id = uuid.uuid4().hex
    stored_result = dict(result)
    stored_result["analysis_id"] = analysis_id
    symbol = stored_result.get("prediction", {}).get("summary", {}).get("symbol") or form_data.get("symbol", "")
    with closing(_connect(config)) as conn:
        conn.execute(
            """
            INSERT INTO analysis_runs (id, user_id, symbol, form_json, result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                user_id,
                symbol,
                json.dumps(form_data, ensure_ascii=False),
                json.dumps(stored_result, ensure_ascii=False),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    return stored_result


def get_analysis_run(config: ToolkitConfig, user_id: int, analysis_id: str) -> dict | None:
    with closing(_connect(config)) as conn:
        row = conn.execute(
            """
            SELECT id, symbol, form_json, result_json, created_at
            FROM analysis_runs
            WHERE id = ? AND user_id = ?
            """,
            (analysis_id, user_id),
        ).fetchone()
    if not row:
        return None
    form_data = json.loads(row["form_json"])
    result = json.loads(row["result_json"])
    result["analysis_id"] = row["id"]
    return {
        "id": row["id"],
        "symbol": row["symbol"],
        "form": form_data,
        "result": result,
        "created_at": row["created_at"],
    }


def get_latest_analysis_run(config: ToolkitConfig, user_id: int, symbol: str) -> dict | None:
    with closing(_connect(config)) as conn:
        row = conn.execute(
            """
            SELECT id, symbol, form_json, result_json, created_at
            FROM analysis_runs
            WHERE user_id = ? AND symbol = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id, symbol),
        ).fetchone()
    if not row:
        return None
    form_data = json.loads(row["form_json"])
    result = json.loads(row["result_json"])
    result["analysis_id"] = row["id"]
    return {
        "id": row["id"],
        "symbol": row["symbol"],
        "form": form_data,
        "result": result,
        "created_at": row["created_at"],
    }


def record_broker_order(config: ToolkitConfig, user_id: int, order_result: dict) -> int:
    """Record a broker order in the database. Returns the row id."""
    with closing(_connect(config)) as conn:
        cur = conn.execute(
            """INSERT INTO broker_orders
               (user_id, symbol, side, volume, price, order_type, status,
                broker_order_id, message, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                (order_result.get("details") or {}).get("symbol", ""),
                (order_result.get("details") or {}).get("side", ""),
                (order_result.get("details") or {}).get("volume", 0),
                (order_result.get("details") or {}).get("price", 0.0),
                (order_result.get("details") or {}).get("order_type", "LIMIT"),
                order_result.get("status", "UNKNOWN"),
                order_result.get("order_id", ""),
                order_result.get("message", ""),
                json.dumps(order_result.get("details") or {}, ensure_ascii=False),
                order_result.get("timestamp", datetime.now().isoformat()),
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_broker_orders(config: ToolkitConfig, user_id: int, limit: int = 50) -> list[dict]:
    """Return recent broker orders for a user."""
    with closing(_connect(config)) as conn:
        rows = conn.execute(
            """SELECT id, symbol, side, volume, price, order_type, status,
                      broker_order_id, message, created_at
               FROM broker_orders WHERE user_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Schema migration for existing databases
# ---------------------------------------------------------------------------

def _migrate_schema(config: ToolkitConfig) -> None:
    """Add user_id column to legacy tables that don't have it yet."""
    with closing(_connect(config)) as conn:
        migrations = [
            ("portfolio_state", "user_id", "INTEGER NOT NULL DEFAULT 1"),
            ("positions", "user_id", "INTEGER NOT NULL DEFAULT 1"),
            ("paper_trades", "user_id", "INTEGER NOT NULL DEFAULT 1"),
            ("live_syncs", "user_id", "INTEGER NOT NULL DEFAULT 1"),
            ("live_orders", "user_id", "INTEGER NOT NULL DEFAULT 1"),
            ("ta_analyses", "user_id", "INTEGER NOT NULL DEFAULT 1"),
            ("users", "llm_config_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("users", "broker_config_json", "TEXT NOT NULL DEFAULT '{}'"),
        ]
        for table, column, col_type in migrations:
            try:
                conn.execute(f"SELECT {column} FROM {table} LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()


def _state_path(config: ToolkitConfig, mode: str) -> Path:
    if mode == "paper":
        return config.paper_state_path
    if mode == "live":
        return config.live_state_path
    raise ValueError(f"Unsupported mode: {mode}")


def _migrate_legacy_json(config: ToolkitConfig, mode: str) -> None:
    state_path = _state_path(config, mode)
    if not state_path.exists():
        return

    with closing(_connect(config)) as conn:
        row = conn.execute("SELECT 1 FROM portfolio_state WHERE user_id = ? AND mode = ?", (DEFAULT_USER_ID, mode)).fetchone()
        if row:
            return

        payload = json.loads(state_path.read_text(encoding="utf-8"))
        save_portfolio_state(
            config=config,
            mode=mode,
            state={
                "initial_cash": float(payload.get("initial_cash", config.default_paper_cash if mode == "paper" else 0.0)),
                "initial_equity": float(
                    payload.get(
                        "initial_equity",
                        payload.get("initial_cash", config.default_paper_cash if mode == "paper" else 0.0),
                    )
                ),
                "cash": float(payload.get("cash", 0.0)),
                "positions": payload.get("positions", {}),
                "realized_pnl": float(payload.get("realized_pnl", 0.0)),
            },
            user_id=DEFAULT_USER_ID,
        )

        if mode == "paper":
            for trade in payload.get("trade_history", []):
                conn.execute(
                    """
                    INSERT INTO paper_trades (
                        user_id, timestamp, symbol, action, execution_price, shares_delta, cash_after,
                        position_shares, position_avg_price, status, strategy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        DEFAULT_USER_ID,
                        trade.get("timestamp", ""),
                        trade.get("symbol", ""),
                        trade.get("action", ""),
                        float(trade.get("execution_price", 0.0)),
                        int(trade.get("shares_delta", 0)),
                        float(trade.get("cash_after", 0.0)),
                        int(trade.get("position_after", {}).get("shares", 0)),
                        float(trade.get("position_after", {}).get("avg_price", 0.0)),
                        trade.get("status", ""),
                        trade.get("strategy", ""),
                    ),
                )
        else:
            for sync in payload.get("sync_history", []):
                total_equity = round(float(sync.get("cash", 0.0)) + int(sync.get("shares", 0)) * float(sync.get("market_price", 0.0)), 2)
                unrealized_pnl = round((float(sync.get("market_price", 0.0)) - float(sync.get("avg_price", 0.0))) * int(sync.get("shares", 0)), 2)
                conn.execute(
                    """
                    INSERT INTO live_syncs (
                        user_id, timestamp, symbol, shares, avg_price, cash, market_price, total_equity, unrealized_pnl
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        DEFAULT_USER_ID,
                        sync.get("timestamp", ""),
                        sync.get("symbol", ""),
                        int(sync.get("shares", 0)),
                        float(sync.get("avg_price", 0.0)),
                        float(sync.get("cash", 0.0)),
                        float(sync.get("market_price", 0.0)),
                        total_equity,
                        unrealized_pnl,
                    ),
                )
        conn.commit()


def _migrate_legacy_watchlist(config: ToolkitConfig) -> None:
    """Migrate data/watchlist.json into the watchlist_items table."""
    watchlist_path = config.root_dir / "data" / "watchlist.json"
    if not watchlist_path.exists():
        return
    with closing(_connect(config)) as conn:
        row = conn.execute("SELECT 1 FROM watchlist_items WHERE user_id = ? LIMIT 1", (DEFAULT_USER_ID,)).fetchone()
        if row:
            return
        try:
            data = json.loads(watchlist_path.read_text(encoding="utf-8"))
            items = data.get("items", [])
            now = datetime.now().isoformat(timespec="seconds")
            for item in items:
                conn.execute(
                    "INSERT OR IGNORE INTO watchlist_items (user_id, symbol, name, added_at) VALUES (?, ?, ?, ?)",
                    (DEFAULT_USER_ID, item.get("symbol", ""), item.get("name", ""), now),
                )
            conn.commit()
        except Exception:
            pass


def save_portfolio_state(config: ToolkitConfig, mode: str, state: dict, *, user_id: int = DEFAULT_USER_ID) -> None:
    updated_at = datetime.now().isoformat(timespec="seconds")
    positions = state.get("positions", {})
    with closing(_connect(config)) as conn:
        conn.execute(
            """
            INSERT INTO portfolio_state (user_id, mode, initial_cash, initial_equity, cash, realized_pnl, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, mode) DO UPDATE SET
                initial_cash = excluded.initial_cash,
                initial_equity = excluded.initial_equity,
                cash = excluded.cash,
                realized_pnl = excluded.realized_pnl,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                mode,
                float(state.get("initial_cash", 0.0)),
                float(state.get("initial_equity", state.get("initial_cash", 0.0))),
                float(state.get("cash", 0.0)),
                float(state.get("realized_pnl", 0.0)),
                updated_at,
            ),
        )
        conn.execute("DELETE FROM positions WHERE user_id = ? AND mode = ?", (user_id, mode))
        for symbol, position in positions.items():
            conn.execute(
                """
                INSERT INTO positions (user_id, mode, symbol, shares, avg_price, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    mode,
                    symbol,
                    int(position.get("shares", 0)),
                    float(position.get("avg_price", 0.0)),
                    updated_at,
                ),
            )
        conn.commit()


def _load_positions(conn: sqlite3.Connection, mode: str, user_id: int = DEFAULT_USER_ID) -> dict:
    rows = conn.execute(
        "SELECT symbol, shares, avg_price FROM positions WHERE user_id = ? AND mode = ? ORDER BY symbol",
        (user_id, mode),
    ).fetchall()
    return {
        row["symbol"]: {"shares": int(row["shares"]), "avg_price": float(row["avg_price"])}
        for row in rows
    }


def _load_history(conn: sqlite3.Connection, mode: str, user_id: int = DEFAULT_USER_ID) -> list[dict]:
    if mode == "paper":
        rows = conn.execute(
            """
            SELECT timestamp, symbol, action, execution_price, shares_delta, cash_after,
                   position_shares, position_avg_price, status, strategy
            FROM paper_trades
            WHERE user_id = ?
            ORDER BY id
            """,
            (user_id,),
        ).fetchall()
        return [
            {
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "action": row["action"],
                "execution_price": float(row["execution_price"]),
                "shares_delta": int(row["shares_delta"]),
                "cash_after": float(row["cash_after"]),
                "position_after": {
                    "shares": int(row["position_shares"]),
                    "avg_price": float(row["position_avg_price"]),
                },
                "status": row["status"],
                "strategy": row["strategy"],
            }
            for row in rows
        ]

    rows = conn.execute(
        """
        SELECT timestamp, symbol, shares, avg_price, cash, market_price, total_equity, unrealized_pnl
        FROM live_syncs
        WHERE user_id = ?
        ORDER BY id
        """,
        (user_id,),
    ).fetchall()
    return [
        {
            "timestamp": row["timestamp"],
            "symbol": row["symbol"],
            "shares": int(row["shares"]),
            "avg_price": float(row["avg_price"]),
            "cash": float(row["cash"]),
            "market_price": float(row["market_price"]),
            "total_equity": float(row["total_equity"]),
            "unrealized_pnl": float(row["unrealized_pnl"]),
        }
        for row in rows
    ]


def load_portfolio_state(config: ToolkitConfig, mode: str, *, user_id: int = DEFAULT_USER_ID) -> dict:
    ensure_storage(config)
    with closing(_connect(config)) as conn:
        row = conn.execute(
            """
            SELECT initial_cash, initial_equity, cash, realized_pnl
            FROM portfolio_state
            WHERE user_id = ? AND mode = ?
            """,
            (user_id, mode),
        ).fetchone()
        if not row:
            if mode == "paper":
                return {
                    "initial_cash": config.default_paper_cash,
                    "cash": config.default_paper_cash,
                    "positions": {},
                    "trade_history": [],
                    "realized_pnl": 0.0,
                }
            return {
                "initial_cash": 0.0,
                "initial_equity": 0.0,
                "cash": config.default_paper_cash,
                "positions": {},
                "sync_history": [],
                "realized_pnl": 0.0,
            }

        history_key = "trade_history" if mode == "paper" else "sync_history"
        initial_cash = float(row["initial_cash"])
        initial_equity = float(row["initial_equity"])
        if mode == "paper" and initial_cash <= 0:
            initial_cash = config.default_paper_cash
        if mode == "paper" and initial_equity <= 0:
            initial_equity = initial_cash
        return {
            "initial_cash": initial_cash,
            "initial_equity": initial_equity,
            "cash": float(row["cash"]),
            "positions": _load_positions(conn, mode, user_id),
            "realized_pnl": float(row["realized_pnl"]),
            history_key: _load_history(conn, mode, user_id),
        }


def record_paper_trade(config: ToolkitConfig, trade: dict, *, user_id: int = DEFAULT_USER_ID) -> None:
    ensure_storage(config)
    with closing(_connect(config)) as conn:
        conn.execute(
            """
            INSERT INTO paper_trades (
                user_id, timestamp, symbol, action, execution_price, shares_delta, cash_after,
                position_shares, position_avg_price, status, strategy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                trade["timestamp"],
                trade["symbol"],
                trade["action"],
                float(trade["execution_price"]),
                int(trade["shares_delta"]),
                float(trade["cash_after"]),
                int(trade["position_after"]["shares"]),
                float(trade["position_after"]["avg_price"]),
                trade["status"],
                trade["strategy"],
            ),
        )
        conn.commit()


def record_live_sync(config: ToolkitConfig, sync: dict, *, user_id: int = DEFAULT_USER_ID) -> None:
    ensure_storage(config)
    with closing(_connect(config)) as conn:
        conn.execute(
            """
            INSERT INTO live_syncs (
                user_id, timestamp, symbol, shares, avg_price, cash, market_price, total_equity, unrealized_pnl
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                sync["timestamp"],
                sync["symbol"],
                int(sync["shares"]),
                float(sync["avg_price"]),
                float(sync["cash"]),
                float(sync["market_price"]),
                float(sync["total_equity"]),
                float(sync["unrealized_pnl"]),
            ),
        )
        conn.commit()


def save_ta_job(config: ToolkitConfig, job_id: str, symbol: str, trade_date: str, *, user_id: int = DEFAULT_USER_ID) -> None:
    ensure_storage(config)
    created_at = datetime.now().isoformat(timespec="seconds")
    with closing(_connect(config)) as conn:
        conn.execute(
            """
            INSERT INTO ta_analyses (job_id, user_id, symbol, trade_date, status, created_at)
            VALUES (?, ?, ?, ?, 'pending', ?)
            """,
            (job_id, user_id, symbol, trade_date, created_at),
        )
        conn.commit()


def update_ta_job(
    config: ToolkitConfig,
    job_id: str,
    status: str,
    decision: str | None = None,
    reports: dict | None = None,
    error: str | None = None,
) -> None:
    ensure_storage(config)
    completed_at = datetime.now().isoformat(timespec="seconds") if status in ("done", "failed") else None
    with closing(_connect(config)) as conn:
        conn.execute(
            """
            UPDATE ta_analyses
            SET status = ?,
                decision = COALESCE(?, decision),
                reports_json = COALESCE(?, reports_json),
                error = COALESCE(?, error),
                completed_at = COALESCE(?, completed_at)
            WHERE job_id = ?
            """,
            (
                status,
                decision,
                json.dumps(reports, ensure_ascii=False) if reports is not None else None,
                error,
                completed_at,
                job_id,
            ),
        )
        conn.commit()


def load_ta_job(config: ToolkitConfig, job_id: str) -> dict | None:
    ensure_storage(config)
    with closing(_connect(config)) as conn:
        row = conn.execute(
            "SELECT job_id, symbol, trade_date, status, decision, reports_json, error, created_at, completed_at "
            "FROM ta_analyses WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    if not row:
        return None
    return _ta_row_to_dict(row)


def get_latest_ta_analysis(config: ToolkitConfig, symbol: str, *, user_id: int = DEFAULT_USER_ID) -> dict | None:
    ensure_storage(config)
    with closing(_connect(config)) as conn:
        row = conn.execute(
            "SELECT job_id, symbol, trade_date, status, decision, reports_json, error, created_at, completed_at "
            "FROM ta_analyses WHERE user_id = ? AND symbol = ? AND status = 'done' "
            "ORDER BY completed_at DESC LIMIT 1",
            (user_id, symbol),
        ).fetchone()
    if not row:
        return None
    return _ta_row_to_dict(row)


def _ta_row_to_dict(row: sqlite3.Row) -> dict:
    reports = None
    if row["reports_json"]:
        try:
            reports = json.loads(row["reports_json"])
        except Exception:
            reports = {}
    return {
        "job_id":       row["job_id"],
        "symbol":       row["symbol"],
        "trade_date":   row["trade_date"],
        "status":       row["status"],
        "decision":     row["decision"],
        "reports":      reports,
        "error":        row["error"],
        "created_at":   row["created_at"],
        "completed_at": row["completed_at"],
    }


def record_live_order(config: ToolkitConfig, order: dict, order_file: str, *, user_id: int = DEFAULT_USER_ID) -> None:
    ensure_storage(config)
    with closing(_connect(config)) as conn:
        conn.execute(
            """
            INSERT INTO live_orders (
                user_id, created_at, symbol, strategy, recommended_action, execution_price_reference,
                current_shares, current_avg_price, target_shares, order_shares_delta,
                available_cash, note, rationale_json, order_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                order["created_at"],
                order["symbol"],
                order["strategy"],
                order["recommended_action"],
                float(order["execution_price_reference"]),
                int(order["current_shares"]),
                float(order["current_avg_price"]),
                int(order["target_shares"]),
                int(order["order_shares_delta"]),
                float(order["available_cash"]),
                order["note"],
                json.dumps(order.get("rationale", []), ensure_ascii=False),
                order_file,
            ),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Watchlist CRUD
# ---------------------------------------------------------------------------

def get_watchlist(config: ToolkitConfig, *, user_id: int = DEFAULT_USER_ID) -> list[dict]:
    ensure_storage(config)
    with closing(_connect(config)) as conn:
        rows = conn.execute(
            "SELECT symbol, name FROM watchlist_items WHERE user_id = ? ORDER BY added_at",
            (user_id,),
        ).fetchall()
        return [{"symbol": row["symbol"], "name": row["name"]} for row in rows]


def add_watchlist_item(config: ToolkitConfig, symbol: str, name: str = "", *, user_id: int = DEFAULT_USER_ID) -> None:
    ensure_storage(config)
    with closing(_connect(config)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist_items (user_id, symbol, name, added_at) VALUES (?, ?, ?, ?)",
            (user_id, symbol, name or symbol, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()


def remove_watchlist_item(config: ToolkitConfig, symbol: str, *, user_id: int = DEFAULT_USER_ID) -> None:
    ensure_storage(config)
    with closing(_connect(config)) as conn:
        conn.execute(
            "DELETE FROM watchlist_items WHERE user_id = ? AND symbol = ?",
            (user_id, symbol),
        )
        conn.commit()
