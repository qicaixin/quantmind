from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ToolkitConfig:
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    kronos_root: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "KRONOS_ROOT",
                str(Path(__file__).resolve().parent / "temp" / "Kronos"),
            )
        )
    )
    qlib_data_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "QLIB_DATA_PATH",
                str(Path(__file__).resolve().parent / "data" / "qlib_cn"),
            )
        )
    )
    output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "KRONOS_TOOLKIT_OUTPUT",
                str(Path(__file__).resolve().parent / "outputs"),
            )
        )
    )
    model_name: str = "NeoQuasar/Kronos-small"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    model_revision: str = "901c26c1332695a2a8f243eb2f37243a37bea320"
    tokenizer_revision: str = "0e0117387f39004a9016484a186a908917e22426"
    # Kronos-small's internal attention/time mask supports 240 steps.
    max_context: int = 240
    default_lookback: int = 400
    default_pred_len: int = 20
    signal_threshold: float = 0.0
    transaction_cost_rate: float = 0.0015
    recommended_strategy: str = "forecast_trend"
    default_paper_cash: float = 100000.0
    lot_size: int = 100

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.qlib_data_path.mkdir(parents=True, exist_ok=True)

    def user_output_dir(self, user_id: int) -> Path:
        """Per-user output directory: outputs/{user_id}/"""
        d = self.output_dir / str(user_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def paper_state_path(self) -> Path:
        return self.output_dir / "paper_portfolio.json"

    @property
    def trading_db_path(self) -> Path:
        return self.output_dir / "trading.sqlite3"

    @property
    def live_state_path(self) -> Path:
        return self.output_dir / "live_portfolio.json"

    @property
    def live_order_dir(self) -> Path:
        return self.output_dir / "manual_live_orders"

    def user_live_order_dir(self, user_id: int) -> Path:
        d = self.user_output_dir(user_id) / "manual_live_orders"
        d.mkdir(parents=True, exist_ok=True)
        return d
