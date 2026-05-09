from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

from config import ToolkitConfig
from data_sources import future_trading_days

_PREDICTOR_CACHE: dict[tuple[str, str, int], object] = {}


def load_predictor(config: ToolkitConfig, device: str = "cpu"):
    cache_key = (str(config.kronos_root), device, config.max_context)
    if cache_key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[cache_key]

    kronos_root = Path(config.kronos_root)
    if not kronos_root.exists():
        raise FileNotFoundError(f"Kronos root not found: {kronos_root}")

    sys.path.insert(0, str(kronos_root))
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(
        config.tokenizer_name,
        revision=config.tokenizer_revision,
    )
    model = Kronos.from_pretrained(
        config.model_name,
        revision=config.model_revision,
    )
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_context=config.max_context,
    )
    _PREDICTOR_CACHE[cache_key] = predictor
    return predictor


def predict_future(
    config: ToolkitConfig,
    history_df: pd.DataFrame,
    pred_len: int,
    device: str = "cpu",
) -> pd.DataFrame:
    predictor = load_predictor(config, device=device)
    features = ["open", "high", "low", "close", "volume", "amount"]
    context_df = history_df[features].reset_index(drop=True)
    x_timestamp = history_df["timestamps"].reset_index(drop=True)
    y_timestamp = future_trading_days(config, history_df["timestamps"].iloc[-1], pred_len)

    with torch.no_grad():
        pred_df = predictor.predict(
            df=context_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_k=1,
            top_p=1.0,
            sample_count=1,
            verbose=False,
        )

    return pred_df.reset_index().rename(columns={"index": "timestamps"})
