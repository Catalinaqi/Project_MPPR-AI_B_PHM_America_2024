# src/phm_america_2024/data/load_utils_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Centralized CSV loading with a simple "read_strategy" (full / sample / chunked).
#
# Program flow expectation:
# - Stage2 loads X_train and Y_train for reporting.
# - Stage3 loads X_train + Y_train for join/split/preprocess.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Data Access utility (not a DAO, but centralized IO)
# =============================================================================


@dataclass(frozen=True)
class ReadStrategy:
    mode: str = "full"          # full | sample | chunked
    sample_rows: int = 200000
    random_state: int = 42
    chunk_size: int = 200000


def read_csv(path: str | Path,
             csv_params: Optional[Dict[str, Any]] = None,
             strategy: Optional[ReadStrategy] = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    csv_params = csv_params or {}
    strategy = strategy or ReadStrategy()

    log.info("read_csv: path=%s mode=%s", p, strategy.mode)

    if strategy.mode == "full":
        df = pd.read_csv(p, **csv_params)
        log.info("read_csv(full): shape=%s", df.shape)
        return df

    if strategy.mode == "sample":
        # Efficient: read first N rows
        df = pd.read_csv(p, nrows=strategy.sample_rows, **csv_params)
        log.info("read_csv(sample): nrows=%s shape=%s", strategy.sample_rows, df.shape)
        return df

    if strategy.mode == "chunked":
        # Read first K chunks and concat (here: just one chunk_size for simplicity)
        it = pd.read_csv(p, chunksize=strategy.chunk_size, **csv_params)
        df = next(it)
        log.info("read_csv(chunked): chunk_size=%s shape=%s", strategy.chunk_size, df.shape)
        return df

    raise ValueError(f"Unknown read_strategy.mode: {strategy.mode}")


def safe_inner_join_x_y(x_df: pd.DataFrame, y_df: pd.DataFrame, join_key: str) -> pd.DataFrame:
    if join_key not in x_df.columns:
        raise KeyError(f"join_key '{join_key}' not in X columns: {list(x_df.columns)}")
    if join_key not in y_df.columns:
        raise KeyError(f"join_key '{join_key}' not in Y columns: {list(y_df.columns)}")

    before = (x_df.shape, y_df.shape)
    df = x_df.merge(y_df, on=join_key, how="inner")
    log.info("join_x_y: join_key=%s before=%s after=%s", join_key, before, df.shape)
    return df
