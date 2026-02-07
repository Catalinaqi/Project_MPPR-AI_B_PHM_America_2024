# src/phm_america_2024/features/split_service_features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Centralized train/holdout splitting with stratification for classification.
#
# Program flow expectation:
# - Stage3 takes joined dataset and splits into train/valid sets.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Feature service responsible for dataset splitting
# =============================================================================


@dataclass(frozen=True)
class HoldoutConfig:
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


def split_holdout(df: pd.DataFrame, label_col: str, cfg: HoldoutConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if label_col not in df.columns:
        raise KeyError(f"label_col '{label_col}' not in df columns")

    strat = df[label_col] if cfg.stratify else None

    train_df, valid_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=strat
    )

    log.info("split_holdout: train=%s valid=%s label=%s stratify=%s",
             train_df.shape, valid_df.shape, label_col, cfg.stratify)
    return train_df, valid_df
