# src/phm_america_2024/data/quality_rules_utils_data.py
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Data utilities for "schema/quality" rules that are reused across stages.
#
# Examples:
# - Selecting numeric columns for statistics/EDA/quality checks
# - Avoiding duplicated dtype logic across stage runners
#
# Program flow expectation:
# - Stage runners (e.g., Stage2) call these helpers to decide which columns
#   can be used for numeric-only operations (percentiles, histograms, etc.)
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Utility module (shared, stateless helpers)
# =============================================================================


def numeric_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """
    Return the list of numeric column names from a DataFrame.

    Why this exists
    ---------------
    In reporting/EDA and quality steps we often compute statistics
    (mean/std/percentiles) or draw histograms only for numeric features.
    This helper centralizes the dtype logic and avoids repeating checks
    across stages.

    Parameters
    ----------
    df:
        Input DataFrame.
    exclude:
        Optional list of column names to exclude (e.g., ["id", "faulty"]).

    Returns
    -------
    List[str]
        Numeric column names in the same order as df.columns (minus excluded).
    """
    if df is None or df.shape[1] == 0:
        log.debug("[numeric_cols] empty dataframe -> returning []")
        return []

    exclude_set = set(exclude or [])
    cols: List[str] = []

    for c in df.columns:
        if c in exclude_set:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    log.debug(
        "[numeric_cols] selected %d/%d numeric cols (excluded=%s)",
        len(cols), df.shape[1], sorted(exclude_set) if exclude_set else [],
    )
    return cols
