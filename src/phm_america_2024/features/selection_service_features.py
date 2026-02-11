from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Stage 3.1 (Data Selection / Feature Selection):
# - Remove non-informative features (constant / near-constant).
# - Remove redundant features (duplicate columns).
#
# IMPORTANT:
# - This module does NOT decide business rules.
# - It does NOT write artifacts to disk.
# - It only transforms DataFrames and reports what changed.
#
# Program flow expectation (typical usage in Stage3 runner):
# 1) Stage3 runner builds `feature_cols` (excluding label + technical keys like "id")
# 2) Stage3 runner calls:
#       res = remove_constant_features(df, feature_cols, threshold_unique)
#       df = res.df
#    then:
#       res = remove_duplicate_features(df, feature_cols, strategy="exact")
#       df = res.df
# 3) Stage3 runner builds before/after tables and saves them using reporting layer.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Service module (pure transformation + change reporting).
# =============================================================================


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Return object used by stage runners to keep code clean and explicit."""
    df: pd.DataFrame
    removed_cols: List[str]


def remove_constant_features(
        df: pd.DataFrame,
        feature_cols: Optional[Iterable[str]] = None,
        threshold_unique: int = 1,
) -> FeatureSelectionResult:
    """
    Remove constant / near-constant features based on number of distinct values.

    Parameters
    ----------
    df:
        Input DataFrame (contains keys/labels + features).
    feature_cols:
        Columns to be evaluated as candidate features.
        NOTE: stage runner should EXCLUDE 'id' and label columns here.
    threshold_unique:
        Remove a feature if nunique <= threshold_unique.
        - 1 => constant features only.

    Returns
    -------
    FeatureSelectionResult:
        - df: transformed df with removed columns dropped
        - removed_cols: list of removed feature names
    """
    # --- flow log
    log.debug("[remove_constant_features] start | threshold_unique=%s", threshold_unique)

    if feature_cols is None:
        feature_cols = df.columns
    feature_cols = list(feature_cols)

    if not feature_cols:
        log.info("[remove_constant_features] feature_cols is empty -> nothing to do.")
        return FeatureSelectionResult(df=df, removed_cols=[])

    nunique = df[feature_cols].nunique(dropna=False)
    to_remove = nunique[nunique <= threshold_unique].index.tolist()

    if not to_remove:
        log.info("[remove_constant_features] no constant features found.")
        return FeatureSelectionResult(df=df, removed_cols=[])

    log.info(
        "[remove_constant_features] removing %d feature(s): %s",
        len(to_remove),
        to_remove,
    )
    df_out = df.drop(columns=to_remove)

    # --- flow log
    log.debug("[remove_constant_features] end | removed=%d", len(to_remove))
    return FeatureSelectionResult(df=df_out, removed_cols=to_remove)

def remove_duplicate_features(
        df: pd.DataFrame,
        feature_cols: Optional[Iterable[str]] = None,
        strategy: str = "exact",
) -> FeatureSelectionResult:
    """
    Fast duplicate-column removal.
    strategy:
      - "exact": drop columns that are exactly identical to a previous column
    """
    log.debug("[remove_duplicate_features] start | strategy=%s", strategy)

    if feature_cols is None:
        feature_cols = df.columns
    feature_cols = list(feature_cols)

    if not feature_cols:
        log.info("[remove_duplicate_features] feature_cols is empty -> nothing to do.")
        return FeatureSelectionResult(df=df, removed_cols=[])

    if strategy != "exact":
        raise ValueError(f"Unsupported strategy={strategy}. Only 'exact' is implemented.")

    # --- FAST PATH: hash each column and detect duplicates by hash
    # Notes:
    # - hash_pandas_object is vectorized and fast.
    # - We hash the entire column (all rows) into a single signature.
    # - If two signatures match, we do an exact equality check to be safe.
    from pandas.util import hash_pandas_object

    seen = {}          # signature -> kept column name
    to_remove = []

    for col in feature_cols:
        s = df[col]
        # create a stable signature for the whole column
        sig = int(hash_pandas_object(s, index=False).sum())

        if sig not in seen:
            seen[sig] = col
            continue

        # collision-safe check (rare, but we do it)
        kept = seen[sig]
        if s.equals(df[kept]):
            to_remove.append(col)

    if not to_remove:
        log.info("[remove_duplicate_features] no duplicate features found.")
        log.debug("[remove_duplicate_features] end | removed=0")
        return FeatureSelectionResult(df=df, removed_cols=[])

    log.info("[remove_duplicate_features] removing %d duplicate feature(s): %s", len(to_remove), to_remove)
    df_out = df.drop(columns=to_remove)

    log.debug("[remove_duplicate_features] end | removed=%d", len(to_remove))
    return FeatureSelectionResult(df=df_out, removed_cols=to_remove)

