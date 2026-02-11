# src/phm_america_2024/reporting/tables_utils_reporting.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Reporting "table shaping" utilities.
#
# This module does NOT write files.
# It only transforms pandas DataFrames into report-friendly table layouts.
#
# Step-by-step flow (typical usage in Stage2/Stage3)
# -----------------------------------------------------------------------------
# 1) Compute a raw report table (e.g., describe, missingness, cardinality, etc.)
# 2) If needed, call safe_describe(df, ...) to get a robust describe() output
# 3) Convert the output to a "by-feature" layout using as_table_by_column_heuristic(...)
#    - This is critical for readable tables when there are many features
# 4) Pass the final DataFrame to the artifact layer:
#    - artifacts_service_reporting.save_table_png_pretty_all(...)
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Reporting utility (table shaper / formatter)
# =============================================================================


def safe_describe(
        df: pd.DataFrame,
        include: Union[str, List[str]] = "all",
        numeric_only: bool = False,
        context: str = "",
) -> pd.DataFrame:
    """
    Safe wrapper around pandas.DataFrame.describe() for reporting.

    Why this method is critical
    ---------------------------
    Reporting stages must not crash because of edge cases:
    - empty DataFrames (sampling/chunking)
    - DataFrames without numeric columns when numeric_only=True
    This helper returns an empty DataFrame instead of raising.

    How it fits into the reporting flow
    -----------------------------------
    Stage2/Stage3:
      safe_describe(df) -> as_table_by_column_heuristic(...) -> save_table_png_pretty_all(...)

    Parameters
    ----------
    df:
        Input DataFrame.
    include:
        Passed to df.describe(include=...). Use "all" to include numeric + non-numeric stats.
    numeric_only:
        If True, describe only numeric columns (stable for statistics tables).
    context:
        Optional label used only for logging (e.g., "Stage2/X" or "Stage2/Y").

    Returns
    -------
    pd.DataFrame
        describe() output or empty DataFrame if not applicable.
    """
    tag = f"[safe_describe]{'[' + context + ']' if context else ''}"

    if df is None:
        log.warning("%s df is None -> returning empty DataFrame()", tag)
        return pd.DataFrame()

    if df.shape[1] == 0:
        log.warning("%s df has 0 columns -> returning empty DataFrame()", tag)
        return pd.DataFrame()

    log.info(
        "%s start: rows=%d cols=%d numeric_only=%s include=%s",
        tag, df.shape[0], df.shape[1], numeric_only, include
    )

    try:
        if numeric_only:
            num = df.select_dtypes(include="number")
            if num.shape[1] == 0:
                log.warning("%s numeric_only=True but no numeric columns -> empty DataFrame()", tag)
                return pd.DataFrame()
            out = num.describe()
            log.info("%s end: produced describe table shape=%s (numeric_only)", tag, out.shape)
            return out

        out = df.describe(include=include)
        log.info("%s end: produced describe table shape=%s", tag, out.shape)
        return out

    except Exception as e:
        # Reporting MUST NOT crash the pipeline.
        log.exception("%s describe failed -> returning empty DataFrame(). Error: %s", tag, str(e))
        return pd.DataFrame()


def as_table_by_column_heuristic(
        desc_or_indexed: pd.DataFrame,
        original_columns: Optional[List[str]] = None,
        name_col: str = "NameColumn",
        context: str = "",
) -> pd.DataFrame:
    """
    Convert a DataFrame into a "by-feature" (by-column) reporting table.

    Why this method is critical
    ---------------------------
    - Pandas describe() returns a table shaped as:
        rows = statistics (count/mean/std/...)
        cols = features
      This is hard to read in a report when there are many columns.
    - This method turns it into:
        rows = features
        cols = statistics
      plus a dedicated column (NameColumn) for the feature name.

    Heuristic behavior
    ------------------
    Case 1) Input looks like describe() output (stats x features) -> transpose.
    Case 2) Input already looks like a per-feature table:
        - If name_col is already present -> keep as-is
        - Else -> move index into name_col

    Ordering behavior
    -----------------
    If original_columns is provided, rows are ordered to match the original dataset
    column order. This is important for consistency across outputs.

    Parameters
    ----------
    desc_or_indexed:
        DataFrame to convert (describe output or already indexed by feature names).
    original_columns:
        Optional list of dataset columns used to preserve original ordering.
    name_col:
        Column name used to store the feature name.
    context:
        Optional label used only for logging (e.g., "Stage2/X").

    Returns
    -------
    pd.DataFrame
        A DataFrame with a name column and rows ordered for reporting.
    """
    tag = f"[as_table_by_column_heuristic]{'[' + context + ']' if context else ''}"

    if desc_or_indexed is None or desc_or_indexed.empty:
        log.debug("%s input is empty -> returning empty DataFrame()", tag)
        return pd.DataFrame()

    df = desc_or_indexed.copy()

    # Detect typical describe() index labels -> transpose
    stats_like = {"count", "mean", "std", "min", "25%", "50%", "75%", "max", "top", "freq", "unique"}
    idx_as_str = set(map(str, df.index))
    is_describe_like = bool(idx_as_str.intersection(stats_like))

    if is_describe_like:
        # Case 1: stats x features -> transpose into features x stats
        tbl = df.T.reset_index().rename(columns={"index": name_col})
        log.info("%s Case1: describe-like detected -> transposed. in=%s out=%s", tag, df.shape, tbl.shape)
    else:
        # Case 2: already feature-indexed or already has name column
        if name_col in df.columns:
            tbl = df.reset_index(drop=True)
            log.info("%s Case2a: '%s' already present -> as-is. in=%s out=%s", tag, name_col, df.shape, tbl.shape)
        else:
            tbl = df.reset_index().rename(columns={"index": name_col})
            log.warning("%s Case2b: index moved into '%s'. in=%s out=%s", tag, name_col, df.shape, tbl.shape)

    # Keep original dataset column ordering if provided
    if original_columns:
        tbl[name_col] = tbl[name_col].astype(str)
        order = [c for c in original_columns if c in set(tbl[name_col])]
        if order:
            tbl[name_col] = pd.Categorical(tbl[name_col], categories=order, ordered=True)
            tbl = tbl.sort_values(name_col)
            log.debug("%s ordered rows by original_columns (matched=%d)", tag, len(order))
        else:
            log.debug("%s original_columns provided but no matches -> no reordering applied", tag)

    return tbl.reset_index(drop=True)


# -----------------------------------------------------------------------------#
# Before/After helpers (used in Stage3/Feature selection reporting)
# -----------------------------------------------------------------------------#

def build_before_after_tables(
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        step_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two report-friendly tables:
    1) KPI table (single row)
    2) Changes table (removed/added columns)

    This module does NOT write files.
    Stage runners decide how/where to persist artifacts.
    """
    before_cols = list(df_before.columns)
    after_cols = list(df_after.columns)

    removed = sorted(list(set(before_cols) - set(after_cols)))
    added = sorted(list(set(after_cols) - set(before_cols)))

    kpi_df = pd.DataFrame([{
        "step": step_name,
        "rows_before": int(df_before.shape[0]),
        "cols_before": int(df_before.shape[1]),
        "rows_after": int(df_after.shape[0]),
        "cols_after": int(df_after.shape[1]),
        "cols_removed": int(len(removed)),
        "cols_added": int(len(added)),
    }])

    n = max(len(removed), len(added), 1)
    changes_df = pd.DataFrame({
        "step": [step_name] * n,
        "removed_col": removed + [""] * (n - len(removed)),
        "added_col": added + [""] * (n - len(added)),
    })

    log.debug(
        "[build_before_after_tables] step=%s | removed=%d | added=%d",
        step_name, len(removed), len(added)
    )

    return kpi_df, changes_df


def concat_reports(parts: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate report parts safely.
    """
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0, ignore_index=True)


