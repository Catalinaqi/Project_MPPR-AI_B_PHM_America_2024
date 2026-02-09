# src/phm_america_2024/reporting/tables_utils_reporting.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.core.helpers_utils_core import ensure_dir

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Centralized table-to-image utilities for consistent reporting artifacts.
#
# Program flow expectation:
# - Stage2/Stage3/... compute pandas DataFrames for reporting (describe, quality,
#   cardinality, missingness, etc.) and call save_table_png_pretty(...).
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Reporting utility (artifact renderer)
# =============================================================================


def as_table_by_column_heuristic(desc_or_indexed: pd.DataFrame,
                       original_columns: list[str] | None = None,
                       name_col: str = "NameColumn") -> pd.DataFrame:
    """
    Convert a DataFrame to a "by column" table with an explicit NameColumn.

    Supported inputs:
    1) Output of df.describe() -> stats as rows, features as columns.
       We transpose: rows become features, columns become stats.
    2) Any DataFrame whose index already represents feature names (e.g. created with index=cols).
       We reset index into NameColumn.

    Parameters
    ----------
    desc_or_indexed:
        DataFrame to convert.
    original_columns:
        Optional ordering reference (usually list(X.columns)).
    name_col:
        Name for the feature-name column.
    """
    if desc_or_indexed is None or desc_or_indexed.empty:
        return pd.DataFrame()

    df = desc_or_indexed.copy()

    # Case A: typical describe() output (stats x features) -> transpose
    # Heuristic: if most column names look like feature names and index looks like stats
    # we still support generic by always allowing transpose when index contains stats-like labels.
    stats_like = {"count", "mean", "std", "min", "25%", "50%", "75%", "max", "top", "freq", "unique"}
    if set(map(str, df.index)).intersection(stats_like):
        tbl = df.T.reset_index().rename(columns={"index": name_col})
    else:
        # Case B: already indexed by feature names
        if name_col in df.columns:
            tbl = df.reset_index(drop=True)
        else:
            tbl = df.reset_index().rename(columns={"index": name_col})

    # Keep original dataset column ordering if provided
    if original_columns:
        order = [c for c in original_columns if c in set(tbl[name_col].astype(str))]
        if order:
            tbl[name_col] = tbl[name_col].astype(str)
            tbl[name_col] = pd.Categorical(tbl[name_col], categories=order, ordered=True)
            tbl = tbl.sort_values(name_col)

    return tbl.reset_index(drop=True)


def save_table_png_pretty_all(
        df: pd.DataFrame,
        output_root: Path,
        rel_path: str,
        title: str = "",
        max_rows: int = 60,
        dpi: int = 250,
        align: str = "left",
        float_fmt: str = "{:.3f}",
        add_row_ellipsis: bool = True,
) -> Path:
    """
    Save a DataFrame as a high-quality PNG table (gray header, borders, good spacing).

    Notes
    -----
    - This function is for reporting only: do not use it for data processing.
    - It formats numeric values for readability and avoids oversized figures.
    """
    p = output_root / rel_path
    ensure_dir(p.parent)

    if df is None or df.empty:
        # Still create a small image saying "empty"
        fig, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        ax.axis("off")
        ax.text(0.5, 0.5, "Empty table", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        log.info("Saved pretty table PNG (empty): %s", p)
        return p

    view = df.copy()

    # ---- row limit with optional ellipsis row
    if len(view) > max_rows:
        head = view.head(max_rows).copy()
        if add_row_ellipsis:
            ell = pd.DataFrame([["..."] * view.shape[1]], columns=view.columns)
            view = pd.concat([head, ell], ignore_index=True)
        else:
            view = head

    # ---- add sequential "No" column
    view = view.reset_index(drop=True)
    view.insert(0, "No", range(1, len(view) + 1))
    if add_row_ellipsis and len(view) >= 1 and (view.iloc[-1].astype(str) == "...").any():
        view.loc[len(view) - 1, "No"] = "..."

    # ---- format values (avoid very long floats)
    def _fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, (np.floating, float)):
            if np.isnan(x):
                return ""
            try:
                return float_fmt.format(float(x))
            except Exception:
                return str(x)
        if isinstance(x, (np.integer, int)):
            return str(int(x))
        # keep strings as-is
        return str(x)

    view = view.applymap(_fmt)

    # ---- adaptive figure size
    n_rows, n_cols = view.shape

    # Estimate widths based on max string length per column (cap to avoid giant images)
    max_lens = []
    for j in range(n_cols):
        col_vals = view.iloc[:, j].astype(str)
        max_lens.append(max(col_vals.map(len).max(), len(str(view.columns[j]))))

    # Convert length to inches roughly; cap each column
    col_widths = [min(0.20 * L, 4.0) for L in max_lens]
    fig_width = max(10.0, min(sum(col_widths), 28.0))  # cap total width
    fig_height = max(2.0, min(0.42 * (n_rows + 2), 20.0))  # cap height

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    align_map = {"left": "left", "center": "center", "right": "right"}
    cell_loc = align_map.get(align, "left")

    tbl = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        cellLoc=cell_loc,
        loc="center",
    )

    # ---- style
    tbl.auto_set_font_size(False)
    if n_cols <= 8:
        fs = 10
    elif n_cols <= 12:
        fs = 9
    else:
        fs = 8
    tbl.set_fontsize(fs)
    tbl.scale(1.0, 1.25)

    header_bg = "#E6E6E6"
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)

        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(weight="bold")

        # Center the "No" column
        if c == 0 and r > 0:
            cell._text.set_ha("center")

    fig.tight_layout()
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    log.info("Saved pretty table PNG: %s", p)
    return p
