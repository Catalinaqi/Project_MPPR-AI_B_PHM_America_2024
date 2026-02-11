# src/phm_america_2024/reporting/artifacts_service_reporting.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.core.helpers_utils_core import ensure_dir, write_json

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Write reporting artifacts (tables/figures/metrics) under output_root.
#
# Program flow expectation:
# - Stages compute DataFrames/Figures/Metrics
# - Stages call save_* functions here to persist artifacts (PNG/JSON) to disk
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Reporting/Artifacts service (I/O boundary)
# =============================================================================


def save_metrics_json(output_root: Path, rel_path: str, metrics: Dict[str, Any]) -> Path:
    """
    Persist a metrics dictionary as JSON under output_root/rel_path.
    """
    p = output_root / rel_path
    ensure_dir(p.parent)
    out = write_json(p, metrics)
    log.info("[save_metrics_json] saved metrics: %s", out)
    return out


def save_table_png(
        df: pd.DataFrame,
        output_root: Path,
        rel_path: str,
        title: str = "",
        max_rows: int = 25,
        dpi: int = 150,
) -> Path:
    """
    Save a basic DataFrame table as PNG (simple renderer).

    Notes
    -----
    - This is a lightweight renderer (useful for small tables).
    - For nicer tables with formatting and adaptive sizing, use save_table_png_pretty_all().
    """
    p = output_root / rel_path
    ensure_dir(p.parent)

    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        ax.axis("off")
        ax.text(0.5, 0.5, "Empty table", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        log.info("[save_table_png] saved empty table: %s", p)
        return p

    view = df.head(max_rows).copy()

    fig, ax = plt.subplots(figsize=(12, 0.45 * (len(view) + 2)), dpi=dpi)
    ax.axis("off")
    if title:
        ax.set_title(title)

    tbl = ax.table(cellText=view.values, colLabels=view.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)

    fig.tight_layout()
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    log.info("[save_table_png] saved table: %s (rows=%d cols=%d)", p, df.shape[0], df.shape[1])
    return p


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
    Render a DataFrame as a high-quality PNG table and save it under output_root.

    This is a reporting artifact renderer:
    - It should not be used for data processing.
    - It formats values, caps table size, and creates readable images.
    """
    p = output_root / rel_path
    ensure_dir(p.parent)

    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(8, 2), dpi=dpi)
        ax.axis("off")
        ax.text(0.5, 0.5, "Empty table", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        log.info("[save_table_png_pretty_all] saved empty table: %s", p)
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

    # ---- format values
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
        return str(x)

    view = view.applymap(_fmt)

    # ---- adaptive figure size
    n_rows, n_cols = view.shape

    max_lens = []
    for j in range(n_cols):
        col_vals = view.iloc[:, j].astype(str)
        max_lens.append(max(col_vals.map(len).max(), len(str(view.columns[j]))))

    col_widths = [min(0.20 * L, 4.0) for L in max_lens]
    fig_width = max(10.0, min(sum(col_widths), 28.0))
    fig_height = max(2.0, min(0.42 * (n_rows + 2), 20.0))

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
    fs = 10 if n_cols <= 8 else 9 if n_cols <= 12 else 8
    tbl.set_fontsize(fs)
    tbl.scale(1.0, 1.25)

    header_bg = "#E6E6E6"
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(weight="bold")
        if c == 0 and r > 0:
            cell._text.set_ha("center")

    fig.tight_layout()
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    log.info("[save_table_png_pretty_all] saved table: %s (rows=%d cols=%d)", p, df.shape[0], df.shape[1])
    return p
