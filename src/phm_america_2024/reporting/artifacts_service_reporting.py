# src/phm_america_2024/reporting/artifacts_service_reporting.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.core.helpers_utils_core import ensure_dir, write_json
from phm_america_2024.reporting.plots_utils_reporting import save_fig

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Write artifacts (tables/figures/metrics) under output_root.
#
# Program flow expectation:
# - Stages call: save_table_png / save_metrics_json
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Reporting/Artifacts service
# =============================================================================


def save_metrics_json(output_root: Path, rel_path: str, metrics: Dict[str, Any]) -> Path:
    return write_json(output_root / rel_path, metrics)


def save_table_png(
        df: pd.DataFrame,
        output_root: Path,
        rel_path: str,
        title: str = "",
        max_rows: int = 25,
        dpi: int = 150) -> Path:

    p = output_root / rel_path
    ensure_dir(p.parent)

    view = df.head(max_rows).copy()

    fig, ax = plt.subplots(figsize=(12, 0.45 * (len(view) + 2)))
    ax.axis("off")
    ax.set_title(title)

    tbl = ax.table(cellText=view.values, colLabels=view.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)

    plt.tight_layout()
    fig.savefig(p, dpi=dpi)
    plt.close(fig)

    log.info("Saved table PNG: %s", p)
    return p



def save_table_png_pretty(
        df: pd.DataFrame,
        output_root: Path,
        rel_path: str,
        title: str = "",
        max_rows: int = 25,
        dpi: int = 250,
        align: str = "left",
        float_fmt: str = "{:.4f}",
        add_row_ellipsis: bool = True,
) -> Path:
    p = output_root / rel_path
    ensure_dir(p.parent)

    view = df.copy()

    # ---- 1) limita filas con "..."
    if len(view) > max_rows:
        head = view.head(max_rows).copy()
        if add_row_ellipsis:
            ell = pd.DataFrame([["..."] * view.shape[1]], columns=view.columns)
            view = pd.concat([head, ell], ignore_index=True)
        else:
            view = head

    # ---- 2) añade columna No. (bonita)
    view = view.reset_index(drop=True)
    view.insert(0, "No", range(1, len(view) + 1))
    if add_row_ellipsis and len(view) >= 1 and (view.iloc[-1].astype(str) == "...").any():
        view.loc[len(view) - 1, "No"] = "..."

    # ---- 3) formatea números para que no salgan 99999.557735171...
    def _fmt(x):
        if x is None:
            return ""
        if isinstance(x, (np.floating, float)):
            if np.isnan(x):
                return ""
            return float_fmt.format(x)
        if isinstance(x, (np.integer, int)):
            return str(int(x))
        return str(x)

    view = view.applymap(_fmt)

    # ---- 4) calcula tamaño figura (ancho basado en nº de columnas y texto)
    n_rows, n_cols = view.shape
    # ancho aproximado según longitud de strings (cap para que no explote)
    max_lens = [max(view.iloc[:, j].astype(str).map(len).max(), len(str(view.columns[j]))) for j in range(n_cols)]
    col_widths = [min(0.22 * L, 4.0) for L in max_lens]  # cap
    fig_width = max(10, sum(col_widths))
    fig_height = max(2.0, 0.42 * (n_rows + 2))

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

    # ---- 5) estilo: header gris, bordes, fuentes
    tbl.auto_set_font_size(False)

    # font size adaptativo por columnas
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

        # centrar la columna No
        if c == 0 and r > 0:
            cell._text.set_ha("center")

    # ---- 6) evita recortes feos
    fig.tight_layout()
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    log.info("Saved pretty table PNG: %s", p)
    return p
