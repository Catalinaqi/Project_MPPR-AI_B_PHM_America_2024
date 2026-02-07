# src/phm_america_2024/reporting/artifacts_service_reporting.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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


def save_table_png(df: pd.DataFrame, output_root: Path, rel_path: str, title: str = "", max_rows: int = 25, dpi: int = 150) -> Path:
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
