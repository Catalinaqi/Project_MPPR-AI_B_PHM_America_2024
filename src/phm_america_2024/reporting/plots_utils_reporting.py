# src/phm_america_2024/reporting/plots_utils_reporting.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.core.helpers_utils_core import ensure_dir

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Centralized plotting save utilities for consistent artifact creation.
#
# Program flow expectation:
# - Stage2/Stage5 call save_fig(...) after creating matplotlib figures.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Reporting utility
# =============================================================================


def save_fig(path: str | Path, dpi: int = 150) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    plt.tight_layout()
    plt.savefig(p, dpi=dpi)
    plt.close()
    log.info("Saved figure: %s", p)
    return p
