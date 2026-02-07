# src/phm_america_2024/core/helpers_utils_core.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Small, reusable helpers (filesystem + JSON utilities) used across stages.
#
# Program flow expectation:
# - Stage runners call helpers to create directories and write artifacts.
#
# Design patterns
# - GoF: none (utility module).
# - Enterprise/Architectural:
#   - Cross-cutting utility layer shared across services/runners.
# =============================================================================


def ensure_dir(path: str | Path) -> Path:
    """Create directory (parents=True) if missing and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    log.debug("ensure_dir: %s", p)
    return p


def write_json(path: str | Path, data: Dict[str, Any], indent: int = 2) -> Path:
    """Write JSON to disk (UTF-8)."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    log.info("write_json: %s", p)
    return p
