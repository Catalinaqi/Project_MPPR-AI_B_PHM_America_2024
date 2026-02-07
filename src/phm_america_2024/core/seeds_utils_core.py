# src/phm_america_2024/core/seeds_utils_core.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Reproducibility utilities: seed Python random + NumPy (and optionally others).
#
# Program flow expectation:
# - Pipeline runner sets seed once at the beginning of a run.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Cross-cutting concern / Reproducibility
# =============================================================================


def set_global_seed(seed: int, set_hash_seed: bool = True) -> None:
    if set_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    log.info("Global seed set: %s", seed)
