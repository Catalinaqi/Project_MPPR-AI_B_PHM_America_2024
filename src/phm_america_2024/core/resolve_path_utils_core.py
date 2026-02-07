# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Path handling is cross-cutting in notebooks vs package execution.
# We centralize "resolve relative/config paths -> absolute paths" here so
# stages/services never guess the working directory.
#
# Program flow expectation:
# - Stage runners receive (cfg, variables, output_root)
# - Any path coming from YAML may be:
#   - a template "${x_train_path}"
#   - a relative path "data/raw/train/X_train.csv"
#   - an absolute path
# - resolve_path() makes it absolute and stable.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Utility module (cross-cutting concern)
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union


def resolve_path(p: Union[str, Path], variables: Dict[str, Any], output_root: Union[str, Path]) -> Path:
    """
    Resolve a path that may be:
    - a template like "${x_train_path}"
    - a relative path like "data/raw/train/X_train.csv"
    - an absolute path
    """
    if not isinstance(output_root, Path):
        output_root = Path(output_root)

    # Template: "${x_train_path}"
    if isinstance(p, str) and p.startswith("${") and p.endswith("}"):
        key = p[2:-1]
        p = variables.get(key, p)

    pp = Path(str(p))

    # If already absolute -> done
    if pp.is_absolute():
        return pp

    # Assume output_root = ".../out" -> project_root = parent of out
    project_root = output_root.resolve().parent
    return (project_root / pp).resolve()
