# src/phm_america_2024/config/load_loader_config.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import yaml

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Load YAML configs and resolve simple ${var} placeholders from a variables dict.
# (No framework, just predictable config loading.)
#
# Program flow expectation:
# - Notebook loads dataset_config.yml and pipeline_config.yml
# - Pipeline runner merges variables and resolves placeholders.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Configuration Loader / DTO-ish dict config
# =============================================================================

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    log.info("Loading YAML: %s", p)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {p}")
    return data


def resolve_placeholders(obj: Any, variables: Dict[str, Any]) -> Any:
    """
    Recursively resolve ${var} placeholders in strings.
    Unknown vars are left as-is (so you notice them).
    """
    if isinstance(obj, dict):
        return {k: resolve_placeholders(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_placeholders(v, variables) for v in obj]
    if isinstance(obj, str):
        def repl(m: re.Match) -> str:
            key = m.group(1)
            val = variables.get(key)
            return str(val) if val is not None else m.group(0)

        return _VAR_PATTERN.sub(repl, obj)
    return obj
