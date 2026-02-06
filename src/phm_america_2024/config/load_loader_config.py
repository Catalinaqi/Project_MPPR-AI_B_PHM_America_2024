# src/crispdm/config/load_loader_config.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from crispdm.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# "load.py" layer:
# - Loads YAML files as Python dicts.
# - Resolves ${var} placeholders using runtime_vars injected by notebook/builders.
# - Keeps raw YAML reading concerns isolated from the rest of the pipeline.
#
# Program flow:
# - build_factory_config/build_preview_config:
#     -> load_yaml() / load_and_resolve()
#     -> returns resolved dict
# - schema_dto_config.ProjectConfig.from_dict() consumes resolved dict
#
# Design patterns
# - GoF: none
# - Enterprise/Architectural:
#   - Configuration Loader
#   - Configuration Templating (placeholder resolution)
# =============================================================================

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


@dataclass(frozen=True)
class LoadedYaml:
    """
    raw:
      YAML loaded as dict (may contain ${var} placeholders)
    resolved:
      YAML dict after placeholder substitution
    variables:
      merged variables map (YAML defaults + notebook overrides)
    """
    raw: Dict[str, Any]
    resolved: Dict[str, Any]
    variables: Dict[str, Any]
    source_path: Path


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load YAML from disk. Root must be a mapping (dict).
    """
    p = Path(path)
    log.debug("load_yaml: path=%s", p)

    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")

    content = p.read_text(encoding="utf-8")
    data = yaml.safe_load(content) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict. Got: {type(data)}")

    log.debug("load_yaml: loaded keys=%s", list(data.keys()))
    return data


def merge_variables(raw: Dict[str, Any], runtime_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge variables from YAML with overrides from notebook/runtime.

    Expected YAML location:
      raw['pipeline']['variables']

    Rule:
      notebook/runtime vars override YAML defaults.
    """
    runtime_vars = runtime_vars or {}

    pipeline = raw.get("pipeline", {})
    if not isinstance(pipeline, dict):
        pipeline = {}

    yaml_vars = pipeline.get("variables", {})
    if not isinstance(yaml_vars, dict):
        yaml_vars = {}

    merged = dict(yaml_vars)
    merged.update(runtime_vars)

    log.debug("merge_variables: yaml_vars=%s runtime_vars=%s",
              sorted(list(yaml_vars.keys())), sorted(list(runtime_vars.keys())))
    return merged


def _resolve_string(template: str, variables: Dict[str, Any]) -> Any:
    """
    Resolve ${var} placeholders inside a string.

    Special case:
    - if the string is exactly "${var}", return the raw value (can be None/list/int),
      not a string.
    """
    matches = list(_VAR_PATTERN.finditer(template))
    if not matches:
        return template

    if len(matches) == 1 and matches[0].span() == (0, len(template)):
        key = matches[0].group(1).strip()
        val = variables.get(key)
        log.debug("_resolve_string: whole-token key=%s -> %r", key, val)
        return val

    def repl(m: re.Match) -> str:
        key = m.group(1).strip()
        val = variables.get(key)
        return "" if val is None else str(val)

    resolved = _VAR_PATTERN.sub(repl, template)
    return resolved


def resolve_placeholders(obj: Any, variables: Dict[str, Any]) -> Any:
    """
    Recursively resolve placeholders in dict/list/str.
    """
    if isinstance(obj, dict):
        return {k: resolve_placeholders(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_placeholders(v, variables) for v in obj]
    if isinstance(obj, str):
        return _resolve_string(obj, variables)
    return obj


def find_unresolved_placeholders(obj: Any) -> Tuple[bool, int]:
    """
    Scan a resolved object and detect leftover ${...} placeholders.
    Returns: (has_unresolved, count)
    """
    count = 0

    def _walk(x: Any) -> None:
        nonlocal count
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)
        elif isinstance(x, str):
            if _VAR_PATTERN.search(x):
                count += 1

    _walk(obj)
    return (count > 0, count)


def load_and_resolve(path: str | Path,
                     runtime_vars: Optional[Dict[str, Any]] = None) -> LoadedYaml:
    """
    Main entrypoint:
    - load YAML
    - merge variables
    - resolve ${var} placeholders
    """
    log.info("load_and_resolve: start path=%s", path)
    log.debug("load_and_resolve: start path=%s", path)

    raw = load_yaml(path)
    variables = merge_variables(raw, runtime_vars)
    resolved = resolve_placeholders(raw, variables)

    has_unresolved, n = find_unresolved_placeholders(resolved)
    log.debug("load_and_resolve: unresolved placeholders=%s", has_unresolved)
    #print("Unresolved placeholders?", has_unresolved, "count:", n)
    log.debug("load_and_resolve: unresolved placeholders found=%d", n)
    if has_unresolved:
        log.warning("load_and_resolve: unresolved placeholders found=%d (check notebook vars)", n)
    else:
        log.debug("load_and_resolve: all placeholders resolved")

    log.info("load_and_resolve: done")
    return LoadedYaml(
        raw=raw,
        resolved=resolved,
        variables=variables,
        source_path=Path(path),
    )
