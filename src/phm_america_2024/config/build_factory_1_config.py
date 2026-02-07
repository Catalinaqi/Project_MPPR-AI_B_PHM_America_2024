
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.config.load_loader_1_config import load_and_resolve, load_yaml
from phm_america_2024.config.schema_dto_1_config import ProjectConfig
from phm_america_2024.config.validate_validator_1_config import validate_config_dict
from phm_america_2024.config.enums_utils_1_config import ProblemType, normalize_problem_type
#from phm_america_2024.reporting.audit_service_reporting import save_config_used

import json

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# This module builds the final "resolved configuration" used by the program.
# It is the assembly point for:
# - pipeline config YAML
# - dataset config YAML
# - notebook runtime variables overrides
#
# Program flow:
# - preview_facade_api -> build_preview_config(...)
#   -> load YAMLs
#   -> compute runtime_vars (dataset_path, etc.)
#   -> resolve ${vars} in pipeline YAML
#   -> validate (preview mode)
#   -> convert to DTO (ProjectConfig)
#   -> save audit snapshot (config_used.yml)
#
# Design patterns
# - GoF:
#   - Factory (creates final ProjectConfig)
# - Enterprise/Architectural:
#   - Builder (assemble config from multiple sources)
#   - Single Source of Truth (typed ProjectConfig)
# =============================================================================


@dataclass(frozen=True)
class BuiltConfig:
    """
    Returned by builder functions to keep both typed config and raw dicts.
    """
    project_config: ProjectConfig
    resolved_dict: Dict[str, Any]
    #audit_path: Path

def _select_dataset_path(dataset_cfg: Dict[str, Any], *, split: str = "train") -> str:
    """
    Dataset config format expected:
      datasets:
        <key>:
          paths:
            train: "..."
            test: "..."
    """
    log.info("[_select_dataset_path] START dataset_cfg transformed to json: %s", json.dumps(dataset_cfg, indent=2,ensure_ascii=False))
    datasets = dataset_cfg.get("datasets") or {} # no exist en el dict
    log.info("[_select_dataset_path] datasets keys: %s", list(datasets.keys()))
    paths = dataset_cfg.get("paths") or {}
    log.info("Selecting dataset path for split='%s' from paths: %s", split, paths)
    if not isinstance(paths, dict):
        raise ValueError("dataset entry must contain paths:{train:,test:} mapping.")
    p = paths.get(split)
    if not p:
        raise ValueError(f"Dataset paths missing '{split}'.")
    log.info("End [_select_dataset_path]: selected dataset path='%s'", p)
    return str(p)


def _load_dataset_entry(dataset_config_path: Path, dataset_key: str) -> Dict[str, Any]:
    """
    Load a single dataset entry from dataset_config.yml.
    Raises KeyError if dataset_key not found.
    :param dataset_config_path:
    :param dataset_key:
    :return:
    """
    log.info("[_load_dataset_entry] START Loading dataset entry for key='%s' and path='%s'", dataset_key, dataset_config_path)

    raw = load_yaml(dataset_config_path)

    datasets = raw.get("datasets") or {}



    if dataset_key not in datasets:
        raise KeyError(f"dataset_key '{dataset_key}' not found in {dataset_config_path}. Available: {list(datasets.keys())}")
    entry = datasets[dataset_key] or {}
    if not isinstance(entry, dict):
        raise ValueError(f"datasets.{dataset_key} must be a dict.")
    elif dataset_key in datasets:
        log.info("End [_load_dataset_entry]: Loading dataset entry for key='%s'", dataset_key)

    return {"version": raw.get("version", "1.0"), "key": dataset_key, **entry}