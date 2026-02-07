# src/phm_america_2024/pipelines/pipeline_context.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from phm_america_2024.config.load_loader_config import load_yaml, resolve_placeholders
from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.core.seeds_utils_core import set_global_seed

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Notebook-friendly context builder:
# - loads dataset_config.yml + pipeline_config.yml
# - builds variables dict
# - resolves ${...} placeholders
# - sets global seed
#
# Program flow expectation:
# - Notebook calls build_context()
# - Then calls run_stage2, run_stage3, run_stage4, run_stage5 manually.
#
# Design patterns
# - GoF: Builder (context builder).
# - Enterprise/Architectural:
#   - Orchestration helper / Config assembly
# =============================================================================


@dataclass
class PipelineContext:
    cfg: Dict[str, Any]
    variables: Dict[str, Any]
    output_root: Path


def build_context(dataset_config_path: str | Path,
                  pipeline_config_path: str | Path,
                  output_root: str | Path = "out") -> PipelineContext:
    output_root = Path(output_root)

    dataset_cfg = load_yaml(dataset_config_path)
    pipe_cfg_raw = load_yaml(pipeline_config_path)

    ds = dataset_cfg["datasets"]
    train = ds["phm_north_america_2024_train"]["paths"]
    test = ds["phm_north_america_2024_test"]["paths"]
    val = ds["phm_north_america_2024_validation"]["paths"]

    variables: Dict[str, Any] = {
        "x_train_path": train["x_train_path"],
        "y_train_path": train["y_train_path"],
        "x_test_path": test["x_test_path"],
        "x_validation_path": val["x_validation_path"],
        "output_root": str(output_root),
    }

    # merge pipeline.variables (you hardcode join_key + label_col there)
    pipeline_vars = pipe_cfg_raw.get("pipeline", {}).get("variables", {}) or {}
    pipeline_vars = resolve_placeholders(pipeline_vars, variables)
    variables.update(pipeline_vars)

    # resolve entire pipeline config
    cfg = resolve_placeholders(pipe_cfg_raw, variables)

    # set seed
    seed = int(cfg.get("runtime", {}).get("random_seed", 42))
    set_global_seed(seed)

    log.info("Context built. output_root=%s", output_root)
    log.info("Variables keys=%s", sorted(list(variables.keys())))

    return PipelineContext(cfg=cfg, variables=variables, output_root=output_root)
