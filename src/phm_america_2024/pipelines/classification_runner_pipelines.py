# src/phm_america_2024/pipelines/classification_runner_pipelines.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.core.seeds_utils_core import set_global_seed
from phm_america_2024.config.load_loader_config import load_yaml, resolve_placeholders
from phm_america_2024.stages.stage2_understanding_runner_stages import run_stage2
from phm_america_2024.stages.stage3_preparation_runner_stages import run_stage3
from phm_america_2024.stages.stage4_modeling_runner_stages import run_stage4
from phm_america_2024.stages.stage5_evaluation_runner_stages import run_stage5

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Orchestrates the classification pipeline using your YAML configuration.
#
# Program flow expectation:
# - Notebook calls run_classification_pipeline(dataset_yml, pipeline_yml, output_root)
#
# Design patterns
# - GoF: Facade (pipeline runner facade).
# - Enterprise/Architectural:
#   - Pipeline Orchestrator / Runner
# =============================================================================


def run_classification_pipeline(dataset_config_path: str | Path,
                                pipeline_config_path: str | Path,
                                output_root: str | Path = "out") -> Dict[str, Any]:
    output_root = Path(output_root)

    dataset_cfg = load_yaml(dataset_config_path)
    pipe_cfg_raw = load_yaml(pipeline_config_path)

    # Build variables from dataset_config.yml + pipeline defaults
    ds = dataset_cfg["datasets"]
    train = ds["phm_north_america_2024_train"]["paths"]
    test = ds["phm_north_america_2024_test"]["paths"]
    val = ds["phm_north_america_2024_validation"]["paths"]

    variables = {
        "x_train_path": train["x_train_path"],
        "y_train_path": train["y_train_path"],
        "x_test_path": test["x_test_path"],
        "x_validation_path": val["x_validation_path"],
        "output_root": str(output_root),
    }

    # Merge YAML pipeline variables (like join_key/label_col set hardcoded there)
    pipeline_vars = pipe_cfg_raw.get("pipeline", {}).get("variables", {}) or {}
    # Resolve placeholders inside those too (if any)
    pipeline_vars = resolve_placeholders(pipeline_vars, variables)
    variables.update(pipeline_vars)

    # Resolve the entire pipeline config with final variables
    pipe_cfg = resolve_placeholders(pipe_cfg_raw, variables)

    # Seed
    seed = int(pipe_cfg.get("runtime", {}).get("random_seed", 42))
    set_global_seed(seed)

    log.info("Run classification pipeline: output_root=%s", output_root)
    log.info("Variables: %s", {k: variables[k] for k in sorted(variables.keys())})

    # STAGES
    _ = run_stage2(pipe_cfg, variables, output_root)
    s3 = run_stage3(pipe_cfg, variables, output_root)
    s4 = run_stage4(pipe_cfg, s3, output_root)
    s5 = run_stage5(pipe_cfg, s3, s4, output_root)

    return {"stage3": s3, "stage4": s4, "stage5": s5}
