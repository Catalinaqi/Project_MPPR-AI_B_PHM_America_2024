from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.config.enums_utils_1_config import ProblemType, normalize_problem_type

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# This module validates configuration dictionaries (resolved YAML) before they
# are converted into typed DTOs.
#
# It enforces:
# - minimal required structure (version/pipeline/runtime/stages)
# - basic consistency rules (task known, paths present, etc.)
# - strictness levels: preview vs run
#
# Program flow:
# - load_loader_config.load_and_resolve() -> resolved dict
# - validate_validator_config.validate_config_dict(resolved, mode="preview"|"run")
# - schema_dto_config.ProjectConfig.from_dict(resolved) -> typed config
#
# Design patterns
# - GoF: none
# - Enterprise/Architectural:
#   - Validation Layer (fail-fast)
#   - Configuration Gatekeeper
# =============================================================================


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]

    def raise_if_invalid(self) -> None:
        if not self.ok:
            msg = "Config validation failed:\n- " + "\n- ".join(self.errors)
            raise ValueError(msg)


def _get(d: Dict[str, Any], path: str) -> Any:
    """
    Get nested key using dot path: "pipeline.task".
    """
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def validate_config_dict(
        resolved: Dict[str, Any],
        *,
        mode: str = "preview",
) -> ValidationResult:
    """
    Validate a resolved YAML config dictionary.

    mode:
    - "preview": allow missing target_col/time_col (Stage2 will suggest them)
    - "run": stricter; requires required fields for the chosen task
    """
    log.info("[validate_config_dict] START mode=%s", mode)

    errors: List[str] = []
    warnings: List[str] = []

    # ---- Root structure ----
    if not isinstance(resolved, dict):
        errors.append(f"Root config must be a dict. Got: {type(resolved)}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    version = resolved.get("version")
    if not version:
        warnings.append("Missing 'version'. Default will be assumed by schema layer.")

    pipeline = resolved.get("pipeline")
    runtime = resolved.get("runtime")
    stages = resolved.get("stages")

    if not isinstance(pipeline, dict):
        errors.append("Missing or invalid 'pipeline' block (must be a dict).")
    if not isinstance(runtime, dict):
        errors.append("Missing or invalid 'runtime' block (must be a dict).")
    if not isinstance(stages, dict):
        errors.append("Missing or invalid 'stages' block (must be a dict).")

    if errors:
        log.info("[validate_config_dict] FAILED early errors=%d", len(errors))
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    # ---- Pipeline basic fields ----
    pipe_name = pipeline.get("name")
    pipe_task_raw = pipeline.get("task")
    if not pipe_name:
        errors.append("pipeline.name is required.")
    if not pipe_task_raw:
        errors.append("pipeline.task is required.")

    # Normalize task
    task: Optional[ProblemType] = None
    if pipe_task_raw:
        try:
            task = normalize_problem_type(pipe_task_raw)
        except Exception as e:
            errors.append(f"pipeline.task invalid: {pipe_task_raw}. Error: {e}")

    # ---- Runtime ----
    output_root = runtime.get("output_root")
    if not output_root:
        warnings.append("runtime.output_root missing; default 'out' will be used.")
    #log_level = runtime.get("log_level", "DEBUG")
    #if not isinstance(log_level, str):
        #errors.append("runtime.log_level must be a string (e.g. DEBUG/INFO).")

    # ---- Stage2 presence ----
    stage2 = stages.get("stage2_understanding")
    if stage2 is None:
        warnings.append("stages.stage2_understanding not found. Stage2 preview may not run.")
    elif not isinstance(stage2, dict):
        errors.append("stages.stage2_understanding must be a dict.")
    # ---- Stage3 presence ----
    stage3 = stages.get("stage3_preparation")
    if stage3 is None:
        warnings.append("stages.stage3_preparation not found. Modeling stage may not run.")
    elif not isinstance(stage3, dict):
        errors.append("stages.stage3_preparation must be a dict.")
    # ---- Stage4 presence ----
    stage4 = stages.get("stage4_modeling")
    if stage4 is None:
        warnings.append("stages.stage4_modeling not found. Evaluation stage may not run.")
    elif not isinstance(stage4, dict):
        errors.append("stages.stage4_modeling must be a dict.")
    # ---- Stage5 presence ----
    stage5 = stages.get("stage5_evaluation_and_interpretation")
    if stage5 is None:
        warnings.append("stages.stage5_evaluation_and_interpretation not found. Deployment stage may not run.")
    elif not isinstance(stage5, dict):
        errors.append("stages.stage5_evaluation_and_interpretation must be a dict.")


    # ---- Variables ----
    variables = pipeline.get("variables") or {}
    if not isinstance(variables, dict):
        errors.append("pipeline.variables must be a dict if provided.")

    dataset_path = variables.get("dataset_path")
    # In preview mode, dataset_path must still exist because Stage2 must read CSV.
    if mode == "preview" and not dataset_path:
        errors.append("pipeline.variables.dataset_path is required in preview mode (Stage2 needs it).")

    target_col = variables.get("target_col")
    time_col = variables.get("time_col")

    # ---- Task-specific requirements ----
    if task is not None:
        if mode == "run":
            # stricter:
            if task in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION) and not target_col:
                errors.append(f"target_col is required for task={task.value} in run mode.")
            if task == ProblemType.TIMESERIES and not time_col:
                errors.append("time_col is required for timeseries in run mode.")
            # clustering never requires target
        else:
            # preview: allow missing target/time
            if task in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION) and not target_col:
                warnings.append(f"target_col missing for {task.value} (ok in preview; Stage2 will suggest).")
            if task == ProblemType.TIMESERIES and not time_col:
                warnings.append("time_col missing for timeseries (ok in preview; Stage2 will suggest).")

    ok = len(errors) == 0
    log.info("[validate_config_dict] DONE ok=%s errors=%d warnings=%d", ok, len(errors), len(warnings))
    for w in warnings:
        log.debug("[validate_config_dict][warning] %s", w)
    for e in errors:
        log.debug("[validate_config_dict][error] %s", e)

    return ValidationResult(ok=ok, errors=errors, warnings=warnings)
