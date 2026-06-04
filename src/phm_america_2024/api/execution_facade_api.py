# src/phm_america_2024/api/execution_facade_api.py
from __future__ import annotations

# 1-> Imports: Standard-library
#----------------------------------------------
from typing import Any, Optional

# 2-> Imports: Internal imports
#----------------------------------------------
from phm_america_2024.common.logging_adapter_common import get_logger, config_run_logging
from phm_america_2024.common.context_facade_common import RunContext, create_run_context
from phm_america_2024.common.path_service_common import find_project_root
from phm_america_2024.configuration.build_factory_config import build_config, BuiltConfig
from phm_america_2024.configuration.enum_registry_config import StepsPhase,ProblemType
from phm_america_2024.data.download_extractor_data import download_phm_2024_dataset

from phm_america_2024.pipeline.regression_runner_pipeline_back import run_regression_pipeline
#from phm_america_2024.pipeline.classification_runner_pipeline import run_classification_pipeline


# =============================================================================
# Application API for PHM America 2024 competition pipeline
# -----------------------------------------------------------------------------
# Purpose:
#   Unified interface (API) for main.py and notebooks.  Encapsulates
#   initialisation (download, config, logging) and per‑step orchestration.
#
# No hardcoded values — every parameter comes from YAML config files,
# environment variables, or CLI arguments.
#
# Design patterns:
# - GoF: Facade (single entrypoint)
# - Architectural: Application Service / Orchestrator (thin)
#
# Responsibilities:
#   1. init_run_facade_api()  – returns initialised RunContext
#   2. run_2_X_*()            – executes a single CRISP‑DM step
# =============================================================================

log = get_logger(__name__)


def init_run_facade_api(
    pipeline_name: str,
    dataset_key: str,
    notebook_vars: Optional[dict[str, Any]] = None,
) -> RunContext:
    """Initialize a run context for a given pipeline and dataset.

    Step 1: CALL find_project_root() — detect project root.
    Step 2: CALL download_phm_2024_dataset() — download dataset if not already present.
    Step 3: CALL build_config(pipeline_name, dataset_key, notebook_vars) — build pipeline config into built: BuiltConfig (class)
    Step 4: CALL create_run_context(config, dataset_key) — create run context into ctx: RunContext (class)
    Step 5: CALL config_run_logging() — configure logging.
    Step 6: Return initialized RunContext.

    Parameters
    ----------
    pipeline_name : str
        Pipeline name (e.g. ``"regression"``, ``"classification"``).
        Corresponds to ``config/pipeline/<pipeline_name>_pipeline_config.yml``.
    dataset_key : str
        Dataset key in ``dataset_config.yml``.
    notebook_vars : dict[str, Any], optional
        Runtime variables from the notebook context.

    Returns
    -------
    RunContext
        Initialized run context ready for phase execution.
    """
    notebook_vars = notebook_vars or {}

    log.info(
        "[init_run_facade_api] start pipeline=%s dataset_key=%s",
        pipeline_name,
        dataset_key,
    )

    # Step 1: CALL find_project_root() — detect project root
    _ = find_project_root()
    log.debug("[init_run_facade_api] project root detected")

    # Step 2: CALL download_phm_2024_dataset() — download dataset if not already present
    download_phm_2024_dataset()
    log.debug("[init_run_facade_api] dataset available")

    # Step 3: CALL build_config() — store resolved pipeline config into built: BuiltConfig (class)
    built: BuiltConfig = build_config(
        pipeline_name=pipeline_name,
        dataset_key=dataset_key,
        notebook_vars=notebook_vars,
    )
    log.debug("[init_run_facade_api] built config task=%s", built.pipeline_config.common_base_config.problem_type)

    # Step 4: CALL create_run_context() — create run context into ctx: RunContext (class)
    ctx = create_run_context(
        config=built.pipeline_config,
        dataset_key=dataset_key,
    )
    log.debug("[init_run_facade_api] run context created run_id=%s", ctx.run_id)

    # Step 5: CALL config_run_logging() — configure logging
    log_level = built.pipeline_config.common_base_config.runtime.log_level
    output_root = built.pipeline_config.common_base_config.runtime.output_root
    run_name = f"run_{ctx.task}_{dataset_key}_{ctx.run_id}"
    log_file = config_run_logging(
        output_root=output_root,
        run_name=run_name,
        log_level=log_level,
    )

    log.info("[init_run_facade_api] done run_id=%s run_dir=%s log=%s",
             ctx.run_id, ctx.run_dir, log_file)
    return ctx


# -----------------------------------------------------------------
# Phase 2 – Data Understanding : per‑step execution
# -----------------------------------------------------------------

def _dispatch_pipeline(ctx: RunContext, steps: list[str]) -> RunContext:
    """Execute one or more steps via the correct pipeline runner.

    Step 1: Determine task runner based on ctx.task.
    Step 2: CALL runner(ctx, steps) — delegate execution.
    Step 3: Return updated context.

    Parameters
    ----------
    ctx : RunContext
        Run context from init.
    steps : list[str]
        Step names (e.g. ``["step_2_1_data_acquisition"]``).

    Returns
    -------
    RunContext
        Updated context after step execution.
    """
    log.debug("[_dispatch_pipeline] task=%s steps=%s", ctx.task, steps)

    if ctx.task == ProblemType.CLASSIFICATION.value:
        #ctx = run_classification_pipeline(ctx, steps=steps)
        ctx
    elif ctx.task == ProblemType.REGRESSION.value:
        ctx = run_regression_pipeline(ctx, steps=steps)
    else:
        log.error("[_dispatch_pipeline] unknown task=%s", ctx.task)
        raise ValueError(f"Unknown task: {ctx.task}")

    log.debug("[_dispatch_pipeline] completed steps=%s", steps)
    return ctx


def _dispatch_step(ctx: RunContext, step_enum: StepsPhase) -> RunContext:
    """Execute a single CRISP‑DM step.

    Step 1: Extract step string from enum.
    Step 2: CALL _dispatch_pipeline(ctx, [step_value]).
    Step 3: Return updated context.
    """
    step_value = step_enum.value
    log.debug("[_dispatch_step] step=%s", step_value)
    return _dispatch_pipeline(ctx, [step_value])


# ──────────────────────────────────────────────────────────────────────────────
# Public step runners — one per CRISP‑DM sub‑step (Phase 2)
# ──────────────────────────────────────────────────────────────────────────────

def run_2_1_data_acquisition(ctx: RunContext) -> RunContext:
    """Run Step 2.1 – Data Acquisition (load & merge).

    Step 1: CALL _dispatch_step(ctx, StepsPhase.STEP_2_1).
    Step 2: Log shape information if available.
    Step 3: Return updated context.
    """
    log.info("[run_2_1_data_acquisition] start task=%s run_id=%s", ctx.task, ctx.run_id)
    ctx = _dispatch_step(ctx, StepsPhase.STEP_2_1)
    log.debug("[run_2_1_data_acquisition] after dispatch")

    if hasattr(ctx, 'df_train') and ctx.df_train is not None:
        log.info("[run_2_1_data_acquisition] done df_train_shape=%s", ctx.df_train.shape)
    else:
        log.warning("[run_2_1_data_acquisition] done df_train not available")

    return ctx


def run_2_2_data_description(ctx: RunContext) -> RunContext:
    """Run Step 2.2 – Data Description.

    Step 1: CALL _dispatch_step(ctx, StepsPhase.STEP_2_2).
    Step 2: Return updated context.
    """
    log.info("[run_2_2_data_description] start task=%s run_id=%s", ctx.task, ctx.run_id)
    ctx = _dispatch_step(ctx, StepsPhase.STEP_2_2)
    log.debug("[run_2_2_data_description] completed")
    log.info("[run_2_2_data_description] done")
    return ctx


def run_2_3_data_quality_verification(ctx: RunContext) -> RunContext:
    """Run Step 2.3 – Data Quality Verification.

    Step 1: CALL _dispatch_step(ctx, StepsPhase.STEP_2_3).
    Step 2: Return updated context.
    """
    log.info("[run_2_3_data_quality_verification] start task=%s run_id=%s", ctx.task, ctx.run_id)
    ctx = _dispatch_step(ctx, StepsPhase.STEP_2_3)
    log.debug("[run_2_3_data_quality_verification] completed")
    log.info("[run_2_3_data_quality_verification] done")
    return ctx


def run_2_4_data_exploration(ctx: RunContext) -> RunContext:
    """Run Step 2.4 – Data Exploration.

    Step 1: CALL _dispatch_step(ctx, StepsPhase.STEP_2_4).
    Step 2: Return updated context.
    """
    log.info("[run_2_4_data_exploration] start task=%s run_id=%s", ctx.task, ctx.run_id)
    ctx = _dispatch_step(ctx, StepsPhase.STEP_2_4)
    log.debug("[run_2_4_data_exploration] completed")
    log.info("[run_2_4_data_exploration] done")
    return ctx

# ──────────────────────────────────────────────────────────────────────────────
# Public step runners — one per CRISP‑DM sub‑step (Phase 3 – Data Preparation)
# ──────────────────────────────────────────────────────────────────────────────

def run_3_1_data_selection(ctx: RunContext) -> RunContext:
    """Run Step 3.1 – Data Selection (filter columns & drop constants).

    Step 1: CALL _dispatch_step(ctx, StepsPhase.STEP_3_1).
    Step 2: Log shape information if available.
    Step 3: Return updated context.
    """
    log.info("[run_3_1_data_selection] start task=%s run_id=%s", ctx.task, ctx.run_id)
    ctx = _dispatch_step(ctx, StepsPhase.STEP_3_1)
    log.debug("[run_3_1_data_selection] after dispatch")

    if hasattr(ctx, 'df_selected') and ctx.df_selected is not None:
        log.info("[run_3_1_data_selection] done df_selected_shape=%s", ctx.df_selected.shape)
    else:
        log.warning("[run_3_1_data_selection] done df_selected not available")

    return ctx


def run_3_2_data_cleaning(ctx: RunContext) -> RunContext:
    """Run Step 3.2 – Data Cleaning (outlier clipping, duplicate removal).

    Step 1: CALL _dispatch_step(ctx, StepsPhase.STEP_3_2).
    Step 2: Log shape information if available.
    Step 3: Return updated context.
    """
    log.info("[run_3_2_data_cleaning] start task=%s run_id=%s", ctx.task, ctx.run_id)
    ctx = _dispatch_step(ctx, StepsPhase.STEP_3_2)
    log.debug("[run_3_2_data_cleaning] after dispatch")

    if hasattr(ctx, 'df_cleaned') and ctx.df_cleaned is not None:
        log.info("[run_3_2_data_cleaning] done df_cleaned_shape=%s", ctx.df_cleaned.shape)
    else:
        log.warning("[run_3_2_data_cleaning] done df_cleaned not available")

    return ctx


def run_3_3_data_transformation(ctx: RunContext) -> RunContext:
    """Run Step 3.3 – Data Transformation (scaling & feature engineering).

    Step 1: CALL _dispatch_step(ctx, StepsPhase.STEP_3_3).
    Step 2: Log shape and scaler info if available.
    Step 3: Return updated context.
    """
    log.info("[run_3_3_data_transformation] start task=%s run_id=%s", ctx.task, ctx.run_id)
    ctx = _dispatch_step(ctx, StepsPhase.STEP_3_3)
    log.debug("[run_3_3_data_transformation] after dispatch")

    if hasattr(ctx, 'df_transformed') and ctx.df_transformed is not None:
        log.info("[run_3_3_data_transformation] done df_transformed_shape=%s", ctx.df_transformed.shape)
    else:
        log.warning("[run_3_3_data_transformation] done df_transformed not available")

    return ctx


def run_3_5_data_formatting(ctx: RunContext) -> RunContext:
    """Run Step 3.5 – Data Formatting (internal train/val split & type casting).

    Step 1: CALL _dispatch_step(ctx, StepsPhase.STEP_3_5).
    Step 2: Log split shapes if available.
    Step 3: Return updated context.
    """
    log.info("[run_3_5_data_formatting] start task=%s run_id=%s", ctx.task, ctx.run_id)
    ctx = _dispatch_step(ctx, StepsPhase.STEP_3_5)
    log.debug("[run_3_5_data_formatting] after dispatch")

    if hasattr(ctx, 'df_train_split') and ctx.df_train_split is not None:
        log.info("[run_3_5_data_formatting] done train_shape=%s val_shape=%s",
                 ctx.df_train_split.shape,
                 ctx.df_val_split.shape if hasattr(ctx, 'df_val_split') else "N/A")
    else:
        log.warning("[run_3_5_data_formatting] done df_train_split not available")

    return ctx

def run_4_1_algorithm_selection(ctx: RunContext) -> RunContext:
    log.info("[run_4_1_algorithm_selection] start task=%s run_id=%s", ctx.task, ctx.run_id)
    ctx = _dispatch_step(ctx, StepsPhase.STEP_4_1)
    log.debug("[run_4_1_algorithm_selection] completed")
    log.info("[run_4_1_algorithm_selection] done")
    return ctx

def run_4_2_model_training(ctx: RunContext) -> RunContext:
    log.info("[run_4_2_model_training] start task=%s run_id=%s", ctx.task, ctx.run_id)
    ctx = _dispatch_step(ctx, StepsPhase.STEP_4_2)
    log.debug("[run_4_2_model_training] completed")
    log.info("[run_4_2_model_training] done")
    return ctx

def run_4_4_model_evaluation(ctx: RunContext) -> RunContext:
    log.info("[run_4_4_model_evaluation] start task=%s run_id=%s", ctx.task, ctx.run_id)
    ctx = _dispatch_step(ctx, StepsPhase.STEP_4_4)
    log.debug("[run_4_4_model_evaluation] completed")
    log.info("[run_4_4_model_evaluation] done")
    return ctx