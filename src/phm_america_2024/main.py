# src/phm_america_2024/main.py
from __future__ import annotations

import sys
import argparse
from typing import Any

# Import all facade step orchestration API functions
from phm_america_2024.api.execution_facade_api import (
    init_run_facade_api,
    run_2_1_data_acquisition,
    run_2_2_data_description,
    run_2_3_data_quality_verification,
    run_2_4_data_exploration,
    run_3_1_data_selection,
    run_3_2_data_cleaning,
    run_3_3_data_transformation,
    run_3_5_data_formatting,
    run_4_1_algorithm_selection,
    run_4_2_model_training,
    run_4_4_model_evaluation,
    run_5_1_interpretation,
    run_5_2_probabilistic_evaluation,
    run_5_3_process_audit,
    run_5_4_decision_making

)
from phm_america_2024.common.logging_adapter_common import get_logger

# ── INVERSION OF CONTROL: CENTRALIZED ARTIFACT REGISTRY ─────────────────────
# By importing these modules, Python runs their internal @register_artifact
# decorators to dynamically populate the runtime generator dictionary.
from phm_america_2024.registry import phase2_generator_registry
from phm_america_2024.registry import phase3_generator_registry
from phm_america_2024.registry import phase4_generator_registry
from phm_america_2024.registry import phase5_generator_registry

from phm_america_2024.registry.generator_registry_registry import get_registered_generators

# Connect with the centralized configuration infrastructure repository
from phm_america_2024.configuration.yml_repository_config import YmlRepository

log = get_logger(__name__)

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Step 1: Define --pipeline (required), --dataset (required), --steps (optional).
    Step 2: Return parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="PHM America 2024 – CRISP-DM Pipeline CLI"
    )

    parser.add_argument(
        "--resume_run",
        type=str,
        default=None,
        help="Run ID to resume (e.g. '20260604_153111'). If provided, it won't create a new run folder.",
    )

    parser.add_argument(
        "--pipeline",
        required=True,
        help="Pipeline name (e.g. 'regression', 'classification')",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset key in dataset_config.yml (e.g. 'phm2024')",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        default=["2.1"],
        choices=["2.1", "2.2", "2.3", "2.4", "3.1", "3.2", "3.3", "3.5", "4.1",
                 "4.2", "4.4", "4.5","5.1","5.2","5.3","5.4"],
        help="Phase steps to execute (e.g. '2.1' '2.2'). Default: '2.1'",
    )
    return parser.parse_args()


def _execute_pipeline_steps(ctx: Any, steps: list[str]) -> Any:
    """Execute requested Phase 2 and Phase 3 steps based on logical DAG order.

    Step 1: Define step metadata map and execution order.
    Step 2: Sequentially process each step requested by user CLI arguments.
    Step 3: Return updated run context.
    """
    step_map = {
        "2.1": ("Data Acquisition", run_2_1_data_acquisition),
        "2.2": ("Data Description", run_2_2_data_description),
        "2.3": ("Data Quality Verification", run_2_3_data_quality_verification),
        "2.4": ("Data Exploration", run_2_4_data_exploration),
        "3.1": ("Data Selection", run_3_1_data_selection),
        "3.2": ("Data Cleaning", run_3_2_data_cleaning),
        "3.3": ("Data Transformation", run_3_3_data_transformation),
        "3.5": ("Data Formatting", run_3_5_data_formatting),
        "4.1": ("Algorithm Selection", run_4_1_algorithm_selection),
        "4.2": ("Model Training", run_4_2_model_training),
        "4.4": ("Model Evaluation", run_4_4_model_evaluation),
        "5.1": ("Interpretation", run_5_1_interpretation),
        "5.2": ("Probabilistic Evaluation", run_5_2_probabilistic_evaluation),
        "5.3": ("Process Audit", run_5_3_process_audit),
        "5.4": ("Decision Making", run_5_4_decision_making),
    }

    # Strict operational sequence order for the pipeline execution loop
    execution_sequence = ["2.1", "2.2", "2.3", "2.4", "3.1", "3.2", "3.3", "3.5",
                          "4.1", "4.2", "4.4","5.1", "5.2", "5.3", "5.4",]

    for step_key in execution_sequence:
        if step_key in steps:
            name, func = step_map[step_key]
            log.info("Executing Pipeline Step %s – %s", step_key, name)
            ctx = func(ctx)
            log.info("Pipeline Step %s completed", step_key)

    return ctx


def main() -> int:
    """CLI entry point.

    Step 1: Parse arguments.
    Step 2: Load merged pipeline and dataset configs via YmlRepository.
    Step 3: CALL init_run_facade_api(pipeline_name, dataset_key).
    Step 4: CALL _execute_pipeline_steps(ctx, steps).
    Step 5: Log success and return exit code.
    """

    args = _parse_args()

    log.info("=" * 60)
    log.info("PHM America 2024 – Pipeline Execution Started")
    log.info(f"Pipeline: {args.pipeline}, Dataset: {args.dataset}, Steps: {args.steps}")
    log.info("=" * 60)

    try:
        # Load all target configurations using the centralized configuration repository.
        # This triggers automatic deep merges and profile resolution tasks.
        log.info("[main] Resolving configuration files via YmlRepository...")
        pipeline_cfg = YmlRepository.load_pipeline_config(args.pipeline)
        dataset_cfg = YmlRepository.get_dataset_by_key(args.dataset)
        active_profile = YmlRepository.get_active_profile()

        log.info(f"[main] Active Configuration Profile resolved to: '{active_profile}'")

        # Initialize the runtime facade orchestration engine context payload
        ctx = init_run_facade_api(
            pipeline_name=args.pipeline,
            dataset_key=args.dataset,
            resume_run_id=args.resume_run
        )
        # 🔄 POSICIÓN CORRECTA: Ahora que los logs están activos, el mensaje se registrará perfectamente
        log.info("DEBUG: Registered generators: %s", list(get_registered_generators().keys()))

        # Trigger execution loop for all requested CRISP-DM workflow blocks
        ctx = _execute_pipeline_steps(ctx, args.steps)

    except Exception as exc:
        log.error("Execution failed: %s", exc, exc_info=True)
        return 1

    log.info("=" * 60)
    log.info("Execution finished successfully.")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())