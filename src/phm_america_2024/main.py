# src/phm_america_2024/main.py
from __future__ import annotations

import sys
import argparse
from typing import Any

from phm_america_2024.api.execution_facade_api import (
    init_run_facade_api,
    run_2_1_data_acquisition,
    run_2_2_data_description,
    run_2_3_data_quality_verification,
    run_2_4_data_exploration,
)
from phm_america_2024.common.logging_adapter_common import get_logger

from phm_america_2024.registry import phase2_generator_registry


# IMPORTACIÓN REQUERIDA: Conectamos el repositorio de configuraciones centralizado
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
        choices=["2.1", "2.2", "2.3", "2.4", "3.1", "3.2", "3.3", "3.5", "4.1", "4.2", "4.3", "4.4", "4.5"],
        help="Phase steps to execute (e.g. '2.1' '2.2'). Default: '2.1'",
    )
    return parser.parse_args()


def _execute_phase2_steps(ctx: Any, steps: list[str]) -> Any:
    """Execute requested Phase 2 steps.

    Step 1: For each step in the list, call the corresponding API function.
    Step 2: Log completion and return updated context.
    """
    step_map = {
        "2.1": ("Data Acquisition", run_2_1_data_acquisition),
        "2.2": ("Data Description", run_2_2_data_description),
        "2.3": ("Data Quality Verification", run_2_3_data_quality_verification),
        "2.4": ("Data Exploration", run_2_4_data_exploration),
    }

    for step_key in ["2.1", "2.2", "2.3", "2.4"]:
        if step_key in steps:
            name, func = step_map[step_key]
            log.info("Executing Phase 2.%s – %s", step_key, name)
            ctx = func(ctx)
            log.info("Phase 2.%s completed", step_key)

    return ctx


def main() -> int:
    """CLI entry point.

    Step 1: Parse arguments.
    Step 2: Load merged pipeline and dataset configs via YmlRepository.
    Step 3: CALL init_run_facade_api(pipeline_name, dataset_key, configuration).
    Step 4: CALL _execute_phase2_steps(ctx, steps).
    Step 5: Log success and return exit code.
    """

    args = _parse_args()

    log.info("=" * 60)
    log.info("PHM America 2024 – Pipeline Execution Started")
    log.info(f"Pipeline: {args.pipeline}, Dataset: {args.dataset}, Steps: {args.steps}")
    log.info("=" * 60)

    try:
        # INTEGRACIÓN: Cargamos las configuraciones usando el Repositorio centralizado.
        # Esto ejecuta el Deep Merge automático y resuelve perfiles dinámicamente en RAM.
        log.info("[main] Resolving configuration files via YmlRepository...")
        pipeline_cfg = YmlRepository.load_pipeline_config(args.pipeline)
        dataset_cfg = YmlRepository.get_dataset_by_key(args.dataset)
        active_profile = YmlRepository.get_active_profile()

        log.info(f"[main] Active Configuration Profile resolved to: '{active_profile}'")

        # Inicializamos la fachada pasando las configuraciones ya procesadas y validadas
        ctx = init_run_facade_api(
            pipeline_name=args.pipeline,
            dataset_key=args.dataset,
            # NOTA: Asegúrate de que tu función init_run_facade_api reciba estos objetos
            # o los inyecte directo en el RunContext si tu API interna así lo soporta.
        )

        # Ejecutamos las etapas CRISP-DM solicitadas
        ctx = _execute_phase2_steps(ctx, args.steps)

    except Exception as exc:
        log.error("Execution failed: %s", exc, exc_info=True)
        return 1

    log.info("=" * 60)
    log.info("Execution finished successfully.")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())