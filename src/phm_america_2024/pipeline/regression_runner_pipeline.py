# src/phm_america_2024/pipeline/regression_runner_pipeline.py
from __future__ import annotations

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.context_facade_common import RunContext
from phm_america_2024.configuration.enum_registry_config import StepsPhase
from phm_america_2024.phase.phase2_understanding_runner_phase import Phase2DataUnderstandingRunner
from phm_america_2024.phase.phase3_preparation_runner_phase import Phase3PreparationRunner


log = get_logger(__name__)


def run_regression_pipeline(ctx: RunContext, steps: list[str]) -> RunContext:
    """Execute one or more CRISP‑DM steps for the regression task.

    Step 1: Log step names at DEBUG level.
    Step 2: For each step, CALL _exec_step() in try/except block.
    Step 3: Log completion at INFO level.
    Step 4: Return updated context (or modified ctx on partial failure).
    """
    log.debug("[run_regression_pipeline] entry steps=%s", steps)

    for step_name in steps:
        # Step 2: execute step with exception guard
        try:
            if step_name in (
                    StepsPhase.STEP_2_1.value,
                    StepsPhase.STEP_2_2.value,
                    StepsPhase.STEP_2_3.value,
                    StepsPhase.STEP_2_4.value,
            ):
                ctx = _exec_step(ctx, step_name)
            else:
                log.warning(
                    "[run_regression_pipeline] step '%s' not recognised – skipped",
                    step_name,
                )
                continue

            log.info("[run_regression_pipeline] step '%s' completed successfully", step_name)

        except Exception as exc:
            log.exception(
                "[run_regression_pipeline] step '%s' FAILED – error: %s",
                step_name,
                exc,
            )
            # Re-raise? The spec says independently executable, but a failed step
            # should not block subsequent steps unless critical.
            # We choose to raise to alert the caller.
            raise

    log.info("[run_regression_pipeline] all requested steps completed: %s", steps)
    log.debug("[run_regression_pipeline] exit")
    return ctx


def _exec_step(ctx: RunContext, step_key: str) -> RunContext:
    """Extract step config, inject global data structures, and delegate to Runner.

    Step 1: Access phase configuration from context.
    Step 2: Validate and extract global phase-level settings to avoid hardcoding.
    Step 3: Deep copy/prepare step config and inject global dependencies dynamically.
    Step 4: Instantiate runner and return updated context.
    """
    log.debug("[_exec_step] entry step_key='%s'", step_key)

    # Step 1: retrieve phase config
    phase_cfg = ctx.config.phases.phase2_data_understanding
    log.debug("[_exec_step] phase config keys: %s", list(phase_cfg.keys()))

    # Validación Estricta: Garantizar que la raíz de la fase contenga los bloques obligatorios
    try:
        global_dataset_input = phase_cfg["dataset_input"]
        global_read_strategy = phase_cfg["read_strategy"]
    except KeyError as err:
        log.error("[_exec_step] Critical missing configuration block at phase level: %s", err)
        raise ValueError(f"Phase configuration is missing required global section: {err}") from err

    # Step 2: get step config – may raise KeyError if missing
    step_cfg_raw = phase_cfg.steps[step_key]
    log.debug("[_exec_step] raw step config keys: %s", list(step_cfg_raw.keys()))

    # Crear una copia de trabajo para mutar el diccionario del paso sin corromper la config estática original
    step_cfg = dict(step_cfg_raw)

    # Inyección dinámica de dependencias (Mapeo limpio basado en datos del YAML)
    step_cfg["dataset_input"] = global_dataset_input
    step_cfg["read_strategy"] = global_read_strategy
    log.debug("[_exec_step] injected phase-level context blocks into step_cfg smoothly")

    # Step 3: instantiate runner and run
    runner = Phase2DataUnderstandingRunner(ctx, step_key, step_cfg)
    log.debug("[_exec_step] runner created – calling runner.run()")
    ctx = runner.run()
    log.debug("[_exec_step] runner.run() returned")

    log.info("[_exec_step] step '%s' executed", step_key)
    return ctx