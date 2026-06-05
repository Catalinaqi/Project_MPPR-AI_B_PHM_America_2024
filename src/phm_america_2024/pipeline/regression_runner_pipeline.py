from __future__ import annotations

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.pipeline.utils.context_facade_common import RunContext
from phm_america_2024.domain.enum_registry_domain import StepsPhase
from phm_america_2024.phase.phase2_understanding_runner_phase import Phase2DataUnderstandingRunner
from phm_america_2024.phase.phase3_preparation_runner_phase import Phase3PreparationRunner
from phm_america_2024.phase.phase4_modeling_runner_phase import Phase4ModelingRunner
from phm_america_2024.phase.phase5_evaluation_and_interpretation_phase import (
    Phase5EvaluationAndInterpretationRunner)

log = get_logger(__name__)

# Build prefix sets from StepsPhase enum
_PHASE2_STEPS: set[str] = {m.value for m in StepsPhase if m.value.startswith("step_2_")}
_PHASE3_STEPS: set[str] = {m.value for m in StepsPhase if m.value.startswith("step_3_")}
_PHASE4_STEPS: set[str] = {m.value for m in StepsPhase if m.value.startswith("step_4_")}
_PHASE5_STEPS: set[str] = {m.value for m in StepsPhase if m.value.startswith("step_5_")}



def run_regression_pipeline(ctx: RunContext, steps: list[str]) -> RunContext:
    """Execute one or more CRISP‑DM steps for the regression task.

    Args:
        ctx: Run context from init.
        steps: Step names to execute (e.g. ``["step_2_1_data_acquisition"]``).

    Returns:
        Updated context after step execution.
    """
    log.debug("[run_regression_pipeline] entry steps=%s", steps)

    for step_name in steps:
        try:
            # Step 1: check if step belongs to Phase 2 or Phase 3 or Phase 4
            if (step_name in _PHASE2_STEPS or step_name in _PHASE3_STEPS  or step_name
                    in _PHASE4_STEPS or step_name in _PHASE5_STEPS):
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
            raise

    log.info("[run_regression_pipeline] all requested steps completed: %s", steps)
    log.debug("[run_regression_pipeline] exit")
    return ctx


def _exec_step(ctx: RunContext, step_key: str) -> RunContext:
    """Extract step config, inject phase-level dependencies, and delegate execution.

    Step 1: Determine phase config key and runner class from step prefix.
    Step 2: Retrieve phase config from context.
    Step 3: Extract global read strategy and input source.
    Step 4: Deep-copy step config and inject global dependencies.
    Step 5: Instantiate runner and call runner.run().

    Args:
        ctx: Run context.
        step_key: Full step identifier (e.g. ``"step_3_1_data_selection"``).

    Returns:
        Updated context after runner execution.
    """
    log.debug("[_exec_step] entry step_key='%s'", step_key)

    # ── Step 1: identify phase config key and runner class ─────────────────
    if step_key.startswith("step_2_"):
        phase_config_key: str = "phase2_data_understanding"
        runner_cls = Phase2DataUnderstandingRunner
    elif step_key.startswith("step_3_"):
        phase_config_key = "phase3_data_preparation"
        runner_cls = Phase3PreparationRunner
    elif step_key.startswith("step_4_"):
        phase_config_key = "phase4_data_modeling"
        runner_cls = Phase4ModelingRunner
    elif step_key.startswith("step_5_"):
        phase_config_key = "phase5_evaluation_and_interpretation"
        runner_cls = Phase5EvaluationAndInterpretationRunner
    else:
        log.error("[_exec_step] unknown phase for step_key='%s'", step_key)
        raise ValueError(f"Unknown phase for step: {step_key}")

    # ── Step 2: retrieve phase configuration ────────────────────────────────
    try:
        phase_cfg = ctx.config.phases[phase_config_key]
    except (KeyError, AttributeError) as err:
        log.error("[_exec_step] phase config '%s' not found: %s", phase_config_key, err)
        raise ValueError(f"Phase configuration '{phase_config_key}' is missing") from err

    log.debug("[_exec_step] phase config keys: %s", list(phase_cfg.keys()))

    # ── Step 3: extract global read strategy (avoid hardcoding structure) ────
    try:
        global_read_strategy = phase_cfg["read_strategy"]
        global_input_source = phase_cfg.get("read_strategy", {}).get("input_source", {})
    except KeyError as err:
        log.error("[_exec_step] missing global configuration: %s", err)
        raise ValueError(f"Phase config missing required section: {err}") from err

    # ── Step 4: prepare step config and inject global dependencies ──────────
    step_cfg_raw = phase_cfg["steps"][step_key]
    log.debug("[_exec_step] raw step config keys: %s", list(step_cfg_raw.keys()))

    step_cfg = dict(step_cfg_raw)              # deep copy to avoid mutation
    step_cfg["read_strategy"] = global_read_strategy
    step_cfg["input_source"] = global_input_source
    log.debug("[_exec_step] injected phase-level context into step_cfg")

    # ── Step 5: instantiate runner and execute ──────────────────────────────
    runner = runner_cls(ctx, step_key, step_cfg)
    log.debug("[_exec_step] runner created – calling runner.run()")
    ctx = runner.run()
    log.debug("[_exec_step] runner.run() returned")

    log.info("[_exec_step] step '%s' executed", step_key)
    return ctx