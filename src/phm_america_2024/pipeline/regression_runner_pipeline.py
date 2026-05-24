# src/phm_america_2024/pipeline/regression_runner_pipeline.py
from __future__ import annotations

from typing import Optional

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.context_facade_common import RunContext

# Import phase runners (each phase runner is responsible for its own registry side‑effects)
from phm_america_2024.phase.phase2_understanding_runner_phase import run_phase2

# =============================================================================
# Regression Pipeline Runner
# -----------------------------------------------------------------------------
# Purpose:
#   Orchestrate the CRISP-DM phases for regression problems. Called by
#   execution_facade_api with a list of step keys (enum values) to execute.
#   Manages the sequential execution of phase runners and accumulates phase
#   results while respecting the requested steps subset.
#
# No hardcoded values — every parameter comes from YAML config files
# via the phase runners.
#
# Design patterns:
# - Enterprise/Architectural:
#   - Runner (specific to regression)
# =============================================================================

log = get_logger(__name__)


def run_regression_pipeline(
    ctx: RunContext,
    steps: Optional[list[str]] = None,
) -> RunContext:
    """Run one or more CRISP-DM phases for regression.

    Step 1: If steps is None, run all phases sequentially.
    Step 2: If steps is provided, run only the requested phase steps (by enum value).
    Step 3: Return updated context with phase results registered.

    Parameters
    ----------
    ctx : RunContext
        Initialised run context with configuration and directory paths.
    steps : list[str] or None, optional
        List of step keys (e.g. ``["step_2_1_data_acquisition"]``).
        If None, all phases are executed.

    Returns
    -------
    RunContext
        Updated context with phase results registered.
    """
    log.info("[run_regression_pipeline] start steps=%s", steps)

    # Step 1: Determine which phases to run based on step prefixes
    run_all = steps is None

    # Determine if any Phase 2 steps are requested
    phase2_steps: list[str] | None = None
    if not run_all:
        # Filter steps that belong to Phase 2 (prefix "step_2_")
        phase2_steps = [s for s in steps if s.startswith("step_2_")]
        log.debug("[run_regression_pipeline] filtered Phase 2 steps: %s", phase2_steps)

    # Step 2: Execute Phase 2 – Data Understanding (if requested or full)
    if run_all or phase2_steps:
        log.info("[run_regression_pipeline] executing Phase 2 with filter=%s",
                 phase2_steps if not run_all else "all")
        # CALL run_phase2 with steps_filter if provided, else None (all)
        phase2_result = run_phase2(ctx, steps_filter=phase2_steps)
        # Step 2.1: Register phase result
        ctx.register_phase_result("phase2", phase2_result)
        log.info("[run_regression_pipeline] Phase 2 completed – status=%s",
                 phase2_result.get("status", "unknown"))
    else:
        log.debug("[run_regression_pipeline] Phase 2 not requested – skip")

    # Future phases: repeat pattern for Phase 3, 4, 5
    # if run_all or any(s.startswith("step_3_") for s in steps):
    #     phase3_result = run_phase3(ctx, steps_filter=...)
    #     ctx.register_phase_result("phase3", phase3_result)

    log.info("[run_regression_pipeline] done")
    return ctx