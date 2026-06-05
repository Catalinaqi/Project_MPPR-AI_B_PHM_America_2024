# src/phm_america_2024/pipeline/clustering_runner_pipeline.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


from phm_america_2024.configuration.build_factory_config import build_config, BuiltConfig
from phm_america_2024.domain.enum_registry_domain import PhaseDir
from phm_america_2024.pipeline.utils.context_facade_common import RunContext, create_run_context
from phm_america_2024.common.logging_adapter_common import get_logger
# from phm_america_2024.phase.phase2_understanding_runner_phase import (
#     run_data_description,
#     run_data_quality_verification,
#     run_exploratory_analysis,
#     run_initial_data_collection,
# )

log = get_logger(__name__)


@dataclass
class ClassificationRunContext(RunContext):
    """Mutable state for clustering pipeline run."""
    cluster_labels: Optional[pd.Series] = field(default=None, repr=False)


def create_clustering_context(
        *,
        pipeline_name: str,
        dataset_key: str,
        notebook_vars: Optional[dict[str, Any]] = None,
) -> ClassificationRunContext:
    """Build ClassificationRunContext ready for Phase 2."""
    log.info("[create_clustering_context] start dataset_key=%s", dataset_key)

    # Step 1: Build pipeline config via build_config()
    built: BuiltConfig = build_config(
        pipeline_name=pipeline_name,
        dataset_key=dataset_key,
        notebook_vars=notebook_vars,
    )

    # Step 2: Create run context via factory helper
    ctx_generic = create_run_context(
        config=built.config,
        dataset_key=dataset_key,
    )

    # Step 3: Create clustering-specific context, copying all fields from generic
    ctx = ClassificationRunContext(
        config=ctx_generic.config,
        run_dir=ctx_generic.run_dir,
        run_id=ctx_generic.run_id,
        dataset_key=ctx_generic.dataset_key,
        df_train=ctx_generic.df_train,
        df_test=ctx_generic.df_test,
        artifacts=ctx_generic.artifacts,
        phase_results=ctx_generic.phase_results,
        errors=ctx_generic.errors,
    )

    log.info("[create_clustering_context] done run_id=%s", ctx.run_id)
    return ctx


def run_clustering_pipeline(ctx: ClassificationRunContext) -> ClassificationRunContext:
    """Execute full CRISP-DM clustering pipeline."""
    log.info("[run_clustering_pipeline] START run_id=%s task=%s", ctx.run_id, ctx.task)

    # Phase 2 - Data Understanding
    log.info("[run_clustering_pipeline] >>> PHASE 2")
    ctx = run_phase2_1(ctx)
    # ctx = run_clustering_pipeline_phase2_2(ctx)
    # ctx = run_clustering_pipeline_phase2_3(ctx)
    # ctx = run_clustering_pipeline_phase2_4(ctx)

    # Phase 3 - Data Preparation
    # log.info("[run_clustering_pipeline] >>> PHASE 3")
    # ctx = run_clustering_pipeline_phase3_1(ctx)
    # ctx = run_clustering_pipeline_phase3_2(ctx)
    # ctx = run_clustering_pipeline_phase3_3(ctx)
    # ctx = run_clustering_pipeline_phase3_5(ctx)
    #
    # # Phase 4 - Modeling
    # log.info("[run_clustering_pipeline] >>> PHASE 4")
    # ctx = run_clustering_pipeline_phase4_1(ctx)
    # ctx = run_clustering_pipeline_phase4_2(ctx)
    # ctx = run_clustering_pipeline_phase4_3(ctx)
    # ctx = run_clustering_pipeline_phase4_4(ctx)
    #
    # # Phase 5 - Evaluation
    # log.info("[run_clustering_pipeline] >>> PHASE 5")
    # ctx = run_clustering_pipeline_phase5_1(ctx)
    # ctx = run_clustering_pipeline_phase5_2(ctx)
    # ctx = run_clustering_pipeline_phase5_3(ctx)
    # ctx = run_clustering_pipeline_phase5_4(ctx)

    log.info("[run_clustering_pipeline] END run_id=%s artifacts=%d", ctx.run_id, len(ctx.artifacts))
    return ctx


# =============================================================================
# PHASE 2 ORCHESTRATORS
# =============================================================================


def run_clustering_pipeline_phase2_1(ctx: ClassificationRunContext) -> ClassificationRunContext:
    """Phase 2.1 - Load train/test CSVs separately."""
    if ctx.df_train is not None:
        log.warning("[2.1] df_train already set shape=%s - skipping", ctx.df_train.shape)
        return ctx

    log.info("[2.1] start run_id=%s", ctx.run_id)
    ctx = run_phase2_1(ctx)
    log.info(
        "[2.1] done train=%s test=%s",
        ctx.df_train.shape if ctx.df_train is not None else None,
        ctx.df_test.shape if ctx.df_test is not None else None,
    )
    return ctx

#
# def run_clustering_pipeline_phase2_2(ctx: ClassificationRunContext) -> ClassificationRunContext:
#     """Phase 2.2 - Data profiling and description."""
#     if ctx.df_train is None:
#         raise RuntimeError("[2.2] no data loaded - run Phase 2.1 first")
#
#     log.info("[2.2] start run_id=%s", ctx.run_id)
#     ctx = run_data_description(ctx)
#     log.info("[2.2] done")
#     return ctx
#
#
# def run_clustering_pipeline_phase2_3(ctx: ClassificationRunContext) -> ClassificationRunContext:
#     """Phase 2.3 - Quality verification and drift detection."""
#     if ctx.df_train is None:
#         raise RuntimeError("[2.3] no data loaded - run Phase 2.1 first")
#
#     log.info("[2.3] start run_id=%s", ctx.run_id)
#     ctx = run_data_quality_verification(ctx)
#     drift_detected = ctx.phase_results.get(StepsPhase.STEP_2_3.value, {}).get("drift_analyzed", False)
#     log.info("[2.3] done drift_detected=%s", drift_detected)
#     return ctx
#
#
# def run_clustering_pipeline_phase2_4(ctx: ClassificationRunContext) -> ClassificationRunContext:
#     """Phase 2.4 - Exploratory Data Analysis."""
#     if ctx.df_train is None:
#         raise RuntimeError("[2.4] no data loaded - run Phase 2.1 first")
#
#     log.info("[2.4] start run_id=%s", ctx.run_id)
#     ctx = run_exploratory_analysis(ctx)
#     log.info("[2.4] done")
#     return ctx
#

# =============================================================================
# PHASE 3 ORCHESTRATORS (STUBS)
# =============================================================================

# =============================================================================
# PHASE 4 ORCHESTRATORS (STUBS)
# =============================================================================


# =============================================================================
# PHASE 5 ORCHESTRATORS (STUBS)
# =============================================================================

