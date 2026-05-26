# src/phm_america_2024/registry/phase2_generator_registry.py
from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

from phm_america_2024.registry.generator_registry_registry import register_artifact
from phm_america_2024.data.persist_persister_data import save_parquet, save_json
#from phm_america_2024.data.profiling_profiler_data import plot_gmm_curve
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.configuration.enum_registry_config import StepsPhase, StepOutputArtifact

from phm_america_2024.reporting.plots_generator_reporting import (
    plot_gmm_analysis,plot_flight_regime_binning,
)
from phm_america_2024.reporting.artifact_persister_reporting import save_figure

log = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Step 2.1 – Data Acquisition artifacts (parquet)
# ──────────────────────────────────────────────────────────────────────────────

@register_artifact("step_2_1_data_acquisition", StepOutputArtifact.sample_x_y_train_parquet.value)
def _save_x_y_train_parquet(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist training dataset as parquet."""
    # Step 1: CALL context_data.get — retrieve dataframe from execution context
    df = context_data.get(StepOutputArtifact.sample_x_y_train_parquet.value)
    if df is None or (hasattr(df, "empty") and df.empty):
        log.warning("[_save_x_y_train_parquet] No dataframe found to persist")
        return
    # Step 2: CALL resolve_path — resolve target artifact path
    full_path: Path = resolve_path(ctx.phase2_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    # Step 3: CALL save_parquet — persist dataframe to disk
    save_parquet(df, str(full_path), compression=getattr(ctx.config.common_base_config.output_policy, "compression", "snappy"))
    log.info("[_save_x_y_train_parquet] Success saved rows=%d", len(df))

@register_artifact("step_2_1_data_acquisition", StepOutputArtifact.sample_x_test_parquet.value)
def _save_x_test_parquet(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist test dataset as parquet."""
    df = context_data.get(StepOutputArtifact.sample_x_test_parquet.value)
    if df is None or (hasattr(df, "empty") and df.empty):
        log.warning("[_save_x_test_parquet] No dataframe found")
        return
    full_path: Path = resolve_path(ctx.phase2_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(df, str(full_path), compression=getattr(ctx.config.common_base_config.output_policy, "compression", "snappy"))
    log.info("[_save_x_test_parquet] Success saved rows=%d", len(df))

@register_artifact("step_2_1_data_acquisition", StepOutputArtifact.sample_x_validation_parquet.value)
def _save_x_validation_parquet(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist validation dataset as parquet."""
    df = context_data.get(StepOutputArtifact.sample_x_validation_parquet.value)
    if df is None or (hasattr(df, "empty") and df.empty):
        log.warning("[_save_x_validation_parquet] No dataframe found")
        return
    full_path: Path = resolve_path(ctx.phase2_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(df, str(full_path), compression=getattr(ctx.config.common_base_config.output_policy, "compression", "snappy"))
    log.info("[_save_x_validation_parquet] Success saved rows=%d", len(df))


@register_artifact("step_2_1_data_acquisition", StepOutputArtifact.load_and_merge_json.value)
def _save_merged_train_json(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """
    Persist training data metadata as a JSON artifact.

    Takes the configuration metadata or summary information from the execution
    context and saves it as a JSON file, enabling traceability of the
    data acquisition parameters.

    Parameters
    ----------
    ctx : Any
        The global execution context.
    artifact_path : str
        Relative path string where the artifact should be persisted.
    **context_data : Any
        Key-value pairs containing the metadata or data to be saved.
    """
    # Step 2: CALL log.debug — log entry and input parameters
    log.debug("[_save_merged_train_json] Entry with artifact_path=%s", artifact_path)

    # Step 3: CALL context_data.get — retrieve metadata dictionary from execution context
    data: Optional[Any] = context_data.get(StepOutputArtifact.load_and_merge_json.value)

    # Step 4: EVALUATE data — verify if target data is available for persistence
    if data is None:
        log.warning("[_save_merged_train_json] No metadata found under key '%s' to persist", StepOutputArtifact.load_and_merge_json.value)
        return

    # Step 5: CALL resolve_path — resolve absolute target destination path
    if not hasattr(ctx, "phase2_dir"):
        log.error("[_save_merged_train_json] RunContext missing required attribute 'phase2_dir'")
        raise AttributeError("RunContext missing 'phase2_dir'")

    full_path: Path = resolve_path(ctx.phase2_dir / artifact_path)
    log.debug("[_save_merged_train_json] Resolved full target path -> %s", full_path)

    # Step 6: CALL mkdir — ensure parent directories exist on file system
    full_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 7: CALL save_json — persist clean dictionary structure to disk
    try:
        save_json(data, str(full_path))
        log.info("[_save_merged_train_json] Successfully saved metadata to %s", full_path.name)
    except Exception as e:
        log.error("[_save_merged_train_json] Failed to save JSON artifact on disk: %s", str(e))
        raise e

    # Step 8: CALL log.debug — log exit of method
    log.debug("[_save_merged_train_json] Exit successfully")

# ──────────────────────────────────────────────────────────────────────────────
# Step 2.2 – Data Description artifacts (JSON)
# ──────────────────────────────────────────────────────────────────────────────

@register_artifact(StepsPhase.STEP_2_2.value, StepOutputArtifact.column_metadata_json.value)
def _save_column_metadata(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist column metadata as JSON."""
    result = context_data.get(StepOutputArtifact.column_metadata_json.value)
    if result is None:
        log.warning("[_save_column_metadata] No result to save")
        return
    # Step 1: CALL save_json — persist result dictionary
    save_json(result, str(resolve_path(ctx.phase2_dir / artifact_path)))
    log.info("[_save_column_metadata] Saved artifact='%s'", artifact_path)

@register_artifact(StepsPhase.STEP_2_2.value, StepOutputArtifact.sensor_stats_json.value)
def _save_sensor_stats(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist sensor statistics as JSON."""
    result = context_data.get(StepOutputArtifact.sensor_stats_json.value)
    if result is None:
        log.warning("[_save_sensor_stats] No result to save")
        return
    save_json(result, str(resolve_path(ctx.phase2_dir / artifact_path)))
    log.info("[_save_sensor_stats] Saved artifact='%s'", artifact_path)

@register_artifact(StepsPhase.STEP_2_2.value, StepOutputArtifact.null_count_json.value)
def _save_null_count(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist null count report as JSON."""
    result = context_data.get(StepOutputArtifact.null_count_json.value)
    if result is None:
        log.warning("[_save_null_count] No result to save")
        return
    save_json(result, str(resolve_path(ctx.phase2_dir / artifact_path)))
    log.info("[_save_null_count] Saved artifact='%s'", artifact_path)

@register_artifact(StepsPhase.STEP_2_2.value, StepOutputArtifact.target_distribution_json.value)
def _save_target_distribution(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist target distribution as JSON."""
    result = context_data.get(StepOutputArtifact.target_distribution_json.value)
    if result is None:
        log.warning("[_save_target_distribution] No result to save")
        return
    save_json(result, str(resolve_path(ctx.phase2_dir / artifact_path)))
    log.info("[_save_target_distribution] Saved artifact='%s'", artifact_path)

# ──────────────────────────────────────────────────────────────────────────────
# Step 2.3 – Data Quality Assessment artifacts (JSON)
# ──────────────────────────────────────────────────────────────────────────────

@register_artifact(StepsPhase.STEP_2_3.value, StepOutputArtifact.zero_or_negative_check_json.value)
def _save_zero_negative_check(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist zero/negative check results as JSON."""
    result = context_data.get(StepOutputArtifact.zero_or_negative_check_json.value)
    if result is None:
        log.warning("[_save_zero_negative_check] No result to save")
        return
    save_json(result, str(resolve_path(ctx.phase2_dir / artifact_path)))
    log.info("[_save_zero_negative_check] Saved artifact='%s'", artifact_path)

@register_artifact(StepsPhase.STEP_2_3.value, StepOutputArtifact.collinearity_json.value)
def _save_collinearity(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist collinearity report as JSON."""
    result = context_data.get(StepOutputArtifact.collinearity_json.value)
    if result is None:
        log.warning("[_save_collinearity] No result to save")
        return
    save_json(result, str(resolve_path(ctx.phase2_dir / artifact_path)))
    log.info("[_save_collinearity] Saved artifact='%s'", artifact_path)

# ──────────────────────────────────────────────────────────────────────────────
# Step 2.4 – Data Exploration artifacts (JSON + PNG)
# ──────────────────────────────────────────────────────────────────────────────

@register_artifact(StepsPhase.STEP_2_4.value, StepOutputArtifact.column_catalog_json.value)
def _save_column_catalog(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist column catalog as JSON."""
    result = context_data.get(StepOutputArtifact.column_catalog_json.value)
    if result is None:
        log.warning("[_save_column_catalog] No result to save")
        return
    save_json(result, str(resolve_path(ctx.phase2_dir / artifact_path)))
    log.info("[_save_column_catalog] Saved artifact='%s'", artifact_path)

@register_artifact(StepsPhase.STEP_2_4.value, StepOutputArtifact.ks_report_json.value)
def _save_ks_report(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist KS test report as JSON."""
    result = context_data.get("ks_test_per_feature")

    if result is None:
        log.warning("[_save_ks_report] No result found in context_data to save")
        return

    save_json(result, str(resolve_path(ctx.phase2_dir / artifact_path)))
    log.info("[_save_ks_report] Saved artifact='%s'", artifact_path)

@register_artifact(StepsPhase.STEP_2_4.value, StepOutputArtifact.gmm_curve_png.value)
def _save_gmm_curve(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Registry ahora gestiona la generación Y la persistencia."""
    report = context_data.get("gmm_exploration") # El resultado del cálculo
    if report and "curve" in report:
        fig = plot_gmm_analysis(report)
        save_figure(fig, out_path=resolve_path(ctx.phase2_dir / artifact_path), dpi=300)
        log.info("[_save_gmm_curve] Persistence completed")

@register_artifact(StepsPhase.STEP_2_4.value, StepOutputArtifact.drift_summary_json.value)
def _save_drift_summary(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist drift summary as JSON."""
    result = context_data.get(StepOutputArtifact.drift_summary_json.value)

    if result is None:
        log.warning("[_save_drift_summary] No result found in context to save")
        return
    target_path = resolve_path(ctx.phase2_dir / artifact_path)
    save_json(result, str(target_path))

    log.info("[_save_drift_summary] Successfully saved drift summary to: %s", artifact_path)

@register_artifact(StepsPhase.STEP_2_4.value, StepOutputArtifact.flight_regimes_png.value)
def _save_flight_regimes(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Registry ahora gestiona el ploteo y la persistencia usando datos inyectados."""
    payload = context_data.get("flight_regime_binning")

    # Validación robusta del payload
    if not payload or "data" not in payload or "df" not in payload:
        log.warning("[_save_flight_regimes] Missing required payload data (data/df/plot_meta)")
        return

    report = payload["data"]
    meta = payload["plot_meta"]
    df = payload["df"]  # <-- EXTRAEMOS EL DF INYECTADO POR EL RUNNER

    # Generación del gráfico
    fig = plot_flight_regime_binning(
        df[meta["column"]],
        title=f"Flight Regime: {meta['column']}",
        bins=meta["bins"],
        plot_type=meta["plot_type"]
    )

    save_figure(fig, out_path=resolve_path(ctx.phase2_dir / artifact_path), dpi=300)
    log.info("[_save_flight_regimes] Saved PNG artifact='%s'", artifact_path)