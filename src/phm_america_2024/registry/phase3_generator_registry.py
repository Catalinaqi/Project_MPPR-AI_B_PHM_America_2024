from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from phm_america_2024.registry.generator_registry_registry import register_artifact
from phm_america_2024.data.persist_persister_data import save_parquet, save_json
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.configuration.enum_registry_config import StepsPhase, StepOutputArtifact

from phm_america_2024.registry.generator_registry_registry import _ARTIFACT_GENERATORS

log = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Step 3.1 – Data Selection artifacts (parquet + traces as JSON)
# ──────────────────────────────────────────────────────────────────────────────

@register_artifact(StepsPhase.STEP_3_1.value, StepOutputArtifact.selected_regression_train_parquet.value)
def _save_selected_regression_train(ctx: Any, artifact_path: Any, **context_data: Any) -> None:
    """Persist the selection-result DataFrame as parquet."""
    df = context_data.get(StepOutputArtifact.selected_regression_train_parquet.value)
    if df is None or (hasattr(df, "empty") and df.empty):
        log.warning("[_save_selected_regression_train] No dataframe to persist")
        return

    # ──── SOPORTE PARA ESTRUCTURA ANIDADA (YAML COMPLEJO) ────
    # Si artifact_path tiene atributos o llaves (es un DictConfig o dict), extraemos 'path'
    if hasattr(artifact_path, "get") or isinstance(artifact_path, dict):
        real_path_str = artifact_path.get("path")
        if not real_path_str:
            log.error("[_save_selected_regression_train] Missing 'path' key in artifact configuration: %s", artifact_path)
            raise ValueError("Artifact configuration block is missing the mandatory 'path' field.")
    else:
        # Fallback por si en algún paso viene como un string plano
        real_path_str = artifact_path

    # Ahora real_path_str es un string puro ("3.3.transformation...parquet")
    full_path: Path = resolve_path(ctx.phase3_dir / str(real_path_str))

    # Persistencia limpia respetando el destino aislado de la ejecución
    full_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(df, str(full_path),compression="snappy")
    log.info("[_save_selected_regression_train] Saved rows=%d to %s", len(df), full_path)

# ──────────────────────────────────────────────────────────────────────────────
# Step 3.2 – Data Cleaning artifacts (parquet)
# ──────────────────────────────────────────────────────────────────────────────

@register_artifact(StepsPhase.STEP_3_2.value, StepOutputArtifact.cleaned_regression_train_parquet.value)
def _save_cleaned_regression_train(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist the cleaned DataFrame (outlier-clipped + deduplicated) as parquet."""

    log.info("DEBUG: Phase3 generators registrati: %s", list(_ARTIFACT_GENERATORS.keys()))

    df = context_data.get(StepOutputArtifact.cleaned_regression_train_parquet.value)
    if df is None or (hasattr(df, "empty") and df.empty):
        log.warning("[_save_cleaned_regression_train] No dataframe to persist")
        return
    full_path: Path = resolve_path(ctx.phase3_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    #save_parquet(df, str(full_path))
    save_parquet(df, str(full_path), compression="snappy")
    log.info("[_save_cleaned_regression_train] Saved rows=%d to %s", len(df), artifact_path)

# ──────────────────────────────────────────────────────────────────────────────
# Step 3.3 – Data Transformation artifacts (parquet + pickle scaler)
# ──────────────────────────────────────────────────────────────────────────────

@register_artifact(StepsPhase.STEP_3_3.value, StepOutputArtifact.transformed_regression_train_parquet.value)
def _save_transformed_regression_train(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist the transformed DataFrame (scaled + engineered) as parquet."""
    df = context_data.get(StepOutputArtifact.transformed_regression_train_parquet.value)
    if df is None or (hasattr(df, "empty") and df.empty):
        log.warning("[_save_transformed_regression_train] No dataframe to persist")
        return
    full_path: Path = resolve_path(ctx.phase3_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    #save_parquet(df, str(full_path))
    save_parquet(df, str(full_path), compression="snappy")
    log.info("[_save_transformed_regression_train] Saved rows=%d to %s", len(df), artifact_path)

@register_artifact(StepsPhase.STEP_3_3.value, StepOutputArtifact.fitted_scaler_regression_artifact.value)
def _save_fitted_scaler(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist the fitted RobustScaler object as a pickle file.

    The scaler object is expected in ``context_data`` under the key
    ``fitted_scaler_regression_artifact``, which should be a dict
    with key ``"scaler"`` mapping to the ``RobustScaler`` instance.
    """
    scaler_data = context_data.get(StepOutputArtifact.fitted_scaler_regression_artifact.value)
    if scaler_data is None:
        log.warning("[_save_fitted_scaler] No scaler object found in context_data")
        return

    scaler = scaler_data.get("scaler")  # The dict produced by feature_scaling
    if scaler is None:
        log.warning("[_save_fitted_scaler] 'scaler' key missing in artifact data")
        return

    full_path: Path = resolve_path(ctx.phase3_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, str(full_path))
    log.info("[_save_fitted_scaler] Saved scaler to %s", artifact_path)

# ──────────────────────────────────────────────────────────────────────────────
# Step 3.5 – Data Formatting artifacts (two parquets: train + val)
# ──────────────────────────────────────────────────────────────────────────────

@register_artifact(StepsPhase.STEP_3_5.value, StepOutputArtifact.engineered_train_split.value)
def _save_engineered_train_split(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist the internal training split as parquet."""
    df = context_data.get(StepOutputArtifact.engineered_train_split.value)
    if df is None or (hasattr(df, "empty") and df.empty):
        log.warning("[_save_engineered_train_split] No dataframe to persist")
        return
    full_path: Path = resolve_path(ctx.phase3_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    #save_parquet(df, str(full_path))
    save_parquet(df, str(full_path), compression="snappy")
    log.info("[_save_engineered_train_split] Saved rows=%d to %s", len(df), artifact_path)

@register_artifact(StepsPhase.STEP_3_5.value, StepOutputArtifact.engineered_val_split.value)
def _save_engineered_val_split(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist the internal validation split as parquet."""
    df = context_data.get(StepOutputArtifact.engineered_val_split.value)
    if df is None or (hasattr(df, "empty") and df.empty):
        log.warning("[_save_engineered_val_split] No dataframe to persist")
        return
    full_path: Path = resolve_path(ctx.phase3_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    #save_parquet(df, str(full_path))
    save_parquet(df, str(full_path), compression="snappy")
    log.info("[_save_engineered_val_split] Saved rows=%d to %s", len(df), artifact_path)