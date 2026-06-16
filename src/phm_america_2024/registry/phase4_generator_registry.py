from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from phm_america_2024.registry.generator_registry_registry import register_artifact
from phm_america_2024.common.io_service_common import save_json
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.domain.enum_registry_domain import StepsPhase, StepOutputArtifact

log = get_logger(__name__)

# regressione
@register_artifact(StepsPhase.STEP_4_2.value, StepOutputArtifact.trained_ngboost_model.value)
def _save_trained_ngboost_model(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    model = context_data.get(StepOutputArtifact.trained_ngboost_model.value)
    if model is None:
        log.warning("[_save_trained_ngboost_model] no model to persist")
        return
    full_path: Path = resolve_path(ctx.phase4_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(full_path))
    log.info("[_save_trained_ngboost_model] saved model to %s", artifact_path)


@register_artifact(StepsPhase.STEP_4_4.value, StepOutputArtifact.best_regression_model_metadata.value)
def _save_best_regression_metadata(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    metadata = context_data.get(StepOutputArtifact.best_regression_model_metadata.value)
    if metadata is None:
        log.warning("[_save_best_regression_metadata] no metadata to persist")
        return
    full_path: Path = resolve_path(ctx.phase4_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(metadata, str(full_path))
    log.info("[_save_best_regression_metadata] saved metadata to %s", artifact_path)

@register_artifact(StepsPhase.STEP_4_2.value, StepOutputArtifact.trained_model.value)
def _save_trained_model(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    model = context_data.get(StepOutputArtifact.trained_model.value)
    if model is None:
        log.warning("[_save_trained_model] no model to persist")
        return
    full_path: Path = resolve_path(ctx.phase4_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(full_path))
    log.info("[_save_trained_model] saved model to %s", artifact_path)


# classificazione

@register_artifact(StepsPhase.STEP_4_4.value, StepOutputArtifact.best_classification_model_metadata.value)
def _save_best_classification_metadata(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    metadata = context_data.get(StepOutputArtifact.best_classification_model_metadata.value)
    if metadata is None:
        log.warning("[_save_best_classification_metadata] no metadata to persist")
        return
    full_path: Path = resolve_path(ctx.phase4_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(metadata, str(full_path))
    log.info("[_save_best_classification_metadata] saved metadata to %s", artifact_path)
