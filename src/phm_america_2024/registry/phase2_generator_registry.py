from __future__ import annotations

from typing import Any
from phm_america_2024.registry.generator_registry_registry import register_artifact
from phm_america_2024.data.persist_persister_data import save_parquet
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.configuration.enum_registry_config import StepsPhase, StepOutputArtifact

@register_artifact(StepsPhase.STEP_2_1.value,
                   StepOutputArtifact.sample_x_y_train_parquet.value)
def _save_x_y_train_parquet(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Save merged train dataset (X + Y) as parquet anchoring via path_service."""
    df = context_data.get("df_merged")

    # Usamos tu servicio común para resolver la ruta absoluta de manera segura
    full_path = resolve_path(ctx.phase2_dir / artifact_path)

    save_parquet(
        df,
        str(full_path),
        compression=ctx.config.common_base_config.output_policy.compression
    )


@register_artifact(StepsPhase.STEP_2_1.value,
                   StepOutputArtifact.sample_x_test_parquet.value)
def _save_x_test_parquet(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Save test dataset as parquet anchoring via path_service."""
    df = context_data.get("df_test")
    full_path = resolve_path(ctx.phase2_dir / artifact_path)

    save_parquet(
        df,
        str(full_path),
        compression=ctx.config.common_base_config.output_policy.compression
    )


@register_artifact(StepsPhase.STEP_2_1.value,
                   StepOutputArtifact.sample_x_validation_parquet.value)
def _save_x_validation_parquet(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Save validation dataset as parquet anchoring via path_service."""
    df = context_data.get("df_val")
    full_path = resolve_path(ctx.phase2_dir / artifact_path)

    save_parquet(
        df,
        str(full_path),
        compression=ctx.config.common_base_config.output_policy.compression
    )