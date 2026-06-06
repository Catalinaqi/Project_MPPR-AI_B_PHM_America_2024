# src/phm_america_2024/registry/phase2_generator_registry.py
from __future__ import annotations

from typing import Any
from pathlib import Path

from phm_america_2024.registry.generator_registry_registry import register_artifact
from phm_america_2024.common.io_service_common import save_parquet

from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.domain.enum_registry_domain import StepOutputArtifact


log = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Step 2.1 – Data Acquisition artifacts (parquet)
# ──────────────────────────────────────────────────────────────────────────────


@register_artifact(
    "step_2_1_data_acquisition", StepOutputArtifact.sample_x_y_train_parquet.value
)
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
    save_parquet(
        df,
        str(full_path),
        compression=getattr(
            ctx.config.common_base_config.output_policy, "compression", "snappy"
        ),
    )
    log.info("[_save_x_y_train_parquet] Success saved rows=%d", len(df))


@register_artifact(
    "step_2_1_data_acquisition", StepOutputArtifact.sample_x_test_parquet.value
)
def _save_x_test_parquet(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist test dataset as parquet."""
    df = context_data.get(StepOutputArtifact.sample_x_test_parquet.value)
    if df is None or (hasattr(df, "empty") and df.empty):
        log.warning("[_save_x_test_parquet] No dataframe found")
        return
    full_path: Path = resolve_path(ctx.phase2_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(
        df,
        str(full_path),
        compression=getattr(
            ctx.config.common_base_config.output_policy, "compression", "snappy"
        ),
    )
    log.info("[_save_x_test_parquet] Success saved rows=%d", len(df))


@register_artifact(
    "step_2_1_data_acquisition", StepOutputArtifact.sample_x_validation_parquet.value
)
def _save_x_validation_parquet(
    ctx: Any, artifact_path: str, **context_data: Any
) -> None:
    """Persist validation dataset as parquet."""
    df = context_data.get(StepOutputArtifact.sample_x_validation_parquet.value)
    if df is None or (hasattr(df, "empty") and df.empty):
        log.warning("[_save_x_validation_parquet] No dataframe found")
        return
    full_path: Path = resolve_path(ctx.phase2_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(
        df,
        str(full_path),
        compression=getattr(
            ctx.config.common_base_config.output_policy, "compression", "snappy"
        ),
    )
    log.info("[_save_x_validation_parquet] Success saved rows=%d", len(df))
