from __future__ import annotations

from pathlib import Path
from typing import Any

from phm_america_2024.registry.generator_registry_registry import register_artifact
from phm_america_2024.common.io_service_common import save_json
from phm_america_2024.reporting.artifact_persister_reporting import save_figure
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.domain.enum_registry_domain import StepsPhase, StepOutputArtifact

log = get_logger(__name__)


# =============================================================================
# PERSISTENCIA GRÁFICA (Fase 5.1 e Interpretación)
# =============================================================================


@register_artifact(
    StepsPhase.STEP_5_1.value, StepOutputArtifact.fi_importance_plot.value
)
def _save_fi_importance_plot(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    fig = context_data.get(StepOutputArtifact.fi_importance_plot.value)
    if fig:
        full_path = resolve_path(ctx.phase5_dir / artifact_path)
        save_figure(fig, out_path=full_path, dpi=300)


@register_artifact(
    StepsPhase.STEP_5_1.value, StepOutputArtifact.fi_permutation_plot.value
)
def _save_fi_permutation_plot(
    ctx: Any, artifact_path: str, **context_data: Any
) -> None:
    fig = context_data.get(StepOutputArtifact.fi_permutation_plot.value)
    if fig:
        full_path = resolve_path(ctx.phase5_dir / artifact_path)
        save_figure(fig, out_path=full_path, dpi=300)


# =============================================================================
# PERSISTENCIA GRÁFICA and JSON (Fase 5.2 - Probabilistic Evaluation)
# =============================================================================


@register_artifact(
    StepsPhase.STEP_5_2.value, StepOutputArtifact.evaluation_summary_json.value
)
def _save_evaluation_summary(ctx: Any, artifact_path: str, **context_data: Any) -> None:
    """Persist the final regression metrics summary JSON."""
    summary = context_data.get(StepOutputArtifact.evaluation_summary_json.value)
    if summary is None:
        log.warning("[_save_evaluation_summary] no evaluation summary to persist")
        return
    full_path: Path = resolve_path(ctx.phase5_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, str(full_path))
    log.info("[_save_evaluation_summary] saved evaluation summary to %s", artifact_path)


@register_artifact(
    StepsPhase.STEP_5_2.value, StepOutputArtifact.eval_calibration_plot.value
)
def _save_eval_calibration_plot(
    ctx: Any, artifact_path: str, **context_data: Any
) -> None:
    fig = context_data.get(StepOutputArtifact.eval_calibration_plot.value)
    if fig:
        full_path = resolve_path(ctx.phase5_dir / artifact_path)
        save_figure(fig, out_path=full_path, dpi=300)


@register_artifact(
    StepsPhase.STEP_5_2.value, StepOutputArtifact.eval_degradation_plot.value
)
def _save_eval_degradation_plot(
    ctx: Any, artifact_path: str, **context_data: Any
) -> None:
    fig = context_data.get(StepOutputArtifact.eval_degradation_plot.value)
    if fig:
        full_path = resolve_path(ctx.phase5_dir / artifact_path)
        save_figure(fig, out_path=full_path, dpi=300)


# =============================================================================
# PERSISTENCIA JSON (Fase 5.4)
# =============================================================================


@register_artifact(
    StepsPhase.STEP_5_4.value, StepOutputArtifact.deployment_sign_off.value
)
def _save_deployment_sign_off(
    ctx: Any, artifact_path: str, **context_data: Any
) -> None:
    """Persist the deployment sign-off certificate JSON."""
    cert = context_data.get(StepOutputArtifact.deployment_sign_off.value)
    if cert is None:
        log.warning("[_save_deployment_sign_off] no sign-off certificate to persist")
        return
    full_path: Path = resolve_path(ctx.phase5_dir / artifact_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(cert, str(full_path))
    log.info(
        "[_save_deployment_sign_off] saved sign-off certificate to %s", artifact_path
    )
