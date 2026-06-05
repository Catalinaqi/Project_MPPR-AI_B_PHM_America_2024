from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import joblib

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.pipeline.utils.context_facade_common import RunContext
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.registry.generator_registry_registry import write_output_artifacts
from phm_america_2024.domain.enum_registry_domain import StepsPhase, StepOutputArtifact

# ── Technique implementations ──
from phm_america_2024.interpretation.cluster_interpreter_interpretation import (
    feature_importance,
    permutation_importance,
)
from phm_america_2024.interpretation.business_alignment_evaluator_interpretation import (
    calibration_audit,
    performance_degradation_benchmarking,
)
from phm_america_2024.interpretation.pipeline_auditor_interpretation import leakage_detection
from phm_america_2024.interpretation.deployment_reporter_interpretation import final_sign_off

log = get_logger(__name__)


class Phase5EvaluationAndInterpretationRunner:
    """Execute all techniques for a single CRISP-DM Phase 5 step.

    - Loads the model, scaler, and test data once per step.
    - Dispatches execution per individual technique defined in YAML.
    - Persists combined artifacts via a centralized writer after execution.
    """

    # Mapping YAML technique names to their Python implementations
    _TECHNIQUE_DISPATCH: Dict[str, Any] = {
        "feature_importance": feature_importance,
        "permutation_importance": permutation_importance,
        "calibration_audit": calibration_audit,
        "performance_degradation_benchmarking": performance_degradation_benchmarking,
        "leakage_detection": leakage_detection,
        "final_sign_off": final_sign_off,
    }

    def __init__(self, ctx: RunContext, step_key: str, step_cfg: Dict[str, Any]) -> None:
        """Initialize the evaluation and interpretation runner."""
        self.ctx = ctx
        self.step_key = step_key
        self.step_cfg = step_cfg

        self.base_dir: Path = getattr(ctx, "phase5_dir", None)
        if self.base_dir is None:
            log.error("[Phase5Runner] YAML key missing or context error: phase5_dir is None")
            raise RuntimeError("phase5_dir missing from RunContext")

        self.base_dir.mkdir(parents=True, exist_ok=True)
        log.debug("[Phase5Runner] init step='%s' base_dir='%s'", self.step_key, self.base_dir)

    def run(self) -> RunContext:
        """Run the configured techniques for this phase step."""
        log.debug("[Phase5Runner] run entry")
        log.info("[Phase5Runner] run step='%s'", self.step_key)

        self._load_artifacts()

        extra_artifacts: Dict[str, Any] = {}

        methods: Dict[str, Any] = self.step_cfg.get("methods", {})
        log.debug("[Phase5Runner] methods to execute: %s", list(methods.keys()))

        for method_name, method_cfg in methods.items():
            if not method_cfg.get("enabled", True):
                log.debug("[Phase5Runner] method '%s' disabled – skip", method_name)
                continue

            log.info("[Phase5Runner] executing method='%s'", method_name)

            techniques: Dict[str, Any] = method_cfg.get("techniques", {})
            for tech_name, tech_cfg in techniques.items():
                if not tech_cfg.get("enabled", True):
                    log.debug("[Phase5Runner] technique '%s' disabled – skip", tech_name)
                    continue

                log.info("[Phase5Runner] executing technique='%s'", tech_name)

                art = self._execute_technique(tech_name, tech_cfg)
                if art is not None:
                    log.debug(
                        "[Phase5Runner] technique '%s' returned artifacts: %s",
                        tech_name,
                        list(art.keys()),
                    )
                    extra_artifacts.update(art)

        # Build aggregated evaluation_summary for step 5.2 if data present
        if self.step_key == StepsPhase.STEP_5_2.value:
            intervals = extra_artifacts.get("calibration_intervals")
            degradation = extra_artifacts.get("degradation_metrics")
            if intervals and degradation:
                extra_artifacts["evaluation_summary"] = {
                    "calibration_intervals": intervals,
                    "degradation": degradation,
                }

        self._persist_artifacts(extra_artifacts)

        log.info("[Phase5Runner] completed step='%s'", self.step_key)
        return self.ctx

    def _execute_technique(
        self,
        technique_name: str,
        tech_cfg: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Dispatch to the specific technique function."""
        log.debug("[_execute_technique] entry technique='%s'", technique_name)

        func = self._TECHNIQUE_DISPATCH.get(technique_name)
        if func is None:
            log.warning("[_execute_technique] unknown technique '%s' – skipping", technique_name)
            return None

        df_test = getattr(self.ctx, "df_test", pd.DataFrame())
        try:
            _, extra = func(df_test, tech_cfg, self.ctx, self.base_dir)
            log.info("[_execute_technique] '%s' completed successfully", technique_name)
            return extra
        except Exception as e:
            log.error("[_execute_technique] '%s' failed: %s", technique_name, e)
            raise

    def _load_artifacts(self) -> None:
        """Load model, scaler, test data (and optionally training data) into context."""
        log.debug("[_load_artifacts] entry step='%s'", self.step_key)

        input_source: Dict[str, Any] = self.step_cfg.get("input_source", {})
        if not input_source:
            log.debug("[_load_artifacts] No input_source in step config – nothing to load.")
            return

        model_path = Path(str(input_source.get("model", "")))
        scaler_path = Path(str(input_source.get("scaler", "")))
        test_path = Path(str(input_source.get("challenge_test_data", "")))
        train_path = Path(str(input_source.get("challenge_train_data", "")))  # optional

        # Resolve relative paths
        if not model_path.is_absolute():
            model_path = resolve_path(self.ctx.run_dir / "phase4_data_modeling" / model_path)
        if not test_path.is_absolute():
            test_path = resolve_path(self.ctx.run_dir / "phase2_data_understanding" / test_path)
        if not scaler_path.is_absolute():
            scaler_path = resolve_path(self.ctx.run_dir / "phase3_data_preparation" / scaler_path)
        if train_path and not train_path.is_absolute():
            train_path = resolve_path(self.ctx.run_dir / "phase2_data_understanding" / train_path)

        # Model
        if model_path.exists() and not hasattr(self.ctx, "model"):
            self.ctx.model = joblib.load(model_path)
            log.info("[_load_artifacts] loaded model from %s", model_path)

        # Test data & prediction
        if test_path.exists() and not hasattr(self.ctx, "df_test"):
            df_test_raw = pd.read_parquet(test_path)
            target_col = self.step_cfg.get("target_col", "trq_margin")

            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                self.ctx.scaler = scaler
                numeric_cols = df_test_raw.select_dtypes(include=[np.number]).columns.tolist()
                scale_cols = [
                    c for c in numeric_cols
                    if c != target_col and c in scaler.feature_names_in_
                ]
                df_test_scaled = df_test_raw.copy()
                if scale_cols:
                    df_test_scaled[scale_cols] = scaler.transform(df_test_raw[scale_cols])
            else:
                df_test_scaled = df_test_raw.copy()

            self.ctx.df_test = df_test_scaled

            X_test = df_test_scaled.drop(columns=[target_col])
            pred_dist = self.ctx.model.pred_dist(X_test)
            self.ctx.y_true = df_test_scaled[target_col].values
            self.ctx.y_pred_mean = pred_dist.loc
            self.ctx.y_pred_std = pred_dist.scale

        # Optional training data (for leakage detection)
        if train_path.exists() and not hasattr(self.ctx, "train_df"):
            self.ctx.train_df = pd.read_parquet(train_path)
            log.info("[_load_artifacts] loaded train data from %s", train_path)

    def _persist_artifacts(self, extra_artifacts: Dict[str, Any]) -> None:
        """Write artifacts via the registry."""
        log.debug(
            "[_persist_artifacts] entry step='%s' artifacts=%s",
            self.step_key,
            list(extra_artifacts.keys()),
        )

        context_data: Dict[str, Any] = {}

        if self.step_key == StepsPhase.STEP_5_2.value:
            summary = extra_artifacts.get("evaluation_summary")
            if summary:
                context_data[StepOutputArtifact.evaluation_summary_json.value] = summary

        elif self.step_key == StepsPhase.STEP_5_4.value:
            cert = extra_artifacts.get("deployment_sign_off")
            if cert:
                context_data[StepOutputArtifact.deployment_sign_off.value] = cert

        # Pass remaining artifacts (they will be handled by the registry if registered)
        for k, v in extra_artifacts.items():
            if k not in context_data:
                context_data[k] = v

        write_output_artifacts(self.ctx, self.step_key, self.step_cfg, self.base_dir, **context_data)
        log.info("[_persist_artifacts] artifacts persisted for step='%s'", self.step_key)