from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.context_facade_common import RunContext
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.data.load_loader_data import load_parquet
from phm_america_2024.registry.generator_registry_registry import write_output_artifacts
from phm_america_2024.configuration.enum_registry_config import StepsPhase, StepOutputArtifact
from phm_america_2024.feature.selection_selector_feature import dataset_definition, feature_selection
from phm_america_2024.feature.cleaning_transformer_feature import outlier_handling, duplicate_handling
from phm_america_2024.feature.transformation_transformer_feature import feature_scaling, feature_engineering
from phm_america_2024.feature.formatting_transformer_feature import data_split, dataset_formatting

log = get_logger(__name__)


class Phase3PreparationRunner:
    """Execute all techniques for a single CRISP‑DM data‑preparation step.

    Each step loads a DataFrame from the previous phase/step, applies
    the configured feature‑engineering functions, and persists the
    result as parquet (and optional pickle) via the central artifact registry.
    """

    # ── Technique dispatch mapping ────────────────────────────────────────────
    _TECHNIQUE_DISPATCH: dict[str, Any] = {
        # step 3.1
        "dataset_definition":   dataset_definition,
        "feature_selection":    feature_selection,
        # step 3.2
        "outlier_handling":     outlier_handling,
        "duplicate_handling":   duplicate_handling,
        # step 3.3
        "feature_scaling":      feature_scaling,
        "feature_engineering":  feature_engineering,
        # step 3.5
        "data_split":           data_split,
        "dataset_formatting":   dataset_formatting,
    }

    # Mapping technique_name → StepOutputArtifact key for the main DataFrame
    # (the key used in output_artifacts in YAML)
    _TECHNIQUE_TO_MAIN_ARTIFACT_KEY: dict[str, str] = {
        "dataset_definition":   StepOutputArtifact.selected_regression_train_parquet.value,
        "feature_selection":    StepOutputArtifact.selected_regression_train_parquet.value,
        "outlier_handling":     StepOutputArtifact.cleaned_regression_train_parquet.value,
        "duplicate_handling":   StepOutputArtifact.cleaned_regression_train_parquet.value,
        "feature_scaling":      StepOutputArtifact.transformed_regression_train_parquet.value,
        "feature_engineering":  StepOutputArtifact.transformed_regression_train_parquet.value,
        "data_split":           StepOutputArtifact.engineered_train_split.value,  # also val
        "dataset_formatting":   StepOutputArtifact.engineered_train_split.value,
    }

    def __init__(self, ctx: RunContext, step_key: str, step_cfg: dict[str, Any]) -> None:
        """Initialize the data‑preparation runner for a specific step."""
        self.ctx: RunContext = ctx
        self.step_key: str = step_key
        self.step_cfg: dict[str, Any] = step_cfg
        self.base_dir: Path = getattr(ctx, "phase3_dir", None)

        # Step 1: CALL validate_dir() – ensure phase3 output directory exists
        if self.base_dir is None:
            log.error("[Phase3PreparationRunner] ctx.phase3_dir is None – cannot resolve artifact paths")
            raise RuntimeError("phase3_dir missing from RunContext")

        log.debug("[Phase3PreparationRunner] init step='%s' base_dir='%s'",
                  self.step_key, self.base_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> RunContext:
        """Execute the configured methods and techniques for this step."""
        log.info("[Phase3PreparationRunner] run step='%s'", self.step_key)

        # ── Step 1: load input DataFrame ───────────────────────────────────
        df = self._load_input_dataframe()

        # ── Step 2: iterate methods and techniques ─────────────────────────
        methods: dict[str, Any] = self.step_cfg.get("methods", {})
        log.debug("[Phase3PreparationRunner] methods=%s", list(methods.keys()))

        extra_artifacts: dict[str, Any] = {}  # holds non-DataFrame artifacts

        for method_name, method_cfg in methods.items():
            if not method_cfg.get("enabled", True):
                log.debug("[Phase3PreparationRunner] method '%s' disabled – skip", method_name)
                continue

            techniques: dict[str, Any] = method_cfg.get("techniques", {})
            for technique_name, tech_cfg in techniques.items():
                if not tech_cfg.get("enabled", True):
                    log.debug("[Phase3PreparationRunner] technique '%s' disabled – skip", technique_name)
                    continue

                # Step 2a: apply the technique function
                df, art = self._execute_technique(technique_name, tech_cfg, df)
                if art is not None:
                    extra_artifacts.update(art)

        # ── Step 4: persist artifacts via registry ─────────────────────────
        self._persist_artifacts(df, extra_artifacts)

        log.info("[Phase3PreparationRunner] completed step='%s'", self.step_key)
        return self.ctx

    # ──────────────────────────────────────────────────────────────────────────
    # Input loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_input_dataframe(self) -> Any:
        """Load the input DataFrame from the previous step's artifact.

        Supports dynamic fallback scanning across past execution workspaces to allow
        isolated step execution commands.
        """
        if self.step_key == StepsPhase.STEP_3_1.value:
            input_config = self.step_cfg.get("input_source", {})
            train_path_rel = input_config.get("train_parquet")
            if not train_path_rel:
                log.error("[_load_input_dataframe] No 'train_parquet' in input_source for step 3.1")
                raise ValueError("Missing train_parquet in input_source for step 3.1")
            full_path = resolve_path(self.ctx.phase2_dir / train_path_rel)
            log.debug("[_load_input_dataframe] loading from phase2 artifact: %s", full_path)
            df = load_parquet(str(full_path))
            log.info("[_load_input_dataframe] loaded shape=%s", df.shape)
            return df
        else:
            # Determine previous target file based on the current step lineage context
            if self.step_key == StepsPhase.STEP_3_2.value:
                prev_file = "3.1.selection.selected_regression_train.parquet"
            elif self.step_key == StepsPhase.STEP_3_3.value:
                prev_file = "3.2.cleaning.cleaned_regression_train.parquet"
            elif self.step_key == StepsPhase.STEP_3_5.value:
                prev_file = "3.3.transformation.transformed_regression_train.parquet"
            else:
                raise ValueError(f"Unsupported step: {self.step_key}")

            # Strategy A: Check local path from current runtime folder first
            full_path = resolve_path(self.base_dir / prev_file)
            if full_path.exists():
                log.debug("[_load_input_dataframe] MVP loading directly from current active run directory: %s", full_path)
                df = load_parquet(str(full_path))
                log.info("[_load_input_dataframe] loaded shape=%s", df.shape)
                return df

            # Strategy B: Dynamic fallback lookback traversal engine over historical runs
            log.warning("[_load_input_dataframe] Artifact '%s' missing in active run workspace. Scanning history...", prev_file)

            # Navigates up from 'outputs/runs/regression/phm2024/TIMESTAMP/phase3_data_preparation' to 'phm2024'
            runs_root = self.base_dir.parent.parent
            if runs_root.exists() and runs_root.is_dir():
                # Sort historical timestamp folders by execution date
                past_runs = sorted(
                    [d for d in runs_root.iterdir() if d.is_dir()],
                    key=os.path.getmtime,
                    reverse=True
                )

                for run_dir in past_runs:
                    candidate_file = run_dir / "phase3_data_preparation" / prev_file
                    if candidate_file.exists():
                        log.info("✅ [Standalone Fallback] Found missing data lineage file at: %s", candidate_file)
                        df = load_parquet(str(candidate_file))
                        log.info("[_load_input_dataframe] loaded shape=%s from fallback", df.shape)
                        return df

            # Absolute fail guard if dependencies cannot be resolved across the system
            log.error("[_load_input_dataframe] Precursor dependency file '%s' could not be resolved.", prev_file)
            raise FileNotFoundError(f"Missing upstream data target artifact: {prev_file}. Run preceding phases once.")

    def _get_input_artifact_key_for_step(self) -> str:
        """Return the artifact key that this step consumes."""
        mapping = {
            StepsPhase.STEP_3_2.value: StepOutputArtifact.selected_regression_train_parquet.value,
            StepsPhase.STEP_3_3.value: StepOutputArtifact.cleaned_regression_train_parquet.value,
            StepsPhase.STEP_3_5.value: StepOutputArtifact.transformed_regression_train_parquet.value,
        }
        key = mapping.get(self.step_key)
        if key is None:
            log.error("[_get_input_artifact_key_for_step] Unknown input artifact for step '%s'",
                      self.step_key)
            raise ValueError(f"No input artifact mapping for step: {self.step_key}")
        return key

    # ──────────────────────────────────────────────────────────────────────────
    # Technique execution
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_technique(
            self,
            technique_name: str,
            tech_cfg: dict[str, Any],
            df: Any,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Apply the feature function for a single technique."""
        func = self._TECHNIQUE_DISPATCH.get(technique_name)
        if func is None:
            log.warning("[_execute_technique] unknown technique '%s' – skipping", technique_name)
            return df, None

        params = tech_cfg.get("params", {})
        output_dir = self.base_dir

        log.debug("[_execute_technique] calling '%s' with params=%s", technique_name, list(params.keys()))

        if technique_name in ("feature_scaling", "feature_engineering", "dataset_formatting"):
            df_new, extra = func(df, params, self.ctx, output_dir)
        elif technique_name == "data_split":
            train_df, val_df, extra = func(df, params, self.ctx, output_dir)
            df_new = train_df
            extra["val_df"] = val_df
            return df_new, extra
        else:
            df_new = func(df, params, self.ctx, output_dir)
            extra = None

        # ── Normalizar llave del artifact extra para coincidir con el generador ──
        if extra is not None and technique_name == "feature_scaling":
            # feature_scaling devuelve {'scaler': scaler_obj}
            # El generador espera context_data['fitted_scaler_regression_artifact'] con {'scaler': ...}
            base_key = StepOutputArtifact.fitted_scaler_regression_artifact.value  # "fitted_scaler_regression_artifact"
            extra = {base_key: {"scaler": extra["scaler"]}}

        return df_new, extra

    # ──────────────────────────────────────────────────────────────────────────
    # Artifact persistence
    # ──────────────────────────────────────────────────────────────────────────

    def _persist_artifacts(self, df: Any, extra_artifacts: dict[str, Any]) -> None:
        """Build the context_data payload and call write_output_artifacts."""
        output_artifacts_cfg = self.step_cfg.get("output_artifacts", {})
        context_data: dict[str, Any] = {}

        if self.step_key == StepsPhase.STEP_3_5.value:
            train_key = StepOutputArtifact.engineered_train_split.value
            val_key = StepOutputArtifact.engineered_val_split.value
            context_data[train_key] = df
            context_data[val_key] = extra_artifacts.get("val_df")
        else:
            main_artifact_key = self._get_main_artifact_key()
            if main_artifact_key:
                context_data[main_artifact_key] = df

        for extra_key, extra_value in extra_artifacts.items():
            if extra_key != "val_df":
                context_data[extra_key] = extra_value

        log.debug("[_persist_artifacts] Dispatching to registry with keys: %s", list(context_data.keys()))
        write_output_artifacts(
            ctx=self.ctx,
            step_key=self.step_key,
            step_cfg=self.step_cfg,
            base_dir=self.base_dir,
            **context_data,
        )


    def _get_main_artifact_key(self) -> str | None:
        """Return the artifact key for the main DataFrame of this step."""
        mapping = {
            StepsPhase.STEP_3_1.value: StepOutputArtifact.selected_regression_train_parquet.value,
            StepsPhase.STEP_3_2.value: StepOutputArtifact.cleaned_regression_train_parquet.value,
            StepsPhase.STEP_3_3.value: StepOutputArtifact.transformed_regression_train_parquet.value,
            StepsPhase.STEP_3_5.value: StepOutputArtifact.engineered_train_split.value,
        }
        return mapping.get(self.step_key)