from __future__ import annotations

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
        """
        Initialize the data‑preparation runner for a specific step.

        Parameters
        ----------
        ctx : RunContext
            Global execution context.
        step_key : str
            Identifier for the current pipeline step.
        step_cfg : dict[str, Any]
            Configuration dictionary for the step (including injected read_strategy).
        """
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
        """Execute the configured methods and techniques for this step.

        Step 1: Load the input DataFrame from the previous phase/step.
        Step 2: Iterate over methods → enabled techniques, apply each.
        Step 3: Collect extra artifacts from techniques (e.g., fitted scaler).
        Step 4: Persist all output artifacts via the registry.
        Step 5: Return updated context.
        """
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

        For step 3.1 the input comes from phase2's train parquet.
        For later steps it comes from the previous step's output artifact
        (registered in ctx.artifacts).

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame ready for the first technique.
        """
        import pandas as pd

        if self.step_key == StepsPhase.STEP_3_1.value:
            # Step 1: load from phase2 output (defined in global read_strategy.input_source)
            input_config = self.step_cfg.get("input_source", {})
            train_path_rel = input_config.get("train_parquet")
            if not train_path_rel:
                log.error("[_load_input_dataframe] No 'train_parquet' in input_source for step 3.1")
                raise ValueError("Missing train_parquet in input_source for step 3.1")
            full_path = resolve_path(self.ctx.phase2_dir / train_path_rel)
            log.debug("[_load_input_dataframe] loading from phase2 artifact: %s", full_path)
        else:
            # Step 2: retrieve artifact path from context registry
            # Determine which artifact key to load based on step
            input_artifact_key = self._get_input_artifact_key_for_step()
            artifact_path = self.ctx.artifacts.get(input_artifact_key)
            if not artifact_path:
                log.error("[_load_input_dataframe] Artifact '%s' not found in context. "
                          "Run previous step first.", input_artifact_key)
                raise RuntimeError(f"Required artifact '{input_artifact_key}' missing. "
                                   "Execute the preceding step.")
            full_path = resolve_path(artifact_path)
            log.debug("[_load_input_dataframe] loading from context artifact: %s", full_path)

        # Step 3: load parquet into DataFrame
        df = load_parquet(str(full_path))
        log.info("[_load_input_dataframe] loaded shape=%s", df.shape)
        return df

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
        """Apply the feature function for a single technique.

        Parameters
        ----------
        technique_name : str
            Name from YAML (e.g. ``"dataset_definition"``).
        tech_cfg : dict
            Technique configuration including ``params`` and ``output``.
        df : pd.DataFrame
            Current DataFrame to process.

        Returns
        -------
        tuple[Any, dict[str, Any] | None]
            - Updated DataFrame.
            - Optional extra artifact dict (e.g. ``{"scaler": fitted_scaler}``).
        """
        func = self._TECHNIQUE_DISPATCH.get(technique_name)
        if func is None:
            log.warning("[_execute_technique] unknown technique '%s' – skipping", technique_name)
            return df, None

        params = tech_cfg.get("params", {})
        output_dir = self.base_dir  # trace logs written by feature functions

        log.debug("[_execute_technique] calling '%s' with params=%s", technique_name, list(params.keys()))

        # Special handling: some techniques return (df, extra_artifact)
        if technique_name in ("feature_scaling", "feature_engineering", "dataset_formatting"):
            df_new, extra = func(df, params, self.ctx, output_dir)
        elif technique_name == "data_split":
            # data_split returns (train_df, val_df, extra)
            train_df, val_df, extra = func(df, params, self.ctx, output_dir)
            # We need to persist both train and val. We'll pass the train_df as main,
            # but we also need to store val_df in extra_artifacts for persistence.
            df_new = train_df
            extra["val_df"] = val_df  # inject val_df for the registry to use
            return df_new, extra
        else:
            df_new = func(df, params, self.ctx, output_dir)
            extra = None

        return df_new, extra

    # ──────────────────────────────────────────────────────────────────────────
    # Artifact persistence
    # ──────────────────────────────────────────────────────────────────────────

    def _persist_artifacts(self, df: Any, extra_artifacts: dict[str, Any]) -> None:
        """Build the context_data payload and call write_output_artifacts.

        The step's output_artifacts from YAML define what must be persisted.
        We map each artifact key to the corresponding DataFrame or extra object.
        """
        output_artifacts_cfg = self.step_cfg.get("output_artifacts", {})
        context_data: dict[str, Any] = {}

        # Step 1: add the main DataFrame(s) to context_data
        # For step 3.5 there are two parquet artifacts: train and val
        if self.step_key == StepsPhase.STEP_3_5.value:
            # The main df is train_df; val_df is in extra_artifacts
            train_key = StepOutputArtifact.engineered_train_split.value
            val_key = StepOutputArtifact.engineered_val_split.value
            context_data[train_key] = df
            context_data[val_key] = extra_artifacts.get("val_df")
        else:
            # Single main artifact: the final DataFrame
            main_artifact_key = self._get_main_artifact_key()
            if main_artifact_key:
                context_data[main_artifact_key] = df


        # Step 2: add extra artifacts (e.g., fitted scaler)
        for extra_key, extra_value in extra_artifacts.items():
            if extra_key != "val_df":  # val_df already handled
                # The extra artifact key from YAML for the scaler is
                # "fitted_scaler_regression_artifact" (see StepOutputArtifact)
                context_data[extra_key] = extra_value

        # Step 3: call registry
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