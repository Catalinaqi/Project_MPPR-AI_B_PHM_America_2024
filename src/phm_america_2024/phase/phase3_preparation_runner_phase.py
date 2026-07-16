# src/phm_america_2024/phase/phase3_preparation_runner_phase.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.io_service_common import load_parquet
from phm_america_2024.pipeline.utils.context_facade_common import RunContext
from phm_america_2024.registry.generator_registry_registry import write_output_artifacts
from phm_america_2024.domain.enum_registry_domain import StepOutputArtifact, StepsPhase
from phm_america_2024.feature.selection_selector_feature import (
    dataset_definition,
    feature_selection,
)
from phm_america_2024.feature.cleaning_transformer_feature import (
    outlier_handling,
    duplicate_handling,
)
from phm_america_2024.feature.transformation_transformer_feature import (
    feature_scaling,
    feature_engineering,
)
from phm_america_2024.feature.formatting_transformer_feature import (
    data_split,
    dataset_formatting,
)

log = get_logger(__name__)


class Phase3PreparationRunner:
    """Execute all techniques for a single step of Stage 3 (Feature Extraction)
    of the data-driven diagnosis/prognosis workflow.

    Each step loads a DataFrame from the previous phase/step, applies
    the configured feature-engineering functions, and persists the
    result as parquet (and optional pickle) via the central artifact registry.
    """

    # ── Technique dispatch mapping ────────────────────────────────────────────
    # NOTE: technique -> function resolution happens by technique NAME (dict
    # key), not by step number, so the step labels below are purely
    # documentation and do not affect execution order or correctness.
    _TECHNIQUE_DISPATCH: dict[str, Any] = {
        # step 3.1 (data_selection)
        "dataset_definition": dataset_definition,
        "feature_selection": feature_selection,
        # step 3.2 (data_cleaning)
        "outlier_handling": outlier_handling,
        "duplicate_handling": duplicate_handling,
        # step 3.3 (data_engineering)
        "feature_engineering": feature_engineering,
        # step 3.4 (data_formatting)
        "data_split": data_split,
        "dataset_formatting": dataset_formatting,
        # step 3.5 (data_transformation)
        "feature_scaling": feature_scaling,
    }

    def __init__(
            self, ctx: RunContext, step_key: str, step_cfg: dict[str, Any]
    ) -> None:
        """Initialize the data-preparation runner for a specific step."""
        self.ctx: RunContext = ctx
        self.step_key: str = step_key
        self.step_cfg: dict[str, Any] = step_cfg
        self.base_dir: Path = getattr(ctx, "phase3_dir", None)

        # Step 1: ensure the Phase 3 output directory exists
        if self.base_dir is None:
            log.error(
                "[Phase3PreparationRunner] ctx.phase3_dir is None – cannot resolve artifact paths"
            )
            raise RuntimeError("phase3_dir missing from RunContext")

        log.debug(
            "[Phase3PreparationRunner] init step='%s' base_dir='%s'",
            self.step_key,
            self.base_dir,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> RunContext:
        """Execute the configured methods and techniques for this step."""
        log.info("[Phase3PreparationRunner] run step='%s'", self.step_key)

        # ── Step 1: load input DataFrame ───────────────────────────────────
        df = self._load_input_dataframe()

        # ── Step 2: iterate methods and techniques ─────────────────────────
        # Execution order follows dict insertion order (Python 3.7+ preserves
        # it), which in turn follows the order techniques are declared in the
        # JSON/YAML config. This is relevant, for example, for the relative
        # order between 'outlier_handling' and 'duplicate_handling' in step 3.2.
        methods: dict[str, Any] = self.step_cfg.get("methods", {})
        log.debug("[Phase3PreparationRunner] methods=%s", list(methods.keys()))

        extra_artifacts: dict[str, Any] = {}  # holds non-DataFrame artifacts

        for method_name, method_cfg in methods.items():
            if not method_cfg.get("enabled", True):
                log.debug(
                    "[Phase3PreparationRunner] method '%s' disabled – skip", method_name
                )
                continue

            techniques: dict[str, Any] = method_cfg.get("techniques", {})
            for technique_name, tech_cfg in techniques.items():
                if not tech_cfg.get("enabled", True):
                    log.debug(
                        "[Phase3PreparationRunner] technique '%s' disabled – skip",
                        technique_name,
                    )
                    continue

                # Step 2a: apply the technique function
                df, art = self._execute_technique(technique_name, tech_cfg, df)
                if art is not None:
                    extra_artifacts.update(art)

        # ── Step 3: persist artifacts via registry ──────────────────────────
        self._persist_artifacts(df, extra_artifacts)

        log.info("[Phase3PreparationRunner] completed step='%s'", self.step_key)
        return self.ctx

    # ──────────────────────────────────────────────────────────────────────────
    # Input loading
    # ──────────────────────────────────────────────────────────────────────────
    #
    # AUDIT NOTE: an earlier version of this file contained two near-identical
    # copies of a legacy loader ('_load_input_dataframe_old'), neither of
    # which was ever called anywhere in the codebase — the active path always
    # used '_load_input_dataframe' below. The exact duplicate has been
    # removed here; a single legacy copy is kept, clearly marked, purely for
    # historical traceability. This does not change runtime behavior in any
    # way, since dead code was never executed.

    def _load_input_dataframe(self) -> Any:
        """[ACTIVE] Load the required input DataFrame strictly based on step
        configuration (explicit path declared in the YAML/JSON config)."""
        log.debug("[_load_input_dataframe] entry step='%s'", self.step_key)

        # 1. Extract the explicit artifact path from the config
        try:
            artifact_path_str = self.step_cfg["input_artifact"]["path"]
        except KeyError:
            log.error(
                "[_load_input_dataframe] 'input_artifact.path' missing in config for step %s",
                self.step_key,
            )
            raise ValueError(f"Step {self.step_key} config lacks 'input_artifact.path'")

        # 2. Determine the source base directory.
        # Step 3.1 reads from Phase 2. The remaining steps (3.2 through 3.5)
        # read from Phase 3 itself.
        if self.step_key == StepsPhase.STEP_3_1.value:
            source_dir = self.ctx.phase2_dir
        else:
            source_dir = self.base_dir

        if source_dir is None:
            raise RuntimeError(
                f"[_load_input_dataframe] Source directory is None for step {self.step_key}"
            )

        full_path: Path = source_dir / artifact_path_str

        # 3. Validate existence and load
        if not full_path.exists():
            log.error(
                "[_load_input_dataframe] Dependency failed. Input data file not found at: %s",
                full_path,
            )
            raise FileNotFoundError(
                f"Configured input artifact does not exist: {full_path}"
            )

        log.info("[_load_input_dataframe] Resolving input data from: %s", full_path)

        # 4. Deserialize
        df = load_parquet(str(full_path))
        log.info("[_load_input_dataframe] Loaded shape=%s", getattr(df, "shape", "N/A"))

        return df

    def _load_input_dataframe_legacy(self) -> Any:
        """[LEGACY - NOT REFERENCED ANYWHERE] Dynamically load the input
        DataFrame using file lineage patterns and a historical-run fallback
        search. Superseded by '_load_input_dataframe' above, which reads the
        explicit path from config instead of guessing via glob patterns.
        Kept only for historical traceability; safe to remove entirely once
        no longer needed as reference."""
        log.debug("[_load_input_dataframe_legacy] entry step='%s'", self.step_key)

        # 1. Define what to search for and where it should live
        if self.step_key == StepsPhase.STEP_3_1.value:
            search_dir = self.ctx.phase2_dir
            pattern = "*_train.parquet"
            target_phase_folder = "phase2_data_understanding"
        else:
            search_dir = self.base_dir
            target_phase_folder = "phase3_data_preparation"
            lineage_map = {
                StepsPhase.STEP_3_2.value: "*.selected_regression_train.parquet",
                StepsPhase.STEP_3_3.value: "*.cleaned_regression_train.parquet",
                StepsPhase.STEP_3_4.value: "*.engineered_regression_train.parquet",
                StepsPhase.STEP_3_5.value: "*.transformed_regression_train.parquet",
            }
            pattern = lineage_map.get(self.step_key)
            if not pattern:
                raise ValueError(
                    f"No lineage mapping defined for step: {self.step_key}"
                )

        # ATTEMPT 1: search within the current run (e.g. full pipeline executed in sequence)
        if search_dir and search_dir.exists():
            matches = list(search_dir.glob(pattern))
            if matches:
                log.info(
                    "[_load_input_dataframe_legacy] Lineage resolved in current run. Loading: %s",
                    matches[0].name,
                )
                return load_parquet(str(matches[0]))

        # ATTEMPT 2: historical search engine (fallback for isolated step runs, e.g. 3.1 alone)
        log.warning(
            "[_load_input_dataframe_legacy] Artifact not in active run. Scanning history for '%s'...",
            pattern,
        )

        runs_root = self.base_dir.parent.parent

        if runs_root.exists() and runs_root.is_dir():
            import os

            # Sort run folders by modification date (most recent first)
            past_runs = sorted(
                [d for d in runs_root.iterdir() if d.is_dir()],
                key=os.path.getmtime,
                reverse=True,
            )

            for run_dir in past_runs:
                historical_phase_dir = run_dir / target_phase_folder
                if historical_phase_dir.exists():
                    historical_matches = list(historical_phase_dir.glob(pattern))
                    if historical_matches:
                        log.info(
                            "[Historical Fallback] Found precursor data at: %s",
                            historical_matches[0],
                        )
                        return load_parquet(str(historical_matches[0]))

        # If the historical engine also fails
        log.error(
            "[_load_input_dataframe_legacy] Missing upstream data. Searched history for '%s'",
            pattern,
        )
        raise FileNotFoundError(
            f"Dependency failed. Could not find precursor data for step {self.step_key}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Technique execution
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_technique(
            self,
            technique_name: str,
            tech_cfg: dict[str, Any],
            df: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Apply the feature function for a single technique."""
        # Step 1: resolve the function via the dispatch dictionary
        func = self._TECHNIQUE_DISPATCH.get(technique_name)
        if func is None:
            log.warning(
                "[_execute_technique] unknown technique '%s' – skipping", technique_name
            )
            return df, {}

        log.debug("[_execute_technique] executing '%s'", technique_name)

        try:
            # All Phase 3 functions must return: (DataFrame, artifacts_dict)

            # Step 2: run the technique
            df_new, extra = func(df, tech_cfg, self.ctx, self.base_dir)

            # Step 3: defensive normalization (in case the function returned None for extra)
            extra = extra if extra is not None else {}

            # Step 4: auto-save audit JSONs (if the config declares an 'output' attribute)
            output_key = tech_cfg.get("output")
            if output_key and str(output_key).endswith(".json"):
                # If the function returned a trace in the dict, save it
                trace_data = extra.pop(
                    "trace", None
                )  # extract the trace without sending it to the Registry
                if trace_data:
                    from phm_america_2024.common.io_service_common import save_json

                    save_json(trace_data, self.base_dir / output_key)

            log.info("[_execute_technique] '%s' completed successfully", technique_name)
            return df_new, extra

        except Exception as e:
            log.exception(
                "[_execute_technique] technique '%s' failed: %s", technique_name, e
            )
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Artifact persistence
    # ──────────────────────────────────────────────────────────────────────────

    def _persist_artifacts(self, df: Any, extra_artifacts: dict[str, Any]) -> None:
        """Write artifacts generated by Phase 3 to disk via the central registry.

        Args:
            df: Main updated DataFrame (or None for specific splitting steps).
            extra_artifacts: Dictionary of additional artifacts to persist.
        """
        log.debug(
            "[_persist_artifacts] entry step='%s' artifacts=%s",
            self.step_key,
            list(extra_artifacts.keys()),
        )
        context_data: dict[str, Any] = {}
        task: str = self.ctx.config.metadata.pipeline_key.task

        # ── Step 1: handle the main DataFrame(s) depending on the step ──
        if self.step_key == StepsPhase.STEP_3_4.value:
            # The data-formatting step (3.4) produces disjoint sets (train, val, test)
            if df is not None:
                context_data[StepOutputArtifact.engineered_train_split.value] = df
            if "val_df" in extra_artifacts:
                context_data[StepOutputArtifact.engineered_val_split.value] = (
                    extra_artifacts["val_df"]
                )
            if "test_df" in extra_artifacts:
                context_data[StepOutputArtifact.engineered_test_split.value] = (
                    extra_artifacts["test_df"]
                )
        else:
            # For all other steps, dynamically extract the parquet key from the config
            output_artifacts_cfg = self.step_cfg.get("output_artifacts", {})
            for key in output_artifacts_cfg.keys():
                if "parquet" in key:
                    context_data[key] = df
                    break

        # ── Step 2: task-aware scaler routing (data-transformation / scaling step) ──
        if self.step_key == StepsPhase.STEP_3_5.value:
            # Capture the scaler object regardless of the internal key it arrives under
            scaler_data = extra_artifacts.get(
                "fitted_scaler_regression_artifact"
            ) or extra_artifacts.get("fitted_scaler_bin")
            if scaler_data is not None:
                if task == "classification":
                    log.debug(
                        "[_persist_artifacts] Routing scaler to classification registry key"
                    )
                    context_data[StepOutputArtifact.fitted_scaler_bin.value] = (
                        scaler_data
                    )
                else:
                    log.debug(
                        "[_persist_artifacts] Routing scaler to regression registry key"
                    )
                    context_data[
                        StepOutputArtifact.fitted_scaler_regression_artifact.value
                    ] = scaler_data

        # ── Step 3: consolidate remaining artifacts (clean exclusion, same pattern as Phase 4) ──
        keys_to_exclude = {
            "trace",
            "val_df",
            "test_df",
            "fitted_scaler_regression_artifact",
            "fitted_scaler_bin",
        }
        remaining = {
            k: v for k, v in extra_artifacts.items() if k not in keys_to_exclude
        }
        context_data.update(remaining)

        # ── Step 4: final save and dispatch to the central Registry ──
        if not context_data:
            log.warning(
                "[_persist_artifacts] No artifacts to persist for step='%s'",
                self.step_key,
            )
            return

        log.debug(
            "[_persist_artifacts] final context_data keys mapped for registry: %s",
            list(context_data.keys()),
        )

        write_output_artifacts(
            ctx=self.ctx,
            step_key=self.step_key,
            step_cfg=self.step_cfg,
            base_dir=self.base_dir,
            **context_data,
        )

        log.info(
            "[_persist_artifacts] artifacts persisted for step='%s': %s",
            self.step_key,
            list(context_data.keys()),
        )
        log.debug("[_persist_artifacts] exit")