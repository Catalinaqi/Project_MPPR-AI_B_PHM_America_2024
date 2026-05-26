# src/phm_america_2024/phase/phase2_understanding_runner_phase.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.context_facade_common import RunContext
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.data.load_loader_data import load_train_merged, load_test, load_validation, load_parquet
from phm_america_2024.data.profiling_profiler_data import (
    analyze_target_distributions,
    zero_or_negative_check,
    collinearity_analysis,
    ks_test_per_feature,
    gmm_exploration,
    flight_regime_binning,
    feature_drift_summary,
    compute_column_metadata,
    compute_descriptive_statistics,
    compute_null_counts,
    categorize_columns,
)

from phm_america_2024.registry.generator_registry_registry import write_output_artifacts
from phm_america_2024.configuration.enum_registry_config import StepsPhase, StepOutputArtifact



log = get_logger(__name__)


class Phase2DataUnderstandingRunner:
    """Execute all techniques for a single CRISP‑DM data‑understanding step.

    All artifact saving is delegated to the central artifact registry.
    """

    # Mapping technique_name (from YAML) → StepOutputArtifact key
    _TECHNIQUE_TO_ARTIFACT_KEY: dict[str, str] = {
        "column_metadata":           StepOutputArtifact.column_metadata_json.value,
        "basic_stats":               StepOutputArtifact.sensor_stats_json.value,
        "null_count_per_column":     StepOutputArtifact.null_count_json.value,
        "distribution_analysis":     StepOutputArtifact.target_distribution_json.value,
        "zero_or_negative_check":    StepOutputArtifact.zero_or_negative_check_json.value,
        "collinearity_analysis":     StepOutputArtifact.collinearity_json.value,
        "column_catalog":            StepOutputArtifact.column_catalog_json.value,
        "ks_test_per_feature":       StepOutputArtifact.ks_report_json.value,
        "gmm_exploration":           StepOutputArtifact.gmm_curve_png.value,
        "flight_regime_binning":     StepOutputArtifact.flight_regimes_png.value,
        "feature_drift_summary":     StepOutputArtifact.drift_summary_json.value,
    }

    def __init__(self, ctx: RunContext, step_key: str, step_cfg: dict[str, Any]) -> None:
        """
        Initialize the data understanding runner for a specific phase step.

        Parameters
        ----------
        ctx : RunContext
            The global execution context.
        step_key : str
            Identifier for the current pipeline step.
        step_cfg : dict[str, Any]
            Configuration dictionary for the current step.
        """
        # Step 1: CALL store_references() — initialize runner instance variables
        self.ctx: RunContext = ctx
        self.step_key: str = step_key
        self.step_cfg: dict[str, Any] = step_cfg

        # Step 2: CALL resolve_output_dir() — validate phase output directory availability
        self.base_dir: Path = getattr(ctx, "phase2_dir", None)

        # Step 3: CALL validate_context() — ensure required directory exists
        if self.base_dir is None:
            # ERROR: Log failure if phase2_dir is missing in the provided context
            log.error("[Phase2DataUnderstandingRunner] ctx.phase2_dir is None – cannot resolve artifact paths")
            raise RuntimeError("phase2_dir missing from RunContext")

        # Step 4: CALL log_debug() — confirm successful initialization
        log.debug("[Phase2DataUnderstandingRunner] init step='%s' base_dir='%s'",
                  self.step_key, self.base_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> RunContext:
        """
        Iterate over methods and techniques, executing each one.

        Orchestrates the execution flow by traversing the step configuration.
        It handles specialized data acquisition for initial steps and iterates
        through enabled methods and techniques, performing profiling operations
        sequentially.

        Parameters
        ----------
        None

        Returns
        -------
        RunContext
            The updated pipeline execution context after processing the step.
        """
        # Step 1: CALL log_info() — register the start of the step execution
        log.info("[Phase2DataUnderstandingRunner] run step='%s'", self.step_key)

        # Step 2: CALL _execute_data_acquisition() — handle special case for step 2.1
        if self.step_key == StepsPhase.STEP_2_1.value:
            self._execute_data_acquisition()
            return self.ctx

        # Step 3: CALL get_methods() — retrieve methods from configuration
        methods: dict[str, Any] = self.step_cfg.get("methods", {})
        log.debug("[Phase2DataUnderstandingRunner] run methods=%s", list(methods.keys()))

        # Step 4: CALL iterate_methods() — loop through enabled methods
        for method_name, method_cfg in methods.items():
            if not method_cfg.get("enabled", True):
                # DEBUG: Log skipping of disabled method
                log.debug("[Phase2DataUnderstandingRunner] method '%s' disabled – skip", method_name)
                continue

            # Step 5: CALL iterate_techniques() — process techniques for the current method
            techniques: dict[str, Any] = method_cfg.get("techniques", {})
            for technique_name, tech_cfg in techniques.items():
                if not tech_cfg.get("enabled", True):
                    # DEBUG: Log skipping of disabled technique
                    log.debug("[Phase2DataUnderstandingRunner] technique '%s' disabled – skip", technique_name)
                    continue

                # Step 6: CALL _execute_technique() — perform profiling for the enabled technique
                self._execute_technique(technique_name, tech_cfg)

        # Step 7: CALL log_info() — register the successful completion of the step
        log.info("[Phase2DataUnderstandingRunner] completed step='%s'", self.step_key)
        return self.ctx

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2.1 – Data acquisition
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_data_acquisition(self) -> None:
        """
        Executes step 2.1 data acquisition and delegates persistence to registry.

        Orchestrates the resolution of dataset configurations, performs the
        loading of training, test, and validation datasets into memory, and
        dispatches them to the artifact registry for persistent storage.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        Exception
            If configuration resolution, data loading, or registry
            persistence fails, logging the specific error for traceability.
        """
        # Step 1: CALL log_info() — signal the start of data acquisition
        log.info("[_execute_data_acquisition] start step='%s'", self.step_key)

        from omegaconf import OmegaConf
        from phm_america_2024.configuration.yml_repository_config import YmlRepository
        from phm_america_2024.configuration.read_strategy_repository_config import DataSourceConfig

        # Step 2: CALL get_step_config() — extract pipeline configuration
        raw_step_cfg = OmegaConf.to_container(self.step_cfg, resolve=True) if OmegaConf.is_config(self.step_cfg) else self.step_cfg

        # Step 3: CALL get_dataset_config() — resolve raw dataset settings
        dataset_key = getattr(self.ctx, "dataset_key", "phm2024")
        dataset_real_cfg = YmlRepository.get_dataset_by_key(dataset_key)
        raw_dataset_cfg = OmegaConf.to_container(dataset_real_cfg, resolve=True) if OmegaConf.is_config(dataset_real_cfg) else dataset_real_cfg

        # Step 4: CALL parse_yaml_paths() — extract paths and CSV parameters
        yaml_paths = raw_dataset_cfg.get("paths", {})
        yaml_csv_params = raw_dataset_cfg.get("csv_params", {"sep": ",", "encoding": "utf-8", "decimal": "."})

        # Step 5: CALL map_dataset_paths() — normalize YAML keys for DataSourceConfig
        train_node = yaml_paths.get("train", {})
        test_node = yaml_paths.get("test", {})
        val_node = yaml_paths.get("validation", {})

        normalized_paths = {
            "train": {
                "x_train": train_node.get("X_TRAIN_FEATURE") or train_node.get("x_train"),
                "y_train": train_node.get("Y_TRAIN_TARGET") or train_node.get("y_train"),
                "join_key": train_node.get("join_key", "id")
            },
            "test": {
                "x_test": test_node.get("X_TEST_FEATURE") or test_node.get("x_test")
            },
            "validation": {
                "x_validation": val_node.get("X_VALIDATION_FEATURE") or val_node.get("x_validation")
            }
        }

        # Step 6: CALL build_payload() — create nested DataSourceConfig dictionary
        nested_payload = {
            "paths": normalized_paths,
            "csv_params": yaml_csv_params,
            "read_strategy": raw_step_cfg.get("read_strategy", {})
        }
        log.debug("[_execute_data_acquisition] Formatted target payload ready for factory.")

        # Step 7: CALL instantiate_source_config() — initialize valid data source configuration
        source_config = DataSourceConfig.from_dict(nested_payload)

        # Step 8: CALL load_dataframes() — stream datasets into memory
        log.info("[_execute_data_acquisition] Loading dataframes from disk into memory...")
        df_train, _ = load_train_merged(source_config)
        df_test, _ = load_test(source_config)
        df_val, _ = load_validation(source_config)

        merged_metadata = {
            "join_key": normalized_paths["train"].get("join_key", "id"),
            "rows": len(df_train),
            "status": "merged_successfully",
            "infer_datetime": False
        }



        # Step 9: CALL prepare_registry_payload() — map artifacts for persistence
        context_payload = {
            StepOutputArtifact.sample_x_y_train_parquet.value: df_train,
            StepOutputArtifact.sample_x_test_parquet.value: df_test,
            StepOutputArtifact.sample_x_validation_parquet.value: df_val,
            StepOutputArtifact.load_and_merge_json.value: merged_metadata
        }

        # Step 10: CALL audit_registry_payload() — log dispatch metadata
        log.info("[_execute_data_acquisition] Dispatching to registry. Keys: %s", list(context_payload.keys()))
        log.debug("[_execute_data_acquisition] Target key existence: %s", StepOutputArtifact.load_and_merge_json.value in context_payload)
        for key, df in context_payload.items():
            log.debug("[_execute_data_acquisition] Payload '%s' shape: %s", key, getattr(df, "shape", "N/A"))
        log.info("DEBUG: Llaves en el payload: %s", list(context_payload.keys()))
        for k in context_payload.keys():
            log.info("DEBUG: Intentando procesar llave: %s", k)

        # Step 11: CALL write_output_artifacts() — persist artifacts to filesystem
        write_output_artifacts(
            ctx=self.ctx,
            step_key=self.step_key,
            step_cfg=self.step_cfg,
            base_dir=Path(self.base_dir),
            **context_payload
        )

        # Step 12: CALL log_registry_status() — record registered artifact paths
        registered = getattr(self.ctx, "artifacts", {})
        log.info("[_execute_data_acquisition] Step '%s' artifacts registry status: %d items registered",
                 self.step_key, len(registered))

        for key, path in registered.items():
            log.debug("[_execute_data_acquisition] Registered artifact path: '%s' -> %s", key, path)

        # Step 13: CALL log_completion() — finish step execution
        log.info("[_execute_data_acquisition] Completed step='%s' successfully.", self.step_key)

    # ──────────────────────────────────────────────────────────────────────────
    # Generic technique executor (steps 2.2, 2.3, 2.4)
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_technique(self, technique_name: str, tech_cfg: dict[str, Any]) -> None:
        """
        Check for existing output artifact, load splits, run profiling, and save via registry.

        Orchestrates the granular execution of a profiling technique. It verifies
        whether the artifact exists to skip re-computation, resolves necessary
        data splits, dispatches the profiling function, and persists the result
        in the central registry.

        Parameters
        ----------
        technique_name : str
            The name of the technique defined in the pipeline configuration.
        tech_cfg : dict[str, Any]
            Configuration parameters for the technique, including input splits
            and output paths.

        Returns
        -------
        None
        """
        # Step 1: CALL log_debug() — log entry of the technique
        log.debug("[_execute_technique] entry technique='%s'", technique_name)
        output_key = tech_cfg.get("output")

        if not output_key:
            log.warning("[_execute_technique] technique '%s' has no 'output' key – skip", technique_name)
            return

        output_path: Path = resolve_path(self.base_dir / output_key)

        # Step 2: CALL _should_skip() — verify artifact existence for skipping
        if self._should_skip(output_path):
            log.info("[_execute_technique] technique '%s' artifact exists – skip computation", technique_name)
            return

        # Step 3: CALL _load_splits() — fetch parquet data for profiling
        split_paths = tech_cfg.get("split", [])
        dfs = self._load_splits(split_paths)
        if dfs is None:
            log.error("[_execute_technique] technique '%s' – could not load all splits", technique_name)
            return

        # Step 4: CALL _dispatch_technique() — execute the profiling business logic
        result = self._dispatch_technique(technique_name, tech_cfg, dfs, output_path)
        if result is None:
            log.warning("[_execute_technique] technique '%s' returned None – nothing to save", technique_name)
            return

        # Step 5: CALL get_artifact_key() — resolve registry mapping
        artifact_key = self._TECHNIQUE_TO_ARTIFACT_KEY.get(technique_name)
        if artifact_key is None:
            log.warning("[_execute_technique] unknown artifact_key for technique='%s' – cannot save", technique_name)
            return

        # Step 6: CALL prepare_payload() — create temporary configuration for registry
        temp_step_cfg = {"output_artifacts": {artifact_key: output_key}}
        context_data = {artifact_key: result}

        # Step 7: CALL log_debug() — capture dispatch parameters for auditing
        log.debug("[_execute_technique] Dispatching to write_output_artifacts:")
        log.debug("[_execute_technique] -> step_key: '%s'", self.step_key)
        log.debug("[_execute_technique] -> context_data keys: %s", list(context_data.keys()))

        # Step 8: CALL write_output_artifacts() — persist artifacts to filesystem
        write_output_artifacts(
            ctx=self.ctx,
            step_key=self.step_key,
            step_cfg=temp_step_cfg,
            base_dir=self.base_dir,
            **context_data,
        )

        # Step 9: CALL log_completion() — record method exit and success
        log.info("[_execute_technique] technique '%s' completed – output='%s'", technique_name, output_path)
        log.debug("[_execute_technique] exit technique='%s'", technique_name)

    def _should_skip(self, output_path: Path) -> bool:
        """
        Return True if artifact exists and overwrite is False.

        Determines if a computational step should be bypassed based on the
        existence of the output file and the global configuration regarding
        artifact overwriting.

        Parameters
        ----------
        output_path : Path
            The file path where the artifact is expected to reside.

        Returns
        -------
        bool
            True if the process should be skipped, False otherwise.
        """
        # Step 1: CALL check_overwrite_policy() — determine if forcing re-computation
        if self.ctx.config.common_base_config.runtime.overwrite_artifacts:
            return False

        # Step 2: CALL check_file_existence() — verify if artifact already exists
        if output_path.exists():
            log.debug("[_should_skip] artifact exists – skip: %s", output_path)
            return True

        # Step 3: CALL log_status() — indicate that computation is required
        log.debug("[_should_skip] artifact does not exist – will compute: %s", output_path)
        return False

    def _load_splits(self, split_paths: list[str]) -> list[Any] | None:
        """
        Load each split path into a pandas DataFrame.

        Iterates through the provided list of file paths, resolves their absolute
        locations, and attempts to load them as parquet files. If any file is
        missing or loading fails, it aborts the process to ensure data integrity.

        Parameters
        ----------
        split_paths : list[str]
            A list of strings representing the relative paths to the data splits.

        Returns
        -------
        list[Any] | None
            A list of loaded DataFrames if all paths are valid and loadable,
            otherwise None.
        """
        import pandas as pd
        loaded: list[pd.DataFrame] = []

        # Step 1: CALL log_debug() — signal start of split loading
        log.debug("[_load_splits] loading %d splits", len(split_paths))

        # Step 2: CALL iterate_paths() — process each file path
        for path_str in split_paths:
            abs_path = resolve_path(path_str)
            log.debug("[_load_splits] split='%s'", abs_path)

            # Step 3: CALL check_existence() — validate file presence
            if not abs_path.exists():
                log.error("[_load_splits] split file not found: %s", abs_path)
                return None

            # Step 4: CALL load_parquet() — fetch individual split into memory
            try:
                df = load_parquet(str(abs_path))
                loaded.append(df)
                log.debug("[_load_splits] loaded rows=%d cols=%d", len(df), df.shape[1])
            except Exception:
                # Step 5: CALL log_exception() — handle numerical or I/O failure
                log.exception("[_load_splits] failed to load: %s", abs_path)
                return None

        # Step 6: CALL log_success() — signal completion of all loads
        log.debug("[_load_splits] all splits loaded successfully")
        return loaded

    def _dispatch_technique(
            self,
            technique_name: str,
            tech_cfg: dict[str, Any],
            dfs: list[Any],
            output_path: Path,
    ) -> Any:
        """
        Run the appropriate profiling function based on the technique name.

        Dispatches the execution to the corresponding profiling or analysis function
        defined in the ``profiler_data`` module. It applies parameter filtering to
        ensure compatibility between the pipeline configuration and the function
        signatures.

        Parameters
        ----------
        technique_name : str
            The identifier of the profiling technique to execute.
        tech_cfg : dict[str, Any]
            Configuration parameters associated with the specific technique.
        dfs : list[Any]
            List of loaded pandas DataFrames required for analysis.
        output_path : Path
            The destination path where the output will be registered.

        Returns
        -------
        Any
            The result of the profiling function, or None if the technique is
            unknown or execution fails.
        """
        params = tech_cfg.get("params", {})

        # Step 1: CALL log_debug() — audit parameter keys for technique dispatch
        log.debug("[_dispatch_technique] calling technique='%s' params_keys=%s",
                  technique_name, list(params.keys()))

        try:
            # Step 2: CALL branch_logic() — route to specific profiling implementation
            if technique_name == "column_metadata":
                return compute_column_metadata(dfs[0], **params)

            elif technique_name == "basic_stats":
                # Step 3: CALL filter_params() — reconcile arguments
                clean_params = self._get_filtered_params(compute_descriptive_statistics, params)
                return compute_descriptive_statistics(dfs[0], **clean_params)

            elif technique_name == "null_count_per_column":
                return compute_null_counts(dfs[0], **params)

            elif technique_name == "distribution_analysis":
                return analyze_target_distributions(dfs[0], **params)

            elif technique_name == "zero_or_negative_check":
                return zero_or_negative_check(dfs[0], **params)

            elif technique_name == "collinearity_analysis":
                return collinearity_analysis(dfs[0], **params)

            elif technique_name == "column_catalog":
                clean_params = self._get_filtered_params(categorize_columns, params)
                return categorize_columns(dfs[0], **clean_params)

            elif technique_name == "ks_test_per_feature":
                if len(dfs) < 2:
                    log.error("[_dispatch_technique] ks_test requires >=2 splits, got %d", len(dfs))
                    return None
                clean_params = self._get_filtered_params(ks_test_per_feature, params)
                return ks_test_per_feature(dfs[0], dfs[1], **clean_params)

            elif technique_name == "gmm_exploration":
                clean_params = self._get_filtered_params(gmm_exploration, params)
                return gmm_exploration(dfs[0], **clean_params)

            elif technique_name == "flight_regime_binning":
                # Step 4: CALL process_binning() — calculate regime and metadata
                clean_params = self._get_filtered_params(flight_regime_binning, params)
                report = flight_regime_binning(dfs[0], **clean_params)

                s = dfs[0][clean_params["column"]].dropna()
                bin_size = params.get("bin_size", 10)
                bins = int((s.max() - s.min()) / bin_size) if len(s) > 0 else 30

                return {
                    "data": report,
                    "plot_meta": {
                        "column": clean_params["column"],
                        "plot_type": params.get("plot_type", "hist"),
                        "bins": bins
                    },
                    "df": dfs[0]
                }

            elif technique_name == "feature_drift_summary":
                # Step 5: CALL get_ks_context() — retrieve prerequisite artifacts
                ks_results = getattr(self.ctx, "artifacts", {}).get("ks_report_json")
                if not ks_results:
                    log.error("[_dispatch_technique] Cannot run summary: KS report not in context.")
                    return None
                clean_params = self._get_filtered_params(feature_drift_summary, params)
                return feature_drift_summary(ks_results, **clean_params)

            else:
                # Step 6: CALL log_warning() — handle unsupported technique
                log.warning("[_dispatch_technique] unknown technique='%s'", technique_name)
                return None

        except Exception:
            # Step 7: CALL log_exception() — capture numerical or runtime failures
            log.exception("[_dispatch_technique] technique '%s' failed", technique_name)
            return None

    def _get_filtered_params(self, func: callable, params: dict[str, Any],) -> dict[
        str, Any]:
        """
        Filters params to match only the arguments accepted by the target function.

        Performs an inspection of the target function's signature to extract its
        valid parameter names, subsequently filtering the provided configuration
        dictionary to exclude any unsupported keys.

        Parameters
        ----------
        func : callable
            The function whose signature will be inspected.
        params : dict[str, Any]
            The dictionary of raw parameters to be filtered.

        Returns
        -------
        dict[str, Any]
            A subset of the original dictionary containing only valid parameters.
        """
        from inspect import signature

        # Step 1: CALL inspect_signature() — identify parameters supported by the function
        valid_params = signature(func).parameters.keys()

        # Step 2: CALL filter_dict() — retain only valid keys
        filtered = {k: v for k, v in params.items() if k in valid_params}

        # Step 3: CALL log_audit() — detect and log ignored configuration keys
        ignored = set(params.keys()) - set(valid_params)
        if ignored:
            log.debug("[_get_filtered_params] Technique '%s' - ignored params: %s",
                      func.__name__, ignored)

        # Step 4: CALL return_filtered() — return the validated parameter set
        return filtered