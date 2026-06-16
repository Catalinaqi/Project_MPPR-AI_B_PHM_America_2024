# src/phm_america_2024/phase/phase2_understanding_runner_phase.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.pipeline.utils.context_facade_common import RunContext
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.common.io_service_common import load_parquet, save_json
from phm_america_2024.registry.generator_registry_registry import write_output_artifacts
from phm_america_2024.domain.enum_registry_domain import StepsPhase
from phm_america_2024.reporting.artifact_persister_reporting import (
    save_figure,
)
from phm_america_2024.reporting.plots_generator_reporting import (
    plot_gmm_analysis,
    plot_flight_regime_binning,
)

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

from phm_america_2024.data.acquisition_extractor_data import load_and_merge


log = get_logger(__name__)


class Phase2DataUnderstandingRunner:
    """Execute all techniques for a single CRISP‑DM data‑understanding step.

    All artifact saving is delegated to the central artifact registry.
    """

    # Mapping technique_name (from YAML) → StepOutputArtifact key
    _TECHNIQUE_TO_ARTIFACT_KEY: dict[str, str] = {
        # 2.1
        "load_and_merge": load_and_merge,
        # 2.2
        "column_metadata": compute_column_metadata,
        "basic_stats": compute_descriptive_statistics,
        "null_count_per_column": compute_null_counts,
        "distribution_analysis": analyze_target_distributions,
        # 2.3
        "zero_or_negative_check": zero_or_negative_check,
        "collinearity_analysis": collinearity_analysis,
        # 2.4
        "column_catalog": categorize_columns,
        "ks_test_per_feature": ks_test_per_feature,
        "gmm_exploration": gmm_exploration,
        "feature_drift_summary": feature_drift_summary,
        "flight_regime_binning": flight_regime_binning,
    }

    # ──────────────────────────────────────────────────────────────────────────
    # The constructor initializes.
    #   the runner with context, step key, and configuration.
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        ctx: RunContext,
        step_key: str,
        step_cfg: dict[str, Any],
    ) -> None:
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
            log.error(
                "[Phase2DataUnderstandingRunner] ctx.phase2_dir is None – cannot resolve artifact paths"
            )
            raise RuntimeError("phase2_dir missing from RunContext")

        # Step 4: CALL log_debug() — confirm successful initialization
        log.debug(
            "[Phase2DataUnderstandingRunner] init step='%s' base_dir='%s'",
            self.step_key,
            self.base_dir,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point.
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

        # Step 2: CALL _load_input_dataframe() — retrieve step input data
        df_input: Any = self._load_input_dataframe()
        extra_artifacts: Dict[str, Any] = {}

        # Step 3: CALL get_methods() — retrieve methods from configuration
        methods: dict[str, Any] = self.step_cfg.get("methods", {})
        log.debug(
            "[Phase2DataUnderstandingRunner] run methods=%s", list(methods.keys())
        )

        # Step 4: CALL items() — iterate over configured methods
        for method_name, method_cfg in methods.items():
            if not method_cfg.get("enabled", True):
                log.debug(
                    "[Phase2DataUnderstandingRunner] method '%s' disabled – skip",
                    method_name,
                )
                continue

            log.info(
                "[Phase2DataUnderstandingRunner] executing method='%s'", method_name
            )

            # Step 5: CALL get() — extract techniques dictionary from method
            techniques: dict[str, Any] = method_cfg.get("techniques", {})
            log.debug(
                "[Phase2DataUnderstandingRunner] techniques in method '%s': %s",
                method_name,
                list(techniques.keys()),
            )

            # Step 6: CALL items() — iterate over configured techniques
            for technique_name, tech_cfg in techniques.items():
                if not tech_cfg.get("enabled", True):
                    log.debug(
                        "[Phase2DataUnderstandingRunner] technique '%s' disabled – skip",
                        technique_name,
                    )
                    continue

                log.info(
                    "[Phase2DataUnderstandingRunner] executing technique='%s'",
                    technique_name,
                )

                # Step 7: CALL _execute_technique() — dispatch logic to target function
                df_input, art = self._execute_technique(
                    technique_name, tech_cfg, df_input
                )

                # Step 8: Accumulate artifacts if the technique yielded something.
                if art:
                    log.debug(
                        "[Phase2DataUnderstandingRunner] technique '%s' returned artifacts: %s",
                        technique_name,
                        list(art.keys()),
                    )

                    # Step 9: CALL update() — merge output artifacts
                    extra_artifacts.update(art)
                else:
                    log.debug(
                        "[Phase2DataUnderstandingRunner] technique '%s' returned no artifacts",
                        technique_name,
                    )

        # Step 10: CALL _persist_artifacts() — serialize phase results
        self._persist_artifacts(df_input, extra_artifacts)

        # Step 11: CALL log_info() — register the successful completion of the step
        log.info("[Phase2DataUnderstandingRunner] completed step='%s'", self.step_key)
        return self.ctx

    # ──────────────────────────────────────────────────────────────────────────
    # Generic technique executor (steps 2.2, 2.3, 2.4).
    #   The process includes artifact existence checks, split loading,
    #   technique dispatch, and registry persistence,
    #   all orchestrated in a modular fashion.
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_technique(
        self,
        technique_name: str,
        tech_cfg: dict[str, Any],
        df: Any,
    ) -> tuple[Any, dict[str, Any]]:  # <-- CORRECCIÓN 1: Firma actualizada
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
        tuple[Any, dict[str, Any]]
            El DataFrame (posiblemente modificado) y un diccionario de artefactos.
        """
        # Step 1: CALL log_debug() — log entry of the technique
        log.debug("[_execute_technique] entry technique='%s'", technique_name)

        # Step 1.1: CALL get() — resolve target technique function
        func = self._TECHNIQUE_TO_ARTIFACT_KEY.get(technique_name)

        if func is None:
            log.warning(
                "[_execute_technique] unknown technique '%s' – skipping", technique_name
            )
            return df, {}  # <-- CORRECCIÓN 2: Retorna tupla vacía

        output_dir: Path = self.base_dir
        log.debug(
            "[_execute_technique] dispatching to '%s' output_dir='%s'",
            technique_name,
            output_dir,
        )

        # Extraer la ruta de salida definida a nivel de técnica
        output_key = tech_cfg.get("output")

        if not output_key:
            log.debug(
                "[_execute_technique] '%s' has no output key – executing in memory only",
                technique_name,
            )
        else:
            output_path: Path = resolve_path(self.base_dir / output_key)
            log.info("[_execute_technique] output_path: '%s' ", output_path)
            if self._should_skip(output_path):
                log.info(
                    "[_execute_technique] '%s' artifact exists – skipping computation",
                    technique_name,
                )
                return df, {}  # <-- CORRECCIÓN 3: Previene error de unpacking en run()

        try:
            # Ejecutar la función matemática/ingesta
            result = func(df, tech_cfg, self.ctx, self.base_dir)
            log.info("[_execute_technique] '%s' completed successfully", result)

            # CASO A: Es el Paso 2.1 (Retorna una tupla: df_train, diccionario_de_parquets)
            if isinstance(result, tuple) and len(result) == 2:
                df_out, artifacts_dict = result
                log.info(
                    "[_execute_technique] '%s' completed successfully. Train shape: %s, Total artifacts: %d",
                    technique_name,
                    df_out.shape,
                    len(artifacts_dict),
                )
                log.debug(
                    "[_execute_technique] Artifacts keys in payload: %s",
                    list(artifacts_dict.keys()),
                )
                return df_out, artifacts_dict

            # CASO B: Es una técnica de Profiling (Retorna un JSON/diccionario o dibuja un PNG)
            if output_key and result is not None:
                # --- GUARDAR JSON ---
                if str(output_key).endswith(".json"):
                    save_json(result, output_path)

                # --- NUEVO: GUARDAR PNG ---
                elif str(output_key).endswith(".png"):
                    fig = None

                    # 1. Gráfico para GMM Exploration
                    if technique_name == "gmm_exploration":
                        if "error" not in result:
                            fig = plot_gmm_analysis(result)
                        else:
                            log.warning(
                                "[_execute_technique] GMM returned error, skipping plot."
                            )

                    # 2. Gráfico para Flight Regime Binning
                    elif technique_name == "flight_regime_binning":
                        if "error" not in result:
                            col_name = result["column"]
                            n_bins = result["n_bins"]
                            # Le pasamos la columna del DataFrame original
                            fig = plot_flight_regime_binning(
                                data=df[col_name],
                                title=f"Flight Regimes Distribution ({col_name})",
                                bins=n_bins,
                                plot_type="hist",
                            )
                        else:
                            log.warning(
                                "[_execute_technique] Binning returned error, skipping plot."
                            )

                    # 3. Persistir la figura en disco y liberar memoria
                    if fig is not None:
                        dpi = getattr(
                            self.ctx.config.common_base_config.output_policy, "dpi", 150
                        )
                        save_figure(fig, out_path=output_path, dpi=dpi)

            # Retornar el DF intacto y un diccionario vacío
            return df, {}

        except Exception as e:
            log.exception(
                "[_execute_technique] technique '%s' failed: %s", technique_name, e
            )
            raise  # <-- ¡CRÍTICO! Fail-Fast para detener el orquestador

    def _should_skip(
        self,
        output_path: Path,
    ) -> bool:
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
        log.debug(
            "[_should_skip] artifact does not exist – will compute: %s", output_path
        )
        return False

    def _load_input_dataframe(self) -> Any:
        """Carga el DataFrame de entrenamiento una sola vez para los pasos de profiling."""
        log.debug("[_load_input_dataframe] entry step='%s'", self.step_key)

        # Si es el paso 2.1, no cargamos nada porque este paso es el que CREA los datos
        if self.step_key == StepsPhase.STEP_2_1.value:
            return None

        # 1. INTENTO PRIMARIO: Obtener el nombre inyectado dinámicamente por el ConfigBuilder
        # Usamos navegación segura con .get() para evitar KeyErrors
        phase2_cfg = self.ctx.config.phases.get("phase2_data_understanding", {})
        step_2_1_cfg = phase2_cfg.get("steps", {}).get("step_2_1_data_acquisition", {})
        output_artifacts = step_2_1_cfg.get("output_artifacts", {})

        # Buscamos el nombre del artefacto inyectado en la config
        train_parquet_name = output_artifacts.get("sample_x_y_train_parquet")

        # 2. FALLBACK DINÁMICO (A prueba de balas y sin hardcodear)
        if not train_parquet_name:
            log.warning(
                "Artifact name not in config. Searching dynamically in output directory..."
            )
            # Buscamos físicamente cualquier archivo que coincida con el patrón de train en la carpeta
            matches = list(self.base_dir.glob("2.1.data_acquisition.*_train.parquet"))

            if not matches:
                raise FileNotFoundError(
                    f"Could not find any training parquet dynamically in {self.base_dir}"
                )

            # Tomamos el archivo encontrado (ej. prod_300000_stratified_train.parquet)
            train_parquet_name = matches[0].name
            log.info(
                "Dynamically resolved artifact name via glob: %s", train_parquet_name
            )

        # 3. Carga final del archivo
        path_train = self.base_dir / train_parquet_name

        if not path_train.exists():
            log.error(
                "[_load_input_dataframe] Base dataset not found at: %s", path_train
            )
            log.error("Did you run step 2.1 first or use --resume_run?")
            raise FileNotFoundError(
                f"Missing base training data for Phase 2: {path_train}"
            )

        df = load_parquet(str(path_train))
        log.info("[_load_input_dataframe] Loaded base dataset shape: %s", df.shape)

        return df

    def _persist_artifacts(self, df: Any, extra_artifacts: dict[str, Any]) -> None:
        """Escribe los artefactos de Fase 2 mapeándolos para el Registry."""
        log.debug(
            "[_persist_artifacts] entry step='%s' artifacts=%s",
            self.step_key,
            list(extra_artifacts.keys()),
        )

        if not extra_artifacts:
            log.warning("[_persist_artifacts] No artifacts received for Step 2.1.")
            return

        log.debug(
            "[_persist_artifacts] Dispatching Step 2.1 output_artifacts to Registry: %s",
            list(extra_artifacts.keys()),
        )

        context_data: dict[str, Any] = {}

        if self.step_key != StepsPhase.STEP_2_1.value:
            log.debug(
                "[_persist_artifacts] Step '%s' uses technique-level outputs. Skipping registry.",
                self.step_key,
            )
            return

        log.debug(
            "[_persist_artifacts] Dispatching Step 2.1 output_artifacts to Registry."
        )

        if not extra_artifacts:
            log.warning("[_persist_artifacts] No artifacts received for Step 2.1.")
            return

        log.debug(
            "[_persist_artifacts] Dispatching Step 2.1 output_artifacts to Registry: %s",
            list(extra_artifacts.keys()),
        )

        # Agregamos cualquier sobrante por seguridad
        remaining = {
            k: v for k, v in extra_artifacts.items() if k not in context_data.values()
        }
        context_data.update(remaining)

        log.debug(
            "[_persist_artifacts] final context_data keys mapped for registry: %s",
            list(context_data.keys()),
        )

        # Despachar al Registry central
        write_output_artifacts(
            ctx=self.ctx,
            step_key=self.step_key,
            step_cfg=self.step_cfg,
            base_dir=self.base_dir,
            **extra_artifacts,
        )

        log.info(
            "[_persist_artifacts] artifacts persisted for step='%s'", self.step_key
        )
        log.debug("[_persist_artifacts] exit")
