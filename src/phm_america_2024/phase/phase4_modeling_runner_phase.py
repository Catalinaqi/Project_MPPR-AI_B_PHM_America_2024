from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.pipeline.utils.context_facade_common import RunContext
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.common.io_service_common import load_parquet, load_pickle_joblib
from phm_america_2024.registry.generator_registry_registry import write_output_artifacts
from phm_america_2024.domain.enum_registry_domain import StepsPhase, StepOutputArtifact

from phm_america_2024.model.algorithm_selector_model import (
    single_probabilistic_architecture,
)
from phm_america_2024.model.regression_trainer_model import (
    cross_validation as cross_validation_regression,
)
from phm_america_2024.model.classification_trainer_model import (
    cross_validation as cross_validation_classification,
)
from phm_america_2024.model.regression_evaluator_model import (
    model_selection_criteria as model_selection_criteria_regression,
)
from phm_america_2024.model.classification_evaluator_model import (
    model_selection_criteria as model_selection_criteria_classification,
)

#

log = get_logger(__name__)


class Phase4ModelingRunner:
    """Execute all techniques for a single CRISP-DM modeling step."""

    _TECHNIQUE_DISPATCH: Dict[str, Any] = {
        # "single_calibrated_architecture": single_calibrated_architecture,
        "single_probabilistic_architecture": single_probabilistic_architecture,
        "cross_validation_regression": cross_validation_regression,
        "cross_validation_classification": cross_validation_classification,
        "model_selection_criteria_regression": model_selection_criteria_regression,
        "model_selection_criteria_classification": model_selection_criteria_classification,
    }

    def __init__(
        self,
        ctx: RunContext,
        step_key: str,
        step_cfg: Dict[str, Any],
    ) -> None:
        """Initialize the modeling runner phase.

        Args:
            ctx: Run context holding shared execution state.
            step_key: String identifier for the current step.
            step_cfg: Dictionary containing step configuration.
        Returns:
            None
        """
        self.ctx: RunContext = ctx
        self.step_key: str = step_key
        self.step_cfg: Dict[str, Any] = step_cfg

        self._TECHNIQUE_DISPATCH["model_selection_criteria"] = self._TECHNIQUE_DISPATCH[
            "model_selection_criteria_" + ctx.config.metadata.pipeline_key.task
        ]
        self._TECHNIQUE_DISPATCH["cross_validation"] = self._TECHNIQUE_DISPATCH[
            "cross_validation_" + ctx.config.metadata.pipeline_key.task
        ]

        # Step 1: CALL getattr() — retrieve phase4 directory from context
        self.base_dir: Path = getattr(ctx, "phase4_dir", None)

        if self.base_dir is None:
            log.error(
                "[Phase4ModelingRunner] YAML key missing or context error: phase4_dir is None"
            )
            raise RuntimeError("phase4_dir missing from RunContext")

        log.debug(
            "[Phase4ModelingRunner] init step='%s' base_dir='%s'",
            self.step_key,
            self.base_dir,
        )

    def run(self) -> RunContext:
        """Run the configured techniques for this phase.

        Args:
            None
        Returns:
            Updated RunContext object.
        """
        log.debug("[Phase4ModelingRunner] run entry")
        log.info("[Phase4ModelingRunner] run step='%s'", self.step_key)

        # Step 1: CALL _load_input_dataframe() — retrieve step input data
        df: Any = self._load_input_dataframe()
        extra_artifacts: Dict[str, Any] = {}

        # Step 2: CALL getattr() — retrieve cached algorithm configuration from memory
        algorithm_config_cache: Optional[Dict[str, Any]] = getattr(
            self.ctx, "algorithm_config", None
        )
        if algorithm_config_cache is not None:
            log.debug("[Phase4ModelingRunner] algorithm_config found in context cache")
        else:
            log.debug("[Phase4ModelingRunner] algorithm_config not in cache")

        # Step 3: CALL _load_algorithm_config_from_disk() — fallback to disk if running granularly
        if (
            algorithm_config_cache is None
            and self.step_key == StepsPhase.STEP_4_2.value
        ):
            log.warning(
                "[Phase4ModelingRunner] algorithm_config missing from cache – attempting disk load for independent execution"
            )
            algorithm_config_cache = self._load_algorithm_config_from_disk()
            if algorithm_config_cache is not None:
                log.info(
                    "[Phase4ModelingRunner] algorithm_config successfully loaded from disk"
                )
            else:
                log.error(
                    "[Phase4ModelingRunner] algorithm_config could not be loaded from disk – step will fail"
                )

        # Step 4: CALL get() — extract methods dictionary from config
        methods: Dict[str, Any] = self.step_cfg.get("methods", {})
        log.debug("[Phase4ModelingRunner] methods to execute: %s", list(methods.keys()))

        # Step 5: CALL items() — iterate over configured methods
        for method_name, method_cfg in methods.items():
            if not method_cfg.get("enabled", True):
                log.debug(
                    "[Phase4ModelingRunner] method '%s' disabled – skip", method_name
                )
                continue

            log.info("[Phase4ModelingRunner] executing method='%s'", method_name)

            # Step 6: CALL get() — extract techniques dictionary from method
            techniques: Dict[str, Any] = method_cfg.get("techniques", {})
            log.debug(
                "[Phase4ModelingRunner] techniques in method '%s': %s",
                method_name,
                list(techniques.keys()),
            )

            # Step 7: CALL items() — iterate over configured techniques
            for tech_name, tech_cfg in techniques.items():
                if not tech_cfg.get("enabled", True):
                    log.debug(
                        "[Phase4ModelingRunner] technique '%s' disabled – skip",
                        tech_name,
                    )
                    continue

                log.info("[Phase4ModelingRunner] executing technique='%s'", tech_name)

                # Step 8: CALL _execute_technique() — dispatch logic to target function
                df, art = self._execute_technique(
                    tech_name, tech_cfg, df, algorithm_config=algorithm_config_cache
                )

                if art is not None:
                    log.debug(
                        "[Phase4ModelingRunner] technique '%s' returned artifacts: %s",
                        tech_name,
                        list(art.keys()),
                    )

                    # Step 9: CALL update() — merge output artifacts
                    extra_artifacts.update(art)

                    if "algorithm_config" in art:
                        algorithm_config_cache = art["algorithm_config"]
                        # Step 10: CALL setattr() — persist algorithm configuration in context
                        setattr(self.ctx, "algorithm_config", algorithm_config_cache)
                        log.debug(
                            "[Phase4ModelingRunner] algorithm_config cached in context from technique '%s'",
                            tech_name,
                        )
                else:
                    log.debug(
                        "[Phase4ModelingRunner] technique '%s' returned no artifacts",
                        tech_name,
                    )

        log.debug(
            "[Phase4ModelingRunner] all techniques completed – extra_artifacts keys: %s",
            list(extra_artifacts.keys()),
        )

        # Step 11: CALL _persist_artifacts() — serialize phase results
        self._persist_artifacts(df, extra_artifacts)

        log.info("[Phase4ModelingRunner] completed step='%s'", self.step_key)
        log.debug("[Phase4ModelingRunner] run exit")
        return self.ctx

    def _load_algorithm_config_from_disk(self) -> Optional[Dict[str, Any]]:
        """Load Step 4.1 configuration artifact from disk to comply with independent execution rule."""
        log.debug("[_load_algorithm_config_from_disk] entry")
        try:
            # En lugar de depender de parsear el YAML (que falla al buscar la ruta),
            # buscamos el archivo JSON que sabemos que generó el Paso 4.1.
            # self.base_dir ya apunta correctamente a la carpeta reutilizada gracias a --resume_run.

            # El nombre exacto que vimos en tus logs (4.1.modeling.algo_setup_trace.json)
            artifact_filename = "4.1.modeling.algo_setup_trace.json"
            artifact_path: Path = self.base_dir / artifact_filename

            log.debug(
                "[_load_algorithm_config_from_disk] Looking for artifact at: %s",
                artifact_path,
            )

            if not artifact_path.exists():
                log.error(
                    "[_load_algorithm_config_from_disk] Artifact file not found: %s",
                    artifact_path,
                )
                log.error(
                    "Did you forget to pass --resume_run <run_id> or execute step 4.1 first?"
                )
                return None

            # Deserializar JSON artifact
            with open(artifact_path, "r", encoding="utf-8") as f:
                trace_data: Dict[str, Any] = json.load(f)

            # Extraemos la configuración del modelo de adentro del JSON
            model_configured: Optional[Dict[str, Any]] = trace_data.get(
                "model_configured"
            )

            log.debug(
                "[_load_algorithm_config_from_disk] model_configured keys: %s",
                list(model_configured.keys()) if model_configured else None,
            )
            log.info(
                "[_load_algorithm_config_from_disk] successfully loaded config from disk"
            )
            log.debug("[_load_algorithm_config_from_disk] exit")

            return model_configured

        except Exception as e:
            log.error(
                "[_load_algorithm_config_from_disk] failed to load configuration from disk: %s",
                e,
            )
            return None

    def _get_explicit_train_path(self) -> str:
        """Extract explicit absolute path directly from the YAML config.

        Args:
            None
        Returns:
            Absolute path string for the input dataset.
        """
        log.debug("[_get_explicit_train_path] entry")

        # Step 1: CALL get() — extract read strategy from step config
        read_strategy: Dict[str, Any] = self.step_cfg.get("read_strategy", {})
        log.debug(
            "[_get_explicit_train_path] read_strategy from step_cfg: %s",
            bool(read_strategy),
        )

        if not read_strategy and hasattr(self.ctx, "phase_cfg"):
            log.debug(
                "[_get_explicit_train_path] read_strategy not in step_cfg – falling back to phase_cfg"
            )
            # Step 2: CALL getattr() — fallback to phase config
            read_strategy = getattr(self.ctx, "phase_cfg", {}).get("read_strategy", {})
            log.debug(
                "[_get_explicit_train_path] read_strategy from phase_cfg: %s",
                bool(read_strategy),
            )

        if not read_strategy:
            log.error(
                "[_get_explicit_train_path] YAML key missing: read_strategy not found in step or phase config"
            )
            raise ValueError("read_strategy must be defined in YAML configuration.")

        # Step 3: CALL get() — extract input source dictionary
        input_source: Dict[str, Any] = read_strategy.get("input_source", {})

        # Step 4: CALL get() — extract train_data path string
        explicit_train: Optional[str] = input_source.get("train_data")
        log.debug(
            "[_get_explicit_train_path] resolved train_data path: %s", explicit_train
        )

        if not explicit_train:
            log.error(
                "[_get_explicit_train_path] YAML key missing: train_data not found under input_source"
            )
            raise ValueError("train_data must be defined under input_source in YAML.")

        log.debug("[_get_explicit_train_path] exit")
        return explicit_train

    # ──────────────────────────────────────────────────────────────────────────
    # Input loading
    # ──────────────────────────────────────────────────────────────────────────
    def _load_input_dataframe(self) -> Any:
        """Load the required DataFrame based on step configuration.

        Args:
            None
        Returns:
            DataFrame or Model required for the step.
        """
        log.debug("[_load_input_dataframe] entry step='%s'", self.step_key)
        step_val: str = self.step_key

        if step_val in (StepsPhase.STEP_4_1.value, StepsPhase.STEP_4_2.value):
            # Step 1: CALL _get_explicit_train_path() — resolve yaml train data path
            # 1. Obtener solo el nombre del archivo del YAML
            explicit_train: str = self._get_explicit_train_path()

            # Step 2: Construir la ruta completa usando phase3_dir
            # phase3_dir es la carpeta donde la Fase 3 guardó los resultados
            # self.ctx.phase3_dir debe existir en tu objeto RunContext
            path: Path = Path(self.ctx.phase3_dir) / explicit_train

            # Step 2: CALL exists() — verify file presence
            if not path.exists():
                log.error(
                    "[_load_input_dataframe] train data file not found: %s",
                    path,
                )
                raise FileNotFoundError(
                    f"Configured train_data path does not exist: {path}"
                )

            log.info("[_load_input_dataframe] using train_data: %s", path)

            # Step 4: CALL load_parquet() — deserialize dataframe
            df: Any = load_parquet(str(path))
            log.info(
                "[_load_input_dataframe] loaded shape=%s", getattr(df, "shape", "N/A")
            )
            return df

        elif step_val == StepsPhase.STEP_4_4.value:
            # Step 5: CALL extract dynamic path from configuration
            try:
                phase_cfg = self.ctx.config.phases["phase4_data_modeling"]
                step_4_2_cfg = phase_cfg["steps"]["step_4_2_model_training"]
                model_path_str: str = step_4_2_cfg["output_artifacts"]["trained_model"][
                    "path"
                ]
            except (KeyError, AttributeError) as e:
                log.error(
                    "[_load_input_dataframe] Error dynamically resolving model path: %s",
                    e,
                )
                raise ValueError(
                    "Could not extract trained_model path from configuration."
                )

            log.debug(
                "[_load_input_dataframe] model_data path from config: %s",
                model_path_str,
            )

            # Step 6: CALL resolve_path() — normalize model path object
            path_model: Path = resolve_path(self.ctx.phase4_dir / model_path_str)

            # Step 7: CALL load_pickle_joblib() — deserialize model object
            model: Any = load_pickle_joblib(str(path_model))
            log.info("[_load_input_dataframe] model loaded from: %s", path_model)
            df = model

        else:
            log.error("[_load_input_dataframe] unknown step_key: %s", step_val)
            raise ValueError(f"Unknown step: {step_val}")

        log.debug("[_load_input_dataframe] exit")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Technique execution
    # ──────────────────────────────────────────────────────────────────────────
    def _execute_technique(
        self,
        technique_name: str,
        tech_cfg: Dict[str, Any],
        df: Any,
        algorithm_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Dispatch to the specific technique function.

        Args:
            technique_name: Name of technique to execute.
            tech_cfg: Technique configuration dict.
            df: Input data structure.
            algorithm_config: Optional configuration injected from prior steps.
        Returns:
            Tuple containing updated data and artifact dictionary.
        """
        log.debug("[_execute_technique] entry technique='%s'", technique_name)

        # Step 1: CALL get() — resolve target technique function
        func = self._TECHNIQUE_DISPATCH.get(technique_name)

        if func is None:
            log.warning(
                "[_execute_technique] unknown technique '%s' – skipping", technique_name
            )
            return df, None

        output_dir: Path = self.base_dir
        log.debug(
            "[_execute_technique] dispatching to '%s' output_dir='%s'",
            technique_name,
            output_dir,
        )

        if technique_name == "model_selection_criteria":
            # Step 2: CALL func() — execute evaluation logic passing full tech_cfg
            model, extra = func(df, tech_cfg, self.ctx, output_dir)
            log.info("[_execute_technique] '%s' completed", technique_name)
            log.debug("[_execute_technique] exit")
            return model, extra

        elif technique_name == "cross_validation":
            log.debug(
                "[_execute_technique] algorithm_config present: %s",
                algorithm_config is not None,
            )
            # Step 3: CALL func() — execute training logic passing full tech_cfg
            df_new, extra = func(
                df, tech_cfg, self.ctx, output_dir, algorithm_config=algorithm_config
            )
            log.info("[_execute_technique] '%s' completed", technique_name)
            log.debug("[_execute_technique] exit")
            return df_new, extra

        else:
            # Step 4: CALL func() — execute standard algorithm configuration passing full tech_cfg
            df_new, extra = func(df, tech_cfg, self.ctx, output_dir)
            log.info("[_execute_technique] '%s' completed", technique_name)
            log.debug("[_execute_technique] exit")
            return df_new, extra

    def _persist_artifacts(
        self,
        df: Any,
        extra_artifacts: Dict[str, Any],
    ) -> None:
        """Write artifacts generated by the phase to disk.

        Args:
            df: Updated data or model object.
            extra_artifacts: Dictionary of additional artifacts to persist.
        Returns:
            None
        """
        log.debug(
            "[_persist_artifacts] entry step='%s' artifacts=%s",
            self.step_key,
            list(extra_artifacts.keys()),
        )
        context_data: Dict[str, Any] = {}

        if self.step_key == StepsPhase.STEP_4_2.value:
            # Step 1: CALL get() — extract trained model from artifacts
            trained_model = extra_artifacts.get("trained_model")
            context_data[StepOutputArtifact.trained_model.value] = trained_model
            log.debug(
                "[_persist_artifacts] trained_model present: %s",
                trained_model is not None,
            )

        elif self.step_key == StepsPhase.STEP_4_4.value:
            # Step 2: CALL get() — extract evaluation metadata from artifacts
            best_metadata = extra_artifacts.get("best_model_metadata")
            context_data[StepOutputArtifact.best_regression_model_metadata.value] = (
                best_metadata
            )
            log.debug(
                "[_persist_artifacts] best_model_metadata present: %s",
                best_metadata is not None,
            )

        # Step 3: CALL update() — consolidate remaining artifacts
        remaining = {k: v for k, v in extra_artifacts.items() if k != "trained_model"}
        context_data.update(remaining)
        log.debug(
            "[_persist_artifacts] final context_data keys: %s",
            list(context_data.keys()),
        )

        # Step 4: CALL write_output_artifacts() — serialize data to output directory
        write_output_artifacts(
            self.ctx,
            self.step_key,
            self.step_cfg,
            self.base_dir,
            **context_data,
        )

        log.info(
            "[_persist_artifacts] artifacts persisted for step='%s'", self.step_key
        )
        log.debug("[_persist_artifacts] exit")
