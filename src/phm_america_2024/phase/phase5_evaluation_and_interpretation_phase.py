# src/phm_america_2024/phase/phase5_evaluation_and_interpretation_phase.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.pipeline.utils.context_facade_common import RunContext
from phm_america_2024.registry.generator_registry_registry import write_output_artifacts
from phm_america_2024.domain.enum_registry_domain import StepsPhase, StepOutputArtifact

# ── Technique implementations ──
# step_5_1_interpretation
from phm_america_2024.interpretation.cluster_interpreter_interpretation import (
    feature_importance,
    permutation_importance,
)

# step_5_2_probabilistic_evaluation
from phm_america_2024.interpretation.business_alignment_evaluator_interpretation import (
    calibration_audit,
    performance_degradation_benchmarking,
)

# step_5_3_process_audit
from phm_america_2024.interpretation.pipeline_auditor_interpretation import (
    leakage_detection,
)

# step_5_4_decision_making
from phm_america_2024.interpretation.deployment_reporter_interpretation import (
    final_sign_off,
)


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

    def __init__(
        self,
        ctx: RunContext,
        step_key: str,
        step_cfg: Dict[str, Any],
    ) -> None:
        """Initialize the evaluation and interpretation runner."""
        self.ctx = ctx
        self.step_key = step_key
        self.step_cfg = step_cfg

        self.base_dir: Path = getattr(ctx, "phase5_dir", None)
        if self.base_dir is None:
            log.error("[Phase5Runner] CRÍTICO: phase5_dir is None en el contexto.")
            raise RuntimeError("phase5_dir missing from RunContext")

        self.base_dir.mkdir(parents=True, exist_ok=True)
        log.debug(
            "[Phase5Runner] Inicializado step='%s' en base_dir='%s'",
            self.step_key,
            self.base_dir,
        )

    def run(self) -> RunContext:
        """Run the configured techniques for this phase step."""
        log.debug("[Phase5Runner] run() - ENTRY")
        log.info("[Phase5Runner] Iniciando ejecución del step='%s'", self.step_key)

        # 1. Cargar dependencias (Model, Scaler, Data)
        df_test: Any = self._load_artifacts()

        if df_test is None:
            log.error(
                "[Phase5Runner] run() cancelado: Falló la carga de artefactos. Verifique los directorios de fase y la configuración."
            )
            return self.ctx

        extra_artifacts: Dict[str, Any] = {}
        methods: Dict[str, Any] = self.step_cfg.get("methods", {})
        log.debug(
            "[Phase5Runner] Métodos configurados en YAML: %s", list(methods.keys())
        )

        for method_name, method_cfg in methods.items():
            if not method_cfg.get("enabled", True):
                log.info(
                    "[Phase5Runner] Método '%s' está deshabilitado en YAML. Saltando...",
                    method_name,
                )
                continue

            log.info("[Phase5Runner] --- Procesando Método: '%s' ---", method_name)
            techniques: Dict[str, Any] = method_cfg.get("techniques", {})
            log.debug(
                "[Phase5Runner] Técnicas encontradas para '%s': %s",
                method_name,
                list(techniques.keys()),
            )

            for tech_name, tech_cfg in techniques.items():
                if not tech_cfg.get("enabled", True):
                    log.info(
                        "[Phase5Runner] Técnica '%s' está deshabilitada en YAML. Saltando...",
                        tech_name,
                    )
                    continue

                log.info("[Phase5Runner] Ejecutando técnica: '%s'", tech_name)

                # 2. Ejecutar técnica
                try:
                    df_test, art = self._execute_technique(tech_name, tech_cfg, df_test)

                    if art is not None:
                        log.debug(
                            "[Phase5Runner] Técnica '%s' retornó artefactos: %s",
                            tech_name,
                            list(art.keys()),
                        )
                        extra_artifacts.update(art)
                    else:
                        log.warning(
                            "[Phase5Runner] Técnica '%s' no retornó artefactos extra (art is None).",
                            tech_name,
                        )
                except Exception:
                    log.error(
                        "[Phase5Runner] CRÍTICO: La técnica '%s' falló durante su ejecución.",
                        tech_name,
                    )
                    raise

        # Consolidación final del summary para 5.2
        if self.step_key == StepsPhase.STEP_5_2.value:
            log.debug(
                "[Phase5Runner] Step 5.2 detectado. Intentando consolidar evaluation_summary."
            )
            intervals = extra_artifacts.get("calibration_intervals")
            degradation = extra_artifacts.get("degradation_metrics")
            if intervals and degradation:
                # CAMBIO: Usar la clave exacta registrada en el StepOutputArtifact 'evaluation_summary_json'
                extra_artifacts["evaluation_summary_json"] = {
                    "calibration_intervals": intervals,
                    "degradation": degradation,
                }
                log.info(
                    "[Phase5Runner] evaluation_summary_json consolidado con éxito."
                )
            else:
                log.warning(
                    "[Phase5Runner] No se pudo consolidar evaluation_summary. Faltan intervals o degradation."
                )

        # 3. Persistir
        log.debug(
            "[Phase5Runner] Iniciando persistencia de %d artefactos extra.",
            len(extra_artifacts),
        )
        self._persist_artifacts(extra_artifacts)

        log.info("[Phase5Runner] Finalizado step='%s'", self.step_key)
        log.debug("[Phase5Runner] run() - EXIT")
        return self.ctx

    def _execute_technique_old(
        self,
        technique_name: str,
        tech_cfg: Dict[str, Any],
        df_test: Any,
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Dispatch to the specific technique function and auto-save JSON traces."""
        log.debug("[_execute_technique] ENTRY - técnica='%s'", technique_name)

        func = self._TECHNIQUE_DISPATCH.get(technique_name)
        if func is None:
            log.error(
                "[_execute_technique] Técnica desconocida '%s' no encontrada en _TECHNIQUE_DISPATCH.",
                technique_name,
            )
            return df_test, None

        if df_test is None:
            log.error(
                "[_execute_technique] df_test es None ANTES de llamar a la función de la técnica '%s'.",
                technique_name,
            )

        try:
            # Ejecutamos la función inyectando el df_test
            log.debug(
                "[_execute_technique] Llamando a la función '%s'...", technique_name
            )
            _, extra = func(df_test, tech_cfg, self.ctx, self.base_dir)

            # Auto-guardado JSON
            output_key = tech_cfg.get("output")
            if output_key and str(output_key).endswith(".json") and extra:
                log.debug(
                    "[_execute_technique] output_key='.json' detectado. Preparando autoguardado para '%s'",
                    output_key,
                )
                from phm_america_2024.common.io_service_common import save_json

                # Excluir plots
                json_data = {k: v for k, v in extra.items() if not k.endswith("_plot")}
                save_path = self.base_dir / str(output_key)

                log.debug(
                    "[_execute_technique] Enviando %d llaves a save_json",
                    len(json_data),
                )
                save_json(json_data, save_path)
                log.info(
                    "[_execute_technique] Trazabilidad JSON auto-guardada en: %s",
                    save_path.name,
                )
            elif output_key:
                log.debug(
                    "[_execute_technique] output_key presente '%s' pero no califica para autoguardado JSON.",
                    output_key,
                )

            log.info(
                "[_execute_technique] Técnica '%s' completada con éxito.",
                technique_name,
            )
            log.debug("[_execute_technique] EXIT")
            return df_test, extra
        except Exception as e:
            log.error(
                "[_execute_technique] Fallo crítico ejecutando la técnica '%s': %s",
                technique_name,
                str(e),
            )
            raise

    def _load_artifacts(self) -> Any:
        """Load model, scaler, test data into context and return the test DataFrame."""
        log.debug(
            "[_load_artifacts] ENTRY - Iniciando carga de dependencias para step='%s'",
            self.step_key,
        )

        # 1. Búsqueda de la estrategia de lectura (Local vs Global)
        read_strategy = self.step_cfg.get("read_strategy", {})

        if read_strategy:
            log.debug(
                "[_load_artifacts] 'read_strategy' encontrado a nivel local (dentro del step)."
            )
        else:
            log.warning(
                "[_load_artifacts] 'read_strategy' no encontrado a nivel local. Intentando fallback a configuración global..."
            )
            try:
                if hasattr(self.ctx, "config") and hasattr(self.ctx.config, "phases"):
                    phase5_cfg = self.ctx.config.phases.get(
                        "phase5_evaluation_and_interpretation", {}
                    )
                    read_strategy = phase5_cfg.get("read_strategy", {})

                    if read_strategy:
                        log.info(
                            "[_load_artifacts] Fallback exitoso: 'read_strategy' global recuperado correctamente."
                        )
                    else:
                        log.error(
                            "[_load_artifacts] Fallback fallido: 'read_strategy' global también está vacío o no definido."
                        )
                else:
                    log.warning(
                        "[_load_artifacts] El contexto (ctx) no tiene la estructura de configuración esperada."
                    )
            except Exception as e:
                log.error(
                    "[_load_artifacts] Error inesperado extrayendo read_strategy global: %s",
                    str(e),
                    exc_info=True,
                )

        # 2. Validación y extracción de los archivos de entrada
        input_source = read_strategy.get("input_source", {})

        if not input_source:
            log.error(
                "[_load_artifacts] CRÍTICO: No hay 'input_source' definido en el YAML (ni local ni global). "
                "El orquestador no sabe qué archivos cargar. Abortando proceso."
            )
            return None
        else:
            log.info(
                "[_load_artifacts] 'input_source' cargado con éxito. Archivos requeridos por el YAML: %s",
                list(input_source.keys()),
            )

        # 3. Función anidada corregida para resolver dinámicamente los datasets
        def _resolve_dynamic_dataset(
            base_dir: Path, yaml_filename: str, fallback_pattern: str
        ) -> Path:
            if yaml_filename:
                exact_path = base_dir / yaml_filename
                if exact_path.exists():
                    log.debug(
                        "[_resolve_dynamic_dataset] Archivo exacto encontrado: %s",
                        exact_path.name,
                    )
                    return exact_path

                log.warning(
                    "[_resolve_dynamic_dataset] Archivo exacto '%s' no encontrado. Buscando patrón '%s' en %s",
                    yaml_filename,
                    fallback_pattern,
                    base_dir.name,
                )
            else:
                log.debug(
                    "[_resolve_dynamic_dataset] yaml_filename vacío. Pasando directamente a buscar por patrón: '%s'",
                    fallback_pattern,
                )

            matches = list(base_dir.glob(fallback_pattern))
            if matches:
                log.info(
                    "[_resolve_dynamic_dataset] Fallback exitoso. Usando: '%s'",
                    matches[0].name,
                )
                return matches[0]

            log.error(
                "[_resolve_dynamic_dataset] Patrón '%s' tampoco encontró resultados.",
                fallback_pattern,
            )
            return base_dir / (yaml_filename or "missing_file")

        # 4. Construcción de las rutas
        model_path = Path(self.ctx.phase4_dir) / str(input_source.get("model", ""))
        scaler_path = Path(self.ctx.phase3_dir) / str(input_source.get("scaler", ""))

        log.debug("[_load_artifacts] Resolviendo rutas dinámicas de datasets...")

        # --- CORRECCIÓN CRÍTICA ---
        # Leer primero 'internal_test_data' (esperado en Fase 3).
        # Si no existe, intentar 'challenge_test_data' (esperado en Fase 2).
        internal_test_file = input_source.get("internal_test_data", "")
        challenge_test_file = input_source.get("challenge_test_data", "")

        if internal_test_file:
            log.debug(
                "[_load_artifacts] Detectado 'internal_test_data' en YAML. Buscando en phase3_dir."
            )
            test_dir = Path(self.ctx.phase3_dir)
            test_target_file = internal_test_file
        else:
            log.debug(
                "[_load_artifacts] 'internal_test_data' no encontrado. Intentando 'challenge_test_data' en phase2_dir."
            )
            test_dir = Path(self.ctx.phase2_dir)
            test_target_file = challenge_test_file

        test_path = _resolve_dynamic_dataset(
            test_dir,
            str(test_target_file),
            "*_test.parquet",
        )

        # Mantener validación y entrenamiento apuntando a Fase 2 (por si alguna técnica los requiere)
        val_path = _resolve_dynamic_dataset(
            Path(self.ctx.phase2_dir),
            str(input_source.get("challenge_val_data", "")),
            "*_validation.parquet",
        )
        train_path = _resolve_dynamic_dataset(
            Path(self.ctx.phase2_dir),
            str(input_source.get("challenge_train_data", "")),
            "*_train.parquet",
        )

        # 5. Carga del Modelo
        if model_path.exists() and getattr(self.ctx, "model", None) is None:
            log.debug("[_load_artifacts] Cargando modelo desde %s...", model_path.name)
            import joblib

            self.ctx.model = joblib.load(model_path)
            log.info("[_load_artifacts] Modelo cargado exitosamente.")
        elif not model_path.exists():
            log.error("[_load_artifacts] Modelo no encontrado en: %s", model_path)
            raise FileNotFoundError(f"Model missing: {model_path}")

        # 6. Carga Test Data y Scaler
        if test_path.exists() and getattr(self.ctx, "df_test", None) is None:
            log.debug(
                "[_load_artifacts] Leyendo parquet de test desde %s...", test_path.name
            )
            import pandas as pd
            import joblib

            df_test_raw = pd.read_parquet(test_path)
            target_col = self.step_cfg.get("target_col", "trq_margin")
            log.debug("[_load_artifacts] target_col definido como '%s'", target_col)

            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                self.ctx.scaler = scaler
                log.info(
                    "[_load_artifacts] Scaler cargado exitosamente: %s",
                    scaler_path.name,
                )

                features_expected = getattr(scaler, "feature_names_in_", [])
                valid_features = [
                    c for c in features_expected if c in df_test_raw.columns
                ]

                if len(valid_features) < len(features_expected):
                    missing = set(features_expected) - set(df_test_raw.columns)
                    log.warning(
                        "[_load_artifacts] Columnas faltantes en test_data que el escalador espera: %s",
                        missing,
                    )

                df_test_scaled = df_test_raw.copy()
                if valid_features:
                    log.info(
                        "[_load_artifacts] Escalando columnas presentes: %s",
                        valid_features,
                    )
                    df_test_scaled[valid_features] = scaler.transform(
                        df_test_raw[valid_features]
                    )
                else:
                    log.warning(
                        "[_load_artifacts] No se encontraron columnas coincidentes para escalar."
                    )
            else:
                log.warning(
                    "[_load_artifacts] Scaler NO encontrado. Datos se procesarán sin escalar."
                )
                df_test_scaled = df_test_raw.copy()

            self.ctx.df_test = df_test_scaled

            if hasattr(self.ctx.model, "pred_dist"):
                log.debug(
                    "[_load_artifacts] El modelo soporta 'pred_dist'. Calculando predicciones base..."
                )
                try:
                    X_test = df_test_scaled.drop(columns=[target_col], errors="ignore")
                    pred_dist = self.ctx.model.pred_dist(X_test)
                    self.ctx.y_true = df_test_scaled[target_col].values
                    self.ctx.y_pred_mean = pred_dist.loc
                    self.ctx.y_pred_std = pred_dist.scale
                    log.info(
                        "[_load_artifacts] Predicciones base (media y desviación) guardadas en ctx."
                    )
                except Exception as e:
                    log.error("[_load_artifacts] Error fatal durante pred_dist: %s", e)
                    raise
            else:
                log.warning(
                    "[_load_artifacts] El modelo no tiene 'pred_dist'. Las métricas probabilísticas podrían fallar."
                )
        elif not test_path.exists():
            log.error(
                "[_load_artifacts] Datos de Test no encontrados en: %s", test_path
            )
            raise FileNotFoundError(f"Test data missing: {test_path}")

        # 7. Carga de datos Opcionales (Validación y Train)
        if val_path.exists() and getattr(self.ctx, "val_df", None) is None:
            log.debug("[_load_artifacts] Cargando datos de validación...")
            import pandas as pd

            self.ctx.val_df = pd.read_parquet(val_path)
            log.info("[_load_artifacts] Datos de validación cargados.")

        if train_path.exists() and getattr(self.ctx, "train_df", None) is None:
            log.debug("[_load_artifacts] Cargando datos de entrenamiento...")
            import pandas as pd

            self.ctx.train_df = pd.read_parquet(train_path)
            log.info("[_load_artifacts] Datos de entrenamiento cargados.")

        log.debug("[_load_artifacts] EXIT")
        return getattr(self.ctx, "df_test", None)

    # def _persist_artifacts(
    #     self,
    #     extra_artifacts: Dict[str, Any],
    # ) -> None:
    #     """Persiste los artefactos dinámicamente según la configuración YAML."""
    #     log.debug("[_persist_artifacts] ENTRY - step='%s'", self.step_key)
    #
    #     context_data: Dict[str, Any] = {}
    #
    #     output_artifacts_cfg = self.step_cfg.get("output_artifacts", {})
    #     log.debug(
    #         "[_persist_artifacts] Artefactos esperados por YAML: %s",
    #         list(output_artifacts_cfg.keys()),
    #     )
    #
    #     # Mapeo especial
    #     if "evaluation_summary" in extra_artifacts:
    #         context_data[StepOutputArtifact.evaluation_summary_json.value] = (
    #             extra_artifacts["evaluation_summary"]
    #         )
    #         log.debug("[_persist_artifacts] Mapeado evaluation_summary.")
    #
    #     if "deployment_sign_off" in extra_artifacts:
    #         context_data[StepOutputArtifact.deployment_sign_off.value] = (
    #             extra_artifacts["deployment_sign_off"]
    #         )
    #         log.debug("[_persist_artifacts] Mapeado deployment_sign_off.")
    #
    #     # Match exacto con YAML
    #     for artifact_key in output_artifacts_cfg.keys():
    #         if artifact_key in extra_artifacts and artifact_key not in context_data:
    #             context_data[artifact_key] = extra_artifacts[artifact_key]
    #             log.debug(
    #                 "[_persist_artifacts] Artefacto YAML '%s' acoplado para registro.",
    #                 artifact_key,
    #             )
    #
    #     # Residuales
    #     for k, v in extra_artifacts.items():
    #         if k not in context_data and k not in [
    #             "evaluation_summary",
    #             "deployment_sign_off",
    #         ]:
    #             context_data[k] = v
    #
    #     if context_data:
    #         log.info(
    #             "[_persist_artifacts] Enviando %d artefactos al Registry para escritura...",
    #             len(context_data),
    #         )
    #         write_output_artifacts(
    #             self.ctx, self.step_key, self.step_cfg, self.base_dir, **context_data
    #         )
    #         log.info(
    #             "[_persist_artifacts] Artefactos persistidos exitosamente: %s",
    #             list(context_data.keys()),
    #         )
    #     else:
    #         log.warning(
    #             "[_persist_artifacts] No hay artefactos en context_data para persistir en '%s'.",
    #             self.step_key,
    #         )
    #
    #     log.debug("[_persist_artifacts] EXIT")

    def _persist_artifacts(
        self,
        extra_artifacts: Dict[str, Any],
    ) -> None:
        """Persiste los artefactos dinámicamente según la configuración YAML."""
        log.debug("[_persist_artifacts] ENTRY - step='%s'", self.step_key)

        context_data: Dict[str, Any] = {}

        output_artifacts_cfg = self.step_cfg.get("output_artifacts", {})
        log.debug(
            "[_persist_artifacts] Artefactos esperados por YAML: %s",
            list(output_artifacts_cfg.keys()),
        )

        # 1. MATCH DINÁMICO UNIVERSAL (El corazón de la arquitectura)
        # Revisa todo lo que pide el YAML y si la técnica lo generó, lo acopla.
        # Esto funciona para 5.1 (plots de importancia), 5.2 (calibración), 5.4 (sign_off), etc.
        for artifact_key in output_artifacts_cfg.keys():
            if artifact_key in extra_artifacts:
                context_data[artifact_key] = extra_artifacts[artifact_key]
                log.debug(
                    "[_persist_artifacts] Artefacto YAML '%s' acoplado para registro.",
                    artifact_key,
                )

        # 2. MAPEO LEGACY DE SEGURIDAD
        # Por si en el método run() todavía armas el diccionario usando la clave vieja "evaluation_summary"
        # en lugar de "evaluation_summary_json" que es la que usa el YAML.
        if (
            "evaluation_summary" in extra_artifacts
            and StepOutputArtifact.evaluation_summary_json.value not in context_data
        ):
            context_data[StepOutputArtifact.evaluation_summary_json.value] = (
                extra_artifacts["evaluation_summary"]
            )
            log.debug(
                "[_persist_artifacts] Mapeo legacy aplicado para evaluation_summary_json."
            )

        # 3. ARTEFACTOS RESIDUALES
        # Copia cualquier otro dato devuelto por la técnica que NO esté en el YAML
        # (por ejemplo, métricas internas, diccionarios crudos "calibration_intervals", etc.)
        for k, v in extra_artifacts.items():
            if k not in context_data and k != "evaluation_summary":
                context_data[k] = v

        # 4. ENVÍO AL REGISTRY CENTRALIZADO
        if context_data:
            log.info(
                "[_persist_artifacts] Enviando %d artefactos al Registry para escritura...",
                len(context_data),
            )
            write_output_artifacts(
                self.ctx, self.step_key, self.step_cfg, self.base_dir, **context_data
            )
            log.info(
                "[_persist_artifacts] Artefactos persistidos exitosamente: %s",
                list(context_data.keys()),
            )
        else:
            log.warning(
                "[_persist_artifacts] No hay artefactos en context_data para persistir en '%s'.",
                self.step_key,
            )

        log.debug("[_persist_artifacts] EXIT")

    def _execute_technique(
        self,
        technique_name: str,
        tech_cfg: Dict[str, Any],
        df_test: Any,
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Dispatch to the specific technique function and auto-save JSON & PNG traces."""
        log.debug("[_execute_technique] ENTRY - técnica='%s'", technique_name)

        func = self._TECHNIQUE_DISPATCH.get(technique_name)
        if func is None:
            log.error(
                "[_execute_technique] Técnica desconocida '%s' no encontrada en _TECHNIQUE_DISPATCH.",
                technique_name,
            )
            return df_test, None

        if df_test is None:
            log.error(
                "[_execute_technique] df_test es None ANTES de llamar a la función de la técnica '%s'.",
                technique_name,
            )

        try:
            # Ejecutamos la función inyectando el df_test
            log.debug(
                "[_execute_technique] Llamando a la función '%s'...", technique_name
            )
            _, extra = func(df_test, tech_cfg, self.ctx, self.base_dir)

            if extra:
                # --- 1. AUTO-GUARDADO JSON ---
                output_key = tech_cfg.get("output")
                if output_key and str(output_key).endswith(".json"):
                    log.debug(
                        "[_execute_technique] output_key='.json' detectado. Preparando autoguardado JSON para '%s'",
                        output_key,
                    )
                    from phm_america_2024.common.io_service_common import save_json

                    # Excluir los plots del JSON para evitar errores de serialización (TypeError)
                    json_data = {
                        k: v for k, v in extra.items() if not k.endswith("_plot")
                    }
                    save_path = self.base_dir / str(output_key)

                    log.debug(
                        "[_execute_technique] Enviando %d llaves a save_json",
                        len(json_data),
                    )
                    save_json(json_data, save_path)
                    log.info(
                        "[_execute_technique] Trazabilidad JSON auto-guardada en: %s",
                        save_path.name,
                    )

                # --- 2. NUEVO: AUTO-GUARDADO DE PLOTS (PNG) ---
                # from phm_america_2024.common.io_service_common import save_figure
                from phm_america_2024.reporting.artifact_persister_reporting import (
                    save_figure,
                )

                # Rescatamos la configuración global de output_artifacts del YAML
                # para buscar el path exacto de las figuras (ej. 5.1.plots.feature_importance_plot.png)
                output_artifacts_cfg = self.step_cfg.get("output_artifacts", {})

                for k, v in extra.items():
                    # Si la llave termina en '_plot' y la figura no es None, la guardamos
                    if k.endswith("_plot") and v is not None:
                        log.debug(
                            "[_execute_technique] Objeto plot detectado para la llave: '%s'",
                            k,
                        )

                        # Buscar la ruta definida en YAML, si no existe usa el nombre de la llave
                        artifact_cfg = output_artifacts_cfg.get(k, {})
                        plot_filename = artifact_cfg.get("path", f"{k}.png")
                        plot_path = self.base_dir / plot_filename

                        try:
                            # Intentar resolver los DPI desde el contexto general (fallback a 150)
                            dpi = 150
                            if hasattr(self.ctx, "config") and hasattr(
                                self.ctx.config, "common_base_config"
                            ):
                                dpi = getattr(
                                    self.ctx.config.common_base_config.output_policy,
                                    "dpi",
                                    150,
                                )

                            # Guardamos la figura
                            save_figure(v, out_path=plot_path, dpi=dpi)
                            log.info(
                                "[_execute_technique] Figura PNG auto-guardada con éxito en: %s",
                                plot_path.name,
                            )
                        except Exception as e:
                            log.error(
                                "[_execute_technique] Fallo al guardar la figura PNG '%s': %s",
                                k,
                                e,
                            )

            log.info(
                "[_execute_technique] Técnica '%s' completada con éxito.",
                technique_name,
            )
            log.debug("[_execute_technique] EXIT")
            return df_test, extra

        except Exception as e:
            log.error(
                "[_execute_technique] Fallo crítico ejecutando la técnica '%s': %s",
                technique_name,
                str(e),
            )
            raise
