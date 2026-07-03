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
    """Execute all techniques for a single CRISP‑DM data‑preparation step.

    Each step loads a DataFrame from the previous phase/step, applies
    the configured feature‑engineering functions, and persists the
    result as parquet (and optional pickle) via the central artifact registry.
    """

    # ── Technique dispatch mapping ────────────────────────────────────────────
    _TECHNIQUE_DISPATCH: dict[str, Any] = {
        # step 3.1
        "dataset_definition": dataset_definition,
        "feature_selection": feature_selection,
        # step 3.2
        "outlier_handling": outlier_handling,
        "duplicate_handling": duplicate_handling,
        # step 3.3
        "feature_scaling": feature_scaling,
        "feature_engineering": feature_engineering,
        # step 3.5
        "data_split": data_split,
        "dataset_formatting": dataset_formatting,
    }

    def __init__(
        self, ctx: RunContext, step_key: str, step_cfg: dict[str, Any]
    ) -> None:
        """Initialize the data‑preparation runner for a specific step."""
        self.ctx: RunContext = ctx
        self.step_key: str = step_key
        self.step_cfg: dict[str, Any] = step_cfg
        self.base_dir: Path = getattr(ctx, "phase3_dir", None)

        # Step 1: CALL validate_dir() – ensure phase3 output directory exists
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

        # ── Step 4: persist artifacts via registry ─────────────────────────
        self._persist_artifacts(df, extra_artifacts)

        log.info("[Phase3PreparationRunner] completed step='%s'", self.step_key)
        return self.ctx

    # ──────────────────────────────────────────────────────────────────────────
    # Input loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_input_dataframe_old(self) -> Any:
        """Carga el DataFrame de entrada de manera dinámica usando linaje de archivos y búsqueda histórica."""
        log.debug("[_load_input_dataframe] entry step='%s'", self.step_key)

        # 1. Definir qué buscar y dónde debería estar
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
                StepsPhase.STEP_3_5.value: "*.transformed_regression_train.parquet",
            }
            pattern = lineage_map.get(self.step_key)
            if not pattern:
                raise ValueError(
                    f"No lineage mapping defined for step: {self.step_key}"
                )

        # INTENTO 1: Buscar en la corrida actual (ej. si corres todo el pipeline de corrido)
        if search_dir and search_dir.exists():
            matches = list(search_dir.glob(pattern))
            if matches:
                log.info(
                    "[_load_input_dataframe] Lineage resolved in current run. Loading: %s",
                    matches[0].name,
                )
                return load_parquet(str(matches[0]))

        # INTENTO 2: Motor de búsqueda histórica (Fallback para cuando aíslas pasos como el 3.1)
        log.warning(
            "[_load_input_dataframe] Artifact not in active run. Scanning history for '%s'...",
            pattern,
        )

        runs_root = self.base_dir.parent.parent

        if runs_root.exists() and runs_root.is_dir():
            import os

            # Ordenar las carpetas de corridas por fecha (las más recientes primero)
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
                            "✅ [Historical Fallback] Found precursor data at: %s",
                            historical_matches[0],
                        )
                        return load_parquet(str(historical_matches[0]))

        # Si el motor histórico también falla
        log.error(
            "[_load_input_dataframe] Missing upstream data. Searched history for '%s'",
            pattern,
        )
        raise FileNotFoundError(
            f"Dependency failed. Could not find precursor data for step {self.step_key}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Input loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_input_dataframe(self) -> Any:
        """Load the required input DataFrame strictly based on step configuration."""
        log.debug("[_load_input_dataframe] entry step='%s'", self.step_key)

        # 1. Extraer la ruta explícita del artefacto desde el YAML
        try:
            artifact_path_str = self.step_cfg["input_artifact"]["path"]
        except KeyError:
            log.error(
                "[_load_input_dataframe] 'input_artifact.path' missing in config for step %s",
                self.step_key,
            )
            raise ValueError(f"Step {self.step_key} config lacks 'input_artifact.path'")

        # 2. Determinar el directorio base origen
        # El paso 3.1 lee de la Fase 2. El resto (3.2, 3.3, 3.5) leen de la propia Fase 3.
        if self.step_key == StepsPhase.STEP_3_1.value:
            source_dir = self.ctx.phase2_dir
        else:
            source_dir = self.base_dir

        if source_dir is None:
            raise RuntimeError(
                f"[_load_input_dataframe] Source directory is None for step {self.step_key}"
            )

        full_path: Path = source_dir / artifact_path_str

        # 3. Validar existencia y cargar
        if not full_path.exists():
            log.error(
                "[_load_input_dataframe] Dependency failed. Input data file not found at: %s",
                full_path,
            )
            raise FileNotFoundError(
                f"Configured input artifact does not exist: {full_path}"
            )

        log.info("[_load_input_dataframe] Resolving input data from: %s", full_path)

        # 4. Deserializar
        df = load_parquet(str(full_path))
        log.info("[_load_input_dataframe] Loaded shape=%s", getattr(df, "shape", "N/A"))

        return df

    def _load_input_dataframe_old(self) -> Any:
        """Carga el DataFrame de entrada de manera dinámica usando linaje de archivos y búsqueda histórica."""
        log.debug("[_load_input_dataframe] entry step='%s'", self.step_key)

        # 1. Definir qué buscar y dónde debería estar
        pattern = self.step_cfg["input_artifact"]["path"]

        if self.step_key == StepsPhase.STEP_3_1.value:
            print(f"dioporto {pattern}")
            search_dir = self.ctx.phase2_dir
            target_phase_folder = "phase2_data_understanding"
        else:
            search_dir = self.base_dir
            target_phase_folder = "phase3_data_preparation"

        if not pattern:
            raise ValueError(f"No lineage mapping defined for step: {self.step_key}")

        # INTENTO 1: Buscar en la corrida actual (ej. si corres todo el pipeline de corrido)
        if search_dir and search_dir.exists():
            matches = list(search_dir.glob(pattern))
            if matches:
                log.info(
                    "[_load_input_dataframe] Lineage resolved in current run. Loading: %s",
                    matches[0].name,
                )
                return load_parquet(str(matches[0]))

        # INTENTO 2: Motor de búsqueda histórica (Fallback para cuando aíslas pasos como el 3.1)
        log.warning(
            "[_load_input_dataframe] Artifact not in active run. Scanning history for '%s'...",
            pattern,
        )

        runs_root = self.base_dir.parent.parent

        if runs_root.exists() and runs_root.is_dir():
            import os

            # Ordenar las carpetas de corridas por fecha (las más recientes primero)
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
                            "✅ [Historical Fallback] Found precursor data at: %s",
                            historical_matches[0],
                        )
                        return load_parquet(str(historical_matches[0]))

        # Si el motor histórico también falla
        log.error(
            "[_load_input_dataframe] Missing upstream data. Searched history for '%s'",
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
        func = self._TECHNIQUE_DISPATCH.get(technique_name)
        if func is None:
            log.warning(
                "[_execute_technique] unknown technique '%s' – skipping", technique_name
            )
            return df, {}

        output_dir = self.base_dir
        log.debug("[_execute_technique] executing '%s' a ciegas", technique_name)

        try:
            # Todas las funciones de la Fase 3 deben retornar: (DataFrame, diccionario_artefactos)
            # df_new, extra = func(df, tech_cfg, self.ctx, output_dir)

            # Ejecución universal
            df_new, extra = func(df, tech_cfg, self.ctx, self.base_dir)

            # Normalización defensiva (por si la función devolvió None en extra)
            extra = extra if extra is not None else {}

            # Autoguardado de JSONs de auditoría (si el YAML tiene el atributo 'output')
            output_key = tech_cfg.get("output")
            if output_key and str(output_key).endswith(".json"):
                # Si la función devolvió la traza en el diccionario, la guardamos
                trace_data = extra.pop(
                    "trace", None
                )  # Extraemos la traza sin enviarla al Registry
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

        # ── Step 1: Gestionar DataFrames principales según el paso ──
        if self.step_key == StepsPhase.STEP_3_5.value:
            # El paso 3.5 genera conjuntos disjuntos (train, val, test)
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
            # Para los pasos 3.1, 3.2 y 3.3 extraemos dinámicamente el Parquet desde la config del YAML
            output_artifacts_cfg = self.step_cfg.get("output_artifacts", {})
            for key in output_artifacts_cfg.keys():
                if "parquet" in key:
                    context_data[key] = df
                    break

        # ── Step 2: Mapeo inteligente del Scaler según la tarea (Paso 3.3) ──
        if self.step_key == StepsPhase.STEP_3_3.value:
            # Capturamos el objeto scaler sin importar la llave interna con la que venga
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

        # ── Step 3: Consolidar artefactos restantes (Exclusión limpia estilo Fase 4) ──
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

        # ── Step 4: Guardado final y despacho al Registry central ──
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
