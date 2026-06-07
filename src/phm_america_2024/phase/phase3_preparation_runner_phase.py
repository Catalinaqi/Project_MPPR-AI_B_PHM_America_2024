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

    def _load_input_dataframe(self) -> Any:
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
        """Build the context_data payload and call write_output_artifacts."""
        log.debug("[_persist_artifacts] entry step='%s'", self.step_key)

        context_data: dict[str, Any] = {}

        # 1. Gestionar el DataFrame principal y artefactos específicos
        # Paso 3.5 tiene lógica especial porque genera DOS parquets (train y val)
        # if self.step_key == StepsPhase.STEP_3_5.value:
        #     # Validación: df debe existir para ser el train split
        #     if df is not None:
        #         context_data[StepOutputArtifact.engineered_train_split.value] = df
        #
        #     # Buscamos el val_df en los artefactos extra
        #     if "val_df" in extra_artifacts:
        #         context_data[StepOutputArtifact.engineered_val_split.value] = (
        #             extra_artifacts["val_df"]
        #         )
        if self.step_key == StepsPhase.STEP_3_5.value:
            if df is not None:
                context_data[StepOutputArtifact.engineered_train_split.value] = df

            if "val_df" in extra_artifacts:
                context_data[StepOutputArtifact.engineered_val_split.value] = (
                    extra_artifacts["val_df"]
                )

            # 👇 NUEVO: Capturar y guardar el dataset de test interno
            if "test_df" in extra_artifacts:
                # Si tienes un enum, úsalo: StepOutputArtifact.engineered_test_split.value
                # Si no, usa el string directamente (o actualiza tu enum_registry_domain.py)
                context_data["engineered_test_split"] = extra_artifacts["test_df"]
        else:
            # Para los demás pasos, buscamos el nombre del parquet en la config del YAML
            output_artifacts_cfg = self.step_cfg.get("output_artifacts", {})
            for key in output_artifacts_cfg.keys():
                if "parquet" in key:
                    context_data[key] = df
                    break

        # 2. Inyectar todo lo demás (Scalers, indices, etc.)
        for key, value in extra_artifacts.items():
            if key in [
                "trace",
                "val_df",
            ]:  # 'val_df' ya se trató arriba, 'trace' se ignoró
                continue
            context_data[key] = value

        # 3. Guardado final
        if not context_data:
            log.warning(
                "[_persist_artifacts] No artifacts to persist for %s", self.step_key
            )
            return

        log.debug(
            "[_persist_artifacts] Dispatching to registry with keys: %s",
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
            "Artifacts persisted for step '%s': %s",
            self.step_key,
            list(context_data.keys()),
        )
