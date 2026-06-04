from __future__ import annotations

import os
import json
import yaml  # <-- Importante para leer el archivo crudo
from pathlib import Path
from typing import Any

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.context_facade_common import RunContext
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.data.load_loader_data import load_parquet, load_pickle
from phm_america_2024.registry.generator_registry_registry import write_output_artifacts
from phm_america_2024.configuration.enum_registry_config import StepsPhase, StepOutputArtifact

# --- técnica functions ---
from phm_america_2024.model.algorithm_selector_model import single_probabilistic_architecture
from phm_america_2024.model.regression_trainer_model import cross_validation
from phm_america_2024.model.regression_evaluator_model import model_selection_criteria

log = get_logger(__name__)


class Phase4ModelingRunner:
    """Execute all techniques for a single CRISP‑DM modeling step."""

    _TECHNIQUE_DISPATCH: dict[str, Any] = {
        "single_probabilistic_architecture": single_probabilistic_architecture,
        "cross_validation": cross_validation,
        "model_selection_criteria": model_selection_criteria,
    }

    def __init__(self, ctx: RunContext, step_key: str, step_cfg: dict[str, Any]) -> None:
        self.ctx: RunContext = ctx
        self.step_key: str = step_key
        self.step_cfg: dict[str, Any] = step_cfg
        self.base_dir: Path = getattr(ctx, "phase4_dir", None)
        if self.base_dir is None:
            log.error("[Phase4ModelingRunner] ctx.phase4_dir is None")
            raise RuntimeError("phase4_dir missing from RunContext")
        log.debug("[Phase4ModelingRunner] init step='%s' base_dir='%s'", self.step_key, self.base_dir)

    def run(self) -> RunContext:
        log.info("[Phase4ModelingRunner] run step='%s'", self.step_key)
        # Step 1: load inputs
        df = self._load_input_dataframe()
        extra_artifacts: dict[str, Any] = {}

        # Step 2: iterate methods / techniques
        methods = self.step_cfg.get("methods", {})
        for method_name, method_cfg in methods.items():
            if not method_cfg.get("enabled", True):
                continue
            techniques = method_cfg.get("techniques", {})
            for tech_name, tech_cfg in techniques.items():
                if not tech_cfg.get("enabled", True):
                    continue
                df, art = self._execute_technique(tech_name, tech_cfg, df)
                if art is not None:
                    extra_artifacts.update(art)

        # Step 3: persist artifacts
        self._persist_artifacts(df, extra_artifacts)
        log.info("[Phase4ModelingRunner] completed step='%s'", self.step_key)
        return self.ctx

    def _get_explicit_train_path(self) -> str | None:
        """Helper to extract explicit absolute path directly from the YAML file."""
        # First, try the standard framework injection
        read_strategy = self.step_cfg.get("read_strategy", {})
        if not read_strategy and hasattr(self.ctx, "phase_cfg"):
            read_strategy = getattr(self.ctx, "phase_cfg", {}).get("read_strategy", {})

        # If it's empty, let's go straight to the source YAML file
        if not read_strategy:
            log.warning("[Phase4ModelingRunner] step_cfg no contiene read_strategy. Buscando en YAML crudo...")
            try:
                # Localizar el archivo en el proyecto (Asume que está en root/config/phm_america_2024/regression.yml)
                # Ajusta las subidas (.parents[3]) si tu estructura de carpetas es distinta
                project_root = Path(__file__).resolve().parents[3]
                config_path = project_root / "config" / "phm_america_2024" / "regression.yml"

                if config_path.exists():
                    with open(config_path, "r", encoding="utf-8") as f:
                        raw_yaml = yaml.safe_load(f)

                    # Navegar por el YAML para encontrar read_strategy
                    phase4_config = raw_yaml.get("phases", {}).get("phase4_data_modeling", {})
                    if not phase4_config:
                        phase4_config = raw_yaml.get("phase4_data_modeling", {})

                    read_strategy = phase4_config.get("read_strategy", {})
                    if read_strategy:
                        log.info("[Phase4ModelingRunner] Extracción exitosa de read_strategy desde YAML crudo.")
            except Exception as e:
                log.error("[Phase4ModelingRunner] Fallo al leer YAML crudo: %s", e)

        input_source = read_strategy.get("input_source", {}) if isinstance(read_strategy, dict) else {}
        explicit_train = input_source.get("train_data")

        return explicit_train if explicit_train and isinstance(explicit_train, str) else None

    def _load_input_dataframe(self) -> Any:
        step_val = self.step_key

        # =========================================================
        # BYPASS ABSOLUTO: Ignoramos el YAML para asegurar la ejecución
        # =========================================================
        override_train_path = r"K:\00_Code\Manutenzione\Project_MPPR-AI_B_PHM_America_2024\outputs\runs\regression\phm2024\20260604_110122\phase3_data_preparation\3.5.formatting.regression_internal_train.parquet"

        if step_val == StepsPhase.STEP_4_1.value or step_val == StepsPhase.STEP_4_2.value:
            log.info("[Phase4ModelingRunner] 🚀 MODO OVERRIDE ACTIVO. Forzando lectura directa desde: %s", override_train_path)

            # Verificación de seguridad rápida
            if not os.path.exists(override_train_path):
                raise FileNotFoundError(f"¡ALERTA! El archivo realmente no existe en tu disco duro en esta ruta: {override_train_path}")

            path = resolve_path(Path(override_train_path))
            df = load_parquet(str(path))

        elif step_val == StepsPhase.STEP_4_4.value:
            # Cargar el modelo del paso anterior en la fase 4
            prev = "4.2.training.ngboost_regressor.pkl"
            path = resolve_path(self.ctx.phase4_dir / prev)
            model = load_pickle(str(path))
            df = model

        else:
            raise ValueError(f"Unknown step: {step_val}")

        log.info("[Phase4ModelingRunner] loaded input shape=%s", getattr(df, "shape", "N/A"))
        return df

    def _execute_technique(self, technique_name: str, tech_cfg: dict, df: Any):
        func = self._TECHNIQUE_DISPATCH.get(technique_name)
        if func is None:
            log.warning("unknown technique '%s' – skip", technique_name)
            return df, None
        params = tech_cfg.get("params", {})
        output_dir = self.base_dir
        log.debug("[_execute_technique] calling '%s'", technique_name)

        # special handling for model_selection_criteria which receives a trained model
        if technique_name == "model_selection_criteria":
            # df is actually the trained model object
            model, extra = func(df, params, self.ctx, output_dir)
            return model, extra
        else:
            df_new, extra = func(df, params, self.ctx, output_dir)
            return df_new, extra

    def _persist_artifacts(self, df: Any, extra_artifacts: dict[str, Any]) -> None:
        context_data: dict = {}
        output_artifacts_cfg = self.step_cfg.get("output_artifacts", {})

        if self.step_key == StepsPhase.STEP_4_2.value:
            # need to pass trained model from extra
            context_data[StepOutputArtifact.trained_ngboost_model.value] = extra_artifacts.get("trained_model")
        elif self.step_key == StepsPhase.STEP_4_4.value:
            context_data[StepOutputArtifact.best_regression_model_metadata.value] = extra_artifacts.get("best_model_metadata")
        # step_4_1 produces only traces (no main artifact)
        # add any extra artifacts
        context_data.update({k: v for k, v in extra_artifacts.items() if k != "trained_model"})
        write_output_artifacts(self.ctx, self.step_key, self.step_cfg, self.base_dir, **context_data)