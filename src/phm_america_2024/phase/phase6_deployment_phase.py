# src/phm_america_2024/phase/phase6_deployment_phase.py
"""
Phase 6 – Deployment orchestration.

Handles two steps:
  - step_6_1_academic_scoring: cascade inference (classification → regression).
  - step_6_2_package_deliverables: ZIP packaging of artefacts.

Follows the same structural contract as Phase 5 runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.pipeline.utils.context_facade_common import RunContext

# Technique implementations
from phm_america_2024.deployment.academic_scoring_deployment import (
    run_cascade_inference,
    run_zip_delivery,
)
from phm_america_2024.domain.enum_registry_domain import Phase

log = get_logger(__name__)


class Phase6DeploymentRunner:
    """Execute all techniques for a single CRISP‑DM Phase 6 step.

    - Loads models, calibrator, and test data once per step.
    - Dispatches execution per individual technique defined in YAML.
    - Relies on the technique functions to persist their own output artefacts
      (parquet predictions, ZIP archive).
    """

    # Mapping YAML technique names to their Python implementations.
    # Each function receives (ctx, input_source, params) and returns the
    # updated RunContext (convention: the function enriches ctx with e.g.
    # ctx.predictions_path / ctx.zip_path).
    _TECHNIQUE_DISPATCH: Dict[str, Any] = {
        "cascade_inference": run_cascade_inference,
        "zip_delivery": run_zip_delivery,
    }

    def __init__(
        self,
        ctx: RunContext,
        step_key: str,
        step_cfg: Dict[str, Any],
    ) -> None:
        """Initialize the deployment runner."""
        self.ctx = ctx
        self.step_key = step_key
        self.step_cfg = step_cfg

        # Base directory: prefer ctx.phase6_dir (set earlier), else fallback
        # to a sub‑directory inside the run dir.
        self.base_dir: Path = getattr(ctx, "phase6_dir", None)
        if self.base_dir is None:
            self.base_dir = Path(ctx.run_dir) / Phase.PHASE6.value
            # Save it back to context so downstream functions (like zip_delivery) can use it
            setattr(ctx, "phase6_dir", self.base_dir)
            log.debug(
                "[Phase6Runner] phase6_dir not found on ctx; using fallback %s",
                self.base_dir,
            )
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Store the global read_strategy and input_source for this phase
        # (injected by _exec_step in regression_runner_pipeline.py)
        self.read_strategy: Dict[str, Any] = step_cfg.get("read_strategy", {})
        self.input_source: Dict[str, str] = step_cfg.get("input_source", {})

        log.debug(
            "[Phase6Runner] Initialised step='%s' base_dir='%s'",
            self.step_key,
            self.base_dir,
        )

    def run(self) -> RunContext:
        """Run the configured techniques for this phase step."""
        log.debug("[Phase6Runner] run() - ENTRY")
        log.info("[Phase6Runner] Starting execution of step='%s'", self.step_key)

        # 1. Load required artefacts (models, calibrator, test data)
        #    The load method returns None on failure.
        loaded_ok = self._load_artifacts()
        if not loaded_ok:
            log.error(
                "[Phase6Runner] run() cancelled – failed to load artefacts. "
                "Check phase directories and configuration."
            )
            return self.ctx

        # 2. Iterate over methods → techniques
        methods: Dict[str, Any] = self.step_cfg.get("methods", {})
        log.debug("[Phase6Runner] Methods configured: %s", list(methods.keys()))

        for method_name, method_cfg in methods.items():
            if not method_cfg.get("enabled", True):
                log.info(
                    "[Phase6Runner] Method '%s' is disabled in YAML. Skipping...",
                    method_name,
                )
                continue

            log.info("[Phase6Runner] --- Processing Method: '%s' ---", method_name)
            techniques: Dict[str, Any] = method_cfg.get("techniques", {})
            log.debug(
                "[Phase6Runner] Techniques for '%s': %s",
                method_name,
                list(techniques.keys()),
            )

            for tech_name, tech_cfg in techniques.items():
                if not tech_cfg.get("enabled", True):
                    log.info(
                        "[Phase6Runner] Technique '%s' is disabled in YAML. Skipping...",
                        tech_name,
                    )
                    continue

                log.info("[Phase6Runner] Executing technique: '%s'", tech_name)
                try:
                    self.ctx = self._execute_technique(tech_name, tech_cfg)
                except Exception:
                    log.exception(
                        "[Phase6Runner] CRITICAL: technique '%s' failed during execution.",
                        tech_name,
                    )
                    raise

        # 3. Persist any extra artefacts (if the technique returned them)
        #    Most artefacts are saved directly by the technique functions,
        #    but we can still trace them via the registry.
        self._persist_artifacts()

        log.info("[Phase6Runner] Completed step='%s'", self.step_key)
        log.debug("[Phase6Runner] run() - EXIT")
        return self.ctx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_technique(
        self,
        technique_name: str,
        tech_cfg: Dict[str, Any],
    ) -> RunContext:
        """Dispatch to the specific technique function."""
        log.debug("[_execute_technique] ENTRY - technique='%s'", technique_name)

        func = self._TECHNIQUE_DISPATCH.get(technique_name)
        if func is None:
            log.error(
                "[_execute_technique] Unknown technique '%s' not found in dispatch map.",
                technique_name,
            )
            return self.ctx

        params = tech_cfg.get("params", {})

        # The technique functions expect (ctx, input_source, params)
        log.debug(
            "[_execute_technique] Calling '%s' with params keys: %s",
            technique_name,
            list(params.keys()),
        )
        updated_ctx = func(self.ctx, self.input_source, params)

        # Update self.ctx if a new object was returned
        if updated_ctx is not None:
            self.ctx = updated_ctx

        log.info("[_execute_technique] Technique '%s' completed.", technique_name)
        log.debug("[_execute_technique] EXIT")
        return self.ctx

    def _load_artifacts(self) -> bool:
        """Load models, calibrator, and test data into the context.

        For step_6_1, we need:
          - classification_model
          - regression_model
          - classification_calibrator
          - classification_test_data
          - regression_test_data

        For step_6_2, the technique `zip_delivery` does not require loading
        models; it just collects files from the run directory. The function
        will still be called for consistency but the input_source is not
        essential for step_6_2; we can skip loading.

        Returns True on success, False if required artefacts are missing.
        """
        log.debug("[_load_artifacts] ENTRY - step='%s'", self.step_key)

        # For step_6_2, loading is not necessary; skip.
        if "package_deliverables" in self.step_key:
            log.info("[_load_artifacts] Step 6.2 – no models to load.")
            return True

        # Step 6.1 – validate that required keys exist in input_source
        required_keys = [
            "classification_model",
            "regression_model",
            "classification_calibrator",
            "classification_test_data",
            "regression_test_data",
        ]
        missing_keys = [k for k in required_keys if k not in self.input_source]
        if missing_keys:
            log.error(
                "[_load_artifacts] Missing keys in input_source: %s",
                missing_keys,
            )
            return False

        # We do not actually load models here because cascade_inference
        # does it internally using _resolve_artifact_path. However, we can
        # perform an existence check to fail early.
        # (Optional) Verify files exist.
        from pathlib import Path

        run_dir = Path(self.ctx.run_dir).resolve()
        for key in required_keys:
            rel_path = self.input_source[key]
            abs_path = (run_dir / rel_path).resolve()
            if not abs_path.exists():
                log.error(
                    "[_load_artifacts] Required artefact not found: '%s' at %s",
                    key,
                    abs_path,
                )
                return False

        log.info("[_load_artifacts] All required artefacts exist.")
        log.debug("[_load_artifacts] EXIT - success")
        return True

    def _persist_artifacts(self) -> None:
        """Write artefact metadata for tracing (optional).

        The technique functions themselves save the primary output files.
        This method can be used to register them with the centralized
        artifact registry if desired (e.g., for trace logs).

        Currently a no‑op; extend as needed.
        """
        log.debug("[_persist_artifacts] ENTRY - step='%s'", self.step_key)

        # Example: register paths stored on ctx (predictions_path, zip_path)
        # with the registry system.
        # For now, just log context attributes that were set.
        if hasattr(self.ctx, "predictions_path"):
            log.info(
                "[_persist_artifacts] Predictions available at: %s",
                self.ctx.predictions_path,
            )
        if hasattr(self.ctx, "zip_path"):
            log.info(
                "[_persist_artifacts] ZIP archive available at: %s",
                self.ctx.zip_path,
            )

        log.debug("[_persist_artifacts] EXIT")
