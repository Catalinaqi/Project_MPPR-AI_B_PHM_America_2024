# src/phm_america_2024/registry/generator_registry_registry.py
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Optional
from pathlib import Path
from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.context_facade_common import RunContext
from phm_america_2024.common.path_service_common import resolve_path

log = get_logger(__name__)

# Registry mapping step and artifact keys to generator functions
_ARTIFACT_GENERATORS: Dict[Tuple[str, str], Callable] = {}

def register_artifact(step_key: str, artifact_key: str) -> Callable:
    """Register generator function for a specific step and artifact key.

    Args:
        step_key: Identifier for the pipeline step.
        artifact_key: Identifier for the output artifact.

    Returns:
        Decorator function to wrap the generator.
    """
    def decorator(func: Callable) -> Callable:
        # Step 1: CALL log.warning — check for existing registry overrides
        key = (step_key, artifact_key)
        if key in _ARTIFACT_GENERATORS:
            log.warning("[register_artifact] Overwriting key='%s'", key)
        _ARTIFACT_GENERATORS[key] = func
        return func
    return decorator

def write_output_artifacts(
        ctx: RunContext,
        step_key: str,
        step_cfg: Dict[str, Any],
        base_dir: Optional[Path] = None,
        **context_data: Any,
) -> None:
    """Dispatch and execute artifact generators defined in step configuration.

    Args:
        ctx: Current RunContext (class).
        step_key: Active step identifier.
        step_cfg: YAML step configuration dictionary.
        base_dir: Optional override for phase output directory.
        context_data: Arbitrary data needed for generation.
    """
    log.info("DEBUG: Registro actual contiene: %s", list(_ARTIFACT_GENERATORS.keys()))
    # Step 1: CALL step_cfg.get — extract artifacts mapping from configuration
    output_artifacts: Dict[str, Any] = step_cfg.get("output_artifacts") or {}

    log.info("DEBUG: Llaves encontradas en output_artifacts: %s", list(output_artifacts.keys()))

    for artifact_key, artifact_path in output_artifacts.items():

        # === NORMALIZACIÓN DEFINITIVA ===
        # Atrapamos DictConfig, dicts normales y strings puros para extraer solo el texto
        if hasattr(artifact_path, "path"):
            clean_path = str(artifact_path.path)
        elif isinstance(artifact_path, dict):
            clean_path = str(artifact_path.get("path", artifact_path))
        else:
            clean_path = str(artifact_path)
        # ================================

        # Step 2: CALL _ARTIFACT_GENERATORS.get — retrieve mapped generator function
        generator = _ARTIFACT_GENERATORS.get((step_key, artifact_key))

        if generator:
            log.debug("[write_output_artifacts] Executing key='%s'", artifact_key)
            try:
                # Usamos el path 100% limpio (clean_path) para el generador
                log.info(f"DEBUG: Llamando al generador para {artifact_key} con path: {clean_path}")
                generator(ctx, clean_path, **context_data)
                log.info(f"DEBUG: Generador {artifact_key} finalizado con éxito.")

                # Inspección de tipos para asegurarnos de que todo es texto
                log.info(f"DEBUG: Inspeccionando tipos antes del registro:")
                log.info(f"  artifact_key: {type(artifact_key)} -> {artifact_key}")
                log.info(f"  clean_path: {type(clean_path)} -> {clean_path}")
                log.info(f"  base_dir: {type(base_dir)} -> {base_dir}")

                # ¡Usamos el path limpio (clean_path) para el registro!
                _register_artifact_path(ctx, artifact_key, clean_path, base_dir)
                log.info(f"DEBUG: Registro en contexto exitoso para {artifact_key}")

            except Exception as e:
                # Esto atrapará si algo falla en el generador O en el registro
                log.error(f"DEBUG: ERROR CRÍTICO en el generador {artifact_key}: {e}", exc_info=True)
        else:
            # Step 5: CALL log.warning — handle missing generator registration
            log.warning("[write_output_artifacts] No generator for '%s'", artifact_key)

def _register_artifact_path(
        ctx: RunContext,
        artifact_key: str,
        artifact_path: str | Path,
        base_dir: Optional[Path] = None,
) -> None:
    """Resolve and persist the absolute path of a generated artifact."""
    # Step 1: CALL getattr — resolve phase-level base directory
    target_dir = base_dir or getattr(ctx, "phase2_dir", None)
    if not target_dir:
        log.warning("[_register_artifact_path] No base_dir for '%s'", artifact_key)
        return

    # Step 2: CALL resolve_path — canonicalize the file system path
    abs_path: Path = resolve_path(Path(target_dir) / artifact_path)
    # Step 3: CALL ctx.register_artifact — update the run context
    ctx.register_artifact(artifact_key, abs_path)
    log.debug("[_register_artifact_path] Registered '%s' to '%s'", artifact_key, abs_path)