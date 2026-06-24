# src/phm_america_2024/deployment/academic_scoring_deployment.py
"""
Implements the deployment logic for the PHM America 2024 pipeline:

- **zip_delivery**:        Packages all required artefacts (models, datasets,
                           code, predictions) into a single ZIP archive for
                           academic submission.

Every path is resolved relative to the current run directory (``ctx.run_dir``).
Logging follows the project‑wide pattern via ``get_logger``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any


from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.path_service_common import find_project_root
from phm_america_2024.pipeline.utils.context_facade_common import RunContext

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_artifact_path(ctx: RunContext, relative_path: str) -> Path:
    """Convert a relative path (as stored in the YAML config) to an absolute
    ``Path`` anchored at the current run directory.

    The YAML config refers to artefacts relative to the run output folder
    (e.g. ``"4.2.training.ngboost_regressor.pkl"``).  This helper appends them
    to ``ctx.run_dir`` and returns the resolved absolute path.

    Args:
        ctx:             Run context (must have a ``run_dir`` attribute).
        relative_path:   Dot‑separated path from the YAML config.

    Returns:
        Absolute ``Path`` pointing to the artefact.
    """
    run_dir = Path(ctx.run_dir).resolve()
    full_path = (run_dir / relative_path).resolve()
    log.debug("[_resolve_artifact_path] %s → %s", relative_path, full_path)
    return full_path


def run_zip_delivery_old(
    ctx: RunContext,
    input_source: dict[str, str],
    params: dict[str, Any],
) -> RunContext:
    """Package artefacts (models, datasets, predictions, source code) into a ZIP.

    **Flow**:
    1. Collect the list of files to include from ``params['files_to_include']``.
    2. Optionally include the whole ``src/`` directory tree if
       ``params['include_code']`` is ``True``.
    3. Create a ZIP archive at the path specified in ``params['output']``.
    4. Log warnings for any missing files (non‑fatal).

    Args:
        ctx:            Run context (used to resolve run‑relative paths and
                        detect the project root).
        input_source:   (Unused but kept for interface consistency.)
        params:         Parameters of the ``zip_delivery`` technique
                        (see ``phase6_deployment.steps.step_6_2_package_deliverables
                              .methods.package_delivery.techniques.zip_delivery.params``).

    Returns:
        The updated ``RunContext`` with ``zip_path`` attribute.
    """
    log.info("[zip_delivery] ===== Step 6.2 – Package Deliverables ===== start")

    # ── Extract parameters ───────────────────────────────────────────────────
    files_to_include = params.get("files_to_include", [])
    include_code = params.get("include_code", False)
    output_rel = params.get("output", "Consegna_PHM_America_2024.zip")

    log.info("[zip_delivery] Configuration:")
    log.info("    output ZIP      : %s", output_rel)
    log.info("    include_code    : %s", include_code)
    log.info("    files_to_include: %s", files_to_include)

    # ── Resolve output path ──────────────────────────────────────────────────
    # output_abs = _resolve_artifact_path(ctx, output_rel)
    # output_abs.parent.mkdir(parents=True, exist_ok=True)
    # log.info("[zip_delivery] Output ZIP resolved to: %s", output_abs)

    # NUEVO: Forzar que el archivo ZIP se guarde en phase6
    phase6_dir = Path(
        getattr(ctx, "phase6_dir", Path(ctx.run_dir) / "phase6")
    ).resolve()
    output_abs = phase6_dir / output_rel
    output_abs.parent.mkdir(parents=True, exist_ok=True)

    log.info("[zip_delivery] Output ZIP resolved to: %s", output_abs)

    # ── Create ZIP archive ───────────────────────────────────────────────────
    with zipfile.ZipFile(output_abs, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # ── Add explicitly listed artefacts ────────────────────────────────
        for rel_path in files_to_include:
            abs_path = _resolve_artifact_path(ctx, rel_path)
            if abs_path.exists():
                arcname = abs_path.name
                zf.write(abs_path, arcname=arcname)
                log.debug("[zip_delivery] Added: %s → %s", abs_path, arcname)
            else:
                log.warning("[zip_delivery] Skipped (not found): %s", abs_path)

        # ── Optionally add source code ─────────────────────────────────────
        if include_code:
            # Determine the project root directory (contains src/)
            try:
                project_root = find_project_root()
            except RuntimeError:
                log.warning(
                    "[zip_delivery] Cannot locate project root – code inclusion skipped"
                )
                project_root = None

            if project_root is not None:
                src_dir = project_root / "src"
                if src_dir.is_dir():
                    for py_file in src_dir.rglob("*.py"):
                        arcname = str(py_file.relative_to(project_root))
                        zf.write(py_file, arcname=arcname)
                        log.debug("[zip_delivery] Added source: %s", arcname)
                else:
                    log.warning(
                        "[zip_delivery] src/ directory not found at %s", src_dir
                    )

    log.info("[zip_delivery] ZIP archive created at %s", output_abs)
    ctx.zip_path = output_abs

    log.info("[zip_delivery] ===== Step 6.2 – Package Deliverables ===== done")
    return ctx


def run_zip_delivery_v2(
    ctx: RunContext,
    input_source: dict[str, str],
    params: dict[str, Any],
) -> RunContext:
    """Package artefacts (models, datasets, predictions, source code) into a ZIP."""
    log.info("[zip_delivery] ===== Step 6.2 – Package Deliverables ===== start")

    # ── Extract parameters ───────────────────────────────────────────────────
    files_to_include = params.get("files_to_include", [])
    include_code = params.get("include_code", False)
    output_rel = params.get("output", "Consegna_PHM_America_2024.zip")

    log.info("[zip_delivery] Configuration:")
    log.info("    output ZIP      : %s", output_rel)
    log.info("    include_code    : %s", include_code)
    log.info("    files_to_include: %s", files_to_include)

    # ── Resolve output path (Forzar guardado en phase6_deployment) ───────────
    phase6_dir = Path(
        getattr(ctx, "phase6_deployment_dir", Path(ctx.run_dir) / "phase6_deployment")
    ).resolve()
    output_abs = phase6_dir / output_rel
    output_abs.parent.mkdir(parents=True, exist_ok=True)
    log.info("[zip_delivery] Output ZIP resolved to: %s", output_abs)

    # Definir el directorio global de "runs" para buscar en clasificación y regresión
    try:
        project_root = find_project_root()
        search_dir = project_root / "outputs" / "runs"
    except RuntimeError:
        # Fallback si falla find_project_root
        search_dir = Path(ctx.run_dir).resolve().parents[2]

    # ── Create ZIP archive ───────────────────────────────────────────────────
    with zipfile.ZipFile(output_abs, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # ── 1. Buscar y agregar los Parquets listados de forma inteligente ──
        for filename in files_to_include:
            # Primero: verificamos si es el archivo que acabamos de crear en la fase 6
            direct_file = phase6_dir / filename
            if direct_file.exists():
                zf.write(direct_file, arcname=filename)
                log.info("[zip_delivery] Added (from phase6): %s", filename)
                continue

            # Segundo: buscamos globalmente en la carpeta de outputs/runs
            found_files = list(search_dir.rglob(filename))

            if found_files:
                # Si hay varios con el mismo nombre (varias ejecuciones), tomamos el más reciente
                found_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                file_to_zip = found_files[0]
                zf.write(file_to_zip, arcname=filename)
                log.info("[zip_delivery] Added (found in runs): %s", filename)
            else:
                log.warning(
                    "[zip_delivery] SKIPPED! File not found anywhere in runs: %s",
                    filename,
                )

        # ── 2. Opcionalmente agregar código fuente ───────────────────────────
        if include_code:
            try:
                if project_root is not None:
                    src_dir = project_root / "src"
                    if src_dir.is_dir():
                        for py_file in src_dir.rglob("*.py"):
                            arcname = str(py_file.relative_to(project_root))
                            zf.write(py_file, arcname=arcname)
                            log.debug("[zip_delivery] Added source: %s", arcname)
                    else:
                        log.warning(
                            "[zip_delivery] src/ directory not found at %s", src_dir
                        )
            except Exception as e:
                log.warning("[zip_delivery] Failed to include source code: %s", e)

    log.info("[zip_delivery] ZIP archive created at %s", output_abs)
    ctx.zip_path = output_abs

    log.info("[zip_delivery] ===== Step 6.2 – Package Deliverables ===== done")
    return ctx


def run_zip_delivery_v3(
    ctx: RunContext,
    input_source: dict[str, str],
    params: dict[str, Any],
) -> RunContext:
    """Package ONLY the requested Parquet datasets and predictions into a ZIP."""
    log.info("[zip_delivery] ===== Step 6.2 – Package Deliverables ===== start")

    # ── Extract parameters ───────────────────────────────────────────────────
    files_to_include = params.get("files_to_include", [])
    output_rel = params.get("output", "Consegna_PHM_America_2024.zip")

    log.info("[zip_delivery] Configuration:")
    log.info("    output ZIP      : %s", output_rel)
    log.info("    files_to_include: %s", files_to_include)

    # ── Resolve output path (Coincidir con la carpeta automática phase6) ─────
    phase6_dir = Path(
        getattr(ctx, "phase6_dir", Path(ctx.run_dir) / "phase6")
    ).resolve()
    output_abs = phase6_dir / output_rel
    output_abs.parent.mkdir(parents=True, exist_ok=True)
    log.info("[zip_delivery] Output ZIP resolved to: %s", output_abs)

    # Definir el directorio global de "runs" para buscar en clasificación y regresión
    try:
        project_root = find_project_root()
        search_dir = project_root / "outputs" / "runs"
    except RuntimeError:
        search_dir = Path(ctx.run_dir).resolve().parents[2]

    # ── Create ZIP archive (MODO ESTRICTO: Solo archivos explícitos) ─────────
    with zipfile.ZipFile(output_abs, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename in files_to_include:
            # 1. Buscar primero en la carpeta actual phase6
            direct_file = phase6_dir / filename
            if direct_file.exists():
                zf.write(direct_file, arcname=filename)
                log.info("[zip_delivery] Added (from phase6): %s", filename)
                continue

            # 2. Buscar globalmente en las carpetas de clasificación y regresión
            found_files = list(search_dir.rglob(filename))

            if found_files:
                # Tomamos el más reciente
                found_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                file_to_zip = found_files[0]
                zf.write(file_to_zip, arcname=filename)
                log.info("[zip_delivery] Added (found in runs): %s", filename)
            else:
                log.warning(
                    "[zip_delivery] SKIPPED! File not found anywhere: %s", filename
                )

        # NOTA: La lógica del código fuente (src) ha sido completamente eliminada.

    log.info("[zip_delivery] ZIP archive created at %s", output_abs)
    ctx.zip_path = output_abs

    log.info("[zip_delivery] ===== Step 6.2 – Package Deliverables ===== done")
    return ctx


def run_zip_delivery(
    ctx: RunContext,
    input_source: dict[str, str],
    params: dict[str, Any],
) -> RunContext:
    """Package ONLY the requested Parquet datasets and predictions into a ZIP."""
    log.info("[zip_delivery] ===== Step 6.2 – Package Deliverables ===== start")

    # ── Extract parameters ───────────────────────────────────────────────────
    files_to_include = params.get("files_to_include", [])
    output_rel = params.get("output", "Consegna_PHM_America_2024.zip")

    log.info("[zip_delivery] Configuration:")
    log.info("    output ZIP      : %s", output_rel)
    log.info("    files_to_include: %s", files_to_include)

    # ── Resolve output path (Relying on the Runner's context) ────────────────
    phase6_dir = Path(ctx.phase6_dir).resolve()
    output_abs = phase6_dir / output_rel
    output_abs.parent.mkdir(parents=True, exist_ok=True)
    log.info("[zip_delivery] Output ZIP resolved to: %s", output_abs)

    # Definir el directorio global de "runs" para buscar en clasificación y regresión
    try:
        project_root = find_project_root()
        search_dir = project_root / "outputs" / "runs"
    except RuntimeError:
        search_dir = Path(ctx.run_dir).resolve().parents[2]

    # ── Create ZIP archive (MODO ESTRICTO: Solo archivos explícitos) ─────────
    with zipfile.ZipFile(output_abs, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename in files_to_include:
            # 1. Buscar primero en la carpeta actual (phase6_deployment)
            direct_file = phase6_dir / filename
            if direct_file.exists():
                zf.write(direct_file, arcname=filename)
                log.info("[zip_delivery] Added (from phase6_deployment): %s", filename)
                continue

            # 2. Buscar globalmente en las carpetas de clasificación y regresión
            found_files = list(search_dir.rglob(filename))

            if found_files:
                # Tomamos el más reciente
                found_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                file_to_zip = found_files[0]
                zf.write(file_to_zip, arcname=filename)
                log.info("[zip_delivery] Added (found in runs): %s", filename)
            else:
                log.warning(
                    "[zip_delivery] SKIPPED! File not found anywhere: %s", filename
                )

    log.info("[zip_delivery] ZIP archive created at %s", output_abs)
    ctx.zip_path = output_abs

    log.info("[zip_delivery] ===== Step 6.2 – Package Deliverables ===== done")
    return ctx
