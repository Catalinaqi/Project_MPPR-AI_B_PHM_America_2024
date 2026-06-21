# src/phm_america_2024/deployment/academic_scoring_deployment.py
"""
Implements the deployment logic for the PHM America 2024 pipeline:

- **cascade_inference**:   Executes a two‑stage probabilistic inference
                           (classification → regression) on internal test splits.
                           Only samples that pass the classification threshold
                           receive a full regression prediction (torque margin
                           distribution parameters).

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

import numpy as np
import pandas as pd
import joblib

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


# ---------------------------------------------------------------------------
# Public orchestration functions
# ---------------------------------------------------------------------------


def run_cascade_inference(
    ctx: RunContext,
    input_source: dict[str, str],
    params: dict[str, Any],
) -> RunContext:
    """Execute cascade inference on the internal test splits.

    **Flow**:
    1. Load pre‑trained classification model, calibrator, and regression model.
    2. Load classification and regression test DataFrames.
    3. Predict failure probability (calibrated) for every test sample.
    4. Samples with probability >= ``filter_threshold`` move to the regression stage.
    5. For those samples, predict the conditional mean (μ) and standard deviation (σ)
       of the torque margin distribution using NGBoost.
    6. Save the consolidated predictions to a Parquet file.

    Args:
        ctx:            Run context containing the run directory and project info.
        input_source:   Dictionary of artefact keys → YAML relative paths
                        (coming from ``phase6_deployment.read_strategy.input_source``).
        params:         Parameters of the ``cascade_inference`` technique
                        (see ``phase6_deployment.steps.step_6_1_academic_scoring
                              .methods.batch_scoring.techniques.cascade_inference.params``).

    Returns:
        The updated ``RunContext`` with ``predictions_df`` and ``predictions_path``
        attributes.
    """
    log.info("[cascade_inference] ===== Step 6.1 – Cascade Inference ===== start")

    # ── Resolve all input paths from the YAML input_source ──────────────────
    clf_path = _resolve_artifact_path(ctx, input_source["classification_model"])
    reg_path = _resolve_artifact_path(ctx, input_source["regression_model"])
    cal_path = _resolve_artifact_path(ctx, input_source["classification_calibrator"])
    clf_test_path = _resolve_artifact_path(
        ctx, input_source["classification_test_data"]
    )
    reg_test_path = _resolve_artifact_path(ctx, input_source["regression_test_data"])

    log.info("[cascade_inference] Input paths resolved:")
    log.info("    classification_model   : %s", clf_path)
    log.info("    regression_model       : %s", reg_path)
    log.info("    calibrator             : %s", cal_path)
    log.info("    classification_test    : %s", clf_test_path)
    log.info("    regression_test        : %s", reg_test_path)

    # ── Assert artefacts exist ──────────────────────────────────────────────
    for name, path in [
        ("classification_model", clf_path),
        ("regression_model", reg_path),
        ("calibrator", cal_path),
        ("classification_test_data", clf_test_path),
        ("regression_test_data", reg_test_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"[cascade_inference] Required artefact missing: {name} at {path}"
            )

    # ── Load models ─────────────────────────────────────────────────────────
    log.info("[cascade_inference] Loading models …")
    clf_model = joblib.load(clf_path)
    reg_model = joblib.load(reg_path)
    calibrator = joblib.load(cal_path)
    log.info("[cascade_inference] All models loaded successfully")

    # ── Load test DataFrames ────────────────────────────────────────────────
    log.info("[cascade_inference] Loading test DataFrames …")
    df_clf_test = pd.read_parquet(clf_test_path)
    df_reg_test = pd.read_parquet(reg_test_path)
    log.info("[cascade_inference] Classification test rows: %d", len(df_clf_test))
    log.info(
        "[cascade_inference] Classification test columns name-> df_clf_test: %s",
        list(df_clf_test.columns),
    )
    log.info("[cascade_inference] Regression test rows   : %d", len(df_reg_test))
    log.info(
        "[cascade_inference] Regression test columns name -> df_reg_test: %s",
        list(df_reg_test.columns),
    )

    # Ensure both DataFrames have the same row count (they should, but be safe)
    if len(df_clf_test) != len(df_reg_test):
        log.warning(
            "[cascade_inference] Test set size mismatch: clf=%d, reg=%d. Truncating to %d.",
            len(df_clf_test),
            len(df_reg_test),
            min(len(df_clf_test), len(df_reg_test)),
        )
        n = min(len(df_clf_test), len(df_reg_test))
        df_clf_test = df_clf_test.iloc[:n]
        df_reg_test = df_reg_test.iloc[:n]

    # ── Extract parameters ───────────────────────────────────────────────────
    filter_threshold = params.get("filter_threshold", 0.5)  # default 0.5
    execution_order = params.get("execution_order", ["classification", "regression"])
    log.info(
        "[cascade_inference] Configuration: filter_threshold=%s, execution_order=%s",
        filter_threshold,
        execution_order,
    )

    # ── Stage 1: Classification ──────────────────────────────────────────────
    log.info(
        "[cascade_inference] Stage 1 – Classification (predicting failure probability)"
    )

    # FILTRO: Dejamos exactamente las 10 columnas que usó LightGBM
    cols_to_drop_clf = ["faulty"]
    features_clf = [c for c in df_clf_test.columns if c not in cols_to_drop_clf]
    X_clf_test = df_clf_test[features_clf]

    probas = clf_model.predict_proba(X_clf_test)[:, 1]  # positive class
    probas_cal = calibrator.transform(probas.reshape(-1, 1)).ravel()

    mask_pass = probas_cal >= filter_threshold
    n_total = len(probas_cal)
    n_pass = mask_pass.sum()
    pass_ratio = 100.0 * n_pass / n_total if n_total > 0 else 0.0
    log.info(
        "[cascade_inference] Classification done: %d / %d samples pass threshold (%.2f%%)",
        n_pass,
        n_total,
        pass_ratio,
    )

    # ── Stage 2: Regression (only for samples that passed the threshold) ─────
    log.info("[cascade_inference] Stage 2 – Probabilistic Regression (NGBoost)")

    # Pre‑allocate output arrays (None means “no regression prediction”)
    trq_mean = [None] * n_total
    trq_std = [None] * n_total

    if n_pass > 0:
        df_reg_pass = df_reg_test.loc[mask_pass]

        # FILTRO: Dejamos exactamente las 10 columnas que usó NGBoost
        cols_to_drop_reg = ["trq_margin"]
        features_reg = [c for c in df_reg_pass.columns if c not in cols_to_drop_reg]
        X_reg_pass = df_reg_pass[features_reg]

        # NGBoost returns [μ, log(σ)] for the Normal distribution
        # pred_params = reg_model.predict(X_reg_pass)
        # mu = pred_params[:, 0]
        # sigma = np.exp(pred_params[:, 1])

        # NGBoost predict() devuelve solo un array 1D (la media).
        # Para regresión probabilística usamos pred_dist() que devuelve el objeto de distribución.
        dists = reg_model.pred_dist(X_reg_pass)

        # Extraemos mu (loc) y sigma (scale) directamente
        # NGBoost ya calcula internamente el exponencial para scale, no hace falta np.exp()
        mu = dists.loc
        sigma = dists.scale

        # Write results back to the correct rows (using original indices)
        # for idx, (m, s) in zip(mask_pass.index[mask_pass], zip(mu, sigma)):
        #     trq_mean[idx] = m
        #     trq_std[idx] = s

        # Obtenemos los índices numéricos donde mask_pass es True usando NumPy
        pass_indices = np.where(mask_pass)[0]

        # Write results back to the correct rows
        for idx, (m, s) in zip(pass_indices, zip(mu, sigma)):
            trq_mean[idx] = m
            trq_std[idx] = s

        log.info(
            "[cascade_inference] Regression predictions generated for %d samples",
            n_pass,
        )
    else:
        log.info(
            "[cascade_inference] No samples pass threshold – regression stage skipped"
        )

    # ── Assemble results DataFrame ───────────────────────────────────────────
    # predictions_df = pd.DataFrame(
    #     {
    #         "failure_probability": probas_cal,
    #         "passes_threshold": mask_pass,
    #         "trq_margin_mean": trq_mean,
    #         "trq_margin_std": trq_std,
    #     }
    # )

    # ── Assemble results DataFrame ───────────────────────────────────────────
    # 1. Extraer ID si existe en los datos, de lo contrario usamos el índice
    if "id" in df_clf_test.columns:
        ids = df_clf_test["id"].values
    else:
        ids = df_clf_test.index.values

    # 2. Construir el DataFrame con los nombres exactos del Challenge
    predictions_df = pd.DataFrame(
        {
            "id": ids,
            "faulty_true": df_clf_test["faulty"].values,
            "faulty_pred_prob": probas_cal,
            "faulty_pred": mask_pass.astype(int),  # Convierte True/False a 1 y 0
            "trq_margin_true": df_reg_test["trq_margin"].values,
            "trq_margin_pred_mu": trq_mean,
            "trq_margin_pred_sigma": trq_std,
        }
    )

    # ── Persist predictions ──────────────────────────────────────────────────
    # output_rel = params.get("output", "6.1.final_academic_predictions.parquet")
    # output_abs = _resolve_artifact_path(ctx, output_rel)
    # output_abs.parent.mkdir(parents=True, exist_ok=True)

    # NUEVO: Forzar que el directorio base sea phase6
    # output_rel = params.get("output", "6.1.final_academic_predictions.parquet")
    # phase6_dir = Path(
    #     getattr(ctx, "phase6_deployment_dir", Path(ctx.run_dir) / "phase6_deployment")
    # ).resolve()
    # output_abs = phase6_dir / output_rel
    # output_abs.parent.mkdir(parents=True, exist_ok=True)
    #
    # predictions_df.to_parquet(output_abs, index=False)
    # log.info("[cascade_inference] Predictions saved to %s", output_abs)

    # ── Persist predictions ──────────────────────────────────────────────────
    # output_rel = params.get("output", "6.1.final_academic_predictions.parquet")
    #
    # # Usamos "phase6" para coincidir con la carpeta que el Runner crea automáticamente
    # phase6_dir = Path(
    #     getattr(ctx, "phase6_dir", Path(ctx.run_dir) / "phase6")
    # ).resolve()
    # output_abs = phase6_dir / output_rel
    # output_abs.parent.mkdir(parents=True, exist_ok=True)
    #
    # predictions_df.to_parquet(output_abs, index=False)
    # log.info("[cascade_inference] Predictions saved to %s", output_abs)

    # ── Persist predictions ──────────────────────────────────────────────────
    output_rel = params.get("output", "6.1.final_academic_predictions.parquet")

    # RELY ON CONTEXT: The Phase runner guarantees this is set correctly.
    phase6_dir = Path(ctx.phase6_dir).resolve()
    output_abs = phase6_dir / output_rel
    output_abs.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_parquet(output_abs, index=False)
    log.info("[cascade_inference] Predictions saved to %s", output_abs)

    # ── Enrich context ───────────────────────────────────────────────────────
    ctx.predictions_df = predictions_df
    ctx.predictions_path = output_abs

    log.info("[cascade_inference] ===== Step 6.1 – Cascade Inference ===== done")
    return ctx


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
