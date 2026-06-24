# src/phm_america_2024/deployment/academic_scoring_deployment.py
"""
Modulo: academic_scoring_deployment.py
Algoritmo principale: Inferenza a cascata (cascade inference) in due stadi:
  1. Classificazione binaria con LightGBM (probabilità di guasto calibrata via IsotonicRegression).
  2. Regressione probabilistica con NGBoost (distribuzione Normale) per il margine di coppia (torque margin).
   Solo i campioni con probabilità di guasto >= soglia (filter_threshold) passano allo stadio di regressione.

Flusso:
1. Carica i modelli pre-addestrati (classificatore, calibratore, regressore) e i DataFrame di test (classificazione e regressione).
2. Risolve i percorsi assoluti per tutti gli artefatti a partire dal contesto ctx.run_dir.
3. Verifica che tutti gli artefatti esistano.
4. Carica i modelli con joblib e i dati di test con pd.read_parquet.
5. Estrae i parametri dalla configurazione YAML (filter_threshold, execution_order).
6. Stadio 1 – Classificazione:
   - Seleziona le feature di classificazione (esclude la colonna "faulty").
   - Predice le probabilità grezze con LightGBM, poi le calibra con IsotonicRegression.
   - Applica la soglia filter_threshold per creare una maschera binaria (mask_pass).
7. Stadio 2 – Regressione (solo per campioni con mask_pass=True):
   - Seleziona le feature di regressione (esclude la colonna "trq_margin").
   - Chiama model.pred_dist(X_reg_pass) per ottenere la distribuzione Normale.
   - Estrae media (loc) e deviazione standard (scale) dalla distribuzione.
8. Assembla un DataFrame finale con ID, probabilità di guasto, predizione binaria, valori veri e parametri di distribuzione.
9. Salva il DataFrame come Parquet nella directory phase6.
10. Arricchisce il contesto ctx con predictions_df e predictions_path.

Import:
- pathlib.Path: per risolvere percorsi.
- typing.Any: per annotazioni generiche.
- numpy, pandas: manipolazione dati.
- joblib: caricamento modelli serializzati.
- logging_adapter_common: logger personalizzato.
- context_facade_common.RunContext: tipo del contesto di esecuzione.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from phm_america_2024.common.logging_adapter_common import get_logger
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
