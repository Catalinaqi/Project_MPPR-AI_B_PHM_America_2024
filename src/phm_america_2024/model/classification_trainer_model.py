# src/phm_america_2024/model/classification_trainer_model.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


# ---- cross_validation (entrenamiento con GMM + GroupKFold) ----
def cross_validation(
    df: pd.DataFrame,
    tech_cfg: Dict[str, Any],
    ctx: Any,
    output_dir: Path,
    algorithm_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Execute GMM-grouped GroupKFold cross validation and train LightGBM per fold.

    Args:
        df: Input DataFrame.
        tech_cfg: YAML technique configuration containing params and output.
        ctx: Run context holding shared execution state.
        output_dir: Output directory path.
        algorithm_config: Injected configuration from prior Step 4.1 (LightGBM params).

    Returns:
        Best trained model (lowest validation Brier score) and extra artifacts dict.
    """
    log.debug("[classification cross_validation] entry – shape=%s", df.shape)

    # ── 1. Extraer parámetros del YAML (zero hardcoding) ──
    try:
        params: Dict[str, Any] = tech_cfg["params"]
        n_splits: int = params["n_splits"]
        strategy: str = params["strategy"]
        target_col: str = params["target_variable"]
        grouping: Dict[str, Any] = params["grouping_mechanism"]
        gmm_features: list[str] = grouping["features"]
        n_clusters: int = grouping["n_clusters"]
        cov_type: str = grouping["covariance_type"]
        random_seed: int = grouping.get("random_seed", 42)
        output_filename: str = tech_cfg["output"]
    except KeyError as e:
        log.error("[classification cross_validation] YAML key missing: %s", e)
        raise

    # ── 2. Validar consistencia ──
    if strategy != "GroupKFold":
        log.warning(
            "[classification cross_validation] Se esperaba 'GroupKFold' pero se encontró '%s'. Usando GroupKFold.",
            strategy,
        )
    if n_splits > n_clusters:
        log.warning(
            "[classification cross_validation] n_splits (%d) > n_clusters (%d). Ajustando n_splits a %d.",
            n_splits,
            n_clusters,
            n_clusters,
        )
        n_splits = n_clusters

    # ── 3. Sanitizar datos ──
    df = df.replace([np.inf, -np.inf], np.nan)
    initial_rows: int = len(df)
    df = df.dropna()
    dropped: int = initial_rows - len(df)
    if dropped:
        log.warning(
            "[classification cross_validation] Droped %d filas con NaN/Inf.", dropped
        )

    log.info("[classification cross_validation] target_col is: %s", target_col)

    feature_cols: list[str] = [c for c in df.columns if c != target_col]
    # X: np.ndarray = df[feature_cols].values
    # y: np.ndarray = df[target_col].values

    X = df[feature_cols]  # Mantenemos el DataFrame
    y = df[target_col].values

    # ── 4. GMM clustering ──
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type=cov_type, random_state=random_seed
    )
    gmm_idx = [feature_cols.index(f) for f in gmm_features if f in feature_cols]
    if len(gmm_idx) < len(gmm_features):
        raise ValueError(f"GMM features no encontradas: {gmm_features}")
    # X_gmm = X[:, gmm_idx]
    X_gmm = X.iloc[:, gmm_idx]
    cluster_labels: np.ndarray = gmm.fit_predict(X_gmm)

    # ── 5. GroupKFold ──
    gkf = GroupKFold(n_splits=n_splits)

    # ── 6. Obtener configuración del algoritmo (desde step_4_1) ──
    if algorithm_config is None:
        raise ValueError("algorithm_config es obligatorio. Ejecute step_4_1 primero.")
    try:
        lgb_params: Dict[str, Any] = {
            "objective": algorithm_config.get("objective", "binary"),
            "boosting_type": algorithm_config.get("boosting_type", "gbdt"),
            "scale_pos_weight": algorithm_config.get("scale_pos_weight", 1.0),
            "learning_rate": algorithm_config.get("learning_rate", 0.05),
            "n_estimators": algorithm_config.get("n_estimators", 100),
            "max_depth": algorithm_config.get("max_depth", -1),
            "num_leaves": algorithm_config.get("num_leaves", 31),
            "random_state": algorithm_config.get("random_state", 42),
        }
    except KeyError as e:
        log.error(
            "[classification cross_validation] Falta clave en algorithm_config: %s", e
        )
        raise

    log.info("[classification cross_validation] LightGBM params: %s", lgb_params)

    # ── 7. Iterar folds ──
    best_model: Any = None
    best_brier: float = float("inf")
    fold_results: list[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X, y, groups=cluster_labels)
    ):
        log.info("[classification cross_validation] Fold %d/%d", fold_idx + 1, n_splits)
        # X_train, X_val = X[train_idx], X[val_idx]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train)

        # Predecir probabilidades
        y_proba = model.predict_proba(X_val)[:, 1]
        # Brier score
        brier = np.mean((y_val - y_proba) ** 2)

        fold_results.append({"fold": fold_idx, "brier_score": float(brier)})

        if brier < best_brier:
            best_brier = brier
            best_model = model

        log.info(
            "[classification cross_validation] Fold %d Brier = %.6f", fold_idx, brier
        )

    # ── 8. Escribir traza ──
    trace: Dict[str, Any] = {
        "n_folds": n_splits,
        "gmm_features": gmm_features,
        "gmm_n_clusters": n_clusters,
        "fold_results": fold_results,
        "best_fold_brier": float(best_brier),
    }
    output_path = output_dir / output_filename
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    log.info("[classification cross_validation] Trace escrito en %s", output_path)

    extra: Dict[str, Any] = {"trained_model": best_model}
    return best_model, extra


# ---- post_processing_calibration (Isotonic Regression) ----
def post_processing_calibration(
    model: Any,
    tech_cfg: Dict[str, Any],
    ctx: Any,
    output_dir: Path,
    algorithm_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Fit an IsotonicRegression calibrator on the validation split.

    Args:
        model: Trained LightGBM model (from cross_validation).
        tech_cfg: YAML technique configuration.
        ctx: Run context.
        output_dir: Output directory.
        algorithm_config: Not used here, kept for interface consistency.

    Returns:
        Unmodified model and extra artifact dict with calibrator.
    """
    log.debug("[post_processing_calibration] entry")

    try:
        params: Dict[str, Any] = tech_cfg["params"]
        method: str = params["method"]  # debe ser "IsotonicRegression"
        fit_on_validation: bool = params["fit_on_validation_splits"]
        output_filename: str = tech_cfg["output"]
    except KeyError as e:
        log.error("[post_processing_calibration] YAML key missing: %s", e)
        raise

    # Cargar validation split desde el YAML
    try:
        val_data_rel = ctx.config.phases["phase4_data_modeling"]["read_strategy"][
            "input_source"
        ]["val_data"]
        phase3_dir = getattr(ctx, "phase3_dir", ".")
        val_path = Path(phase3_dir) / val_data_rel
        df_val = pd.read_parquet(val_path)
    except Exception as e:
        log.error(
            "[post_processing_calibration] No se pudo cargar validation split: %s", e
        )
        log.warning(
            "[post_processing_calibration] Saltando calibración por falta de datos de validación."
        )
        extra = {"fitted_isotonic_calibrator": None}
        return model, extra

    # Extraer features y target del validation set
    # feature_cols = [c for c in df_val.columns if c != "faulty"]
    # X_val = df_val[feature_cols].values
    # y_val = df_val["faulty"].values

    # Extraer features y target del validation set
    feature_cols = [c for c in df_val.columns if c != "faulty"]
    X_val = df_val[feature_cols]  # <-- ELIMINAR el .values aquí
    y_val = df_val["faulty"].values

    # Obtener predicciones crudas del modelo
    y_pred_raw = model.predict_proba(X_val)[:, 1]

    # Ajustar calibrator isotónico
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_pred_raw, y_val)

    log.info(
        "[post_processing_calibration] IsotonicRegression fitted on %d validation samples.",
        len(y_val),
    )

    # Escribir traza
    trace = {
        "method": method,
        "fit_on_validation_splits": fit_on_validation,
        "n_samples": len(y_val),
        "calibrator_type": "IsotonicRegression",
    }
    output_path = output_dir / output_filename
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")

    extra = {"fitted_isotonic_calibrator": calibrator}
    return model, extra
