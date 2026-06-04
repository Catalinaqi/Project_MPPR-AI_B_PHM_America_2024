from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.mixture import GaussianMixture
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def algorithm_selection(
        df: pd.DataFrame,
        params: dict[str, Any],
        ctx: Any,
        output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return an NGBRegressor configuration dict (no fitting) and log trace.

    This technique corresponds to step_4_1_algorithm_selection.
    It does not transform the dataframe (returns as-is) but returns an
    extra artifact containing the model configuration.
    """
    log.debug("[algorithm_selection] entry – shape=%s", df.shape)

    # Build the model configuration from the YAML params
    model_cfg = {
        "Dist": Normal,
        "Score": LogScore,
        "n_estimators": params.get("n_estimators", 400),
        "learning_rate": params.get("learning_rate", 0.03),
        "Base": params.get("Base", {"type": "DecisionTreeRegressor", "max_depth": 4}),
        "random_state": params.get("random_state", 42),
    }

    # Persist trace JSON
    trace = {
        "library": "ngboost",
        "estimator": "NGBRegressor",
        "model_configured": model_cfg,
    }
    output_path = output_dir / "4.1.modeling.algo_setup_trace.json"
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    log.debug("[algorithm_selection] trace written to %s", output_path)

    extra = {"algorithm_config": model_cfg}
    return df, extra


def cross_validation(
        df: pd.DataFrame,
        params: dict[str, Any],
        ctx: Any,
        output_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Execute GMM-grouped GroupKFold cross validation and train NGBoost per fold.

    Returns the best model (lowest validation NLL) and extra artifacts.
    """
    log.debug("[cross_validation] entry – shape=%s", df.shape)

    # ---------- Configuration ----------
    # CORRECCIÓN: 'params' ya contiene los valores definidos en el YAML bajo la llave 'params'.
    n_splits = params.get("n_splits", 5)
    grouping = params.get("grouping_mechanism", {})

    gmm_features = grouping.get("features", ["oat", "mgt", "ias"])
    n_clusters = grouping.get("n_clusters", 4)
    cov_type = grouping.get("covariance_type", "full")

    # PROTECCIÓN MATEMÁTICA: n_splits nunca puede ser mayor a n_clusters para GroupKFold
    if n_splits > n_clusters:
        log.warning(
            "[cross_validation] CONFLICTO EN YAML: n_splits (%d) es mayor que n_clusters (%d). "
            "Ajustando n_splits a %d de forma automática para evitar que GroupKFold falle.",
            n_splits, n_clusters, n_clusters
        )
        n_splits = n_clusters

    # Extract target and features
    target_col = "trq_margin"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    # ---------- GMM clustering on selected features ----------
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=cov_type, random_state=42)
    gmm_features_idx = [feature_cols.index(f) for f in gmm_features if f in feature_cols]
    if len(gmm_features_idx) < len(gmm_features):
        raise ValueError("Some GMM features not found in dataframe: %s", gmm_features)
    X_gmm = X[:, gmm_features_idx]
    cluster_labels = gmm.fit_predict(X_gmm)

    # Use cluster labels as groups for GroupKFold
    gkf = GroupKFold(n_splits=n_splits)

    # ---------- Training loop ----------
    best_model = None
    best_nll = float("inf")
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=cluster_labels)):
        log.info("[cross_validation] Fold %d/%d", fold_idx+1, n_splits)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Use default NGBRegressor params (can be extended via YAML if needed)
        model = NGBRegressor(
            Dist=Normal,
            Score=LogScore,
            n_estimators=400,
            learning_rate=0.03,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Evaluate NLL
        y_pred_dist = model.pred_dist(X_val)
        nll = -y_pred_dist.logpdf(y_val).mean()
        rmse = np.sqrt(((y_pred_dist.mean - y_val) ** 2).mean())
        fold_results.append({"fold": fold_idx, "nll": nll, "rmse": rmse})

        if nll < best_nll:
            best_nll = nll
            best_model = model

        log.info("[cross_validation] Fold %d NLL=%.4f RMSE=%.4f", fold_idx+1, nll, rmse)

    # ---------- Persist trace ----------
    trace = {
        "n_folds": n_splits,
        "gmm_features": gmm_features,
        "gmm_n_clusters": n_clusters,
        "fold_results": fold_results,
        "best_fold_nll": best_nll,
    }
    output_path = output_dir / "4.2.training.cv_fold_execution_trace.json"
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")

    extra = {"trained_model": best_model}
    return best_model, extra