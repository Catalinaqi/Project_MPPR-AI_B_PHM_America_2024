from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)

# step_4_2_model_training -> Entrenamiento con K-Fold


def algorithm_selection(
    df: pd.DataFrame,
    tech_cfg: Dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return an NGBRegressor configuration dict (no fitting) and log trace.

    Args:
        df: Input DataFrame.
        tech_cfg: YAML technique configuration containing params and output.
        ctx: Run context holding shared execution state.
        output_dir: Output directory path.

    Returns:
        Unmodified DataFrame and extra artifact dict with configuration.
    """
    log.debug("[algorithm_selection] entry – shape=%s", df.shape)

    try:
        # Step 1: Extract parameters directly from YAML config (Zero hardcoding)
        params: Dict[str, Any] = tech_cfg["params"]
        model_cfg: Dict[str, Any] = {
            "Dist": Normal,
            "Score": LogScore,
            "n_estimators": params["n_estimators"],
            "learning_rate": params["learning_rate"],
            "minibatch_frac": params["minibatch_frac"],
            "Base": params["Base"],
            "random_state": params["random_state"],
        }
        output_filename: str = tech_cfg["output"]
    except KeyError as e:
        log.error("[algorithm_selection] YAML key missing in configuration: %s", e)
        raise

    trace: Dict[str, Any] = {
        "library": "ngboost",
        "estimator": "NGBRegressor",
        "model_configured": model_cfg,
    }

    output_path: Path = output_dir / output_filename

    # Step 2: CALL write_text() — serialize trace to disk
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    log.info("[algorithm_selection] trace written to %s", output_path)

    extra: Dict[str, Any] = {"algorithm_config": model_cfg}

    log.debug("[algorithm_selection] exit")
    return df, extra


def cross_validation(
    df: pd.DataFrame,  # -> *_internal_train.parquet
    tech_cfg: Dict[str, Any],
    ctx: Any,
    output_dir: Path,
    algorithm_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Execute GMM-grouped GroupKFold cross validation and train NGBoost per fold.

    Args:
        df: Input DataFrame.
        tech_cfg: YAML technique configuration containing params and output.
        ctx: Run context holding shared execution state.
        output_dir: Output directory path.
        algorithm_config: Injected configuration from prior Step 4.1.

    Returns:
        Best trained model (lowest validation NLL) and extra artifacts.
    """

    log.debug("[cross_validation] entry – shape=%s", df.shape)

    try:
        # Extraemos el path de train desde la configuración global
        train_data_name = ctx.config.phases["phase4_data_modeling"]["read_strategy"][
            "input_source"
        ]["train_data"]

        log.info("[cross_validation] train_data_name: %s", train_data_name)

        phase3_dir = getattr(ctx, "phase3_dir", "Directorio_Desconocido")
        full_train_path = Path(phase3_dir) / train_data_name
        log.info("[cross_validation] DATASET ORIGEN (Train): %s", full_train_path)

    except Exception as e:
        log.warning(
            "[cross_validation] No se pudo resolver la ruta del dataset en los logs: %s",
            e,
        )

    try:
        # Step 1: Extract parameters directly from YAML config (Zero hardcoding)
        # ----- params-------------
        params: Dict[str, Any] = tech_cfg["params"]
        # ----- params start------------
        random_seed: int = params["random_seed"]
        strategy: str = params["strategy"]
        if strategy != "GroupKFold":
            log.error("[cross_validation] unsupported strategy: %s", strategy)
            raise ValueError(f"Unsupported strategy: {strategy}")
        target_col: str = params["target_variable"]
        n_splits: int = params["n_splits"]
        grouping: Dict[str, Any] = params["grouping_mechanism"]

        gmm_features: list[str] = grouping["features"]
        n_clusters: int = grouping["n_clusters"]
        cov_type: str = grouping["covariance_type"]
        # ----- params end-------------

        output_filename: str = tech_cfg["output"]
    except KeyError as e:
        log.error("[cross_validation] YAML key missing in configuration: %s", e)
        raise

    # Step 2: Validate mathematical boundaries
    if n_splits > n_clusters:
        log.warning(
            "[cross_validation] shift detected: n_splits (%d) > n_clusters (%d). Adjusting n_splits to %d.",
            n_splits,
            n_clusters,
            n_clusters,
        )
        n_splits = n_clusters

    # Step 3: CALL replace() — sanitize infinite values to NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    initial_rows: int = len(df)

    # Step 4: CALL dropna() — execute row dropping for missing values
    df = df.dropna()
    dropped_rows: int = initial_rows - len(df)

    if dropped_rows > 0:
        log.warning(
            "[cross_validation] Dropped %d rows containing NaN or Infinity.",
            dropped_rows,
        )

    feature_cols: list[str] = [c for c in df.columns if c != target_col]
    X: np.ndarray = df[feature_cols].values
    y: np.ndarray = df[target_col].values

    # Step 5: CALL GaussianMixture() — instantiate GMM clusterer
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type=cov_type, random_state=random_seed
    )

    gmm_features_idx: list[int] = [
        feature_cols.index(f) for f in gmm_features if f in feature_cols
    ]
    if len(gmm_features_idx) < len(gmm_features):
        log.error(
            "[cross_validation] numerical failure / missing features: %s", gmm_features
        )
        raise ValueError(f"Some GMM features not found in dataframe: {gmm_features}")

    X_gmm: np.ndarray = X[:, gmm_features_idx]

    # Step 6: CALL fit_predict() — assign flight regime clusters
    cluster_labels: np.ndarray = gmm.fit_predict(X_gmm)

    # Step 7: CALL GroupKFold() — instantiate folder
    gkf = GroupKFold(n_splits=n_splits)

    if not algorithm_config:
        log.error(
            "[cross_validation] algorithm_config missing. Step 4.1 must be executed prior."
        )
        raise ValueError("algorithm_config is required but was None.")

    try:
        # Step 8: Extract model hyper-parameters from injected config
        tree_depth: int = algorithm_config["Base"]["max_depth"]
        min_samples: int = algorithm_config["Base"]["min_samples_leaf"]
        n_est: int = algorithm_config["n_estimators"]
        lr: float = algorithm_config["learning_rate"]
        mini_batch: float = algorithm_config["minibatch_frac"]
        algo_seed: int = algorithm_config["random_state"]
    except KeyError as e:
        log.error(
            "[cross_validation] YAML key missing in injected algorithm_config: %s", e
        )
        raise

    log.info(
        "[cross_validation] NGBoost init: estimators=%d, lr=%s, depth=%d",
        n_est,
        lr,
        tree_depth,
    )

    best_model: Any = None
    best_nll: float = float("inf")
    fold_results: list[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X, y, groups=cluster_labels)
    ):
        log.info("[cross_validation] Fold %d/%d", fold_idx + 1, n_splits)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Step 9: CALL DecisionTreeRegressor() — instantiate robust base learner
        base_learner = DecisionTreeRegressor(
            max_depth=tree_depth, min_samples_leaf=min_samples, random_state=algo_seed
        )

        # Step 10: CALL NGBRegressor() — instantiate probabilistic model
        model = NGBRegressor(
            Dist=Normal,
            Score=LogScore,
            Base=base_learner,
            n_estimators=n_est,
            learning_rate=lr,
            minibatch_frac=mini_batch,
            random_state=algo_seed,
        )

        # Step 11: CALL fit() — train model fold
        # internal_train: Se usa exclusivamente dentro de la función model.fit(X_train, y_train).
        # Es el material de estudio del modelo
        model.fit(X_train, y_train)

        # Step 12: CALL pred_dist() — generate normal distribution parameters
        y_pred_dist = model.pred_dist(X_val)

        # Step 13: CALL logpdf() — evaluate negative log likelihood
        # internal_val: Se usa inmediatamente después en model.pred_dist(X_val).
        # Es el "examen de prueba" que le haces al modelo para calcular el RMSE y el NLL base.
        nll: float = -y_pred_dist.logpdf(y_val).mean()

        # Obtenemos el array de medias (las predicciones puntuales reales)
        pred_means = y_pred_dist.mean()

        # Calculamos RMSE usando numpy puramente para evitar problemas de tipos
        errors = pred_means - y_val
        rmse: float = np.sqrt(np.mean(errors**2))

        fold_results.append({"fold": fold_idx, "nll": nll, "rmse": rmse})

        if nll < best_nll:
            best_nll = nll
            best_model = model

        log.info(
            "[cross_validation] Fold %d results: NLL=%.4f RMSE=%.4f",
            fold_idx + 1,
            nll,
            rmse,
        )

    trace_results: Dict[str, Any] = {
        "n_folds": n_splits,
        "gmm_features": gmm_features,
        "gmm_n_clusters": n_clusters,
        "fold_results": fold_results,
        "best_fold_nll": best_nll,
    }

    output_path = output_dir / output_filename

    # Step 14: CALL write_text() — serialize cross validation trace to disk
    output_path.write_text(
        json.dumps(trace_results, indent=2, default=str), encoding="utf-8"
    )
    log.info("[cross_validation] trace written to %s", output_path)

    extra: Dict[str, Any] = {"trained_model": best_model}

    log.debug("[cross_validation] exit")
    return best_model, extra
