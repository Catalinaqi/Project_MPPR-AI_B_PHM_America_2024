"""
Module: regression_trainer_model.py
Core algorithm: NGBoost (Natural Gradient Boosting) with Normal distribution and LogScore.
Libraries: ngboost, scikit-learn (GaussianMixture, GroupKFold, DecisionTreeRegressor), joblib.

Flow:
1. algorithm_selection():
   - Extracts hyperparameters from the YAML config.
   - Builds a model_cfg dict containing the Normal / LogScore classes and the
     hyperparameters (including their uncertainty ranges for random search).
   - Saves a JSON trace file to the output directory.
   - Returns the DataFrame unchanged plus the model configuration.

2. cross_validation():
   - Resolves the training set path from ctx.config (phase-level read_strategy).
   - Replaces infinities with NaN and drops rows with missing values.
   - Selects model input features: either an explicit list declared in the
     YAML config ('features' param), or falls back to "all columns except
     target" for backward compatibility.
   - Uses GaussianMixture to cluster rows by flight regime (grouping features).
   - Applies GroupKFold respecting those clusters as groups.
   - For each hyperparameter-sampling iteration:
       - Draws a randomized hyperparameter set within the configured
         uncertainty ranges.
       - Fits and evaluates NGBoost on each fold (optionally in parallel via
         joblib, optionally with early stopping against each fold's own
         validation split).
       - Averages NLL/RMSE across folds for that iteration.
   - Selects the hyperparameter configuration with the lowest average NLL.
   - Refits the best configuration on the full training set.
   - Saves the full cross-validation trace (all folds, all iterations) as JSON.
   - Returns the best trained model and the extra artifacts dict.

Imports:
- json, pathlib, typing: serialization, paths, type hints.
- time: wall-clock timing instrumentation (per-fold and total).
- numpy, pandas: data manipulation.
- joblib.Parallel/delayed: parallel execution of independent folds.
- ngboost.NGBRegressor: probabilistic model.
- ngboost.distns.Normal: Normal distribution for the target.
- ngboost.scores.LogScore: log-likelihood-based scoring rule.
- sklearn.mixture.GaussianMixture: clustering used to group similar flights.
- sklearn.model_selection.GroupKFold: K-fold that respects groups.
- sklearn.tree.DecisionTreeRegressor: base learner for NGBoost.
- logging_adapter_common: project logger.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)

# step_4_1_algorithm_selection -> builds the NGBoost hyperparameter config
# step_4_2_model_training      -> GMM-grouped GroupKFold training loop (this module's core)

# Soft time budget for the whole cross_validation() run, used only for
# logging a live projection — it does NOT stop or alter execution.
TARGET_TOTAL_SECONDS = 30 * 60  # ~30 minutes


def algorithm_selection(
        df: pd.DataFrame,
        tech_cfg: Dict[str, Any],
        ctx: Any,
        output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build an NGBRegressor configuration dict (no fitting yet) and log a trace.

    Args:
        df: Input DataFrame (passed through unchanged).
        tech_cfg: YAML technique configuration containing 'params' and 'output'.
        ctx: Run context holding shared execution state.
        output_dir: Output directory path.

    Returns:
        Unmodified DataFrame and extra artifact dict with the model configuration.
    """
    log.debug("[algorithm_selection] entry – shape=%s", df.shape)

    try:
        # Step 1: extract hyperparameters from YAML config (zero hardcoding)
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

    log.info(
        "[algorithm_selection] configured NGBRegressor: n_estimators=%s, "
        "learning_rate=%s, minibatch_frac=%s, Base.max_depth=%s",
        model_cfg["n_estimators"],
        model_cfg["learning_rate"],
        model_cfg["minibatch_frac"],
        model_cfg["Base"].get("max_depth"),
    )

    trace: Dict[str, Any] = {
        "library": "ngboost",
        "estimator": "NGBRegressor",
        "model_configured": model_cfg,
    }

    output_path: Path = output_dir / output_filename

    # Step 2: persist the trace to disk
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    log.info("[algorithm_selection] trace written to %s", output_path)

    extra: Dict[str, Any] = {"algorithm_config": model_cfg}

    log.debug("[algorithm_selection] exit")
    return df, extra


def _resolve_feature_columns(
        df: pd.DataFrame, target_col: str, params: Dict[str, Any]
) -> List[str]:
    """Resolve which columns are used as model input (X).

    If 'features' is explicitly declared in the YAML config, use that list
    (validating that every declared column actually exists in the data).
    Otherwise, fall back to the original behavior: every column except the
    target. The fallback is kept for backward compatibility, but logs a
    warning since it can silently include unintended columns (e.g. a
    classification label, or unvetted engineered features) and inflate
    training time.
    """
    declared_features: Optional[List[str]] = params.get("features")

    if declared_features:
        missing = set(declared_features) - set(df.columns)
        if missing:
            log.error(
                "[cross_validation] declared 'features' not found in training data: %s",
                missing,
            )
            raise ValueError(f"Missing declared features in training data: {missing}")
        log.info(
            "[cross_validation] using explicit feature list from config (%d columns): %s",
            len(declared_features),
            declared_features,
        )
        return declared_features

    fallback_features = [c for c in df.columns if c != target_col]
    log.warning(
        "[cross_validation] no 'features' declared in config – defaulting to "
        "ALL columns except target (%d columns). This may unintentionally "
        "include columns like 'faulty' or unvetted interaction terms as "
        "model input, and increases training time proportionally to the "
        "number of columns.",
        len(fallback_features),
    )
    return fallback_features


def _fit_and_evaluate_fold(
        fold_idx: int,
        n_splits: int,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        tree_depth: int,
        min_samples_leaf: int,
        n_estimators: int,
        learning_rate: float,
        minibatch_frac: float,
        random_state: int,
        early_stopping_rounds: Optional[int],
) -> Dict[str, Any]:
    """Fit NGBoost on a single fold and evaluate it on its own validation split.

    Isolated as a standalone function so it can be dispatched in parallel
    (via joblib) across folds that share the same sampled hyperparameters.
    """
    fold_start = time.time()

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    log.debug(
        "[cross_validation][fold %d/%d] train_rows=%d val_rows=%d "
        "max_depth=%d min_samples_leaf=%d n_estimators=%d lr=%.4f minibatch_frac=%.3f",
        fold_idx + 1, n_splits, len(train_idx), len(val_idx),
        tree_depth, min_samples_leaf, n_estimators, learning_rate, minibatch_frac,
        )

    # Step A: instantiate the base learner (weak tree used at every boosting stage)
    base_learner = DecisionTreeRegressor(
        max_depth=tree_depth, min_samples_leaf=min_samples_leaf, random_state=random_state
    )

    # Step B: instantiate the probabilistic model for this fold
    model = NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        Base=base_learner,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        minibatch_frac=minibatch_frac,
        random_state=random_state,
    )

    # Step C: fit. If early_stopping_rounds is configured, pass this fold's
    # own validation split so NGBoost can stop boosting once the
    # validation NLL stops improving, instead of always running the full
    # n_estimators (this is one of the main time-saving levers).
    if early_stopping_rounds:
        model.fit(
            X_train, y_train,
            X_val=X_val, Y_val=y_val,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        model.fit(X_train, y_train)

    # Step D: evaluate on the held-out validation split for this fold
    y_pred_dist = model.pred_dist(X_val)
    nll: float = -y_pred_dist.logpdf(y_val).mean()

    pred_means = y_pred_dist.mean()
    errors = pred_means - y_val
    rmse: float = float(np.sqrt(np.mean(errors ** 2)))

    fold_seconds = time.time() - fold_start

    log.info(
        "[cross_validation][fold %d/%d] done in %.1fs – NLL=%.4f RMSE=%.4f",
        fold_idx + 1, n_splits, fold_seconds, nll, rmse,
        )

    return {"fold": fold_idx, "nll": nll, "rmse": rmse, "seconds": fold_seconds}


def cross_validation(
        df: pd.DataFrame,  # -> *_internal_train.parquet
        tech_cfg: Dict[str, Any],
        ctx: Any,
        output_dir: Path,
        algorithm_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Run GMM-grouped GroupKFold cross-validation and train NGBoost per fold.

    Args:
        df: Input DataFrame (training split, already scaled).
        tech_cfg: YAML technique configuration containing 'params' and 'output'.
        ctx: Run context holding shared execution state.
        output_dir: Output directory path.
        algorithm_config: Injected configuration from step 4.1.

    Returns:
        Best trained model (lowest validation NLL) and extra artifacts.
    """
    run_start = time.time()
    log.debug("[cross_validation] entry – shape=%s", df.shape)

    # Step 0: resolve and log the source dataset path (informational only —
    # the actual DataFrame is already loaded and passed in as `df`)
    try:
        train_data_name = ctx.config.phases["phase4_data_modeling"]["read_strategy"][
            "input_source"
        ]["train_data"]
        log.info("[cross_validation] train_data_name: %s", train_data_name)

        phase3_dir = getattr(ctx, "phase3_dir", "UnknownDirectory")
        full_train_path = Path(phase3_dir) / train_data_name
        log.info("[cross_validation] source dataset (train): %s", full_train_path)
    except Exception as e:
        log.warning(
            "[cross_validation] could not resolve dataset path for logging purposes: %s",
            e,
        )

    # Step 1: extract parameters from YAML config (zero hardcoding)
    try:
        params: Dict[str, Any] = tech_cfg["params"]

        random_seed: int = params["random_seed"]
        n_iterations: int = params["iterations"]
        strategy: str = params["strategy"]
        if strategy != "GroupKFold":
            log.error("[cross_validation] unsupported strategy: %s", strategy)
            raise ValueError(f"Unsupported strategy: {strategy}")

        target_col: str = params["target_variable"]
        n_splits: int = params["n_splits"]
        grouping: Dict[str, Any] = params["grouping_mechanism"]

        gmm_features: List[str] = grouping["features"]
        n_clusters: int = grouping["n_clusters"]
        cov_type: str = grouping["covariance_type"]

        # Optional optimization params — safe defaults preserve the original
        # sequential, non-early-stopped behavior if not declared in the YAML.
        n_jobs: int = params.get("n_jobs", 1)
        early_stopping_rounds: Optional[int] = params.get("early_stopping_rounds")

        output_filename: str = tech_cfg["output"]
    except KeyError as e:
        log.error("[cross_validation] YAML key missing in configuration: %s", e)
        raise

    log.info(
        "[cross_validation] config loaded: iterations=%d n_splits=%d n_jobs=%d "
        "early_stopping_rounds=%s target=%s",
        n_iterations, n_splits, n_jobs, early_stopping_rounds, target_col,
    )

    # Step 2: validate mathematical boundaries between splits and clusters
    if n_splits > n_clusters:
        log.warning(
            "[cross_validation] shift detected: n_splits (%d) > n_clusters (%d). "
            "Adjusting n_splits to %d.",
            n_splits, n_clusters, n_clusters,
        )
        n_splits = n_clusters

    # Step 3: sanitize infinite values to NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    initial_rows: int = len(df)

    # Step 4: drop rows with missing values
    df = df.dropna()
    dropped_rows: int = initial_rows - len(df)
    if dropped_rows > 0:
        log.warning(
            "[cross_validation] dropped %d rows containing NaN or Infinity (%.2f%% of input).",
            dropped_rows, 100 * dropped_rows / initial_rows,
                          )
    log.debug("[cross_validation] rows after sanitization: %d", len(df))

    # Step 5: resolve model input features (explicit config list, or fallback)
    feature_cols: List[str] = _resolve_feature_columns(df, target_col, params)
    X: np.ndarray = df[feature_cols].values
    y: np.ndarray = df[target_col].values
    log.info("[cross_validation] training matrix shape: X=%s y=%s", X.shape, y.shape)

    # Step 6: instantiate the GMM clusterer used to group rows by flight regime
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type=cov_type, random_state=random_seed
    )

    gmm_features_idx: List[int] = [
        feature_cols.index(f) for f in gmm_features if f in feature_cols
    ]
    if len(gmm_features_idx) < len(gmm_features):
        log.error(
            "[cross_validation] GMM grouping features not found in the resolved "
            "feature list: %s. Note that grouping_mechanism.features must be a "
            "subset of the model input 'features' list.",
            gmm_features,
        )
        raise ValueError(f"Some GMM features not found in dataframe: {gmm_features}")

    X_gmm: np.ndarray = X[:, gmm_features_idx]

    # Step 7: assign flight-regime cluster labels (used as GroupKFold groups)
    cluster_labels: np.ndarray = gmm.fit_predict(X_gmm)
    log.info(
        "[cross_validation] GMM clustering done: n_clusters=%d, cluster sizes=%s",
        n_clusters,
        np.bincount(cluster_labels).tolist(),
    )

    # Step 8: instantiate the group-aware K-fold splitter
    gkf = GroupKFold(n_splits=n_splits)

    if not algorithm_config:
        log.error(
            "[cross_validation] algorithm_config missing. Step 4.1 must be executed prior."
        )
        raise ValueError("algorithm_config is required but was None.")

    # Step 9: extract base hyperparameters (and their uncertainty ranges)
    # injected from step 4.1
    try:
        tree_depth: int = algorithm_config["Base"]["max_depth"]
        min_samples: int = algorithm_config["Base"]["min_samples_leaf"]
        n_est: int = algorithm_config["n_estimators"]
        lr: float = algorithm_config["learning_rate"]
        mini_batch: float = algorithm_config["minibatch_frac"]
        algo_seed: int = algorithm_config["random_state"]
        tree_depth_uncertainty: int = algorithm_config["Base"]["max_depth_uncertainty"]
        min_samples_uncertainty: int = algorithm_config["Base"]["min_samples_leaf_uncertainty"]
        n_est_uncertainty: int = algorithm_config["n_estimators_uncertainty"]
        lr_uncertainty: float = algorithm_config["learning_rate_uncertainty"]
        mini_batch_uncertainty: float = algorithm_config["minibatch_frac_uncertainty"]
    except KeyError as e:
        log.error(
            "[cross_validation] YAML key missing in injected algorithm_config: %s", e
        )
        raise

    log.info(
        "[cross_validation] NGBoost base hyperparameters: n_estimators=%d (±%d), "
        "learning_rate=%.4f (±%.4f), max_depth=%d (±%d)",
        n_est, n_est_uncertainty, lr, lr_uncertainty, tree_depth, tree_depth_uncertainty,
    )

    best_model: Any = None
    best_nll: float = float("inf")
    best_config: Optional[Dict[str, Any]] = None
    # Accumulates fold results across ALL iterations, for the persisted trace file.
    all_fold_results: List[Dict[str, Any]] = []

    total_planned_fits = n_iterations * n_splits
    completed_fits = 0

    # Step 10: hyperparameter-sampling loop. Each iteration draws one
    # randomized configuration (within the configured uncertainty ranges)
    # and evaluates it via GroupKFold cross-validation.
    for iteration_idx in range(n_iterations):
        iteration_start = time.time()

        tree_depth_u: int = tree_depth + random.randrange(-tree_depth_uncertainty, tree_depth_uncertainty)
        min_samples_u: int = min_samples + random.randrange(-min_samples_uncertainty, min_samples_uncertainty)
        n_est_u: int = n_est + random.randrange(-n_est_uncertainty, n_est_uncertainty)
        lr_u: float = lr + ((random.random() - 0.5) * 2 * lr_uncertainty)
        mini_batch_u: float = mini_batch + ((random.random() - 0.5) * 2 * mini_batch_uncertainty)

        log.info(
            "[cross_validation] iteration %d/%d – sampled config: max_depth=%d "
            "min_samples_leaf=%d n_estimators=%d lr=%.4f minibatch_frac=%.3f",
            iteration_idx + 1, n_iterations,
            tree_depth_u, min_samples_u, n_est_u, lr_u, mini_batch_u,
            )

        splits = list(gkf.split(X, y, groups=cluster_labels))

        # Step 10a: fit+evaluate every fold for this iteration. Folds are
        # independent given fixed hyperparameters, so they can run in
        # parallel via joblib. n_jobs=1 (default) preserves the original
        # sequential behavior exactly.
        fold_results: List[Dict[str, Any]] = Parallel(n_jobs=n_jobs)(
            delayed(_fit_and_evaluate_fold)(
                fold_idx, n_splits, X, y, train_idx, val_idx,
                tree_depth_u, min_samples_u, n_est_u, lr_u, mini_batch_u,
                algo_seed, early_stopping_rounds,
            )
            for fold_idx, (train_idx, val_idx) in enumerate(splits)
        )
        all_fold_results.extend(fold_results)
        completed_fits += len(fold_results)

        # Step 10b: average this iteration's fold metrics (BUGFIX: fold_results
        # is now local to this iteration — previously it was never reset,
        # so nll_avg from iteration 2 onward was silently contaminated with
        # fold results from earlier, unrelated hyperparameter configurations)
        nll_list = [f["nll"] for f in fold_results]
        nll_avg = sum(nll_list) / len(nll_list)

        iteration_seconds = time.time() - iteration_start
        elapsed_total = time.time() - run_start

        log.info(
            "[cross_validation] iteration %d/%d done in %.1fs – avg NLL=%.4f (best so far=%.4f)",
            iteration_idx + 1, n_iterations, iteration_seconds, nll_avg,
            best_nll if best_nll != float("inf") else nll_avg,
            )

        # Step 10c: live time projection against the ~30 min soft target.
        # This is informational only — it does not alter execution.
        if completed_fits > 0:
            avg_seconds_per_fit = elapsed_total / completed_fits
            projected_total = avg_seconds_per_fit * total_planned_fits
            log.info(
                "[cross_validation] time projection: elapsed=%.1fs, "
                "avg=%.1fs/fit, projected_total=%.1fs (%.1f min) vs target=%.0f min",
                elapsed_total, avg_seconds_per_fit, projected_total,
                projected_total / 60, TARGET_TOTAL_SECONDS / 60,
                )
            if projected_total > TARGET_TOTAL_SECONDS * 1.5:
                log.warning(
                    "[cross_validation] projected total runtime (%.1f min) is "
                    "well above the ~30 min target. Consider reducing "
                    "'iterations', 'n_estimators', enabling/tightening "
                    "'early_stopping_rounds', increasing 'n_jobs', or "
                    "reducing the number of declared 'features'.",
                    projected_total / 60,
                    )

        # Step 10d: track the best configuration seen so far (lowest avg NLL)
        if nll_avg < best_nll:
            best_nll = nll_avg
            best_config = {
                "Base": {
                    "max_depth": tree_depth_u,
                    "min_samples_leaf": min_samples_u,
                },
                "n_estimators": n_est_u,
                "learning_rate": lr_u,
                "minibatch_frac": mini_batch_u,
                "random_state": algo_seed,
            }
            log.info(
                "[cross_validation] new best config found at iteration %d/%d (avg NLL=%.4f)",
                iteration_idx + 1, n_iterations, best_nll,
                )

    if best_config is None:
        log.error("[cross_validation] no valid configuration was found across all iterations.")
        raise RuntimeError("cross_validation completed without a valid best_config.")

    # Step 11: refit the best configuration on the FULL training set (not
    # just one fold) — this is the model that gets persisted and consumed
    # downstream by evaluation/interpretation.
    log.info(
        "[cross_validation] refitting best configuration on full training set "
        "(%d rows, %d features): %s",
        X.shape[0], X.shape[1], best_config,
    )
    refit_start = time.time()

    best_model_base = DecisionTreeRegressor(
        max_depth=best_config["Base"]["max_depth"],
        min_samples_leaf=best_config["Base"]["min_samples_leaf"],
        random_state=best_config["random_state"],
    )
    best_model = NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        Base=best_model_base,
        n_estimators=best_config["n_estimators"],
        learning_rate=best_config["learning_rate"],
        minibatch_frac=best_config["minibatch_frac"],
        random_state=best_config["random_state"],
    )
    best_model.fit(X, y)

    refit_seconds = time.time() - refit_start
    log.info("[cross_validation] final refit completed in %.1fs", refit_seconds)

    total_seconds = time.time() - run_start
    log.info(
        "[cross_validation] TOTAL runtime: %.1fs (%.1f min) across %d fits "
        "(%d iterations x %d folds) + 1 final refit",
        total_seconds, total_seconds / 60, completed_fits, n_iterations, n_splits,
                       )

    # Step 12: persist the full trace (all folds, all iterations)
    trace_results: Dict[str, Any] = {
        "n_folds": n_splits,
        "n_iterations": n_iterations,
        "n_features": X.shape[1],
        "feature_cols": feature_cols,
        "gmm_features": gmm_features,
        "gmm_n_clusters": n_clusters,
        "n_jobs": n_jobs,
        "early_stopping_rounds": early_stopping_rounds,
        "fold_results": all_fold_results,
        "best_fold_nll": best_nll,
        "best_config": best_config,
        "total_runtime_seconds": total_seconds,
        "refit_seconds": refit_seconds,
    }

    output_path = output_dir / output_filename
    output_path.write_text(
        json.dumps(trace_results, indent=2, default=str), encoding="utf-8"
    )
    log.info("[cross_validation] trace written to %s", output_path)

    extra: Dict[str, Any] = {"trained_model": best_model}

    log.debug("[cross_validation] exit")
    return best_model, extra