from __future__ import annotations

from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from sklearn.inspection import permutation_importance as sk_perm_importance

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.pipeline.utils.context_facade_common import RunContext
from phm_america_2024.reporting.plots_generator_reporting import (
    plot_feature_importance,
    plot_permutation_importance
)

log = get_logger(__name__)


def feature_importance(
        df: pd.DataFrame,
        tech_cfg: Dict[str, Any],
        ctx: RunContext,
        output_dir: Any, # Ignorado, el registro se encarga del path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute model native feature importance extraction and visualization logic."""
    log.debug("[feature_importance] entry")

    model: NGBRegressor = getattr(ctx, "model", None)
    if model is None:
        raise RuntimeError("Context is missing the trained NGBoost model.")

    target_col: str = getattr(ctx, "target_col", "trq_margin")
    X_test: pd.DataFrame = df.drop(columns=[target_col], errors="ignore")
    feature_names: List[str] = list(X_test.columns)

    params: Dict[str, Any] = tech_cfg.get("params", {})
    strategy: str = params.get("strategy", "model_native")
    top_k: int = params.get("top_k", 15)

    importance_data: Dict[str, Any] = {}
    fig = None

    if strategy == "model_native":
        tree_imps: List[np.ndarray] = []
        for stage in model.ensemble:
            for tree in stage:
                if hasattr(tree, "feature_importances_"):
                    tree_imps.append(tree.feature_importances_)

        if tree_imps:
            avg: np.ndarray = np.mean(tree_imps, axis=0)
            sorted_idx: np.ndarray = np.argsort(avg)[::-1][:top_k]
            importance_data = {feature_names[i]: float(avg[i]) for i in sorted_idx}
            log.info("[feature_importance] successfully calculated top %d features", top_k)
        else:
            importance_data = {"error": "No tree importances found in model ensemble."}
    else:
        importance_data = {"error": f"Unknown strategy '{strategy}'."}

    # Generar la figura en memoria (sin guardarla)
    if "error" not in importance_data:
        fig = plot_feature_importance(importance_data, top_k)

    log.debug("[feature_importance] exit")

    # RETORNO PURO DE OBJETOS EN MEMORIA
    return df, {
        "feature_importance_data": importance_data,
        "feature_importance_plot": fig
    }


def permutation_importance(
        df: pd.DataFrame,
        tech_cfg: Dict[str, Any],
        ctx: RunContext,
        output_dir: Any, # Ignorado
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute permutation importance evaluation and visualization logic."""
    log.debug("[permutation_importance] entry")

    model: NGBRegressor = getattr(ctx, "model", None)
    y_test: np.ndarray = getattr(ctx, "y_true", None)

    if model is None or y_test is None:
        raise RuntimeError("Context is missing the trained model or true targets.")

    target_col: str = getattr(ctx, "target_col", "trq_margin")
    X_test: pd.DataFrame = df.drop(columns=[target_col], errors="ignore")
    feature_names: List[str] = list(X_test.columns)

    params: Dict[str, Any] = tech_cfg.get("params", {})
    n_repeats: int = params.get("n_repeats", 5)
    scoring: str = params.get("scoring", "neg_log_likelihood")

    def nll_scorer(estimator: Any, X: pd.DataFrame, y: np.ndarray) -> float:
        dist = estimator.pred_dist(X)
        return -dist.logpdf(y).sum()

    scoring_func = nll_scorer if scoring == "neg_log_likelihood" else None

    log.info("[permutation_importance] running permutation analysis (repeats=%d)", n_repeats)
    perm_result = sk_perm_importance(
        model, X_test, y_test,
        scoring=scoring_func,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=1,
    )

    perm_data: Dict[str, Any] = {
        "feature_names": feature_names,
        "importances_mean": perm_result.importances_mean.tolist(),
        "importances_std": perm_result.importances_std.tolist(),
        "importances": perm_result.importances.tolist(),
    }

    # Generar la figura en memoria (sin guardarla)
    fig = plot_permutation_importance(perm_data, scoring, top_k=15)

    log.debug("[permutation_importance] exit")

    # RETORNO PURO DE OBJETOS EN MEMORIA
    return df, {
        "permutation_data": perm_data,
        "permutation_plot": fig
    }