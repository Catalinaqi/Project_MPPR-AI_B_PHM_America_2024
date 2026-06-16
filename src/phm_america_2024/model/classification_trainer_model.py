import lightgbm as lgb

# from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from typing import Any, Dict
from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def algorithm_selection(df, tech_cfg, ctx, output_dir):
    params = tech_cfg["params"]
    model_cfg = {}
    output_filename = tech_cfg["output"]
    output_path = output_dir / output_filename
    extra = {"algorithm_config": model_cfg}
    return df, extra


def cross_validation(df, tech_cfg, ctx, output_dir, algorithm_config):
    try:
        # Step 1: Extract parameters directly from YAML config (Zero hardcoding)
        params: Dict[str, Any] = tech_cfg["params"]
        n_splits: int = params["n_splits"]
        strategy: str = params["strategy"]
        if strategy != "StratifiedKFold":
            log.error("[cross_validation] unsupported strategy: %s", strategy)
            raise ValueError(f"Unsupported strategy: {strategy}")
        # grouping: Dict[str, Any] = params["grouping_mechanism"]
        # gmm_features: list[str] = grouping["features"]
        # n_clusters: int = grouping["n_clusters"]
        # cov_type: str = grouping["covariance_type"]
        # random_seed: int = params["random_seed"]
        target_col: str = params["target_variable"]
        output_filename: str = tech_cfg["output"]
    except KeyError as e:
        log.error("[cross_validation] YAML key missing in configuration: %s", e)
        raise

    skf = StratifiedKFold(n_splits=n_splits)

    feature_cols: list[str] = [c for c in df.columns if c != target_col]
    X: np.ndarray = df[feature_cols].values
    Y: np.ndarray = df[target_col].values

    fold_results = []
    best_result = None
    best_model = None
    for fold_index, (train_index, test_index) in enumerate(skf.split(X, Y)):
        gbm_model = lgb.LGBMClassifier()
        #        decision_tree = DecisionTreeClassifier()

        X_train, X_val = X[train_index], X[test_index]
        Y_train, Y_val = Y[train_index], Y[test_index]

        gbm_model.fit(X_train, Y_train)

        score = gbm_model.score(X_val, Y_val)

        if best_result is None or score > best_result:
            best_result = score
            best_model = gbm_model

        fold_results.append(
            {
                "fold": fold_index,
                "score": score,
            }
        )
    trace_result = {
        "n_folds": n_splits,
        "best_model": best_model,
        "fold_results": fold_results,
    }
    extra = {"trained_model": best_model}
    return best_model, extra
