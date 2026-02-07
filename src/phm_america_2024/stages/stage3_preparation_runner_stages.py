# src/phm_america_2024/stages/stage3_preparation_runner_stages.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.core.resolve_path_utils_core import resolve_path
from phm_america_2024.data.load_utils_data import read_csv, safe_inner_join_x_y
from phm_america_2024.features.split_service_features import HoldoutConfig, split_holdout
from phm_america_2024.features.transforms_service_features import PreprocessConfig, build_numeric_preprocessor

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Stage 3 (Data Preparation): join X+Y, clean (impute), transform (scale),
# split (holdout), and output final matrices ready for Stage4.
#
# Program flow expectation:
# - Pipeline runner calls run_stage3(...) and receives train/valid sets + preprocessor.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Stage runner + feature services
# =============================================================================


def run_stage3(cfg: Dict[str, Any], variables: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    stage_cfg = cfg["stages"]["stage3_preparation"]
    if not stage_cfg.get("enabled", False):
        log.info("Stage3 disabled.")
        return {}

    log.info("=== STAGE 3 START [run_stage3] ===")

    x_path = resolve_path(variables["x_train_path"], variables, output_root)
    y_path = resolve_path(variables["y_train_path"], variables, output_root)

    x_path=str(x_path)
    y_path=str(y_path)

    log.info("Stage3 resolved paths: x_path=%s (exists=%s) y_path=%s (exists=%s)",
             x_path, Path(x_path).exists(), y_path, Path(y_path).exists())


    #x_path = variables["x_train_path"]
    #y_path = variables["y_train_path"]
    join_key = variables["join_key"]
    label_col = variables["label_col"]

    csv_params = {"sep": ",", "encoding": "utf-8", "decimal": ".", "low_memory": False}
    X = read_csv(x_path, csv_params=csv_params)
    Y = read_csv(y_path, csv_params=csv_params)

    # Join X + Y (needed because label is in Y_train; modeling needs y aligned with X rows)
    df = safe_inner_join_x_y(X, Y, join_key=join_key)

    # Drop non-target label columns after join (for classification we only keep faulty)
    drop_after_join = (
        cfg["stages"]["stage3_preparation"]["steps"]["step_3_5_data_formatting"]["techniques"]["dataset_formatting"]["methods"]["join_x_y"]["params"]
        .get("drop_after_join", [])
    )
    for col in drop_after_join:
        if col in df.columns:
            log.info("Dropping column after join: %s", col)
            df = df.drop(columns=[col])

    # Split holdout
    split_cfg = cfg["stages"]["stage3_preparation"]["steps"]["step_3_5_data_formatting"]["techniques"]["data_split"]["methods"]["holdout"]["params"]
    hold_cfg = HoldoutConfig(
        test_size=float(split_cfg.get("test_size", 0.2)),
        random_state=int(split_cfg.get("random_state", 42)),
        stratify=bool(split_cfg.get("stratify", True)),
    )
    train_df, valid_df = split_holdout(df, label_col=label_col, cfg=hold_cfg)

    # Build preprocessor (numeric-only dataset: trq_measured,oat,mgt,pa,ias,np,ng)
    feature_cols = [c for c in train_df.columns if c not in (label_col,)]
    # keep join_key as a feature? usually NO -> remove "id" from modeling features
    if join_key in feature_cols:
        feature_cols.remove(join_key)

    prep = build_numeric_preprocessor(
        numeric_features=feature_cols,
        cfg=PreprocessConfig(numeric_imputer="median", scaling="standard")
    )

    X_train = train_df[feature_cols]
    y_train = train_df[label_col].astype(int)

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[label_col].astype(int)

    log.info("Stage3 outputs: X_train=%s y_train=%s X_valid=%s y_valid=%s features=%d",
             X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, len(feature_cols))

    log.info("=== STAGE 3 END [run_stage3] ===")
    return {
        "feature_cols": feature_cols,
        "preprocessor": prep,
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
    }

