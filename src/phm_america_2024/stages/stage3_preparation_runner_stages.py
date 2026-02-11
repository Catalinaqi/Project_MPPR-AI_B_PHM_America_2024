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
from phm_america_2024.core.helpers_utils_core import to_json_log,to_json_log_compact

from phm_america_2024.reporting.artifacts_service_reporting import save_table_png
from phm_america_2024.features.selection_service_features import (
    remove_constant_features,
    remove_duplicate_features,
)
from phm_america_2024.reporting.tables_utils_reporting import (
    build_before_after_tables,
    concat_reports,
)


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

    payload = {"cfg": cfg, "variables": variables, "output_root": str(output_root)}
    log.info("[run_stage2] params=%s", to_json_log_compact(payload))

    # ============================================================
    # STAGE 3 — DATA PREPARATION (PHASE 3)
    # stage3_preparation
    # ============================================================

    stage_cfg = cfg["stages"]["stage3_preparation"]
    if not stage_cfg.get("enabled", False):
        log.info("Stage3 disabled.")
        return {}

    log.info("=== STAGE 3 START [run_stage3] ===")

    # paths
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

    # Basic tables
    out_pol = stage_cfg["output_policy"]
    tables_dir = out_pol["tables_png_dir"]
    figs_dir = out_pol["figures_dir"]



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

    # ============================================================
    # step_3_1_data_selection
    # ============================================================



    # -------------------------------------------------------------------------
    # step_3_1_data_selection | Technique: feature_selection
    # -------------------------------------------------------------------------

    # IMPORTANT: keep 'id' for joins and tracing, but DO NOT treat it as a feature.
    feature_cols = [c for c in df.columns if c not in (label_col, "id")]

    kpi_parts = []
    changes_parts = []

    # -------------------------
    # 3.1 - remove_constant_features
    # -------------------------
    rm_const_cfg = cfg["stages"]["stage3_preparation"]["steps"]["step_3_1_data_selection"]["techniques"]["feature_selection"]["methods"]["remove_constant_features"]
    log.debug("[run_stage3] 3.1 remove_constant_features cfg=%s", rm_const_cfg)

    if rm_const_cfg.get("enabled", False):
        df_before = df.copy(deep=False)

        params = rm_const_cfg.get("params", {})
        threshold_unique = int(params.get("threshold_unique", 1))

        log.info("[run_stage3] applying remove_constant_features(threshold_unique=%s)", threshold_unique)
        res = remove_constant_features(df, feature_cols=feature_cols, threshold_unique=threshold_unique)
        df = res.df

        kpi, changes = build_before_after_tables(df_before, df, step_name="3.1_remove_constant_features")
        kpi_parts.append(kpi)
        changes_parts.append(changes)

    # -------------------------
    # 3.1 - remove_duplicate_features
    # -------------------------
    rm_dup_cfg = cfg["stages"]["stage3_preparation"]["steps"]["step_3_1_data_selection"]["techniques"]["feature_selection"]["methods"]["remove_duplicate_features"]
    log.debug("[run_stage3] 3.1 remove_duplicate_features cfg=%s", rm_dup_cfg)

    if rm_dup_cfg.get("enabled", False):
        df_before = df.copy(deep=False)

        params = rm_dup_cfg.get("params", {})
        strategy = str(params.get("strategy", "exact"))

        log.info("[run_stage3] applying remove_duplicate_features(strategy=%s)", strategy)
        res = remove_duplicate_features(df, feature_cols=feature_cols, strategy=strategy)
        df = res.df

        kpi, changes = build_before_after_tables(df_before, df, step_name="3.1_remove_duplicate_features")
        kpi_parts.append(kpi)
        changes_parts.append(changes)

    # -------------------------
    # 3.1 - save before/after reports as PNG (only if not empty)
    # -------------------------
    kpi_report = concat_reports(kpi_parts)
    changes_report = concat_reports(changes_parts)

    if not kpi_report.empty:
        save_table_png(
            df=kpi_report,
            output_root=output_root,
            rel_path=f"{tables_dir}/stage3_step3_1_feature_selection_kpi.png",
            title="Stage 3.1 - Feature selection KPI (before/after)",
            max_rows=50,
            dpi=200,
        )

    if not changes_report.empty:
        save_table_png(
            df=changes_report,
            output_root=output_root,
            rel_path=f"{tables_dir}/stage3_step3_1_feature_selection_changes.png",
            title="Stage 3.1 - Columns removed/added",
            max_rows=200,
            dpi=200,
        )


    # ============================================================
    # step_3_1_data_selection
    # ============================================================

    # ============================================================
    # step_3_2_data_cleaning
    # ============================================================


    # ============================================================
    # step_3_3_data_transformation
    # ============================================================


    # ============================================================
    # step_3_4_data_integration
    # ============================================================

    # ============================================================
    # step_3_5_data_formatting
    # ============================================================

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

