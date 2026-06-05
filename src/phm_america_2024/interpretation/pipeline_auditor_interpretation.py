"""
MVP – Step 5.3 technique function for leakage detection (train/test overlap, scaler contamination).
Consumes only its own technique configuration from the YAML step.
"""
import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def leakage_detection(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Execute configured leakage detection checks.

    Args:
        df: Unused (signature consistency).
        tech_cfg: Technique configuration dict (e.g. from YAML leakage_detection).
        ctx: Should contain:
            - ``ctx.df_test`` (pd.DataFrame): the test (challenge) data.
            - ``ctx.scaler`` (sklearn scaler, optional): the fitted scaler.
            - ``ctx.train_df`` (pd.DataFrame, optional): training data (if loaded).
        output_dir: Directory to write output files.

    Returns:
        Unmodified df and a dict with key ``leakage_check_results`` containing
        the audit trace.
    """
    log.debug("[leakage_detection] entry")

    params = tech_cfg.get("params", {})
    checks = params.get("checks", ["train_test_overlap"])
    output_file = tech_cfg.get("output", "5.3.audit.leakage_trace.json")

    results: dict[str, Any] = {}

    # ---- train‑test overlap check ----
    if "train_test_overlap" in checks:
        train_df = getattr(ctx, "train_df", None)
        test_df = getattr(ctx, "df_test", None)

        if train_df is None or test_df is None:
            log.warning(
                "[leakage_detection] train_test_overlap skipped: ctx.train_df or ctx.df_test missing."
            )
            results["train_test_overlap"] = {
                "check_performed": False,
                "error": "Training data not loaded in context – cannot perform overlap check.",
            }
        else:
            # exact row overlap (all columns)
            merged = train_df.merge(test_df, how="inner", indicator=False)
            overlap_count = len(merged)
            results["train_test_overlap"] = {
                "check_performed": True,
                "overlap_rows": overlap_count,
                "leakage_detected": overlap_count > 0,
                "message": f"Found {overlap_count} overlapping rows.",
            }

    # ---- scaler contamination check ----
    if "scaler_contamination_check" in checks:
        scaler = getattr(ctx, "scaler", None)
        test_df = getattr(ctx, "df_test", None)

        if scaler is None or test_df is None:
            log.warning(
                "[leakage_detection] scaler_contamination_check skipped: ctx.scaler or ctx.df_test missing."
            )
            results["scaler_contamination_check"] = {
                "check_performed": False,
                "error": "Scaler or test data not loaded in context.",
            }
        else:
            # compare scaler statistics to test data (should be different if scaler was fit on train)
            numeric_test = test_df.select_dtypes(include=[np.number])
            if numeric_test.empty:
                results["scaler_contamination_check"] = {
                    "check_performed": False,
                    "error": "No numeric columns in test data.",
                }
            else:
                test_median = numeric_test.median().values
                test_iqr = (numeric_test.quantile(0.75) - numeric_test.quantile(0.25)).values

                scaler_center = scaler.center_
                scaler_scale = scaler.scale_

                # align lengths (ignore extra features that may be in scaler)
                min_len = min(len(test_median), len(scaler_center), len(scaler_scale))
                diff_center = np.abs(scaler_center[:min_len] - test_median[:min_len]).mean()
                diff_scale = np.abs(scaler_scale[:min_len] - test_iqr[:min_len]).mean()

                # heuristic: if the mean absolute difference is very small, scaler may have seen test data
                tolerance = 0.01 * np.abs(test_median[:min_len]).max() if test_median[:min_len].max() != 0 else 0.01
                contaminated = diff_center < tolerance

                results["scaler_contamination_check"] = {
                    "check_performed": True,
                    "center_mean_diff": float(diff_center),
                    "scale_mean_diff": float(diff_scale),
                    "contamination_detected": bool(contaminated),
                    "message": "Scaler statistics are suspiciously close to test data"
                    if contaminated
                    else "Scaler appears clean (statistics differ from test).",
                }

    # Write trace JSON
    out_path = output_dir / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Leakage detection written to %s", out_path)

    log.debug("[leakage_detection] completed")
    return df, {"leakage_check_results": results}