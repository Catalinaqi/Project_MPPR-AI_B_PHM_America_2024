# src/phm_america_2024/phase/business_alignment_evaluator_interpretation.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def calibration_audit(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Compute empirical coverage for configured prediction intervals.

    Args:
        df: Unused (signature consistency).
        tech_cfg: Technique configuration dict (e.g. from YAML calibration_audit).
        ctx: Must contain y_true, y_pred_mean, y_pred_std (numpy arrays).
        output_dir: Directory to write output files.

    Returns:
        Unmodified df and a dict with key ``calibration_intervals`` containing
        the interval results.
    """
    log.debug("[calibration_audit] entry")

    y_true = ctx.y_true
    y_pred_mean = ctx.y_pred_mean
    y_pred_std = ctx.y_pred_std

    params = tech_cfg.get("params", {})
    intervals = params.get("intervals", [0.10, 0.50, 0.90])
    output_file = tech_cfg.get("output", "5.2.evaluation.intervals_trace.json")

    results = {}
    for alpha in intervals:
        lower = norm.ppf(alpha / 2, loc=y_pred_mean, scale=y_pred_std)
        upper = norm.ppf(1 - alpha / 2, loc=y_pred_mean, scale=y_pred_std)
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        results[f"interval_{alpha}"] = {
            "expected_coverage": 1 - alpha,
            "empirical_coverage": float(coverage),
            "lower_percentile": alpha / 2,
            "upper_percentile": 1 - alpha / 2,
        }

    out_path = output_dir / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Calibration audit written to %s", out_path)

    log.debug("[calibration_audit] completed")
    return df, {"calibration_intervals": results}


def performance_degradation_benchmarking(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Compare model's Negative Log‑Likelihood against a naive baseline.

    Args:
        df: Unused (signature consistency).
        tech_cfg: Technique configuration dict (e.g. from YAML performance_degradation_benchmarking).
        ctx: Must contain y_true, y_pred_mean, y_pred_std (numpy arrays).
        output_dir: Directory to write output files.

    Returns:
        Unmodified df and a dict with key ``degradation_metrics`` containing
        the NLL comparison.
    """
    log.debug("[performance_degradation_benchmarking] entry")

    y_true = ctx.y_true
    y_pred_mean = ctx.y_pred_mean
    y_pred_std = ctx.y_pred_std

    params = tech_cfg.get("params", {})
    baseline = params.get("baseline", "naive_mean")
    expect_degradation = params.get("expect_test_degradation", True)
    output_file = tech_cfg.get("output", "5.2.evaluation.degradation_trace.json")

    # Model NLL
    model_nll = -np.mean(norm.logpdf(y_true, loc=y_pred_mean, scale=y_pred_std))

    if baseline == "naive_mean":
        baseline_mean = np.mean(y_true)
        baseline_std = np.std(y_true)
        baseline_nll = -np.mean(
            norm.logpdf(y_true, loc=baseline_mean, scale=baseline_std)
        )
    else:
        baseline_nll = 0.0

    degradation = {
        "model_nll": float(model_nll),
        "baseline_nll": float(baseline_nll),
        "nll_difference": float(model_nll - baseline_nll),
        "expect_degradation": expect_degradation,
        "degradation_observed": model_nll > baseline_nll,
    }

    out_path = output_dir / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(degradation, indent=2), encoding="utf-8")
    log.info("Degradation benchmark written to %s", out_path)

    log.debug("[performance_degradation_benchmarking] completed")
    return df, {"degradation_metrics": degradation}
