# src/phm_america_2024/stages/stage5_evaluation_runner_stages.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.core.helpers_utils_core import write_json
from phm_america_2024.reporting.plots_utils_reporting import save_fig
from phm_america_2024.interpretation.explain_service_interpretation import (
    get_feature_importance,
    get_permutation_importance,
    compute_confusion_matrix,
)

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Stage 5 (Evaluation & Interpretation): interpretation artifacts for best model.
#
# Program flow expectation:
# - Pipeline runner calls run_stage5(...) with best model + holdout data.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Stage runner + interpretation services
# =============================================================================


def run_stage5(cfg: Dict[str, Any], stage3_out: Dict[str, Any], stage4_out: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    stage_cfg = cfg["stages"]["stage5_evaluation_and_interpretation"]
    if not stage_cfg.get("enabled", False):
        log.info("Stage5 disabled.")
        return {}

    log.info("=== STAGE 5 START ===")

    best_model = stage4_out["best_model"]
    feature_cols = stage3_out["feature_cols"]

    out_pol = stage_cfg["output_policy"]
    figs_dir = out_pol["figures_dir"]

    # Feature importance (tree models)
    fi = get_feature_importance(best_model, feature_cols, top_k=30)

    # Permutation importance
    pi = get_permutation_importance(
        best_model,
        stage3_out["X_valid"], stage3_out["y_valid"],
        feature_cols, n_repeats=10, random_state=42, top_k=30
    )

    # Confusion matrix
    disp = compute_confusion_matrix(best_model, stage3_out["X_valid"], stage3_out["y_valid"], normalize="true")
    disp.plot()
    save_fig(output_root / f"{figs_dir}/confusion_matrix.png", dpi=150)

    # Save interpretation JSON
    interp_path = output_root / "interpretation.json"
    write_json(interp_path, {"feature_importance": fi, "permutation_importance": pi})

    log.info("Saved interpretation: %s", interp_path)
    log.info("=== STAGE 5 END ===")
    return {"interpretation_json": str(interp_path)}
