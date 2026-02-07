# src/phm_america_2024/stages/stage4_modeling_runner_stages.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib

from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.core.helpers_utils_core import ensure_dir, write_json
from phm_america_2024.models.model_registry_models import build_estimators
from phm_america_2024.models.train_service_models import fit_models
from phm_america_2024.models.evaluate_service_models import evaluate_classifier

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Stage 4 (Modeling): train enabled models + evaluate on holdout + select best.
#
# Program flow expectation:
# - Pipeline runner calls run_stage4(...) with Stage3 outputs.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Stage runner + model services
# =============================================================================


def run_stage4(cfg: Dict[str, Any], stage3_out: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    stage_cfg = cfg["stages"]["stage4_modeling"]
    if not stage_cfg.get("enabled", False):
        log.info("Stage4 disabled.")
        return {}

    log.info("=== STAGE 4 START ===")

    algo_cfg = stage_cfg["steps"]["step_4_1_algorithm_selection"]["methods"]
    estimators = build_estimators(algo_cfg)

    trained = fit_models(
        preprocessor=stage3_out["preprocessor"],
        estimators=estimators,
        X_train=stage3_out["X_train"],
        y_train=stage3_out["y_train"],
    )

    # Evaluate each model on holdout
    results: Dict[str, Any] = {}
    best_name = None
    best_score = -1.0

    for name, model in trained.items():
        metrics = evaluate_classifier(model, stage3_out["X_valid"], stage3_out["y_valid"])
        results[name] = metrics

        score = metrics["f1_weighted"]
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is None:
        raise RuntimeError("No best model selected (unexpected).")

    best_model = trained[best_name]
    log.info("Best model: %s (f1_weighted=%.4f)", best_name, best_score)

    # Persist best model
    models_dir = output_root / stage_cfg["output_policy"]["models_dir"]
    ensure_dir(models_dir)
    best_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, best_path)
    log.info("Saved best model: %s", best_path)

    # Persist metrics
    metrics_path = output_root / stage_cfg["output_policy"]["metrics_file"]
    write_json(metrics_path, {"best_model": best_name, "results": results})

    log.info("=== STAGE 4 END ===")
    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "all_results": results,
        "best_model_path": str(best_path),
    }
