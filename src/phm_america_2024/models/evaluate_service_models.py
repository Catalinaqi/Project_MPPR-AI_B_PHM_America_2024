# src/phm_america_2024/models/evaluate_service_models.py
from __future__ import annotations

from typing import Any, Dict, Tuple

from sklearn.metrics import accuracy_score, f1_score, classification_report

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Evaluate trained models with classification metrics.
#
# Program flow expectation:
# - Stage4 evaluates each model on holdout split and selects best.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Evaluation service
# =============================================================================


def evaluate_classifier(model, X_valid, y_valid) -> Dict[str, Any]:
    y_pred = model.predict(X_valid)

    acc = float(accuracy_score(y_valid, y_pred))
    f1w = float(f1_score(y_valid, y_pred, average="weighted"))

    rep = classification_report(y_valid, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "f1_weighted": f1w,
        "classification_report": rep,
    }

    log.info("Eval: acc=%.4f f1_weighted=%.4f", acc, f1w)
    return metrics
