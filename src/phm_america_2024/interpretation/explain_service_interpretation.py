# src/phm_america_2024/interpretation/explain_service_interpretation.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Post-hoc model interpretation artifacts:
# - feature_importances_ (tree-based)
# - permutation importance (model-agnostic)
# - confusion matrix
#
# Program flow expectation:
# - Stage5 runs explainers on the best model using holdout data.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Interpretation service
# =============================================================================


def get_feature_importance(best_model, feature_names: List[str], top_k: int = 30) -> List[Dict[str, Any]]:
    """
    Works for tree-based models where model.named_steps["model"] has feature_importances_.
    """
    model = best_model.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        log.warning("feature_importances_ not available for this model.")
        return []

    importances = np.asarray(model.feature_importances_, dtype=float)
    idx = np.argsort(importances)[::-1][:top_k]

    out = [{"feature": feature_names[i], "importance": float(importances[i])} for i in idx]
    log.info("feature_importance computed: top_k=%s", top_k)
    return out


def get_permutation_importance(best_model, X_valid, y_valid, feature_names: List[str],
                               n_repeats: int = 10, random_state: int = 42, top_k: int = 30) -> List[Dict[str, Any]]:
    r = permutation_importance(best_model, X_valid, y_valid,
                               n_repeats=n_repeats, random_state=random_state, scoring="f1_weighted")
    means = r.importances_mean
    idx = np.argsort(means)[::-1][:top_k]

    out = [{"feature": feature_names[i], "importance_mean": float(means[i])} for i in idx]
    log.info("permutation_importance computed: top_k=%s n_repeats=%s", top_k, n_repeats)
    return out


def compute_confusion_matrix(best_model, X_valid, y_valid, normalize: str | None = "true"):
    y_pred = best_model.predict(X_valid)
    cm = confusion_matrix(y_valid, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    log.info("confusion_matrix computed normalize=%s", normalize)
    return disp
