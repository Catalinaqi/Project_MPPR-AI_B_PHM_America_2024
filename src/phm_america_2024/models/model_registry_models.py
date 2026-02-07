# src/phm_america_2024/models/model_registry_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Registry/factory for ML models enabled in YAML.
#
# Program flow expectation:
# - Stage4 asks registry for enabled estimators and trains them.
#
# Design patterns
# - GoF: Factory Method (simple).
# - Enterprise/Architectural:
#   - Model registry service
# =============================================================================


def build_estimators(algo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    algo_cfg is stages.stage4_modeling.step_4_1_algorithm_selection.methods
    Returns dict: name -> sklearn estimator
    """
    estimators: Dict[str, Any] = {}

    # Decision Tree
    dt = algo_cfg.get("decision_tree_classifier", {})
    if dt.get("enabled"):
        params = dt.get("params", {}) or {}
        estimators["decision_tree_classifier"] = DecisionTreeClassifier(**params)

    # Random Forest
    rf = algo_cfg.get("random_forest_classifier", {})
    if rf.get("enabled"):
        params = rf.get("params", {}) or {}
        estimators["random_forest_classifier"] = RandomForestClassifier(**params)

    if not estimators:
        raise ValueError("No estimators enabled in step_4_1_algorithm_selection")

    log.info("Enabled estimators: %s", list(estimators.keys()))
    return estimators
