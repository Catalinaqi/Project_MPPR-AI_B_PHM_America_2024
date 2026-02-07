# src/phm_america_2024/models/train_service_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from sklearn.pipeline import Pipeline

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Train sklearn pipelines (preprocessor + estimator).
#
# Program flow expectation:
# - Stage4 builds pipelines and fits them on transformed train split.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Training service
# =============================================================================


def fit_models(preprocessor, estimators: Dict[str, Any], X_train, y_train) -> Dict[str, Pipeline]:
    trained: Dict[str, Pipeline] = {}

    for name, est in estimators.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", est),
        ])

        log.info("Training: %s", name)
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        log.info("Trained: %s", name)

    return trained
