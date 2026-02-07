# src/phm_america_2024/features/transforms_service_features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Build the preprocessing pipeline (imputation + scaling) from config.
# Your current dataset is numeric-only -> keep it simple and robust.
#
# Program flow expectation:
# - Stage3 builds preprocessor and transforms X_train/X_valid.
#
# Design patterns
# - GoF: Builder-ish (pipeline builder).
# - Enterprise/Architectural:
#   - Feature engineering/transforms service layer
# =============================================================================


@dataclass(frozen=True)
class PreprocessConfig:
    numeric_imputer: str = "median"
    scaling: str = "standard"   # standard | none


def build_numeric_preprocessor(numeric_features: List[str],
                               cfg: Optional[PreprocessConfig] = None) -> ColumnTransformer:
    cfg = cfg or PreprocessConfig()

    if cfg.numeric_imputer == "median":
        imputer = SimpleImputer(strategy="median")
    elif cfg.numeric_imputer == "mean":
        imputer = SimpleImputer(strategy="mean")
    else:
        raise ValueError(f"Unsupported numeric_imputer: {cfg.numeric_imputer}")

    steps = [("imputer", imputer)]

    if cfg.scaling == "standard":
        steps.append(("scaler", StandardScaler()))
    elif cfg.scaling == "none":
        pass
    else:
        raise ValueError(f"Unsupported scaling: {cfg.scaling}")

    num_pipe = Pipeline(steps=steps)

    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipe, numeric_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    log.info("Built numeric preprocessor: imputer=%s scaling=%s features=%d",
             cfg.numeric_imputer, cfg.scaling, len(numeric_features))

    return preprocessor
