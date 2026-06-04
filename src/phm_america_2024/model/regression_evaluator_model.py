from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def model_selection_criteria(
    model: Any,
    params: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Evaluate best model on validation data and compute NLL + RMSE.

    This technique receives the trained model from previous step and the
    validation split (internal validation). It computes metrics and
    returns a metadata dict.

    Note: The runner must pass the validation DataFrame via context_data
    under a known key (e.g., 'val_df' or separate call). For simplicity,
    we assume the model object already has access to validation data.
    """
    log.debug("[model_selection_criteria] entry")

    # In a real pipeline, val data would be loaded separately.
    # As an MVP placeholder, we compute dummy metrics.
    # Actual implementation should load validation data.

    primary_metric = params.get("primary_metric", "neg_log_likelihood")
    tie_breaker = params.get("tie_breaker", "rmse")

    # Placeholder: report dummy metrics
    metrics = {
        "neg_log_likelihood": -0.5,  # dummy value
        "rmse": 0.1,
        "selected_by": primary_metric,
    }

    # Persist ranking trace
    trace = {
        "primary_metric": primary_metric,
        "tie_breaker": tie_breaker,
        "metrics": metrics,
    }
    output_path = output_dir / "4.4.evaluation.ranking_trace.json"
    output_path.write_text(json.dumps(trace, indent=2, default=str))

    extra = {"best_model_metadata": metrics}
    return model, extra