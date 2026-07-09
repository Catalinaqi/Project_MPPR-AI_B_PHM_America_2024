from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


# step_4_1_algorithm_selection -> Setup de algoritmos (clasificación binaria calibrada)
def single_calibrated_architecture(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Configure LightGBM calibrated architecture from YAML and persist trace.

    Args:
        df: Input DataFrame (unmodified, used for logging only).
        tech_cfg: YAML technique configuration containing params and output.
        ctx: Run context holding shared execution state.
        output_dir: Directory to write the trace JSON artifact.
    Returns:
        Unmodified DataFrame and extra artifact dict with key 'algorithm_config'.
    """
    log.debug("[single_calibrated_architecture] entry – shape=%s", df.shape)

    try:
        # Step 1: Extract params and output filename from YAML config
        params: dict[str, Any] = tech_cfg["params"]
        output_filename: str = tech_cfg["output"]
    except KeyError as e:
        log.error(
            "[single_calibrated_architecture] YAML key missing in configuration: %s",
            e,
        )
        raise

    # Step 2: Build model configuration dict from YAML params
    # LightGBM expects a flat dictionary of parameters
    model_cfg: dict[str, Any] = {
        "objective": params["objective"],
        "boosting_type": params.get("boosting_type", "gbdt"),
        "scale_pos_weight": params.get("scale_pos_weight", 1.0),
        "scale_pos_weight_uncertainty": params.get("scale_pos_weight_uncertainty",0),
        "learning_rate": params["learning_rate"],
        "learning_rate_uncertainty": params["learning_rate_uncertainty"],
        "n_estimators": params["n_estimators"],
        "n_estimators_uncertainty": params["n_estimators_uncertainty"],
        "max_depth": params.get("max_depth", -1),
        "max_depth_uncertainty": params.get("max_depth_uncertainty",0),
        "num_leaves": params.get("num_leaves", 31),
        "num_leaves_uncertainty": params.get("num_leaves_uncertainty",0),
        "random_state": params.get("random_state", None),
        # Additional parameters from YAML (e.g., min_child_samples, subsample, colsample_bytree)
        # could be added generically:
        **{
            k: v
            for k, v in params.items()
            if k not in ("library", "estimator", "output")
        },
    }
    # Remove any keys that are not LightGBM parameters (like "library", "estimator" if they got through)
    for unwanted in ("library", "estimator", "output"):
        model_cfg.pop(unwanted, None)

    # Step 3: Build serializable trace from YAML params
    trace: dict[str, Any] = {
        "library": params["library"],
        "estimator": params["estimator"],
        "model_configured": {
            k: v
            for k, v in model_cfg.items()
            if k not in ("library", "estimator", "output")
        },
    }

    # Step 4: Write trace artifact to disk
    output_path: Path = output_dir / output_filename
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    log.info("[single_calibrated_architecture] trace written to %s", output_path)

    extra: dict[str, Any] = {"algorithm_config": model_cfg}

    log.info("[single_calibrated_architecture] completed – shape=%s", df.shape)
    return df, extra
