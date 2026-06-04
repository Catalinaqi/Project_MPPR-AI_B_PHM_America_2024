from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from ngboost.distns import Normal
from ngboost.scores import LogScore

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def single_probabilistic_architecture(
        df: pd.DataFrame,
        tech_cfg: dict[str, Any],
        ctx: Any,
        output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Configure NGBoost probabilistic architecture from YAML and persist trace.

    Args:
        df: Input DataFrame (unmodified, used for logging only).
        tech_cfg: YAML technique configuration containing params and output.
        ctx: Run context holding shared execution state.
        output_dir: Directory to write the trace JSON artifact.
    Returns:
        Unmodified DataFrame and extra artifact dict with key 'algorithm_config'.
    """
    log.debug("[single_probabilistic_architecture] entry – shape=%s", df.shape)

    try:
        # Step 1: Extract params and output filename from YAML config
        params: dict[str, Any] = tech_cfg["params"]
        output_filename: str = tech_cfg["output"]
        base_params: dict[str, Any] = params["Base"]
    except KeyError as e:
        log.error("[single_probabilistic_architecture] YAML key missing in configuration: %s", e)
        raise

    # Step 2: Build model configuration dict from YAML params
    model_cfg: dict[str, Any] = {
        "Dist": Normal,
        "Score": LogScore,
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "minibatch_frac": params["minibatch_frac"],
        "Base": {
            "type": base_params["type"],
            "max_depth": base_params["max_depth"],
            "min_samples_leaf": base_params["min_samples_leaf"],
        },
        "random_state": params["random_state"],
    }

    # Step 3: Build serializable trace from YAML params
    trace: dict[str, Any] = {
        "library": params["library"],
        "estimator": params["estimator"],
        "model_configured": {
            "Dist": params["Dist"],
            "Score": params["Score"],
            "n_estimators": model_cfg["n_estimators"],
            "learning_rate": model_cfg["learning_rate"],
            "minibatch_frac": model_cfg["minibatch_frac"],
            "Base": {
                "type": model_cfg["Base"]["type"],
                "max_depth": model_cfg["Base"]["max_depth"],
                "min_samples_leaf": model_cfg["Base"]["min_samples_leaf"],
            },
            "random_state": model_cfg["random_state"],
        },
    }

    # Step 4: CALL write_text() — serialize trace artifact to disk
    output_path: Path = output_dir / output_filename
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    log.info("[single_probabilistic_architecture] trace written to %s", output_path)

    extra: dict[str, Any] = {"algorithm_config": model_cfg}

    log.info("[single_probabilistic_architecture] completed – shape=%s", df.shape)
    return df, extra