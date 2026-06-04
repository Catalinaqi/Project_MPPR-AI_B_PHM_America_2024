from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def single_probabilistic_architecture(
    df: pd.DataFrame,
    params: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Configure NGBoost to output Normal distribution parameters.

    This technique corresponds to the ``algorithm_selection`` method
    in the regression pipeline config. It does *not* transform the
    DataFrame; it returns the algorithm configuration as an extra
    artifact for downstream steps.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (used here only for logging, not modified).
    params : dict
        YAML technique parameters:
        ``library`` (str) – ignored, forced to ngboost.
        ``estimator`` (str) – ignored, forced to NGBRegressor.
        ``Dist`` (str) – distribution family, currently ``Normal``.
        ``Score`` (str) – scoring rule, currently ``LogScore``.
        ``n_estimators`` (int)
        ``learning_rate`` (float)
        ``Base`` (dict) – base learner config (e.g. ``max_depth``).
        ``random_state`` (int)
    ctx : Any
        RunContext (unused, interface consistency).
    output_dir : Path
        Directory to write the trace JSON.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any] | None]
        - Unmodified input DataFrame.
        - Extra artifact dict with key ``"algorithm_config"`` containing
          the full model configuration dictionary.
    """
    log.debug("[single_probabilistic_architecture] entry – shape=%s", df.shape)

    # Build model configuration from YAML params (with sensible defaults)
    model_cfg = {
        "library": "ngboost",
        "estimator": "NGBRegressor",
        "Dist": Normal,
        "Score": LogScore,
        "n_estimators": params.get("n_estimators", 400),
        "learning_rate": params.get("learning_rate", 0.03),
        "Base": {
            "type": "DecisionTreeRegressor",
            "max_depth": params.get("Base", {}).get("max_depth", 4),
        },
        "random_state": params.get("random_state", 42),
    }

    # Persist trace log
    trace = {
        "library": "ngboost",
        "estimator": "NGBRegressor",
        "model_configured": {
            "Dist": "Normal",
            "Score": "LogScore",
            "n_estimators": model_cfg["n_estimators"],
            "learning_rate": model_cfg["learning_rate"],
            "Base": {"max_depth": model_cfg["Base"]["max_depth"]},
            "random_state": model_cfg["random_state"],
        },
    }
    output_path = output_dir / "4.1.modeling.algo_setup_trace.json"
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    log.debug("[single_probabilistic_architecture] trace written to %s", output_path)

    # Return unmodified dataframe and the configuration as extra artifact
    extra = {"algorithm_config": model_cfg}
    log.info("[single_probabilistic_architecture] completed – shape=%s", df.shape)
    return df, extra