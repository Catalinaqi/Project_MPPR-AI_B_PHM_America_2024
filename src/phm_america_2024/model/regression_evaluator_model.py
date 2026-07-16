"""
Module: regression_evaluator_model.py
Core algorithm: NGBoost (Natural Gradient Boosting) — already trained in a
prior step (4.2); this module only evaluates it on held-out validation data.
Metrics computed: Negative Log Likelihood (NLL) and Root Mean Squared Error
(RMSE) on the model's probabilistic predictions.

Flow:
1. Read evaluation parameters from the YAML config (primary_metric,
   tie_breaker, target_variable, features, output).
   - target_variable and features are declared EXPLICITLY in this step's own
     config, not derived implicitly from step 4.2's config. This avoids a
     hidden cross-step dependency and makes the evaluation self-contained
     and auditable from its own YAML block.
   - As a non-fatal safety net, this module also attempts to read the same
     values from step 4.2's config and logs a warning if they diverge from
     what step 4.4 declares — this catches accidental drift between the two
     configs without making step 4.4 depend on step 4.2 to function.
2. Resolve the validation dataset path from the global read_strategy
   (phase-level input_source.val_data).
3. Load the validation parquet file.
4. Clean the data: replace infinities with NaN and drop rows with missing
   values.
5. Apply the fitted RobustScaler (ctx.scaler) ONLY to the columns it was
   actually fit on (features + target) — NOT to the full validation
   DataFrame, which may contain additional columns (e.g. unused polynomial
   interaction terms, 'faulty') that the scaler was never trained on and
   would cause a feature-count mismatch if included.
6. Select X_val (declared features) and y_val (declared target) using the
   exact same column list the model was trained on.
7. Call model.pred_dist(X_val) to get the predictive distributions.
8. Compute NLL = mean negative log-pdf of the distributions on the real values.
9. Compute RMSE = root mean squared error between the distribution's mean
   and y_val.
10. Persist the metrics to a JSON trace file in the output directory.
11. Return the unchanged model plus an extra dict containing the computed metrics.

Imports:
- json: trace serialization.
- pathlib.Path: path handling.
- typing.Any: generic type hints.
- numpy: numerical operations (inf handling, sqrt, mean).
- pandas: DataFrame loading and cleaning.
- logging_adapter_common: project logger.
- io_service_common.load_parquet: parquet loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.io_service_common import load_parquet

log = get_logger(__name__)


# step_4_4_model_evaluation -> evaluation of the best model on validation data


def _read_step_4_2_reference(
        ctx: Any,
) -> tuple[Optional[str], Optional[List[str]]]:
    """Best-effort read of step 4.2's declared target_variable / features.

    Used only as a non-fatal drift-detection safety net (Step 1b below) —
    this function must never raise, since step 4.4 is designed to work
    self-contained from its OWN config even if step 4.2's config is
    missing, malformed, or was refactored independently.
    """
    try:
        cv_params = ctx.config.phases["phase4_data_modeling"]["steps"][
            "step_4_2_model_training"
        ]["methods"]["model_training"]["techniques"]["cross_validation"]["params"]
        return cv_params.get("target_variable"), cv_params.get("features")
    except Exception as e:
        log.debug(
            "[model_selection_criteria] could not read step 4.2 config for "
            "drift-check purposes (non-fatal): %s",
            e,
        )
        return None, None


def model_selection_criteria(
        model: Any,
        tech_cfg: dict[str, Any],
        ctx: Any,
        output_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Evaluate the best model on validation data and compute NLL + RMSE.

    Args:
        model: Trained NGBoost model from step 4.2.
        tech_cfg: Technique configuration dict, sourced strictly from YAML.
        ctx: RunContext holding shared execution state (paths, fitted scaler, config).
        output_dir: Output directory path for this pipeline step.

    Returns:
        Tuple of the unchanged model and a metadata dict with the computed metrics.
    """
    log.debug("[model_selection_criteria] ENTRY")

    # ------------------------------------------------------------------
    # Step 1: extract parameters strictly from THIS step's YAML config
    # (zero hardcoding, zero implicit dependency on other steps' configs
    # for values required to run).
    # ------------------------------------------------------------------
    try:
        params = tech_cfg["params"]
        primary_metric: str = params["primary_metric"]
        tie_breaker: str = params["tie_breaker"]
        output_filename: str = tech_cfg["output"]
        target_variable: str = params["target_variable"]
        feature_cols: List[str] = params["features"]
    except KeyError as e:
        log.error(
            "[model_selection_criteria] missing required parameter in step 4.4 "
            "YAML configuration: %s. 'target_variable' and 'features' must be "
            "declared explicitly in this step's params.",
            e,
        )
        raise ValueError(f"YAML configuration error: missing {e}")

    log.info(
        "[model_selection_criteria] Step 1 – config loaded: target='%s', "
        "features=%d columns, primary_metric='%s', tie_breaker='%s'",
        target_variable, len(feature_cols), primary_metric, tie_breaker,
    )
    log.debug("[model_selection_criteria] declared feature list: %s", feature_cols)

    # ------------------------------------------------------------------
    # Step 1b: non-fatal drift check against step 4.2's declared config.
    # This does not gate execution — it only warns, so step 4.4 stays
    # self-contained even if step 4.2's config path changes shape.
    # ------------------------------------------------------------------
    ref_target, ref_features = _read_step_4_2_reference(ctx)
    if ref_target is not None and ref_target != target_variable:
        log.warning(
            "[model_selection_criteria] DRIFT DETECTED: step 4.4 declares "
            "target_variable='%s' but step 4.2 declares '%s'. Verify both "
            "configs are intentionally aligned.",
            target_variable, ref_target,
        )
    if ref_features is not None and list(ref_features) != list(feature_cols):
        log.warning(
            "[model_selection_criteria] DRIFT DETECTED: step 4.4 declares a "
            "different 'features' list than step 4.2. Evaluation will use "
            "step 4.4's list, but the model may have been trained on a "
            "different set. step_4_4=%s | step_4_2=%s",
            feature_cols, ref_features,
        )

    # ------------------------------------------------------------------
    # Step 2: resolve the validation dataset path (phase-level read_strategy)
    # ------------------------------------------------------------------
    try:
        val_data_path: str = ctx.config.phases["phase4_data_modeling"][
            "read_strategy"
        ]["input_source"]["val_data"]
    except KeyError as e:
        log.error(
            "[model_selection_criteria] missing 'val_data' in phase4_data_modeling "
            "read_strategy.input_source: %s",
            e,
        )
        raise ValueError(f"YAML configuration error: missing {e}")

    log.debug("[model_selection_criteria] Step 2 – val_data_path=%s", val_data_path)

    phase3_dir = getattr(ctx, "phase3_dir", None)
    if not phase3_dir:
        log.error(
            "[model_selection_criteria] phase3_dir not found in context. "
            "Cannot resolve validation data path."
        )
        raise ValueError("phase3_dir missing in RunContext")

    full_val_path = Path(phase3_dir) / val_data_path
    log.info("[model_selection_criteria] Step 2 – resolved validation path: %s", full_val_path)

    if not full_val_path.exists():
        log.error(
            "[model_selection_criteria] validation data not found at %s",
            full_val_path,
        )
        raise FileNotFoundError(f"Validation file missing: {full_val_path}")

    # ------------------------------------------------------------------
    # Step 3: load the validation split
    # ------------------------------------------------------------------
    val_data: pd.DataFrame = load_parquet(str(full_val_path))
    log.info(
        "[model_selection_criteria] Step 3 – loaded validation data: shape=%s",
        val_data.shape,
    )

    # ------------------------------------------------------------------
    # Step 4: clean infinities and missing values
    # ------------------------------------------------------------------
    initial_len = len(val_data)
    val_data = val_data.replace([np.inf, -np.inf], np.nan).dropna()
    dropped_rows = initial_len - len(val_data)

    if dropped_rows > 0:
        log.warning(
            "[model_selection_criteria] Step 4 – dropped %d rows containing "
            "NaN or Infinity (%.2f%% of validation data).",
            dropped_rows, 100 * dropped_rows / initial_len,
                          )
    else:
        log.debug("[model_selection_criteria] Step 4 – no NaN/Infinity rows found.")

    # ------------------------------------------------------------------
    # Step 5: apply the fitted RobustScaler ONLY to the columns it was
    # actually fit on (features + target). The scaler was fit in step 3.5
    # on exactly these columns — applying it to the full validation
    # DataFrame (which may include unused interaction terms or 'faulty')
    # would raise a feature-count mismatch, since scikit-learn strictly
    # validates the number of input columns against the fitted scaler.
    # ------------------------------------------------------------------
    # BUGFIX: ctx.scaler (RobustScaler) was fit on exactly 10 columns during
    # step 3.5 (the 9 model features + the target). scikit-learn strictly
    # validates BOTH the count AND THE ORDER of column names when the input
    # is a DataFrame — passing the right columns in the wrong order (e.g.
    # target appended last instead of in its original fit position) raises:
    # "ValueError: The feature names should match those that were passed
    # during fit. Feature names must be in the same order as they were in fit."
    #
    # Fix: read the exact fit-time column order directly from the fitted
    # scaler itself (feature_names_in_), instead of reconstructing it
    # manually as feature_cols + [target_variable]. This makes the order
    # correct by construction, regardless of how features/target happen to
    # be declared in this step's YAML.
    if hasattr(ctx.scaler, "feature_names_in_"):
        scaler_columns: List[str] = list(ctx.scaler.feature_names_in_)
        log.debug(
            "[model_selection_criteria] Step 5 – using fit-time column order "
            "from ctx.scaler.feature_names_in_: %s",
            scaler_columns,
        )
    else:
        # Fallback for scaler objects without feature_names_in_ (e.g. fit on
        # a plain numpy array instead of a DataFrame) — order is NOT
        # guaranteed safe here, only used as a last resort.
        scaler_columns = list(feature_cols) + [target_variable]
        log.warning(
            "[model_selection_criteria] Step 5 – ctx.scaler has no "
            "'feature_names_in_' attribute; falling back to features+target "
            "order, which is NOT guaranteed to match the original fit order."
        )

    # Sanity check: the set of columns the scaler expects should exactly
    # match this step's declared features + target — if not, something is
    # genuinely out of sync between step 3.5 (scaler fit) and step 4.4
    # (this evaluation), not just an ordering artifact.
    expected_set = set(feature_cols) | {target_variable}
    scaler_set = set(scaler_columns)
    if expected_set != scaler_set:
        log.warning(
            "[model_selection_criteria] Step 5 – scaler's fitted columns %s "
            "do not exactly match this step's declared features+target %s. "
            "Symmetric difference: %s",
            scaler_set, expected_set, expected_set.symmetric_difference(scaler_set),
        )

    missing_scaler_cols = set(scaler_columns) - set(val_data.columns)
    if missing_scaler_cols:
        log.error(
            "[model_selection_criteria] Step 5 – columns expected by the "
            "scaler are missing from validation data: %s",
            missing_scaler_cols,
        )
        raise ValueError(f"Validation data missing scaler columns: {missing_scaler_cols}")

    log.info(
        "[model_selection_criteria] Step 5 – applying fitted RobustScaler to "
        "%d columns, in fit-time order: %s",
        len(scaler_columns), scaler_columns,
    )
    val_data[scaler_columns] = ctx.scaler.transform(val_data[scaler_columns])
    log.debug("[model_selection_criteria] Step 5 – scaling applied successfully.")

    # ------------------------------------------------------------------
    # Step 6: select X_val / y_val using the EXACT same feature list the
    # model was trained on. Using anything else (e.g. "all columns except
    # target") would silently mismatch the model's expected input shape
    # and make model.pred_dist() fail or, worse, silently misbehave.
    #
    # Flow reminder: the model was already trained in step 4.2 on the
    # internal train split. Here, in step 4.4, we load the saved model,
    # load the internal validation split, and run model.pred_dist(X_val)
    # to see how the model performs on data it never saw during training
    # — this is the model's "final exam" within the lab environment
    # (val_data is still an internal split, not the external challenge
    # test/val sets, which are inaccessible since the challenge is closed).
    # ------------------------------------------------------------------
    X_val = val_data[feature_cols]
    y_val = val_data[target_variable]
    log.info(
        "[model_selection_criteria] Step 6 – evaluation matrix: X_val=%s y_val=%s",
        X_val.shape, y_val.shape,
    )

    model_path = getattr(ctx, "model_path", None)
    if model_path:
        log.info("[model_selection_criteria] model loaded from: %s", model_path)
    else:
        log.warning("[model_selection_criteria] no model_path found in context")

    log.debug(
        "[model_selection_criteria] model type=%s, params=%s",
        type(model).__name__,
        model.get_params(),
    )

    # ------------------------------------------------------------------
    # Step 7: predict distributions and compute NLL
    # ------------------------------------------------------------------
    dist = model.pred_dist(X_val)
    nll_score: float = float(-dist.logpdf(y_val).mean())
    log.info("[model_selection_criteria] Step 7 – NLL computed: %.4f", nll_score)

    # ------------------------------------------------------------------
    # Step 8: compute RMSE from the distribution's mean
    # ------------------------------------------------------------------
    y_pred_mean = dist.mean()
    rmse_score: float = float(np.sqrt(((y_pred_mean - y_val) ** 2).mean()))
    log.info("[model_selection_criteria] Step 8 – RMSE computed: %.4f", rmse_score)

    log.info(
        "[model_selection_criteria] evaluation metrics: NLL=%.4f, RMSE=%.4f "
        "(scaled space — same space the model was trained in)",
        nll_score, rmse_score,
    )

    # ------------------------------------------------------------------
    # Step 9: build the trace dictionaries
    # ------------------------------------------------------------------
    metrics = {
        "neg_log_likelihood": nll_score,
        "rmse": rmse_score,
        "selected_by": primary_metric,
    }

    trace = {
        "primary_metric": primary_metric,
        "tie_breaker": tie_breaker,
        "target_variable": target_variable,
        "features": feature_cols,
        "n_validation_rows": int(X_val.shape[0]),
        "metrics": metrics,
    }

    # ------------------------------------------------------------------
    # Step 10: persist the trace to disk
    # ------------------------------------------------------------------
    output_path = output_dir / output_filename
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    log.info("[model_selection_criteria] Step 10 – trace written to %s", output_path)

    # ------------------------------------------------------------------
    # Step 11: return the unchanged model plus the extra metadata dict
    # ------------------------------------------------------------------
    extra = {"best_model_metadata": metrics}

    log.debug("[model_selection_criteria] EXIT")
    return model, extra