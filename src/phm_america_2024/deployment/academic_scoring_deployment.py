# src/phm_america_2024/deployment/academic_scoring_deployment.py
"""
Module: academic_scoring_deployment.py
Technique name: dual_model_inference (renamed from the original
'cascade_inference' — the old name became misleading once this module
started supporting two structurally different inference modes, only one
of which is an actual cascade).

Core algorithm: two supported inference modes over the internal test splits,
selected via the `mode` config parameter.

  mode="cascade" (default, backward-compatible with prior runs):
    Two-stage cascade inference:
      1. Binary classification with LightGBM (failure probability,
         calibrated via IsotonicRegression).
      2. Probabilistic regression with NGBoost (Normal distribution) for
         torque margin — ONLY for samples whose calibrated failure
         probability is >= filter_threshold.
    Consequence: regression coverage is bounded by the classifier's recall
    on the faulty class. With the current classifier (~15% recall on
    faulty), only ~7% of rows receive a torque margin prediction; the rest
    are NaN by design (documented pipeline limitation, not a bug).

  mode="independent" (opt-in, does not gate regression on classification):
    Both models run independently on the full test set. Regression
    coverage is 100% regardless of classification outcome. This decouples
    the two tasks, since torque margin is a physically meaningful quantity
    for every engine regardless of predicted health state, not only for
    engines the classifier happens to flag as faulty.

IMPORTANT: classification and regression use DIFFERENT feature sets.
  - regression_features excludes 'trq_margin' (it's the regression target)
    but the classification model DOES use 'trq_margin' as an input feature
    (its target is 'faulty', not 'trq_margin'). Declaring the same feature
    list for both tasks is a config mistake that will raise a feature-count
    error at predict time (e.g. LightGBMError: "number of features in data
    (9) is not the same as it was in training data (10)").

Flow (step-by-step):
1. Resolve absolute paths for all artifacts (models, calibrator, test data)
   from ctx.run_dir + the YAML-declared relative paths.
2. Verify every artifact exists on disk before loading anything.
3. Load the pre-trained models (classifier, calibrator, regressor) via joblib.
4. Load the classification and regression internal test DataFrames.
5. Select model input features EXPLICITLY from config
   (classification_features / regression_features), matching exactly what
   each model was trained on — never inferred implicitly as "all columns
   except target", which silently breaks the moment the underlying dataset
   gains extra columns (e.g. unused polynomial interaction terms).
6. Run classification: predict + calibrate failure probability for every row.
7. Run regression, following the selected mode (cascade or independent).
8. Assemble the final predictions DataFrame with the Challenge's expected
   column names.
9. Persist the predictions to a Parquet file under phase6_dir.
10. Enrich the RunContext with predictions_df / predictions_path.

Imports:
- pathlib.Path: path resolution.
- typing.Any: generic type hints.
- numpy, pandas: data handling.
- joblib: loading serialized models.
- logging_adapter_common: project logger.
- context_facade_common.RunContext: execution context type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import joblib

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.pipeline.utils.context_facade_common import RunContext

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_artifact_path(ctx: RunContext, relative_path: str) -> Path:
    """Convert a relative path (as stored in the YAML config) to an absolute
    ``Path`` anchored at the current run directory.

    The YAML config refers to artifacts relative to the run output folder
    (e.g. ``"4.2.training.ngboost_regressor.pkl"``). This helper appends them
    to ``ctx.run_dir`` and returns the resolved absolute path.

    Args:
        ctx: Run context (must have a ``run_dir`` attribute).
        relative_path: Path from the YAML config, relative to ctx.run_dir.

    Returns:
        Absolute ``Path`` pointing to the artifact.
    """
    run_dir = Path(ctx.run_dir).resolve()
    full_path = (run_dir / relative_path).resolve()
    log.debug("[_resolve_artifact_path] %s -> %s", relative_path, full_path)
    return full_path


def _select_explicit_features(
        df: pd.DataFrame,
        declared_features: Optional[List[str]],
        fallback_drop_cols: List[str],
        context_label: str,
) -> pd.DataFrame:
    """Select model input columns strictly from an explicit config list.

    Falls back to "all columns except fallback_drop_cols" only if no
    explicit list is declared, logging a warning — this fallback is kept
    for backward compatibility but is NOT recommended, since it silently
    mismatches the model's trained feature set whenever the dataset gains
    extra columns upstream (e.g. unused polynomial interaction terms).

    Args:
        df: Source DataFrame to select columns from.
        declared_features: Explicit feature list from config, or None.
        fallback_drop_cols: Columns to drop for the fallback path.
        context_label: Short label used in log messages (e.g. "classification").

    Returns:
        DataFrame restricted to the resolved feature columns.
    """
    if declared_features:
        missing = set(declared_features) - set(df.columns)
        if missing:
            raise ValueError(
                f"[dual_model_inference] declared {context_label}_features not "
                f"found in test data: {missing}"
            )
        log.info(
            "[dual_model_inference] %s features: using explicit list from "
            "config (%d columns): %s",
            context_label, len(declared_features), declared_features,
        )
        return df[declared_features]

    fallback_cols = [c for c in df.columns if c not in fallback_drop_cols]
    log.warning(
        "[dual_model_inference] %s_features not declared in config — "
        "defaulting to 'all columns except %s' (%d columns). This may "
        "silently mismatch the model's actual trained feature set; "
        "declaring an explicit list is strongly recommended.",
        context_label, fallback_drop_cols, len(fallback_cols),
    )
    return df[fallback_cols]


# ---------------------------------------------------------------------------
# Public orchestration function
# ---------------------------------------------------------------------------


def run_dual_model_inference(
        ctx: RunContext,
        input_source: dict[str, str],
        params: dict[str, Any],
) -> RunContext:
    """Run inference on the internal test splits, in cascade or independent mode.

    Args:
        ctx: Run context containing the run directory and project info.
        input_source: Dictionary of artifact keys -> YAML relative paths
            (from ``phase6_deployment.read_strategy.input_source``).
        params: Parameters of the ``cascade_inference`` technique (see
            ``phase6_deployment.steps.step_6_1_academic_scoring.methods
            .batch_scoring.techniques.cascade_inference.params``).

    Returns:
        The updated ``RunContext`` with ``predictions_df`` and
        ``predictions_path`` attributes.
    """
    log.info("[dual_model_inference] ===== Step 6.1 - Academic Scoring ===== START")

    # ------------------------------------------------------------------
    # Step 1: resolve and validate all input artifact paths
    # ------------------------------------------------------------------
    clf_path = _resolve_artifact_path(ctx, input_source["classification_model"])
    reg_path = _resolve_artifact_path(ctx, input_source["regression_model"])
    cal_path = _resolve_artifact_path(ctx, input_source["classification_calibrator"])
    reg_scaler_path = _resolve_artifact_path(ctx, input_source["regression_scaler"])
    clf_test_path = _resolve_artifact_path(ctx, input_source["classification_test_data"])
    reg_test_path = _resolve_artifact_path(ctx, input_source["regression_test_data"])

    log.info(
        "[dual_model_inference] Step 1 - input paths resolved: clf_model=%s, "
        "reg_model=%s, calibrator=%s, reg_scaler=%s, clf_test=%s, reg_test=%s",
        clf_path, reg_path, cal_path, reg_scaler_path, clf_test_path, reg_test_path,
    )

    for name, path in [
        ("classification_model", clf_path),
        ("regression_model", reg_path),
        ("calibrator", cal_path),
        ("regression_scaler", reg_scaler_path),
        ("classification_test_data", clf_test_path),
        ("regression_test_data", reg_test_path),
    ]:
        if not path.exists():
            log.error("[dual_model_inference] Step 1 - missing artifact: %s at %s", name, path)
            raise FileNotFoundError(f"[dual_model_inference] Required artifact missing: {name} at {path}")

    # ------------------------------------------------------------------
    # Step 2: load models
    # ------------------------------------------------------------------
    clf_model = joblib.load(clf_path)
    reg_model = joblib.load(reg_path)
    calibrator = joblib.load(cal_path)
    reg_scaler = joblib.load(reg_scaler_path)
    log.info("[dual_model_inference] Step 2 - all models + regression scaler loaded successfully")

    # ------------------------------------------------------------------
    # Step 3: load test DataFrames
    # ------------------------------------------------------------------
    df_clf_test = pd.read_parquet(clf_test_path)
    df_reg_test = pd.read_parquet(reg_test_path)
    log.info(
        "[dual_model_inference] Step 3 - loaded test data: classification=%s, regression=%s",
        df_clf_test.shape, df_reg_test.shape,
    )
    log.debug("[dual_model_inference] classification test columns: %s", list(df_clf_test.columns))
    log.debug("[dual_model_inference] regression test columns: %s", list(df_reg_test.columns))

    if len(df_clf_test) != len(df_reg_test):
        log.warning(
            "[dual_model_inference] Step 3 - test set size mismatch: clf=%d, "
            "reg=%d. Truncating both to %d rows.",
            len(df_clf_test), len(df_reg_test), min(len(df_clf_test), len(df_reg_test)),
        )
        n = min(len(df_clf_test), len(df_reg_test))
        df_clf_test = df_clf_test.iloc[:n]
        df_reg_test = df_reg_test.iloc[:n]

    # ------------------------------------------------------------------
    # Step 3b: scale df_reg_test with the SAME fitted RobustScaler used to
    # train the NGBoost model (step 3.5). This is critical: unlike
    # classification_test_data (which already comes pre-scaled from step
    # 3.5 in the classification pipeline), regression_test_data comes from
    # step 3.4 — i.e. BEFORE scaling — since scaling happens as a later,
    # separate step (3.5) in the regression pipeline. Predicting on
    # unscaled features with a model trained on scaled features produces
    # degenerate, narrow-range predictions (most rows fall on the same
    # side of nearly every tree split), which is silently wrong rather
    # than an outright crash.
    #
    # scikit-learn validates BOTH column count and column ORDER strictly
    # against the scaler's fit-time columns (feature_names_in_), so we
    # transform using exactly those columns, in that exact order — not a
    # subset, even though we only ultimately need the 9 feature columns.
    # 'trq_margin' is deliberately left in physical units afterwards (we
    # only keep the transformed feature columns, discarding the
    # transformed target column) so trq_margin_true stays interpretable in
    # the final report.
    regression_target_variable: str = params.get("regression_target_variable", "trq_margin")

    scaler_fit_cols: List[str] = list(reg_scaler.feature_names_in_)
    missing_scaler_cols = set(scaler_fit_cols) - set(df_reg_test.columns)
    if missing_scaler_cols:
        log.error(
            "[dual_model_inference] Step 3b - columns expected by the "
            "regression scaler are missing from regression test data: %s",
            missing_scaler_cols,
        )
        raise ValueError(f"regression_test_data missing scaler columns: {missing_scaler_cols}")

    log.info(
        "[dual_model_inference] Step 3b - applying fitted RobustScaler to "
        "%d columns (fit-time order): %s",
        len(scaler_fit_cols), scaler_fit_cols,
    )
    df_scaler_block = df_reg_test[scaler_fit_cols].copy()
    df_scaler_block[:] = reg_scaler.transform(df_scaler_block)

    # Only overwrite the FEATURE columns (leave the target column of
    # df_reg_test untouched, so trq_margin_true stays in physical units).
    scaler_feature_cols = [c for c in scaler_fit_cols if c != regression_target_variable]
    df_reg_test[scaler_feature_cols] = df_scaler_block[scaler_feature_cols]

    # Keep the target's fitted center/scale to inverse-transform mu/sigma
    # back to physical units after prediction (Step 7).
    target_idx = scaler_fit_cols.index(regression_target_variable)
    target_center: float = float(reg_scaler.center_[target_idx])
    target_scale: float = float(reg_scaler.scale_[target_idx])
    log.debug(
        "[dual_model_inference] Step 3b - target scaler params for '%s': "
        "center=%.6f, scale=%.6f",
        regression_target_variable, target_center, target_scale,
    )

    # ------------------------------------------------------------------
    # Step 4: extract configuration parameters
    # ------------------------------------------------------------------
    filter_threshold: float = params.get("filter_threshold", 0.5)
    mode: str = params.get("mode", "cascade")
    execution_order = params.get("execution_order", ["classification", "regression"])

    if mode not in ("cascade", "independent"):
        log.error("[dual_model_inference] Step 4 - unsupported mode: %s", mode)
        raise ValueError(f"Unsupported mode: {mode!r}. Must be 'cascade' or 'independent'.")

    log.info(
        "[dual_model_inference] Step 4 - config: mode=%s, filter_threshold=%s, "
        "execution_order=%s (NOTE: execution_order is informational only in "
        "'cascade' mode — regression is architecturally dependent on the "
        "classification mask and always runs second, regardless of this "
        "param's declared order)",
        mode, filter_threshold, execution_order,
    )

    # ------------------------------------------------------------------
    # Step 5: select model input features explicitly from config
    # ------------------------------------------------------------------
    clf_feature_cols = params.get("classification_features")
    reg_feature_cols = params.get("regression_features")

    if clf_feature_cols and reg_feature_cols and clf_feature_cols == reg_feature_cols:
        log.warning(
            "[dual_model_inference] Step 5 - classification_features and "
            "regression_features are IDENTICAL. This is suspicious: the two "
            "tasks use different targets (faulty vs trq_margin) and are "
            "expected to use different feature sets — in particular, "
            "'trq_margin' is normally a valid classification INPUT feature "
            "but must be EXCLUDED from regression (it's the regression "
            "target). Double-check this is intentional before proceeding."
        )

    X_clf_test = _select_explicit_features(
        df_clf_test, clf_feature_cols, fallback_drop_cols=["faulty"], context_label="classification",
    )
    log.info("[dual_model_inference] Step 5 - X_clf_test shape=%s", X_clf_test.shape)

    # ------------------------------------------------------------------
    # Step 6: classification stage — failure probability for every row
    # ------------------------------------------------------------------
    log.info("[dual_model_inference] Step 6 - running classification stage")
    probas = clf_model.predict_proba(X_clf_test)[:, 1]  # positive class
    probas_cal = calibrator.transform(probas.reshape(-1, 1)).ravel()

    mask_pass = probas_cal >= filter_threshold
    n_total = len(probas_cal)
    n_pass = int(mask_pass.sum())
    pass_ratio = 100.0 * n_pass / n_total if n_total > 0 else 0.0

    log.info(
        "[dual_model_inference] Step 6 - classification done: %d/%d rows pass "
        "filter_threshold=%.2f (%.2f%%)",
        n_pass, n_total, filter_threshold, pass_ratio,
    )
    if mode == "cascade" and pass_ratio < 20.0:
        log.warning(
            "[dual_model_inference] Step 6 - only %.2f%% of rows will receive a "
            "regression prediction in 'cascade' mode. This coverage gap is a "
            "direct consequence of the classifier's recall on the faulty "
            "class combined with filter_threshold=%.2f — consider mode="
            "'independent' if full regression coverage is required "
            "regardless of classification outcome.",
            pass_ratio, filter_threshold,
        )

    # ------------------------------------------------------------------
    # Step 7: regression stage — behavior depends on `mode`
    # ------------------------------------------------------------------
    log.info("[dual_model_inference] Step 7 - running regression stage (mode=%s)", mode)

    trq_mean: List[Optional[float]] = [None] * n_total
    trq_std: List[Optional[float]] = [None] * n_total

    if mode == "cascade":
        # Only rows that passed the classification threshold get scored.
        if n_pass > 0:
            df_reg_pass = df_reg_test.loc[mask_pass]
            X_reg_pass = _select_explicit_features(
                df_reg_pass, reg_feature_cols, fallback_drop_cols=["trq_margin"], context_label="regression",
            )
            log.info("[dual_model_inference] Step 7 - X_reg_pass shape=%s", X_reg_pass.shape)

            dists = reg_model.pred_dist(X_reg_pass)
            mu = dists.loc
            sigma = dists.scale

            pass_indices = np.where(mask_pass)[0]
            for idx, (m, s) in zip(pass_indices, zip(mu, sigma)):
                trq_mean[idx] = m
                trq_std[idx] = s

            log.info(
                "[dual_model_inference] Step 7 - regression predictions generated "
                "for %d/%d rows (cascade mode, gated by classification)",
                n_pass, n_total,
            )
        else:
            log.info(
                "[dual_model_inference] Step 7 - no rows passed the threshold; "
                "regression stage skipped entirely."
            )
    else:
        # mode == "independent": regression runs on every row, regardless of
        # the classification outcome.
        X_reg_all = _select_explicit_features(
            df_reg_test, reg_feature_cols, fallback_drop_cols=["trq_margin"], context_label="regression",
        )
        log.info("[dual_model_inference] Step 7 - X_reg_all shape=%s", X_reg_all.shape)

        dists = reg_model.pred_dist(X_reg_all)
        trq_mean = list(dists.loc)
        trq_std = list(dists.scale)

        log.info(
            "[dual_model_inference] Step 7 - regression predictions generated "
            "for %d/%d rows (independent mode, full coverage)",
            n_total, n_total,
        )

    # ------------------------------------------------------------------
    # Step 7b: inverse-transform mu/sigma from scaled space back to
    # physical units, so trq_margin_pred_mu/sigma are directly comparable
    # to trq_margin_true (both in physical torque-margin units) in the
    # final report. RobustScaler is an affine transform per column
    # (x_scaled = (x - center) / scale), so the inverse for a location
    # parameter (mu) re-applies both center and scale, while a scale
    # parameter (sigma) only re-applies the scale factor (a spread has no
    # position to shift).
    # ------------------------------------------------------------------
    trq_mean = [
        (m * target_scale + target_center) if m is not None else None
        for m in trq_mean
    ]
    trq_std = [
        (s * target_scale) if s is not None else None
        for s in trq_std
    ]
    log.info(
        "[dual_model_inference] Step 7b - mu/sigma converted back to "
        "physical units (center=%.4f, scale=%.4f)",
        target_center, target_scale,
    )

    # ------------------------------------------------------------------
    # Step 8: assemble the final predictions DataFrame
    # ------------------------------------------------------------------
    if "id" in df_clf_test.columns:
        ids = df_clf_test["id"].values
    else:
        ids = df_clf_test.index.values
        log.debug("[dual_model_inference] Step 8 - no 'id' column found; using row index instead")

    predictions_df = pd.DataFrame(
        {
            "id": ids,
            "faulty_true": df_clf_test["faulty"].values,
            "faulty_pred_prob": probas_cal,
            "faulty_pred": mask_pass.astype(int),
            "trq_margin_true": df_reg_test["trq_margin"].values,
            "trq_margin_pred_mu": trq_mean,
            "trq_margin_pred_sigma": trq_std,
        }
    )
    log.info("[dual_model_inference] Step 8 - assembled predictions_df shape=%s", predictions_df.shape)

    # ------------------------------------------------------------------
    # Step 9: persist predictions to disk
    # ------------------------------------------------------------------
    output_rel = params.get("output", "6.1.final_academic_predictions.parquet")
    phase6_dir = Path(ctx.phase6_dir).resolve()
    output_abs = phase6_dir / output_rel
    output_abs.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_parquet(output_abs, index=False)
    log.info("[dual_model_inference] Step 9 - predictions saved to %s", output_abs)

    # ------------------------------------------------------------------
    # Step 10: enrich context and return
    # ------------------------------------------------------------------
    ctx.predictions_df = predictions_df
    ctx.predictions_path = output_abs

    log.info("[dual_model_inference] ===== Step 6.1 - Academic Scoring ===== DONE")
    return ctx