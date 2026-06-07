# src/phm_america_2024/phase/transformation_transformer_feature.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
from sklearn.preprocessing import RobustScaler

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def feature_scaling_old(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],  # params: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> tuple[pd.DataFrame, Optional[dict[str, Any]]]:
    """Apply RobustScaler to selected numerical features, fitted on training data.

    Fits a ``RobustScaler`` (median + IQR) on the features listed in
    ``params["features"]``, transforms them in-place, and returns the scaler
    object inside an extra-artifact dict for downstream persistence.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame after step 3.2 cleaning.
    params : dict
        YAML technique configuration:
        ``method`` (str) – must be ``"RobustScaler"``.
        ``fit_on_train_only`` (bool) – if True (default), fit on current data.
        ``features`` (list[str]) – columns to scale.
    ctx : Any
        RunContext (unused but retained for interface consistency).
    output_dir : Path
        Directory to write the scaler fit trace JSON.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any] | None]
        - Scaled DataFrame.
        - Extra-artifact dict with key ``"scaler"`` containing the fitted
          ``RobustScaler`` instance, or ``None`` if no features were scaled.
    """
    log.debug("[feature_scaling] entry – shape=%s", df.shape)
    log.info("[feature_scaling] Columnas actuales: %s", df.columns.tolist())

    # 1. EXTRACCIÓN CORRECTA: El diccionario real de parámetros vive dentro de "params"
    params = tech_cfg.get("params", {})

    # 2. Ahora sí, accedemos a lo que necesitamos
    method: str = params.get("method", "RobustScaler")
    log.info("[feature_scaling] metodos requeridas en YAML: %s", method)

    features: list[str] = params.get("features", [])
    log.info("[feature_scaling] Columnas requeridas en YAML: %s", features)

    # ----------------

    # 1. INICIALIZACIÓN SEGURA
    # Creamos el diccionario que retornaremos siempre, aunque esté vacío
    extra_artifacts = {}
    artifact_key = "fitted_scaler_regression_artifact"

    # Valores por defecto para el trace (si no se escala nada)
    trace = {
        "method": method,
        "features_scaled": [],
        "scaler_centers_": None,
        "scaler_scales_": None,
    }

    # Step 1: verify method is supported
    if method != "RobustScaler":
        log.warning(
            "[feature_scaling] unknown method='%s' – falling back to RobustScaler",
            method,
        )

    # Step 2: filter only existing numeric features
    existing_features = [
        col
        for col in features
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    missing = set(features) - set(existing_features)
    if missing:
        log.warning(
            "[feature_scaling] requested columns not found or non-numeric: %s", missing
        )

    if existing_features:
        # Step 3: fit scaler on the provided features
        scaler = RobustScaler()
        scaled_values = scaler.fit_transform(df[existing_features])
        df = df.copy()
        df[existing_features] = scaled_values

        log.info(
            "[feature_scaling] scaled %d features: %s",
            len(existing_features),
            existing_features,
        )
        # Actualizamos el trace con los valores reales
        trace["features_scaled"] = existing_features
        trace["scaler_centers_"] = scaler.center_.tolist()
        trace["scaler_scales_"] = scaler.scale_.tolist()

        # Guardamos el objeto escalador en el diccionario de retorno
        extra_artifacts[artifact_key] = {"scaler": scaler}

        log.info("[feature_scaling] scaled %d features", len(existing_features))
    else:
        log.warning(
            "[feature_scaling] no valid features to scale – returning unchanged"
        )

    # Step 4: persist fit trace
    extra_artifacts["trace"] = trace
    output_path = output_dir / "3.3.transformation.scaler_fit_trace.json"
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")

    log.info("[feature_scaling] completed – shape=%s", df.shape)

    # 5. RETORNO SEGURO
    return df, extra_artifacts


def feature_scaling(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],  # params: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> tuple[pd.DataFrame, Optional[dict[str, Any]]]:
    """Apply RobustScaler to selected numerical features, fitted on training data.

    Fits a ``RobustScaler`` (median + IQR) on the features listed in
    ``params["features"]``, transforms them in-place, and returns the scaler
    object inside an extra-artifact dict for downstream persistence.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame after step 3.2 cleaning.
    params : dict
        YAML technique configuration:
        ``method`` (str) – must be ``"RobustScaler"``.
        ``fit_on_train_only`` (bool) – if True (default), fit on current data.
        ``features`` (list[str] or str) – columns to scale, or "all" to auto-detect numeric columns.
    ctx : Any
        RunContext (unused but retained for interface consistency).
    output_dir : Path
        Directory to write the scaler fit trace JSON.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any] | None]
        - Scaled DataFrame.
        - Extra-artifact dict with key ``"scaler"`` containing the fitted
          ``RobustScaler`` instance, or ``None`` if no features were scaled.
    """
    import numpy as np  # Ensure numpy is available for numeric type checking

    log.debug("[feature_scaling] entry – shape=%s", df.shape)
    log.info("[feature_scaling] Columnas actuales: %s", df.columns.tolist())

    # Step 0: EXTRACT CONFIGURATION
    # El diccionario real de parámetros vive dentro de "params"
    params = tech_cfg.get("params", {})

    method: str = params.get("method", "RobustScaler")
    log.info("[feature_scaling] metodos requeridas en YAML: %s", method)

    features_config = params.get("features", [])
    log.info("[feature_scaling] Columnas requeridas en YAML: %s", features_config)

    # ---------------------------------------------------------
    # Step 0.5: DYNAMIC FEATURE SELECTION (Support for "all")
    # ---------------------------------------------------------
    if features_config == "all" or features_config == ["all"] or not features_config:
        # Auto-detect all numeric columns if requested or empty
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        log.info(
            "[feature_scaling] YAML requested 'all'. Auto-detected %d numeric columns.",
            len(features),
        )
    elif isinstance(features_config, str):
        # Handle accidental single string input (e.g., features: "oat")
        features = [features_config]
    else:
        # Use the explicit list provided in YAML
        features = features_config

    log.debug("[feature_scaling] Final features list to process: %s", features)
    # ---------------------------------------------------------

    # Step 1: SAFE INITIALIZATION
    # Creamos el diccionario que retornaremos siempre, aunque esté vacío
    extra_artifacts = {}
    artifact_key = "fitted_scaler_regression_artifact"

    # Default values for the trace (if nothing is scaled)
    trace = {
        "method": method,
        "features_scaled": [],
        "scaler_centers_": None,
        "scaler_scales_": None,
    }

    # Step 1.1: verify method is supported
    if method != "RobustScaler":
        log.warning(
            "[feature_scaling] unknown method='%s' – falling back to RobustScaler",
            method,
        )

    # Step 2: filter only existing numeric features as a final safeguard
    existing_features = [
        col
        for col in features
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Calculate missing columns only to warn the user
    missing = set(features) - set(existing_features)
    if missing:
        log.warning(
            "[feature_scaling] requested columns not found or non-numeric: %s", missing
        )

    if existing_features:
        # Step 3: fit scaler on the provided features
        scaler = RobustScaler()
        scaled_values = scaler.fit_transform(df[existing_features])
        df = df.copy()
        df[existing_features] = scaled_values

        log.info(
            "[feature_scaling] scaled %d features: %s",
            len(existing_features),
            existing_features,
        )

        # Step 3.1: Update the trace with real fitted values
        trace["features_scaled"] = existing_features
        trace["scaler_centers_"] = scaler.center_.tolist()
        trace["scaler_scales_"] = scaler.scale_.tolist()

        # Step 3.2: Save the scaler object in the return dictionary
        extra_artifacts[artifact_key] = {"scaler": scaler}

    else:
        log.warning(
            "[feature_scaling] no valid features to scale – returning unchanged"
        )

    # Step 4: persist fit trace to disk
    extra_artifacts["trace"] = trace
    output_path = output_dir / "3.3.transformation.scaler_fit_trace.json"
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")

    log.info("[feature_scaling] completed – shape=%s", df.shape)

    # Step 5: SAFE RETURN
    return df, extra_artifacts


def feature_engineering(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],  # params: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> tuple[pd.DataFrame, None]:
    """Create physical engineered features: Kelvin-protected ratios and polynomial interactions.

    Applies the formulas and interaction degree defined in the YAML config.
    The ``oat_denominator_protection`` parameter documents the Kelvin offset
    used in formulas (already applied explicitly in the expression strings).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (after scaling).
    params : dict
        YAML technique configuration:
        ``oat_denominator_protection`` (dict) – metadata (method, offset).
        ``formulas`` (dict[str, str]) – name → expression (evaluated with ``eval``).
        ``interactions`` (dict) –
            ``enabled`` (bool) – if True, generate polynomial features.
            ``degree`` (int) – polynomial degree (default 2).
            ``features`` (list[str]) – columns for interactions.
    ctx : Any
        RunContext (unused).
    output_dir : Path
        Directory to write the engineering trace JSON.

    Returns
    -------
    tuple[pd.DataFrame, None]
        - DataFrame with added engineered columns.
        - None (no extra artifacts).
    """

    # 1. EXTRACCIÓN CORRECTA
    params = tech_cfg.get("params", {})

    log.debug("[feature_engineering] entry – shape=%s", df.shape)

    formulas: dict[str, str] = params.get("formulas", {})
    interactions_cfg: dict = params.get("interactions", {})
    oat_protection: dict = params.get("oat_denominator_protection", {})
    _ = oat_protection  # documented in YAML; offset already in formulas

    engineering_log: dict[str, Any] = {
        "formulas_applied": [],
        "interactions": {
            "degree": interactions_cfg.get("degree", 2),
            "features_used": [],
        },
    }

    # Step 1: evaluate each formula safely
    df = df.copy()
    for name, expr in formulas.items():
        try:
            # Evaluate expression using available columns as local variables
            df[name] = eval(expr, {"__builtins__": {}}, df.to_dict("series"))
            engineering_log["formulas_applied"].append(
                {"name": name, "expression": expr}
            )
            log.debug("[feature_engineering] computed formula '%s' -> '%s'", name, expr)
        except Exception as exc:
            log.error("[feature_engineering] formula '%s' failed: %s", name, exc)
            engineering_log.setdefault("formula_errors", []).append(
                {"name": name, "error": str(exc)}
            )

    # Step 2: generate polynomial interactions if enabled
    interactions_enabled: bool = interactions_cfg.get("enabled", False)
    if interactions_enabled:
        degree: int = interactions_cfg.get("degree", 2)
        inter_features: list[str] = interactions_cfg.get("features", [])
        existing_inter = [
            col
            for col in inter_features
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        if degree >= 2 and len(existing_inter) >= 2:
            poly = PolynomialFeatures(
                degree=degree, interaction_only=False, include_bias=False
            )
            poly_values = poly.fit_transform(df[existing_inter])
            poly_feature_names = poly.get_feature_names_out(existing_inter)
            # Add new columns, avoid overwriting existing ones
            for i, poly_name in enumerate(poly_feature_names):
                if poly_name not in df.columns:
                    df[poly_name] = poly_values[:, i]
                else:
                    log.debug(
                        "[feature_engineering] interaction column '%s' already exists – skipping",
                        poly_name,
                    )
            engineering_log["interactions"]["features_used"] = existing_inter
            engineering_log["interactions"]["n_new_columns"] = len(poly_feature_names)
            log.info(
                "[feature_engineering] added %d polynomial interaction columns (degree=%d)",
                len(poly_feature_names),
                degree,
            )
        else:
            log.warning(
                "[feature_engineering] insufficient features for interactions: %s",
                existing_inter,
            )

    # Step 3: persist engineering log
    output_path = output_dir / "3.3.transformation.engineering_formulas_log.json"
    output_path.write_text(
        json.dumps(engineering_log, indent=2, default=str), encoding="utf-8"
    )
    log.debug("[feature_engineering] trace written to %s", output_path)

    log.info("[feature_engineering] completed – shape=%s", df.shape)
    return df, None
