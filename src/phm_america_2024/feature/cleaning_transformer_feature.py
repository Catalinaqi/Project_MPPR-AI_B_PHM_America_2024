from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def outlier_handling(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],  # <- CAMBIO: Recibimos tech_cfg
    ctx: Any,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:  # <- CAMBIO: Retornamos tupla
    """Clip target variable outliers at specified percentiles."""
    log.debug("[outlier_handling] entry – shape=%s", df.shape)

    # Extraemos los parámetros de forma segura
    params = tech_cfg.get("params", {})
    target_clipping: bool = params.get("target_clipping", False)
    target_variable: str = params.get("target_variable", "")
    lower_percentile: float = params.get("lower_percentile", 0.0)
    upper_percentile: float = params.get("upper_percentile", 1.0)

    audit: dict[str, Any] = {
        "target_clipping_enabled": target_clipping,
        "target_variable": target_variable,
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
    }

    if target_clipping and target_variable and target_variable in df.columns:
        col = target_variable
        lower_bound = float(df[col].quantile(lower_percentile))
        upper_bound = float(df[col].quantile(upper_percentile))

        audit["lower_bound_value"] = lower_bound
        audit["upper_bound_value"] = upper_bound
        audit["n_clipped_below"] = int((df[col] < lower_bound).sum())
        audit["n_clipped_above"] = int((df[col] > upper_bound).sum())

        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        log.info(
            "[outlier_handling] clipped '%s' to [%.4f, %.4f] – affected rows: %d below, %d above",
            col,
            lower_bound,
            upper_bound,
            audit["n_clipped_below"],
            audit["n_clipped_above"],
        )
    else:
        if target_clipping:
            log.warning(
                "[outlier_handling] target_variable='%s' not found in columns",
                target_variable,
            )
        audit["n_clipped_below"] = 0
        audit["n_clipped_above"] = 0

    log.info("[outlier_handling] completed – shape=%s", df.shape)

    # <- CAMBIO: Empaquetamos la auditoría bajo la llave "trace"
    return df, {"trace": audit}


def duplicate_handling(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],  # <- CAMBIO: Recibimos tech_cfg
    ctx: Any,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:  # <- CAMBIO: Retornamos tupla
    """Remove duplicate rows from the DataFrame."""
    log.debug("[duplicate_handling] entry – shape=%s", df.shape)

    # Extraemos los parámetros
    params = tech_cfg.get("params", {})
    keep: str | bool = params.get("keep", "first")

    before = len(df)
    df = df.drop_duplicates(keep=keep)
    dropped = before - len(df)

    trace = {
        "keep_strategy": keep,
        "rows_before": before,
        "rows_after": len(df),
        "duplicate_rows_dropped": dropped,
    }
    log.info("[duplicate_handling] dropped %d duplicate rows (keep=%s)", dropped, keep)

    log.info("[duplicate_handling] completed – shape=%s", df.shape)

    # <- CAMBIO: Devolvemos la tupla
    return df, {"trace": trace}
