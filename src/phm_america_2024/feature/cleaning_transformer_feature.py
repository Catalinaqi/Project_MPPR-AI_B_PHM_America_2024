from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def outlier_handling(
    df: pd.DataFrame,
    params: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> pd.DataFrame:
    """Clip target variable outliers at specified percentiles.

    If ``target_clipping`` is True, clips the column named in
    ``target_variable`` to the range [lower_percentile, upper_percentile].
    Persists a clipping audit log with pre/post statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame from step 3.1 (selected features).
    params : dict
        YAML configuration block:
        ``target_clipping`` (bool) – enable clipping.
        ``target_variable`` (str) – column to clip (e.g. ``"trq_margin"``).
        ``lower_percentile`` (float) – lower quantile (e.g. 0.01).
        ``upper_percentile`` (float) – upper quantile (e.g. 0.99).
    ctx : RunContext
        Context for logging (unused here but kept for interface consistency).
    output_dir : Path
        Directory to write audit JSON (``3.2.cleaning.clipping_audit_log.json``).

    Returns
    -------
    pd.DataFrame
        DataFrame with target variable clipped (in-place modification).
    """
    log.debug("[outlier_handling] entry – shape=%s", df.shape)

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
        lower_bound = df[col].quantile(lower_percentile)
        upper_bound = df[col].quantile(upper_percentile)

        n_before = len(df)
        audit["lower_bound_value"] = float(lower_bound)
        audit["upper_bound_value"] = float(upper_bound)
        audit["n_clipped_below"] = int((df[col] < lower_bound).sum())
        audit["n_clipped_above"] = int((df[col] > upper_bound).sum())

        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        log.info(
            "[outlier_handling] clipped '%s' to [%.4f, %.4f] – affected rows: %d below, %d above",
            col, lower_bound, upper_bound,
            audit["n_clipped_below"], audit["n_clipped_above"],
        )
    else:
        if target_clipping:
            log.warning("[outlier_handling] target_variable='%s' not found in columns", target_variable)
        audit["n_clipped_below"] = 0
        audit["n_clipped_above"] = 0

    # Persist audit log
    output_path = output_dir / "3.2.cleaning.clipping_audit_log.json"
    output_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    log.debug("[outlier_handling] audit written to %s", output_path)

    log.info("[outlier_handling] completed – shape=%s", df.shape)
    return df


def duplicate_handling(
    df: pd.DataFrame,
    params: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame.

    Uses the ``keep`` parameter from YAML (default ``"first"``).
    Persists a trace log with the number of duplicate rows removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (after outlier handling).
    params : dict
        YAML configuration block:
        ``keep`` (str) – which duplicate to keep (``"first"``, ``"last"``, or ``False``).
    ctx : RunContext
        Context for logging (unused but kept for interface consistency).
    output_dir : Path
        Directory to write trace JSON (``3.2.cleaning.duplicates_trace.json``).

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicate rows removed.
    """
    log.debug("[duplicate_handling] entry – shape=%s", df.shape)

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

    # Persist trace log
    output_path = output_dir / "3.2.cleaning.duplicates_trace.json"
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    log.debug("[duplicate_handling] trace written to %s", output_path)

    log.info("[duplicate_handling] completed – shape=%s", df.shape)
    return df