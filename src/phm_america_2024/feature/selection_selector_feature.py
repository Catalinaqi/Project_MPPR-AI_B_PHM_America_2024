from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.pipeline.utils.context_facade_common import RunContext

log = get_logger(__name__)


def dataset_definition(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],  # <- CAMBIO 1: Riceviamo la config completa
    ctx: RunContext,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:  # <- CAMBIO 2: Ritorno polimorfico (Tupla)
    """Filter operational features and remove irrelevant technical columns."""
    log.debug("[dataset_definition] entry – input shape=%s", df.shape)

    # Step 1: Estrazione corretta dei parametri
    params = tech_cfg.get("params", {})
    include: list[str] = params.get("include", [])
    patterns_to_drop: list[str] = params.get("patterns_to_drop", [])

    original_shape = df.shape
    initial_cols = list(df.columns)

    # Step 2: build regex pattern for columns to drop
    drop_pattern = "|".join(patterns_to_drop) if patterns_to_drop else None
    if drop_pattern:
        cols_to_drop = [
            col for col in df.columns if re.search(drop_pattern, col, re.IGNORECASE)
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        log.debug("[dataset_definition] dropped columns=%s", cols_to_drop)
    else:
        cols_to_drop = []
        log.debug("[dataset_definition] no drop patterns – skipping")

    # Step 3: keep only explicit include columns that exist
    existing_include = [col for col in include if col in df.columns]
    missing_include = set(include) - set(existing_include)
    if missing_include:
        log.warning(
            "[dataset_definition] requested columns not found: %s", missing_include
        )

    df = df[existing_include]
    log.info(
        "[dataset_definition] after filtering shape=%s, include_columns=%s",
        df.shape,
        existing_include,
    )

    # Step 4: Costruiamo la traccia ma NON la salviamo su disco
    trace = {
        "input_shape": original_shape,
        "columns_present": initial_cols,
        "columns_after_drop_pattern": list(df.columns),
        "dropped_by_pattern": cols_to_drop,
        "include_requested": include,
        "include_kept": existing_include,
    }

    log.info("[dataset_definition] completed – final shape=%s", df.shape)

    # <- CAMBIO 3: Deleghiamo il salvataggio all'Orquestador
    return df, {"trace": trace}


def feature_selection(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],  # <- Riceviamo la config completa
    ctx: RunContext,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:  # <- Ritorno in Tupla
    """Remove constant columns and duplicate rows from the DataFrame."""
    log.debug("[feature_selection] entry – input shape=%s", df.shape)

    # Estrazione corretta dei parametri
    params = tech_cfg.get("params", {})
    remove_constant: bool = params.get("remove_constant", False)
    remove_duplicate: bool = params.get("remove_duplicate", False)

    original_shape = df.shape
    dropped_cols: list[str] = []
    dropped_rows: int = 0

    # Step 1: remove constant columns
    if remove_constant:
        nunique = df.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            df = df.drop(columns=constant_cols)
            dropped_cols = constant_cols
            log.debug("[feature_selection] dropped constant columns=%s", constant_cols)
        else:
            log.debug("[feature_selection] no constant columns found")

    # Step 2: drop duplicate rows
    if remove_duplicate:
        before = len(df)
        df = df.drop_duplicates(keep="first")
        dropped_rows = before - len(df)
        log.debug("[feature_selection] dropped %d duplicate rows", dropped_rows)

    # Step 3: Costruiamo la traccia
    trace = {
        "input_shape": original_shape,
        "output_shape": df.shape,
        "remove_constant": remove_constant,
        "constant_columns_dropped": dropped_cols,
        "remove_duplicate": remove_duplicate,
        "duplicate_rows_dropped": dropped_rows,
    }

    log.info("[feature_selection] completed – final shape=%s", df.shape)

    # Restituiamo la tupla
    return df, {"trace": trace}
