from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def data_split(
    df: pd.DataFrame,
    params: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Partition the DataFrame into internal training and validation splits.

    Uses ``train_test_split`` with ``test_size`` and ``random_state``
    from the YAML configuration.  Persists a trace log with the split
    indices and shapes.

    Parameters
    ----------
    df : pd.DataFrame
        Full transformed DataFrame (from step 3.3).
    params : dict
        YAML technique configuration:
        ``test_size`` (float) – proportion for validation (default 0.2).
        ``random_state`` (int) – random seed (default 42).
    ctx : Any
        RunContext (unused but retained for interface consistency).
    output_dir : Path
        Directory to write the split trace JSON
        (``3.5.formatting.split_indices_trace.json``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]
        - Training DataFrame.
        - Validation DataFrame.
        - Extra-artifact dict with keys ``"train_indices"`` and
          ``"val_indices"`` (list of original index positions).
    """
    log.debug("[data_split] entry – shape=%s", df.shape)

    test_size: float = params.get("test_size", 0.2)
    random_state: int = params.get("random_state", 42)

    # Step 1: perform train/val split (preserving index for traceability)
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    log.info(
        "[data_split] split – train=%s, val=%s",
        train_df.shape,
        val_df.shape,
    )

    # Step 2: collect index positions for traceability
    train_indices = train_df.index.tolist()
    val_indices = val_df.index.tolist()

    # Step 3: persist trace log
    trace = {
        "test_size": test_size,
        "random_state": random_state,
        "train_shape": list(train_df.shape),
        "val_shape": list(val_df.shape),
        "train_indices_sample": train_indices[:5],  # first 5 for sanity
        "val_indices_sample": val_indices[:5],
    }
    output_path = output_dir / "3.5.formatting.split_indices_trace.json"
    output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    log.debug("[data_split] trace written to %s", output_path)

    # Step 4: return DataFrames and extra artifacts (indices)
    # extra = {
    #     "train_indices": train_indices,
    #     "val_indices": val_indices,
    # }
    # log.info("[data_split] completed")
    # return train_df, val_df, extra

    # Empaquetamos todo en el diccionario extra
    extra = {
        "trace": trace,
        "val_df": val_df,  # <- El val_df viaja aquí dentro
        "train_indices": train_df.index.tolist(),
        "val_indices": val_df.index.tolist(),
    }

    return train_df, {
        "trace": trace,
        "val_df": val_df,
        "train_df": train_df,  # <-- Verifica que esto no sea None
    }


def dataset_formatting(
    df: pd.DataFrame,
    params: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cast target column to specified dtype and optionally convert to numpy arrays.

    Ensures the target column (``target_col``) is cast to ``type_casting``
    (e.g. ``float64``). If ``to_numpy`` is True, also converts the entire
    DataFrame to a numpy array representation stored in the extra artifact.
    Persists an arrays manifest JSON with column metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Training or validation split (data_split outputs).
    params : dict
        YAML technique configuration:
        ``target_col`` (str) – name of the target variable (``"trq_margin"``).
        ``type_casting`` (str) – dtype to cast target (``"float64"``).
        ``to_numpy`` (bool) – if True, convert features + target to numpy
        arrays and include them in the extra artifact.
    ctx : Any
        RunContext (unused).
    output_dir : Path
        Directory to write the manifest JSON
        (``3.5.formatting.arrays_manifest.json``).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        - DataFrame with target column cast to the specified type.
        - Extra-artifact dict with keys:
          - ``"target_dtype"`` – the new dtype.
          - ``"to_numpy_applied"`` – whether numpy conversion was done.
          - ``"arrays"`` (optional) – dict with ``"X"`` and ``"y"`` numpy
            arrays if ``to_numpy=True``.
    """
    log.debug("[dataset_formatting] entry – shape=%s", df.shape)

    target_col: str = params.get("target_col", "trq_margin")
    type_casting: str = params.get("type_casting", "float64")
    to_numpy: bool = params.get("to_numpy", False)

    # Step 1: cast target column
    if target_col in df.columns:
        df = df.copy()
        df[target_col] = df[target_col].astype(type_casting)
        log.debug("[dataset_formatting] cast '%s' to %s", target_col, type_casting)
    else:
        log.warning(
            "[dataset_formatting] target_col '%s' not found – skipping cast", target_col
        )

    # Step 2: optionally convert to numpy arrays
    arrays: dict[str, Any] = {}
    if to_numpy and target_col in df.columns:
        # Identify feature columns (exclude target)
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].to_numpy()
        y = df[target_col].to_numpy()
        arrays = {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "feature_names": feature_cols,
        }
        log.debug(
            "[dataset_formatting] converted to numpy – X=%s, y=%s", X.shape, y.shape
        )
    elif to_numpy:
        log.warning("[dataset_formatting] cannot convert to numpy – target_col missing")

    # Step 3: persist arrays manifest
    manifest = {
        "target_col": target_col,
        "target_dtype": type_casting,
        "to_numpy_applied": to_numpy,
        "arrays_metadata": arrays,
    }
    output_path = output_dir / "3.5.formatting.arrays_manifest.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    log.debug("[dataset_formatting] manifest written to %s", output_path)

    # Step 4: build extra artifact
    extra: dict[str, Any] = {
        "target_dtype": type_casting,
        "to_numpy_applied": to_numpy,
    }
    if arrays:
        extra["arrays"] = arrays

    log.info(
        "[dataset_formatting] completed – shape=%s, dtype=%s", df.shape, type_casting
    )
    return df, extra
