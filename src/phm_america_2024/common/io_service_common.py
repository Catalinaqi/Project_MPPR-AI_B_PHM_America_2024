# src/phm_america_2024/common/io_service_common.py
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, cast, Optional
import joblib
import numpy as np
import pandas as pd
from omegaconf import Container, DictConfig, ListConfig, OmegaConf

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.path_service_common import resolve_path


# io: input/output (write/read artifacts to disk)

log = get_logger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder supporting NumPy structures and OmegaConf containers."""

    def default(self, obj: Any) -> Any:
        # Step 1: If it is an OmegaConf container, CALL OmegaConf.to_container to resolve it
        if isinstance(obj, Container):
            return OmegaConf.to_container(obj, resolve=True)

        # Step 2: Handle standard NumPy generic scalars
        if isinstance(obj, (np.generic,)):
            return obj.item()

        # Step 3: Handle NumPy multidimensional arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


def _convert_configs(obj: Any) -> Any:
    """Recursively convert OmegaConf objects into vanilla python structures."""
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: _convert_configs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_configs(v) for v in obj]
    return obj


def save_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    compression: str,
) -> Path:
    """Save a pandas DataFrame into a compressed Parquet artifact."""
    # Step 1: Guard against persisting an empty DataFrame -- likely a bug.
    if df.empty:
        log.error(
            "[save_parquet] DataFrame is empty -- refusing to write "
            "an empty parquet to path=%s. "
            "Check the pipeline step that produced this DataFrame.",
            path,
        )
        raise ValueError(
            f"Cannot save an empty DataFrame to '{path}'. "
            f"An empty DataFrame at this stage indicates a pipeline bug."
        )

    # Step 2: CALL resolve_path() — resolve destination path to absolute location.
    resolved: Path = resolve_path(path)
    log.debug(
        "[save_parquet] resolved path=%s compression=%s rows=%d cols=%d",
        resolved,
        compression,
        len(df),
        df.shape[1],
    )

    # Step 3: CALL mkdir() — create parent directories if they do not exist.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    log.debug("[save_parquet] ensured parent dir=%s", resolved.parent)

    # Step 4: CALL to_parquet() — write parquet to disk.
    try:
        log.info(
            "[save_parquet] writing rows=%d cols=%d compression=%s path=%s",
            len(df),
            df.shape[1],
            compression,
            resolved,
        )
        df.to_parquet(resolved, compression=cast(Any, compression), index=False)
    except Exception:
        log.exception("[save_parquet] to_parquet failed path=%s", resolved)
        raise

    # Step 5: CALL stat() — calculate and log written file size for storage awareness.
    size_mb: float = resolved.stat().st_size / (1024**2)
    log.info(
        "[save_parquet] written size_mb=%.2f compression=%s path=%s",
        size_mb,
        compression,
        resolved,
    )
    return resolved


def save_pickle(
    obj: Any,
    path: str | Path,
) -> Path:
    """Serialize any pipeline artifact using standard pickle protocol."""
    # Step 1: Guard against persisting None -- likely a pipeline bug.
    if obj is None:
        log.error(
            "[save_pickle] object is None -- refusing to serialise None to path=%s. "
            "Check the pipeline step that produced this object.",
            path,
        )
        raise ValueError(
            f"Cannot pickle None to '{path}'. "
            f"A None object at this stage indicates a pipeline bug."
        )

    # Step 2: CALL resolve_path() — resolve destination path to absolute location.
    resolved: Path = resolve_path(path)
    log.debug(
        "[save_pickle] resolved path=%s object_type=%s",
        resolved,
        type(obj).__name__,
    )

    # Step 3: CALL mkdir() — create parent directories if they do not exist.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    log.debug("[save_pickle] ensured parent dir=%s", resolved.parent)

    # Step 4: CALL pickle.dump() — serialise object to disk.
    try:
        log.info(
            "[save_pickle] serialising object_type=%s path=%s",
            type(obj).__name__,
            resolved,
        )
        with resolved.open("wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        log.exception("[save_pickle] pickle.dump failed path=%s", resolved)
        raise

    # Step 5: CALL stat() — log written file size.
    size_mb: float = resolved.stat().st_size / (1024**2)
    log.info(
        "[save_pickle] written size_mb=%.3f object_type=%s path=%s",
        size_mb,
        type(obj).__name__,
        resolved,
    )
    return resolved


def save_json(
    obj: Any,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Save configurations, summaries or operational evaluation metrics to JSON."""
    # Step 1: Guard against persisting None -- likely a pipeline bug.
    if obj is None:
        log.error(
            "[save_json] object is None -- refusing to serialise None to path=%s. "
            "Check the pipeline step that produced this object.",
            path,
        )
        raise ValueError(
            f"Cannot save None to '{path}'. "
            f"A None object at this stage indicates a pipeline bug."
        )

    # Step 2: CALL resolve_path() — resolve destination path to absolute location.
    resolved: Path = resolve_path(path)
    log.debug(
        "[save_json] resolved path=%s object_type=%s indent=%s",
        resolved,
        type(obj).__name__,
        indent,
    )

    # Step 3: CALL mkdir() — create parent directories if they do not exist.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    log.debug("[save_json] ensured parent dir=%s", resolved.parent)

    # Step 4: CALL json.dump() — serialise object with custom _NumpyEncoder and config converters.
    try:
        log.info(
            "[save_json] writing object_type=%s path=%s",
            type(obj).__name__,
            resolved,
        )
        converted = _convert_configs(obj)
        with resolved.open("w", encoding="utf-8") as fh:
            json.dump(
                converted,
                fh,
                indent=indent,
                ensure_ascii=False,
                sort_keys=False,
                cls=_NumpyEncoder,
            )
    except TypeError as err:
        log.error(
            "[save_json] object contains non-JSON-serialisable types path=%s error=%s",
            resolved,
            err,
        )
        raise
    except Exception:
        log.exception("[save_json] json.dump failed path=%s", resolved)
        raise

    # Step 5: CALL stat() — calculate and log written file size.
    size_kb: float = resolved.stat().st_size / 1024
    log.info(
        "[save_json] written size_kb=%.2f object_type=%s path=%s",
        size_kb,
        type(obj).__name__,
        resolved,
    )
    return resolved


def save_numpy(
    arr: Any,
    path: str | Path,
) -> Path:
    """Save raw multidimensional structures in optimized binary NumPy (.npy) format."""
    # Step 1: Guard against None and validate target type.
    if arr is None:
        log.error(
            "[save_numpy] array is None -- refusing to save None to path=%s. "
            "Check the pipeline step that produced this array.",
            path,
        )
        raise ValueError(
            f"Cannot save None to '{path}'. "
            f"A None array at this stage indicates a pipeline bug."
        )

    if not isinstance(arr, np.ndarray):
        log.error(
            "[save_numpy] object is not a numpy array type=%s path=%s",
            type(arr).__name__,
            path,
        )
        raise ValueError(
            f"save_numpy() requires a numpy.ndarray, got {type(arr).__name__}."
        )

    # Step 2: CALL resolve_path() — resolve destination path to absolute location.
    resolved: Path = resolve_path(path)
    log.debug(
        "[save_numpy] resolved path=%s shape=%s dtype=%s",
        resolved,
        arr.shape,
        arr.dtype,
    )

    # Step 3: CALL mkdir() — create parent directories if they do not exist.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    log.debug("[save_numpy] ensured parent dir=%s", resolved.parent)

    # Step 4: CALL np.save() — save array to disk in NumPy binary format.
    try:
        log.info(
            "[save_numpy] saving shape=%s dtype=%s path=%s",
            arr.shape,
            arr.dtype,
            resolved,
        )
        np.save(resolved, arr, allow_pickle=False)
    except Exception:
        log.exception("[save_numpy] np.save failed path=%s", resolved)
        raise

    # Step 5: CALL stat() — calculate and log written file size.
    size_kb: float = resolved.stat().st_size / 1024
    log.info(
        "[save_numpy] written size_kb=%.2f shape=%s dtype=%s path=%s",
        size_kb,
        arr.shape,
        arr.dtype,
        resolved,
    )
    return resolved


# -----------------------------------------------------------------------------
# Existing artifact loaders (Parquet/Pickle) kept below...
# -----------------------------------------------------------------------------


def load_parquet(
    path: str | Path, *, columns: Optional[list[str]] = None
) -> pd.DataFrame:
    # Step 1: Resolve path to absolute location
    resolved = resolve_path(path)
    log.debug("[load_parquet] path=%s columns=%s", resolved, columns)

    # Step 2: Validate file existence
    if not resolved.exists():
        log.error("[load_parquet] parquet file not found path=%s", resolved)
        raise FileNotFoundError(f"Parquet file not found: {resolved}")

    # Step 3: Load parquet dataframe
    try:
        log.info("[load_parquet] loading path=%s", resolved)
        df: pd.DataFrame = pd.read_parquet(resolved, columns=columns)
    except Exception:
        log.exception("[load_parquet] read_parquet failed path=%s", resolved)
        raise

    # Step 4: Return loaded dataframe
    log.info(
        "[load_parquet] loaded rows=%d cols=%d path=%s", len(df), df.shape[1], resolved
    )
    return df


def load_pickle(path: str | Path) -> Any:
    # Step 1: Resolve path to absolute location
    resolved: Path = resolve_path(path)
    log.debug("[load_pickle] resolved path=%s", resolved)

    # Step 2: Validate file existence
    if not resolved.exists():
        log.error("[load_pickle] pickle file not found path=%s", resolved)
        raise FileNotFoundError(f"Pickle file not found: {resolved}")

    # Step 3: Log file size
    size_mb: float = resolved.stat().st_size / (1024**2)
    log.info("[load_pickle] loading size_mb=%.3f path=%s", size_mb, resolved)

    # Step 4: Deserialise object
    try:
        with resolved.open("rb") as fh:
            obj: Any = pickle.load(fh)  # noqa: S301
    except Exception:
        log.exception("[load_pickle] pickle.load failed path=%s", resolved)
        raise

    # Step 5: Return deserialised object
    log.info(
        "[load_pickle] loaded object_type=%s path=%s", type(obj).__name__, resolved
    )
    return obj


def load_pickle_joblib(path: str) -> Any:
    """Load a serialized Python object (pickle/joblib) from disk."""
    resolved = resolve_path(Path(path))

    if not resolved.exists():
        log.error(f"[load_pickle] pickle file not found path={resolved}")
        raise FileNotFoundError(f"Pickle file not found: {resolved}")

    size_mb = resolved.stat().st_size / (1024 * 1024)
    log.info(f"[load_pickle] loading size_mb={size_mb:.3f} path={resolved}")

    try:
        # Usar joblib en lugar de pickle nativo (Estándar para ML/NGBoost)
        obj: Any = joblib.load(str(resolved))
        return obj
    except Exception as e:
        log.error(f"[load_pickle] load failed path={resolved} error={e}")
        raise
