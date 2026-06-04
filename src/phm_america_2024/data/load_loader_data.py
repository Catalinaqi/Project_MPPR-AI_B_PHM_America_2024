# src/phm_america_2024/data/load_loader_data.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional, Tuple
import joblib
import pandas as pd

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.configuration.enum_registry_config import ReadMode
from phm_america_2024.configuration.read_strategy_repository_config import (
    DataSourceConfig,
    ReadStrategyContract
)
from phm_america_2024.data.csv_loader_data import load_by_strategy


log = get_logger(__name__)


def load_train_merged(source: DataSourceConfig) -> Tuple[pd.DataFrame, Path]:
    """Load X_train sampled, and load Y_train efficiently avoiding full scan redundancy."""

    # Step 1: Resolve X_train path
    x_path = resolve_path(source.x_train_path)
    log.info("[load_train_merged] Starting merged load for X=%s, Y=%s", source.x_train_path, source.y_train_path)

    # =========================================================================
    # CORRECCIÓN DE ESTRATIFICACIÓN EN CALIENTE PARA X_train
    # =========================================================================
    # Como 'faulty' vive en Y_train, no podemos estratificar X_train en el chunking inicial.
    # Clonamos el contrato de lectura y lo convertimos a sampleo aleatorio simple temporalmente.
    x_strategy_dict = source.strategy.to_dict() if hasattr(source.strategy, "to_dict") else vars(source.strategy)
    x_tmp_dict = dict(x_strategy_dict)

    if x_tmp_dict.get("sample_method") == "stratified" and x_tmp_dict.get("stratify_column") == "faulty":
        log.info("[load_train_merged] Bypassing initial 'stratified' chunking on X_train (faulty belongs to Y_train). Forcing 'random' sampling.")
        x_tmp_dict["sample_method"] = "random"
        x_tmp_dict["stratify_column"] = "" # Limpiamos la columna para evitar falsos positivos en csv_loader_data

    x_safe_strategy = ReadStrategyContract.from_dict(x_tmp_dict)
    # =========================================================================

    # Step 2: CALL load_by_strategy() — Carga únicamente las 7,000 filas usando el contrato corregido
    df_x, _, _ = load_by_strategy(
        path=x_path,
        csv_params=source.csv_params,
        strategy=x_safe_strategy # <-- Usamos el contrato seguro modificado en caliente
    )

    if df_x is None or df_x.empty:
        log.error("[load_train_merged] X_train loaded empty from %s", x_path)
        raise ValueError(f"X_train is empty: {x_path}")

    # =========================================================================
    # OPTIMIZACIÓN ULTRA-EFICIENTE (CERO DESGASTE COMPUTACIONAL)
    # =========================================================================
    sampled_ids = set(df_x[source.join_key])
    y_path = resolve_path(source.y_train_path)

    log.info("[load_train_merged] Optimizing Y_train read. Streaming via chunks to filter %d target IDs...", len(sampled_ids))

    # Clonamos el contrato a través de su diccionario nativo para Y_train
    y_strategy_dict = dict(x_strategy_dict)
    y_strategy_dict["mode"] = "chunked"

    # Re-instanciamos de forma segura mediante el método factoría de la arquitectura
    y_chunk_strategy = ReadStrategyContract.from_dict(y_strategy_dict)

    # Llamamos al extractor en modo CHUNKED (obtenemos el generador)
    _, y_generator, _ = load_by_strategy(
        path=y_path,
        csv_params=source.csv_params,
        strategy=y_chunk_strategy
    )

    if y_generator is None:
        log.error("[load_train_merged] Failed to initialize Chunked Generator for Y_train.")
        raise ValueError("Y_train chunk generator is None.")

    # Filtramos los chunks en caliente reteniendo solo los IDs de la muestra
    y_filtered_chunks: list[pd.DataFrame] = []
    for chunk in y_generator:
        filtered_chunk = chunk[chunk[source.join_key].isin(sampled_ids)]
        if not filtered_chunk.empty:
            y_filtered_chunks.append(filtered_chunk)

    if not y_filtered_chunks:
        log.error("[load_train_merged] No matching IDs found in Y_train for the selected X sample.")
        raise ValueError("Y_train filtering resulted in an empty dataset.")

    # Consolidamos las 7,000 filas del objetivo extraídas en streaming
    df_y = pd.concat(y_filtered_chunks, ignore_index=True)
    # =========================================================================

    # Step 5: Merge final en memoria de dos dataframes de idéntico tamaño reducido
    log.info(
        "[load_train_merged] Merging optimized subsets on key='%s'. X rows=%d, Y rows=%d",
        source.join_key, len(df_x), len(df_y)
    )

    df_merged: pd.DataFrame = pd.merge(df_x, df_y, on=source.join_key, how="inner")

    # Step 6: Log final combined dimensions and return
    log.info(
        "[load_train_merged] Merge complete. Final dimensions: rows=%d, cols=%d",
        len(df_merged), df_merged.shape[1]
    )
    return df_merged, x_path


def load_test(source: DataSourceConfig) -> Tuple[pd.DataFrame, Path]:
    """Load X_test full using safe architecture deserialization."""
    test_path = resolve_path(source.x_test_path)
    log.info("[load_test] Loading X_test from %s", test_path)

    # Clonación segura e inyección dinámica para Test
    strategy_dict = source.strategy.to_dict() if hasattr(source.strategy, "to_dict") else vars(source.strategy)
    test_strategy_dict = dict(strategy_dict)

    # CORRECCIÓN: Aseguramos el valor correcto aceptado 'full'
    test_strategy_dict["mode"] = "full"

    strategy = ReadStrategyContract.from_dict(test_strategy_dict)

    df_test, _, _ = load_by_strategy(path=test_path, csv_params=source.csv_params, strategy=strategy)
    if df_test is None:
        raise ValueError("X_test is None.")

    log.info("[load_test] Loaded rows=%d, cols=%d", len(df_test), df_test.shape[1])
    return df_test, test_path


def load_validation(source: DataSourceConfig) -> Tuple[pd.DataFrame, Path]:
    """Load X_validation full using safe architecture deserialization."""
    val_path = resolve_path(source.x_validation_path)
    log.info("[load_validation] Loading X_validation from %s", val_path)

    # Clonación segura e inyección dinámica para Validación
    strategy_dict = source.strategy.to_dict() if hasattr(source.strategy, "to_dict") else vars(source.strategy)
    val_strategy_dict = dict(strategy_dict)

    # CORRECCIÓN: Aseguramos el valor correcto aceptado 'full'
    val_strategy_dict["mode"] = "full"

    strategy = ReadStrategyContract.from_dict(val_strategy_dict)

    df_val, _, _ = load_by_strategy(path=val_path, csv_params=source.csv_params, strategy=strategy)
    if df_val is None:
        raise ValueError("X_validation is None.")

    log.info("[load_validation] Loaded rows=%d, cols=%d", len(df_val), df_val.shape[1])
    return df_val, val_path
# -----------------------------------------------------------------------------
# Existing artifact loaders (Parquet/Pickle) kept below...
# -----------------------------------------------------------------------------

def load_parquet(path: str | Path, *, columns: Optional[list[str]] = None) -> pd.DataFrame:
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
    log.info("[load_parquet] loaded rows=%d cols=%d path=%s", len(df), df.shape[1], resolved)
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
    log.info("[load_pickle] loaded object_type=%s path=%s", type(obj).__name__, resolved)
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