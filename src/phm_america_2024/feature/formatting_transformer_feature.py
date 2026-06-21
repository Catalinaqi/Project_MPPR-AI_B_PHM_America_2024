# src/phm_america_2024/phase/formatting_transformer_feature.py
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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Partition the DataFrame into internal training, validation, and test splits."""
    log.debug(
        "[data_split] ENTRY - Iniciando proceso de partición. DataFrame original shape: %s",
        df.shape,
    )

    try:
        # 1. Extracción de parámetros
        test_size: float = params.get("test_size", 0.15)

        # LOG WARNING: Te avisa si te olvidaste de poner val_size en el YAML
        if "val_size" not in params:
            val_size: float = test_size
            log.warning(
                "[data_split] 'val_size' no especificado en YAML. Asumiendo el mismo valor que test_size: %.2f",
                val_size,
            )
        else:
            val_size: float = params["val_size"]

        random_state: int = params.get("random_state", 42)

        # --- NUEVO: Extraer parámetros de estratificación ---
        use_stratify: bool = params.get("stratify", False)
        target_col: str = params.get(
            "target_col", "faulty"
        )  # Por defecto faulty para clasificación

        log.info(
            "[data_split] Parámetros cargados: test_size=%.2f, val_size=%.2f, random_state=%d",
            test_size,
            val_size,
            random_state,
            use_stratify,
        )

        # LOG ERROR preventivo: Validación matemática
        if test_size + val_size >= 1.0:
            err_msg = f"Inconsistencia matemática: sum(test_size, val_size) = {test_size + val_size}. Debe ser menor a 1.0."
            log.error("[data_split] %s", err_msg)
            raise ValueError(err_msg)

        # Configurar la data de estratificación para el PRIMER corte
        stratify_data_1 = (
            df[target_col] if use_stratify and target_col in df.columns else None
        )

        # 2. PRIMER CORTE: Separar el Test Set del resto
        log.debug("[data_split] Ejecutando Primer Corte: Extrayendo Test Set...")
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_data_1,  # <--- SE APLICA AQUÍ
        )
        log.info(
            "[data_split] Primer Corte exitoso. Restante (Train+Val)=%s, Test=%s",
            train_val_df.shape,
            test_df.shape,
        )

        # 3. SEGUNDO CORTE: Separar Validation Set del Training restante
        val_ratio_of_remaining = val_size / (1.0 - test_size)

        # Configurar la data de estratificación para el SEGUNDO corte
        stratify_data_2 = (
            train_val_df[target_col]
            if use_stratify and target_col in train_val_df.columns
            else None
        )

        log.debug(
            "[data_split] Ejecutando Segundo Corte: Extrayendo Val Set (Proporción matemática ajustada: %.4f)...",
            val_ratio_of_remaining,
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_of_remaining,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_data_2,  # <--- SE APLICA AQUÍ
        )

        log.info(
            "[data_split] Partición finalizada con éxito -> Train: %s | Val: %s | Test: %s",
            train_df.shape,
            val_df.shape,
            test_df.shape,
        )

    except Exception as e:
        # LOG ERROR crítico: Captura cualquier falla de Scikit-Learn
        log.error(
            "[data_split] Falla crítica durante la división (train_test_split): %s",
            str(e),
            exc_info=True,
        )
        raise

    # 4. Trazabilidad de Índices
    log.debug(
        "[data_split] Recolectando índices para archivo de auditoría (traceability)..."
    )
    train_indices = train_df.index.tolist()
    val_indices = val_df.index.tolist()
    test_indices = test_df.index.tolist()

    trace = {
        "test_size_original": test_size,
        "val_size_original": val_size,
        "random_state": random_state,
        "train_shape": list(train_df.shape),
        "val_shape": list(val_df.shape),
        "test_shape": list(test_df.shape),
        "train_indices_sample": train_indices[:5],
        "val_indices_sample": val_indices[:5],
        "test_indices_sample": test_indices[:5],
    }

    # 5. Persistencia del JSON
    output_path = output_dir / "3.5.formatting.split_indices_trace.json"

    try:
        log.debug("[data_split] Escribiendo archivo JSON de traza en: %s", output_path)
        output_path.write_text(
            json.dumps(trace, indent=2, default=str), encoding="utf-8"
        )
        log.info("[data_split] Archivo de traza guardado correctamente.")
    except Exception as e:
        # LOG ERROR de I/O (ej. disco lleno o sin permisos)
        log.error(
            "[data_split] Error al escribir el archivo de traza JSON en disco: %s",
            str(e),
        )
        raise

    # 6. Empaquetado
    extra = {
        "trace": trace,
        "val_df": val_df,
        "test_df": test_df,
        "train_df": train_df,
    }

    log.debug(
        "[data_split] EXIT - Retornando DataFrame de entrenamiento y diccionario extra."
    )
    return train_df, extra


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
