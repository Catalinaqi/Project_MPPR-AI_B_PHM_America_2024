# src/phm_america_2024/phase/cluster_interpreter_interpretation.py
from __future__ import annotations

from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from sklearn.inspection import permutation_importance as sk_perm_importance

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.pipeline.utils.context_facade_common import RunContext
from phm_america_2024.reporting.plots_generator_reporting import (
    plot_feature_importance,
    plot_permutation_importance,
)

log = get_logger(__name__)


def feature_importance_old(
    df: pd.DataFrame,
    tech_cfg: Dict[str, Any],
    ctx: RunContext,
    output_dir: Any,  # Ignorado, el registro se encarga del path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute model native feature importance extraction and visualization logic."""
    log.debug("[feature_importance] entry")

    # --- 1. LOGS DE DIAGNÓSTICO Y SEGURIDAD PARA EL DATAFRAME ---
    if df is None:
        log.error(
            "[feature_importance] CRÍTICO: 'df' ha llegado como None. Revisa el paso anterior o la carga de datos."
        )
        raise ValueError(
            "El DataFrame 'df' es None. No se puede calcular la importancia."
        )
    else:
        log.info("[feature_importance] 'df' recibido correctamente. Tipo: %s", type(df))
        if hasattr(df, "shape"):
            log.info("[feature_importance] Forma del 'df': %s", df.shape)
            log.debug("[feature_importance] Columnas del 'df': %s", df.columns.tolist())
        else:
            log.warning(
                "[feature_importance] 'df' NO tiene el atributo 'shape'. ¿Es realmente un DataFrame de Pandas?"
            )
    # -----------------------------------------------------------

    model: NGBRegressor = getattr(ctx, "model", None)
    if model is None:
        log.error(
            "[feature_importance] CRÍTICO: No se encontró el modelo NGBoost en el contexto."
        )
        raise RuntimeError("Context is missing the trained NGBoost model.")

    target_col: str = getattr(ctx, "target_col", "trq_margin")
    log.info("[feature_importance] target_col a remover: '%s'", target_col)

    # --- 2. MANEJO SEGURO DEL DROP ---
    try:
        X_test: pd.DataFrame = df.drop(columns=[target_col], errors="ignore")
        log.debug("[feature_importance] Drop de target_col exitoso.")
    except Exception as e:
        log.error(
            "[feature_importance] Fallo al hacer drop de '%s'. Error: %s", target_col, e
        )
        raise
    # ---------------------------------

    feature_names: List[str] = list(X_test.columns)

    params: Dict[str, Any] = tech_cfg.get("params", {})
    strategy: str = params.get("strategy", "model_native")
    top_k: int = params.get("top_k", 15)

    importance_data: Dict[str, Any] = {}
    fig = None

    if strategy == "model_native":
        tree_imps: List[np.ndarray] = []
        for stage in model.ensemble:
            for tree in stage:
                if hasattr(tree, "feature_importances_"):
                    tree_imps.append(tree.feature_importances_)

        if tree_imps:
            avg: np.ndarray = np.mean(tree_imps, axis=0)
            sorted_idx: np.ndarray = np.argsort(avg)[::-1][:top_k]
            importance_data = {feature_names[i]: float(avg[i]) for i in sorted_idx}
            log.info(
                "[feature_importance] successfully calculated top %d features", top_k
            )
        else:
            log.warning(
                "[feature_importance] No se encontraron 'feature_importances_' en los árboles."
            )
            importance_data = {"error": "No tree importances found in model ensemble."}
    else:
        log.warning("[feature_importance] Estrategia desconocida: '%s'", strategy)
        importance_data = {"error": f"Unknown strategy '{strategy}'."}

    # Generar la figura en memoria (sin guardarla)
    if "error" not in importance_data:
        fig = plot_feature_importance(importance_data, top_k)

    log.debug("[feature_importance] exit")

    # RETORNO PURO DE OBJETOS EN MEMORIA
    return df, {
        "feature_importance_data": importance_data,
        "feature_importance_plot": fig,
    }


def permutation_importance(
    df: pd.DataFrame,
    tech_cfg: Dict[str, Any],
    ctx: RunContext,
    output_dir: Any,  # Ignorado
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute permutation importance evaluation and visualization logic."""
    log.debug("[permutation_importance] ENTRY")

    # --- 1. VALIDACIÓN SEGURA DEL DATAFRAME ---
    if df is None:
        log.error(
            "[permutation_importance] CRÍTICO: 'df' ha llegado como None. Abortando técnica."
        )
        raise ValueError("El DataFrame 'df' es None. No se puede calcular permutación.")
    else:
        log.debug(
            "[permutation_importance] 'df' validado. Shape: %s",
            getattr(df, "shape", "Desconocido"),
        )

    # --- 2. VALIDACIÓN DEL CONTEXTO ---
    model: NGBRegressor = getattr(ctx, "model", None)
    y_test: np.ndarray = getattr(ctx, "y_true", None)

    if model is None or y_test is None:
        log.error(
            "[permutation_importance] CRÍTICO: Modelo o 'y_true' faltante en el contexto."
        )
        raise RuntimeError("Context is missing the trained model or true targets.")

    target_col: str = getattr(ctx, "target_col", "trq_margin")

    # --- 3. DROP SEGURO ---
    log.debug("[permutation_importance] Removiendo target_col: '%s'", target_col)
    try:
        X_test: pd.DataFrame = df.drop(columns=[target_col], errors="ignore")
    except Exception as e:
        log.error(
            "[permutation_importance] Fallo crítico al hacer drop de '%s': %s",
            target_col,
            e,
        )
        raise

    feature_names: List[str] = list(X_test.columns)

    params: Dict[str, Any] = tech_cfg.get("params", {})
    n_repeats: int = params.get("n_repeats", 5)
    scoring: str = params.get("scoring", "neg_log_likelihood")
    log.info(
        "[permutation_importance] Iniciando. Scoring: '%s', Repeats: %d",
        scoring,
        n_repeats,
    )

    # --- 4. FUNCIÓN SCORER ---
    def nll_scorer(estimator: Any, X: pd.DataFrame, y: np.ndarray) -> float:
        dist = estimator.pred_dist(X)
        return float(-dist.logpdf(y).sum())

    scoring_func = nll_scorer if scoring == "neg_log_likelihood" else None

    if scoring_func is None:
        log.warning(
            "[permutation_importance] Fallback al scorer por defecto de Sklearn. Se solicitó scoring='%s'",
            scoring,
        )

    # --- 5. PERMUTACIÓN SEGURA (Puede tardar, necesita trazabilidad) ---
    try:
        log.debug(
            "[permutation_importance] Ejecutando sk_perm_importance... (Esto puede tomar tiempo)"
        )
        perm_result = sk_perm_importance(
            model,
            X_test,
            y_test,
            scoring=scoring_func,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=1,
        )
        log.info("[permutation_importance] Permutación calculada con éxito.")
    except Exception as e:
        log.error(
            "[permutation_importance] Error interno en sklearn permutation_importance: %s",
            e,
        )
        raise

    perm_data: Dict[str, Any] = {
        "feature_names": feature_names,
        "importances_mean": perm_result.importances_mean.tolist(),
        "importances_std": perm_result.importances_std.tolist(),
        "importances": perm_result.importances.tolist(),
    }

    # --- 6. PLOT SEGURO ---
    log.debug("[permutation_importance] Generando plot de permutación en memoria...")
    fig = None
    try:
        fig = plot_permutation_importance(perm_data, scoring, top_k=15)
        log.debug("[permutation_importance] Figura generada exitosamente.")
    except Exception as e:
        log.error("[permutation_importance] Error al graficar permutación: %s", e)

    log.debug("[permutation_importance] EXIT")

    # RETORNO PURO DE OBJETOS EN MEMORIA
    return df, {"permutation_data": perm_data, "permutation_plot": fig}


def feature_importance(
    df: pd.DataFrame,
    tech_cfg: Dict[str, Any],
    ctx: RunContext,
    output_dir: Any,  # Ignorado, el registro se encarga del path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute model native feature importance extraction and visualization logic."""
    import numpy as np

    log.debug("[feature_importance] entry")

    # --- 1. LOGS DE DIAGNÓSTICO Y SEGURIDAD PARA EL DATAFRAME ---
    if df is None:
        log.error(
            "[feature_importance] CRÍTICO: 'df' ha llegado como None. Revisa el paso anterior o la carga de datos."
        )
        raise ValueError(
            "El DataFrame 'df' es None. No se puede calcular la importancia."
        )
    else:
        log.info("[feature_importance] 'df' recibido correctamente. Tipo: %s", type(df))
        if hasattr(df, "shape"):
            log.info("[feature_importance] Forma del 'df': %s", df.shape)
            log.debug("[feature_importance] Columnas del 'df': %s", df.columns.tolist())
        else:
            log.warning(
                "[feature_importance] 'df' NO tiene el atributo 'shape'. ¿Es realmente un DataFrame de Pandas?"
            )
    # -----------------------------------------------------------

    model: Any = getattr(ctx, "model", None)
    if model is None:
        log.error(
            "[feature_importance] CRÍTICO: No se encontró el modelo NGBoost en el contexto."
        )
        raise RuntimeError("Context is missing the trained NGBoost model.")

    target_col: str = getattr(ctx, "target_col", "trq_margin")
    log.info("[feature_importance] target_col a remover: '%s'", target_col)

    # --- 2. MANEJO SEGURO DEL DROP ---
    try:
        X_test: pd.DataFrame = df.drop(columns=[target_col], errors="ignore")
        log.debug("[feature_importance] Drop de target_col exitoso.")
    except Exception as e:
        log.error(
            "[feature_importance] Fallo al hacer drop de '%s'. Error: %s", target_col, e
        )
        raise
    # ---------------------------------

    feature_names: List[str] = list(X_test.columns)

    params: Dict[str, Any] = tech_cfg.get("params", {})
    strategy: str = params.get("strategy", "model_native")
    top_k: int = params.get("top_k", 15)

    importance_data: Dict[str, Any] = {}
    fig = None

    # --- 3. EXTRACCIÓN DE IMPORTANCIAS (NGBOOST COMPATIBLE) ---
    if strategy == "model_native":
        if hasattr(model, "feature_importances_"):
            # Obtenemos las importancias. NGBoost ya las promedia internamente.
            imps = model.feature_importances_

            # NGBoost devuelve un array 2D: (parámetros_distribución, num_features)
            # El índice 0 corresponde a la media (loc). Los modelos tradicionales devuelven 1D.
            if isinstance(imps, np.ndarray) and len(imps.shape) > 1:
                log.debug(
                    "[feature_importance] Array 2D detectado (NGBoost). Extrayendo importancias para 'loc' (índice 0)."
                )
                avg_imps = imps[0]
            else:
                log.debug(
                    "[feature_importance] Array 1D detectado (Modelo tradicional)."
                )
                avg_imps = imps

            # Validamos que el número de importancias coincida con el de columnas
            if len(avg_imps) != len(feature_names):
                log.error(
                    "[feature_importance] Mismatch: el modelo arrojó %d importancias pero hay %d features.",
                    len(avg_imps),
                    len(feature_names),
                )
                importance_data = {"error": "Feature count mismatch."}
            else:
                # Ordenar y tomar el Top K
                sorted_idx: np.ndarray = np.argsort(avg_imps)[::-1][:top_k]
                importance_data = {
                    feature_names[i]: float(avg_imps[i]) for i in sorted_idx
                }
                log.info(
                    "[feature_importance] successfully calculated top %d features",
                    len(importance_data),
                )
        else:
            log.warning(
                "[feature_importance] No se encontró 'feature_importances_' en el modelo."
            )
            importance_data = {
                "error": "Model does not have feature_importances_ attribute."
            }

    else:
        log.warning("[feature_importance] Estrategia desconocida: '%s'", strategy)
        importance_data = {"error": f"Unknown strategy '{strategy}'."}

    # --- 4. GENERACIÓN DEL PLOT ---
    # Generar la figura en memoria (sin guardarla)
    if "error" not in importance_data:
        try:
            fig = plot_feature_importance(importance_data, top_k)
            log.debug("[feature_importance] Figura generada exitosamente.")
        except Exception as e:
            log.error("[feature_importance] Error generando el plot: %s", e)

    log.debug("[feature_importance] exit")

    # RETORNO PURO DE OBJETOS EN MEMORIA
    return df, {
        "feature_importance_data": importance_data,
        "feature_importance_plot": fig,
    }
