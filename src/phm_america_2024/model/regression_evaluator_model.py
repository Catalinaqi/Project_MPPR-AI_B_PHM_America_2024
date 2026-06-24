"""
Modulo: regression_evaluator_model.py
Algoritmo utilizzato: NGBoost (Natural Gradient Boosting) – già addestrato in precedenza, qui solo valutazione.
Metriche calcolate: Negative Log Likelihood (NLL) e Root Mean Squared Error (RMSE) sulle predizioni probabilistiche.

Flusso:
1. Legge i parametri dalla configurazione YAML (primary_metric, tie_breaker, output).
2. Recupera il nome della variabile target dal contesto (dalla configurazione del passo 4.2).
3. Recupera il percorso del dataset di validazione dalla configurazione globale (read_strategy → val_data).
4. Carica il file parquet di validazione usando la funzione load_parquet.
5. Pulisce i dati: sostituisce infiniti con NaN e rimuove righe con valori mancanti.
6. Separa X_val (feature) e y_val (target) escludendo la variabile target.
7. Chiama model.pred_dist(X_val) per ottenere le distribuzioni predittive.
8. Calcola NLL = media negativa del log-pdf delle distribuzioni sui valori reali.
9. Calcola RMSE = radice quadrata della media degli errori quadratici tra la media della distribuzione e y_val.
10. Salva le metriche in un file JSON di trace nella directory di output.
11. Restituisce il modello invariato e un dizionario extra contenente le metriche calcolate.

Import:
- json: per serializzare il trace.
- pathlib.Path: per gestire i percorsi.
- typing.Any: per annotazioni generiche.
- numpy: per operazioni numeriche (inf, sqrt, mean).
- pandas: per la gestione dei DataFrame (caricamento, pulizia).
- logging_adapter_common: logger personalizzato.
- io_service_common.load_parquet: caricamento di file Parquet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.io_service_common import load_parquet

log = get_logger(__name__)


# step_4_4_model_evaluation -> Evaluación del mejor modelo


def model_selection_criteria(
    model: Any,
    tech_cfg: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Evaluate best model on validation data and compute NLL + RMSE.

    Args:
        model: Trained NGBoost model from the previous step.
        tech_cfg: Dictionary containing technique configurations strictly from YAML.
        ctx: RunContext containing execution context.
        output_dir: Path to the pipeline step output directory.

    Returns:
        Tuple with the unchanged model and a metadata dictionary.
    """
    log.debug("[model_selection_criteria] Start - ENTER model_selection_criteria")

    # Step 1: Extract parameters strictly from YAML (Zero hardcoding)
    try:
        params = tech_cfg["params"]
        primary_metric: str = params["primary_metric"]
        tie_breaker: str = params["tie_breaker"]
        output_filename: str = tech_cfg["output"]

        # Extraemos dinámicamente la variable objetivo desde la configuración del paso 4.2
        target_variable: str = ctx.config.phases["phase4_data_modeling"]["steps"][
            "step_4_2_model_training"
        ]["methods"]["model_training"]["techniques"]["cross_validation"]["params"][
            "target_variable"
        ]

        log.info(
            "[model_selection_criteria] name for target_variable is: %s",
            target_variable,
        )

        # Extraemos la ruta de validación de la read_strategy global -> 3.5.formatting.regression_internal_val.parquet
        val_data_path: str = ctx.config.phases["phase4_data_modeling"]["read_strategy"][
            "input_source"
        ]["val_data"]

        log.debug("[model_selection_criteria] val_data_path is: %s", val_data_path)
    except KeyError as e:
        log.error(f"Missing required parameter in YAML configuration: {e}")
        raise ValueError(f"YAML configuration error: missing {e}")

    log.info(
        "[model_selection_criteria] Computing real evaluation metrics (NLL and RMSE)"
    )

    # Step 2: CALL load_parquet() — dynamically load validation split
    # Step 2: Resolve path correctly using phase3_dir
    # Obtenemos el directorio donde la Fase 3 guardó los archivos
    phase3_dir = getattr(ctx, "phase3_dir", None)
    if not phase3_dir:
        log.error(
            "[model_selection_criteria] phase3_dir not found in context. Cannot resolve validation path."
        )
        raise ValueError("phase3_dir missing in RunContext")

    # Construimos la ruta absoluta uniendo el directorio + el nombre del archivo
    full_val_path = Path(phase3_dir) / val_data_path

    log.debug("[model_selection_criteria] full_val_path is: %s", full_val_path)

    if not full_val_path.exists():
        log.error(
            f"[model_selection_criteria] Validation data not found at {full_val_path}"
        )
        raise FileNotFoundError(f"Validation file missing: {full_val_path}")

    val_data: pd.DataFrame = load_parquet(str(full_val_path))
    log.info(f"[model_selection_criteria] Loaded validation data from {full_val_path}")

    # Step 3: CALL replace() and dropna() — Remove Infinity and NaN values before predicting
    initial_len = len(val_data)
    val_data = val_data.replace([np.inf, -np.inf], np.nan).dropna()
    dropped_rows = initial_len - len(val_data)

    if dropped_rows > 0:
        log.warning(
            f"[model_selection_criteria] Dropped {dropped_rows} rows containing NaN "
            f"or Infinity from validation data."
        )

    # ----------------------------------------------------------------
    # X_val -> proviene de cargar el archivo "regression_internal_val.parquet"
    # El flujo: El modelo ya se entrenó en el paso 4.2 usando el
    # train_internal. Ahora, en el paso 4.4, carga el modelo guardado, carga el val_internal,
    # y hace model.pred_dist(X_val) para ver qué la nota del modelo en datos que no
    # usó para entrenar
    # Es su examen final dentro del laboratorio.

    # Step 4: Extract features and target strictly using the YAML target variable
    X_val = val_data.drop(columns=[target_variable])
    y_val = val_data[target_variable]

    # --------------------------
    # AÑADIR: log de la ruta del modelo (si está disponible)
    model_path = getattr(ctx, "model_path", None)
    if model_path:
        log.info("[model_selection_criteria] Model loaded from: %s", model_path)
    else:
        log.warning("[model_selection_criteria] No model_path in context")
    # --------------------------
    log.info(
        "[model_selection_criteria] Model type: %s, parameters: %s",
        type(model).__name__,
        model.get_params(),
    )
    # --------------------------

    # Step 5: CALL pred_dist() — predict distributions and calculate metrics
    dist = model.pred_dist(X_val)

    # NGBoost NLL: Promedio del log-likelihood negativo
    nll_score: float = float(-dist.logpdf(y_val).mean())

    # RMSE: Raíz del error cuadrático medio de la media
    y_pred_mean = dist.mean()
    rmse_score: float = float(np.sqrt(((y_pred_mean - y_val) ** 2).mean()))

    log.info(
        f"[model_selection_criteria] Metrics computed successfully: NLL"
        f"={nll_score:.4f}, RMSE={rmse_score:.4f}"
    )

    # Step 6: Format metrics and trace dictionaries (Model excluded to avoid JSON crash)
    metrics = {
        "neg_log_likelihood": nll_score,
        "rmse": rmse_score,
        "selected_by": primary_metric,
    }

    trace = {
        "primary_metric": primary_metric,
        "tie_breaker": tie_breaker,
        "metrics": metrics,
    }

    # Step 7: CALL write_text() — persist ranking trace to disk
    trace_json = json.dumps(trace, indent=2, default=str)
    output_path = output_dir / output_filename
    output_path.write_text(trace_json)

    log.info(
        "[model_selection_criteria] Model selection criteria evaluated successfully"
    )
    log.debug(f"[model_selection_criteria] Trace saved to {output_path}")

    # Extra returned metadata for the DAG registry
    extra = {"best_model_metadata": metrics}

    log.debug("[model_selection_criteria] End - EXIT model_selection_criteria")
    return model, extra
