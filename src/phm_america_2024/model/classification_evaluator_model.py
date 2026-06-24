"""
Modulo: classification_evaluator_model.py
Algoritmo utilizzato: LightGBM (Gradient Boosting Decision Trees) per classificazione binaria – già addestrato in precedenza, qui solo valutazione.
Metriche calcolate: Brier Score (errore quadratico medio delle probabilità) e ROC AUC (area sotto la curva ROC).

Flusso:
1. Legge i parametri dalla configurazione YAML (primary_metric, tie_breaker, output).
2. Recupera il nome della variabile target dal contesto (dalla configurazione del passo 4.2).
3. Recupera il percorso del dataset di validazione dalla configurazione globale (read_strategy → val_data).
4. Carica il file parquet di validazione usando la funzione load_parquet.
5. Pulisce i dati: sostituisce infiniti con NaN e rimuove righe con valori mancanti.
6. Separa X_val (feature) e y_val (target) escludendo la variabile target.
7. Chiama model.predict_proba(X_val) per ottenere le probabilità della classe positiva.
8. Calcola Brier Score (brier_score_loss) e ROC AUC (roc_auc_score).
9. Salva le metriche in un file JSON di trace nella directory di output.
10. Restituisce il modello invariato e un dizionario extra contenente le metriche calcolate.

Import:
- json: per serializzare il trace.
- pathlib.Path: per gestire i percorsi.
- typing.Any: per annotazioni generiche.
- numpy: per manipolazione numerica.
- pandas: per la gestione dei DataFrame (caricamento, pulizia).
- sklearn.metrics.brier_score_loss: calcolo del Brier Score.
- sklearn.metrics.roc_auc_score: calcolo dell'area sotto la curva ROC.
- logging_adapter_common: logger personalizzato.
- io_service_common.load_parquet: caricamento di file Parquet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.io_service_common import load_parquet

log = get_logger(__name__)


# step_4_4_model_evaluation -> Evaluación del mejor modelo (clasificación calibrada)


def model_selection_criteria(
    model: Any,
    tech_cfg: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Evaluate best classifier on validation data and compute Brier Score + ROC AUC.

    Args:
        model: Trained LightGBM model from the previous step.
        tech_cfg: Dictionary containing technique configurations strictly from YAML.
        ctx: RunContext containing execution context.
        output_dir: Path to the pipeline step output directory.

    Returns:
        Tuple with the unchanged model and a metadata dictionary containing metrics.
    """
    log.debug("ENTER model_selection_criteria (classification)")

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

        # Extraemos la ruta de validación de la read_strategy global
        val_data_path: str = ctx.config.phases["phase4_data_modeling"]["read_strategy"][
            "input_source"
        ]["val_data"]
    except KeyError as e:
        log.error(f"Missing required parameter in YAML configuration: {e}")
        raise ValueError(f"YAML configuration error: missing {e}")

    log.info("Computing classification metrics: Brier Score and ROC AUC")

    # Step 2: Load validation data
    phase3_dir = getattr(ctx, "phase3_dir", None)
    if not phase3_dir:
        log.error("phase3_dir not found in context. Cannot resolve validation path.")
        raise ValueError("phase3_dir missing in RunContext")

    full_val_path = Path(phase3_dir) / val_data_path
    if not full_val_path.exists():
        log.error(f"Validation data not found at {full_val_path}")
        raise FileNotFoundError(f"Validation file missing: {full_val_path}")

    val_data: pd.DataFrame = load_parquet(str(full_val_path))
    log.info(f"Loaded validation data from {full_val_path}")

    # Step 3: Sanitize data
    initial_len = len(val_data)
    val_data = val_data.replace([np.inf, -np.inf], np.nan).dropna()
    dropped_rows = initial_len - len(val_data)
    if dropped_rows > 0:
        log.warning(
            f"Dropped {dropped_rows} rows containing NaN/Inf from validation data."
        )

    # Step 4: Extract features and target
    X_val = val_data.drop(columns=[target_variable])
    y_val = val_data[target_variable]

    # Step 5: Predict probabilities for the positive class
    y_proba = model.predict_proba(X_val)[:, 1]

    # Step 6: Compute metrics
    brier = float(brier_score_loss(y_val, y_proba))
    roc_auc = float(roc_auc_score(y_val, y_proba))

    log.info(f"Metrics computed: Brier={brier:.6f}, ROC AUC={roc_auc:.6f}")

    # Step 7: Build metrics dict
    metrics = {
        "brier_score": brier,
        "roc_auc": roc_auc,
        "selected_by": primary_metric,
    }

    # Step 8: Write trace JSON
    trace = {
        "primary_metric": primary_metric,
        "tie_breaker": tie_breaker,
        "metrics": metrics,
    }
    trace_json = json.dumps(trace, indent=2, default=str)
    output_path = output_dir / output_filename
    output_path.write_text(trace_json)
    log.info(f"Trace written to {output_path}")

    # Step 9: Return metadata for DAG registry
    extra = {"best_model_metadata": metrics}

    log.debug("EXIT model_selection_criteria (classification)")
    return model, extra
