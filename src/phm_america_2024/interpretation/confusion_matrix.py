# src/phm_america_2024/phase/pipeline_auditor_interpretation.py
import json
from pathlib import Path
from typing import Any, Tuple
import sklearn

import numpy as np
import pandas as pd

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)

# step_5_3_process_audit


def confusion_matrix(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    log.debug("[confusion_matrix] entry")

    params = tech_cfg.get("params", {})
    output_file = tech_cfg["output"]
    test_df = getattr(ctx, "df_test", None)
    model = ctx.model
    calibrator = ctx.calibrator

    results: dict[str, Any] = {}

    y_true = test_df[['faulty']]
    
    X = test_df.drop('faulty', axis=1)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    y_calib = calibrator.predict(y_proba[:,0])
    y_calib = np.append(y_calib.reshape(-1,1), 1-y_calib.reshape(-1,1), axis=1)

    matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    results["true_positive"] = int(matrix[1][1])
    results["false_negative"] = int(matrix[1][0])
    results["false_positive"] = int(matrix[0][1])
    results["true_negative"] = int(matrix[0][0])

    brier_score = sklearn.metrics.brier_score_loss(y_true, y_calib)

    # Write trace JSON
    out_path = output_dir / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Leakage detection written to %s", out_path)

    log.debug("[leakage_detection] completed")
    return df, {
            "confusion_matrix_results": results,
            "brier_score": brier_score
            }
