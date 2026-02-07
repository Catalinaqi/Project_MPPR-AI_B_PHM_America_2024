# src/phm_america_2024/stages/stage2_understanding_runner_stages.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.core.resolve_path_utils_core import resolve_path
from phm_america_2024.data.load_utils_data import ReadStrategy, read_csv
from phm_america_2024.reporting.artifacts_service_reporting import save_table_png
from phm_america_2024.reporting.plots_utils_reporting import save_fig

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Stage 2 (Data Understanding): load + describe + quality assessment + EDA.
# IMPORTANT: Must not modify data (report-only).
#
# Program flow expectation:
# - Pipeline runner calls run_stage2(...) and receives loaded sample dfs.
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - Stage runner (orchestrates services)
# =============================================================================


def run_stage2(cfg: Dict[str, Any], variables: Dict[str, Any], output_root: Path) -> Dict[str, Any]:

    log.info("[run_stage2] cfg is: %s and variables is: %s and output is: %s" , cfg, variables, output_root)

    stage_cfg = cfg["stages"]["stage2_understanding"]
    if not stage_cfg.get("enabled", False):
        log.info("Stage2 disabled.")
        return {}

    log.info("=== STAGE 2 START [run_stage2] ===")

    di = stage_cfg["dataset_input"]

    x_path_raw = variables.get("x_train_path", di["path"])
    y_path_raw = variables.get("y_train_path", di["labels_input"]["path"])

    x_path = resolve_path(x_path_raw, variables, output_root)
    y_path = resolve_path(y_path_raw, variables, output_root)


    log.info("[run_stage2] Stage2 paths resolved: x_path=%s (exists=%s) | y_path=%s (exists=%s)",
             x_path, x_path.exists(), y_path, y_path.exists())

    read_cfg = di.get("read_strategy", {})
    strategy = ReadStrategy(
        mode=read_cfg.get("mode", "sample"),
        sample_rows=int(read_cfg.get("sample_rows", 200000)),
        random_state=int(read_cfg.get("random_state", 42)),
        chunk_size=int(read_cfg.get("chunk_size", 200000)),
    )

    # Use defaults from dataset_config if you want; for now keep simple CSV params.
    csv_params = {"sep": ",", "encoding": "utf-8", "decimal": ".", "low_memory": False}

    X = read_csv(x_path, csv_params=csv_params, strategy=strategy)
    Y = read_csv(y_path, csv_params=csv_params, strategy=strategy)


    # Basic tables
    out_pol = stage_cfg["output_policy"]
    dpi = int(out_pol.get("dpi", 150))
    tables_dir = out_pol["tables_png_dir"]
    figs_dir = out_pol["figures_dir"]

    save_table_png(X.describe(include="all"), output_root, f"{tables_dir}/x_describe.png", "X.describe()", dpi=dpi)
    save_table_png(Y.describe(include="all"), output_root, f"{tables_dir}/y_describe.png", "Y.describe()", dpi=dpi)

    # Missing values table
    miss = pd.DataFrame({
        "missing_count": X.isna().sum(),
        "missing_ratio": (X.isna().sum() / len(X)).round(6),
    }).sort_values("missing_ratio", ascending=False)
    save_table_png(miss.reset_index().rename(columns={"index": "column"}),
                   output_root, f"{tables_dir}/x_missing.png", "X missing values", dpi=dpi)

    # Histograms (numeric)
    numeric_cols = [c for c in X.columns if c != "id" and pd.api.types.is_numeric_dtype(X[c])]
    for c in numeric_cols[: min(10, len(numeric_cols))]:
        plt.figure()
        X[c].hist(bins=30)
        plt.title(f"Histogram: {c}")
        save_fig(output_root / f"{figs_dir}/hist_{c}.png", dpi=dpi)

    log.info("=== STAGE 2 END [run_stage2] ===")
    return {"X_sample": X, "Y_sample": Y}


