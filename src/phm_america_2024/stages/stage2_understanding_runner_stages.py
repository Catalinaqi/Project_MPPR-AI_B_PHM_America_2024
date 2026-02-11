# src/phm_america_2024/stages/stage2_understanding_runner_stages.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.core.resolve_path_utils_core import resolve_path
from phm_america_2024.data.load_utils_data import ReadStrategy, read_csv
from phm_america_2024.reporting.artifacts_service_reporting import save_table_png_pretty_all
from phm_america_2024.reporting.plots_utils_reporting import save_fig
from phm_america_2024.reporting.tables_utils_reporting import as_table_by_column_heuristic, safe_describe
from phm_america_2024.core.helpers_utils_core import to_json_log,to_json_log_compact
from phm_america_2024.data.quality_rules_utils_data import numeric_cols

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
    '''
    :param cfg: /config/pipeline/*_pipeline_config.yml
        version [...]
        pipeline [name,task,objective,variables]
        runtime [...]
        stages [...]
    :param variables: /config/pipeline/*_pipeline_config.yml
        x_train_path
        y_train_path
        x_test_path
        x_validation_path
        join_key
        label_col
    :param output_root: /out
        path for folder out
    :return:
    '''


    log.info("[run_stage2] data from the root of the yaml: version [] and pipeline [] (cfg):\n%s", to_json_log(cfg))
    log.info("[run_stage2] data from the root of the yaml: variables ... | runtime [] | stages []:\n%s", to_json_log(variables))
    log.info("[run_stage2] output path is: %s" , output_root)

    payload = {"cfg": cfg, "variables": variables, "output_root": str(output_root)}
    log.info("[run_stage2] params=%s", to_json_log_compact(payload))

    # ============================================================
    # STAGE 2 — DATA UNDERSTANDING (PHASE 2)
    # stage2_understanding
    # ============================================================

    stage_cfg = cfg["stages"]["stage2_understanding"]
    if not stage_cfg.get("enabled", False):
        log.info("Stage2 disabled.")
        return {}

    log.info("=== STAGE 2 START [run_stage2] ===")
    di = stage_cfg["dataset_input"]
    # paths
    x_path_raw = variables.get("x_train_path", di["path"])
    y_path_raw = variables.get("y_train_path", di["labels_input"]["path"])
    x_path = resolve_path(x_path_raw, variables, output_root)
    y_path = resolve_path(y_path_raw, variables, output_root)

    log.info("[run_stage2] Stage2 paths resolved: x_path=%s (exists=%s) | y_path=%s (exists=%s)",
             x_path, x_path.exists(), y_path, y_path.exists())

    # Read strategy
    read_cfg = di.get("read_strategy", {})

    mode = read_cfg.get("mode")  # default neutro
    sample_rows = int(read_cfg.get("sample_rows"))  # None se manca
    random_state = int(read_cfg.get("random_state"))  # 42 ok come default globale
    chunk_size = int(read_cfg.get("chunk_size"))  # None se manca

    # Validazione minima
    if mode not in {"full", "sample", "chunked"}:
        raise ValueError(f"Invalid read_strategy.mode={mode}")

    if sample_rows <= 0:
        raise ValueError("sample_rows must be > 0")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")


    strategy = ReadStrategy(
        mode=mode,
        sample_rows=sample_rows,
        random_state=random_state,
        chunk_size=chunk_size,
    )

    # Read csv params
    read_csv_params = di.get("csv_params", {})
    sep = read_csv_params.get("sep")
    encoding = read_csv_params.get("encoding")
    decimal = read_csv_params.get("decimal")
    low_memory = read_csv_params.get("low_memory")


    csv_params = {
        "sep": sep,
        "encoding": encoding,
        "decimal": decimal,
        "low_memory": low_memory,
    }



    # Basic tables
    out_pol = stage_cfg["output_policy"]

    tables_dir = out_pol["tables_png_dir"]
    figs_dir = out_pol["figures_dir"]

    # add
    pretty_dpi = int(out_pol.get("dpi"))          # fallback su dpi già esistente
    #pretty_max_rows = int(out_pol.get("max_rows"))
    pretty_float_fmt = str(out_pol.get("float_fmt"))
    #pretty_align = str(out_pol.get("align", "left"))



    # ============================================================
    # steps: step_2_1_data_acquisition
    # ============================================================
    # Load data (with strategy)
    X = read_csv(x_path, csv_params=csv_params, strategy=strategy)
    Y = read_csv(y_path, csv_params=csv_params, strategy=strategy)


    steps = stage_cfg.get("steps", {})

    # ============================================================
    # Step 2.2 Describe data (uses YAML params)
    # ============================================================
    s22 = steps.get("step_2_2_describe_data", {})
    if s22.get("enabled", False):
        techs = s22.get("techniques", {})
        name =s22.get("name")

        # --- Technique 1: descriptive_statistics.describe
        ds = techs.get("descriptive_statistics", {})
        if ds.get("enabled", False):
            methods = ds.get("methods", {})

            # -- Method 1: describe() with params
            m_desc = methods.get("describe", {})
            if m_desc.get("enabled", False):
                p = m_desc.get("params", {})
                include = p.get("include", "all")
                numeric_only = bool(p.get("numeric_only", False))

                # --- uso ---
                x_desc = safe_describe(X, include=include, numeric_only=numeric_only)
                y_desc = safe_describe(Y, include=include, numeric_only=numeric_only)

                #x_tbl = as_table_by_column(x_desc, original_columns=list(X.columns))
                #y_tbl = as_table_by_column(y_desc, original_columns=list(Y.columns))

                x_tbl = as_table_by_column_heuristic(x_desc, original_columns=list(X.columns), context="Stage2/X")
                y_tbl = as_table_by_column_heuristic(y_desc, original_columns=list(Y.columns), context="Stage2/Y")


                save_table_png_pretty_all(
                    x_tbl,
                    output_root,
                    f"{tables_dir}/{name}_x_descriptive.png",
                    title=f"X.describe(include={include}, numeric_only={numeric_only})",
                    dpi=pretty_dpi,
                    float_fmt=pretty_float_fmt
                )

                save_table_png_pretty_all(
                    y_tbl, output_root, f"{tables_dir}/{name}_y_descriptive.png",
                    title=f"Y.describe(include={include}, numeric_only={numeric_only})",
                    dpi=pretty_dpi, float_fmt=pretty_float_fmt
                )

            # --- Method 2: min/max/mean/std with numeric-only option
            m_mm = methods.get("min_max_mean_std", {})
            if m_mm.get("enabled", False):
                p = m_mm.get("params", {})
                numeric_only = bool(p.get("numeric_only", True))
                cols_x = numeric_cols(X, exclude=["id"]) if numeric_only else list(X.columns)
                cols_y = numeric_cols(Y, exclude=["id"]) if numeric_only else list(Y.columns)

                def _mm(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
                    d = pd.DataFrame(index=cols)
                    d["min"] = df[cols].min()
                    d["max"] = df[cols].max()
                    d["mean"] = df[cols].mean()
                    d["std"] = df[cols].std()
                    return d

                save_table_png_pretty_all(
                    as_table_by_column_heuristic(_mm(X, cols_x)),
                    output_root,
                    f"{tables_dir}/{name}_x_min_max_mean_std.png",
                    "X min/max/mean/std",
                    dpi=pretty_dpi)
                save_table_png_pretty_all(as_table_by_column_heuristic(_mm(Y, cols_y)), output_root, f"{tables_dir}/{name}_y_min_max_mean_std.png",
                               "Y min/max/mean/std", dpi=pretty_dpi)


        # --- Technique 2: schema_inspection
        si = techs.get("schema_inspection", {})
        if si.get("enabled", False):
            methods = si.get("methods", {})

            # -- Method 1: dtype analysis (count of dtypes in X and Y)
            if methods.get("dtype_analysis", {}).get("enabled", False):
                x_dtype = pd.DataFrame({"dtype": X.dtypes.astype(str)})
                y_dtype = pd.DataFrame({"dtype": Y.dtypes.astype(str)})


                save_table_png_pretty_all(as_table_by_column_heuristic(x_dtype), output_root, f"{tables_dir}/{name}_x_dtype_analysis.png", "X dtypes", dpi=pretty_dpi)
                save_table_png_pretty_all(as_table_by_column_heuristic(y_dtype), output_root, f"{tables_dir}/{name}_y_dtype_analysis.png", "Y dtypes", dpi=pretty_dpi)

            # -- Method 2: cardinality count (number of unique values per column, sorted desc, top N)
            if methods.get("cardinality_count", {}).get("enabled", False):
                p = methods["cardinality_count"].get("params", {})
                max_u = int(p.get("max_unique_to_report", 50))

                x_card = pd.DataFrame({"unique": X.nunique(dropna=False)}).sort_values("unique", ascending=False).head(max_u)
                y_card = pd.DataFrame({"unique": Y.nunique(dropna=False)}).sort_values("unique", ascending=False).head(max_u)


                save_table_png_pretty_all(as_table_by_column_heuristic(x_card), output_root, f"{tables_dir}/{name}_x_cardinality.png", "X cardinality", dpi=pretty_dpi)
                save_table_png_pretty_all(as_table_by_column_heuristic(y_card), output_root, f"{tables_dir}/{name}_y_cardinality.png", "Y cardinality", dpi=pretty_dpi)

            # -- Method 3: missing values count and ratio
            if methods.get("null_count", {}).get("enabled", False):
                x_null = pd.DataFrame({"missing_count": X.isna().sum(), "missing_ratio": (X.isna().sum()/len(X)).round(6)})
                y_null = pd.DataFrame({"missing_count": Y.isna().sum(), "missing_ratio": (Y.isna().sum()/len(Y)).round(6)})


                save_table_png_pretty_all(as_table_by_column_heuristic(x_null.sort_values("missing_ratio", ascending=False)),
                               output_root, f"{tables_dir}/{name}_x_missing.png", "X missing", dpi=pretty_dpi)
                save_table_png_pretty_all(as_table_by_column_heuristic(y_null.sort_values("missing_ratio", ascending=False)),
                           output_root, f"{tables_dir}/{name}_y_missing.png", "Y missing", dpi=pretty_dpi)

    # ============================================================
    # Step 2.3 Data quality assessment
    # ============================================================
    s23 = steps.get("step_2_3_data_quality_assessment", {})
    if s23.get("enabled", False):
        methods = s23.get("methods", {})
        name = s23.get("name")

        # percentile_analysis
        # percentiles: [ 0.01, 0.05, 0.95, 0.99, 0.999 ]
        if methods.get("percentile_analysis", {}).get("enabled", False):
            p = methods["percentile_analysis"].get("params", {})
            percentiles = p.get("percentiles", [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
            #percentiles = p.get("percentiles", [0.25, 0.5, 0.75])
            cols_x = numeric_cols(X, exclude=["id"])
            cols_y = numeric_cols(Y, exclude=["id"])

            def _percentiles(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
                if not cols:
                    return pd.DataFrame()
                qs = df[cols].quantile(percentiles).T  # index = cols, columns = percentiles
                qs.columns = [f"p{int(q*1000)/10:g}" for q in percentiles]  # p1, p5, p99.9, etc.
                return qs

            save_table_png_pretty_all(as_table_by_column_heuristic(_percentiles(X, cols_x)), output_root, f"{tables_dir}/{name}_x_percentiles.png",
                               "X Percentiles", dpi=pretty_dpi)
            save_table_png_pretty_all(as_table_by_column_heuristic(_percentiles(Y, cols_y)), output_root, f"{tables_dir}/{name}_y_percentiles.png",
                               "Y Percentiles", dpi=pretty_dpi)

    # ============================================================
    # Step 2.4 EDA - Histograms for X + Y (including faulty)
    # ============================================================
    s24 = steps.get("step_2_4_eda", {})
    if s24.get("enabled", False):
        methods = s24.get("methods", {})
        name = s24.get("name")

        hist_cfg = methods.get("histograms", {})
        if hist_cfg.get("enabled", False):
            p = hist_cfg.get("params", {})
            bins = int(p.get("bins", 30))
            max_cols = int(p.get("max_columns", 20))

            # X numeric hist (exclude id)
            x_num = numeric_cols(X, exclude=["id"])[:max_cols]
            for c in x_num:
                plt.figure()
                X[c].hist(bins=bins)
                plt.title(f"X Histogram: {c}")
                save_fig(output_root / f"{figs_dir}/{name}_hist_x_{c}.png", dpi=pretty_dpi)

            # Y: faulty as bar, trq_margin as hist (exclude id)
            if "faulty" in Y.columns:
                plt.figure()
                Y["faulty"].value_counts(dropna=False).sort_index().plot(kind="bar")
                plt.title("Y Distribution: faulty")
                save_fig(output_root / f"{figs_dir}/{name}_bar_y_faulty.png", dpi=pretty_dpi)

            y_num = [c for c in numeric_cols(Y, exclude=["id", "faulty"])][:max_cols]
            for c in y_num:
                plt.figure()
                Y[c].hist(bins=bins)
                plt.title(f"Y Histogram: {c}")
                save_fig(output_root / f"{figs_dir}/{name}_hist_y_{c}.png", dpi=pretty_dpi)

    log.info("=== STAGE 2 END [run_stage2] ===")
    return {"X_sample": X, "Y_sample": Y}








