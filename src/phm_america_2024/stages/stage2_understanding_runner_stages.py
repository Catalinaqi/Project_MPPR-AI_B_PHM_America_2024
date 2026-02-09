# src/phm_america_2024/stages/stage2_understanding_runner_stages.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.core.resolve_path_utils_core import resolve_path
from phm_america_2024.data.load_utils_data import ReadStrategy, read_csv
from phm_america_2024.reporting.artifacts_service_reporting import save_table_png, save_table_png_pretty
from phm_america_2024.reporting.plots_utils_reporting import save_fig
from phm_america_2024.reporting.tables_utils_reporting import save_table_png_pretty_all, as_table_by_column_heuristic

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
    # paths
    x_path_raw = variables.get("x_train_path", di["path"])
    y_path_raw = variables.get("y_train_path", di["labels_input"]["path"])
    x_path = resolve_path(x_path_raw, variables, output_root)
    y_path = resolve_path(y_path_raw, variables, output_root)

    log.info("[run_stage2] Stage2 paths resolved: x_path=%s (exists=%s) | y_path=%s (exists=%s)",
             x_path, x_path.exists(), y_path, y_path.exists())

    # Read strategy
    read_cfg = di.get("read_strategy", {})
    strategy = ReadStrategy(
        mode=read_cfg.get("mode", "sample"),
        sample_rows=int(read_cfg.get("sample_rows", 200000)),
        random_state=int(read_cfg.get("random_state", 42)),
        chunk_size=int(read_cfg.get("chunk_size", 200000)),
    )

    csv_params = {"sep": ",", "encoding": "utf-8", "decimal": ".", "low_memory": False}

    # Load data (with strategy)
    X = read_csv(x_path, csv_params=csv_params, strategy=strategy)
    Y = read_csv(y_path, csv_params=csv_params, strategy=strategy)

    # Basic tables
    out_pol = stage_cfg["output_policy"]
    dpi = int(out_pol.get("dpi", 150))
    tables_dir = out_pol["tables_png_dir"]
    figs_dir = out_pol["figures_dir"]

    # add
    pretty_dpi = int(out_pol.get("dpi", dpi))          # fallback su dpi già esistente
    pretty_max_rows = int(out_pol.get("max_rows", 50))
    pretty_float_fmt = str(out_pol.get("float_fmt", "{:.3f}"))
    pretty_align = str(out_pol.get("align", "left"))

    #
    # # Technique: use describe() tables as a proxy for "data understanding" and save as PNGs for the report.
    # save_table_png(X.describe(include="all"), output_root, f"{tables_dir}/x_describe.png", "X.describe()", dpi=dpi)
    # save_table_png(Y.describe(include="all"), output_root, f"{tables_dir}/y_describe.png", "Y.describe()", dpi=dpi)
    #
    # # Missing values table
    # miss = pd.DataFrame({
    #     "missing_count": X.isna().sum(),
    #     "missing_ratio": (X.isna().sum() / len(X)).round(6),
    # }).sort_values("missing_ratio", ascending=False)
    # save_table_png(miss.reset_index().rename(columns={"index": "column"}),
    #                output_root, f"{tables_dir}/x_missing.png", "X missing values", dpi=dpi)
    #
    # # Histograms (numeric)
    # numeric_cols = [c for c in X.columns if c != "id" and pd.api.types.is_numeric_dtype(X[c])]
    # for c in numeric_cols[: min(10, len(numeric_cols))]:
    #     plt.figure()
    #     X[c].hist(bins=30)
    #     plt.title(f"Histogram: {c}")
    #     save_fig(output_root / f"{figs_dir}/hist_{c}.png", dpi=dpi)
    #
    # log.info("=== STAGE 2 END [run_stage2] ===")
    # return {"X_sample": X, "Y_sample": Y}


    steps = stage_cfg.get("steps", {})

    # ============================================================
    # Step 2.2 Describe data (uses YAML params)
    # ============================================================
    s22 = steps.get("step_2_2_describe_data", {})
    if s22.get("enabled", False):
        techs = s22.get("techniques", {})

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

                #x_desc = X.describe(include=include, numeric_only=numeric_only)
                #y_desc = Y.describe(include=include, numeric_only=numeric_only)

                # --- uso ---
                x_desc = safe_describe(X, include=include, numeric_only=numeric_only)
                y_desc = safe_describe(Y, include=include, numeric_only=numeric_only)

                # save tables as PNG for report: method 'describe' with params in title
                # save_table_png(_as_table_by_column(x_desc), output_root, f"{tables_dir}/x_descriptive_.png",
                #                f"X.describe(include={include}, numeric_only={numeric_only})", dpi=dpi)
                # save_table_png(_as_table_by_column(y_desc), output_root, f"{tables_dir}/y_descriptive_.png",
                #                f"Y.describe(include={include}, numeric_only={numeric_only})", dpi=dpi)

                x_tbl = as_table_by_column(x_desc, original_columns=list(X.columns))
                y_tbl = as_table_by_column(y_desc, original_columns=list(Y.columns))

                save_table_png_pretty(
                    x_tbl, output_root, f"{tables_dir}/x_descriptive.png",
                    title=f"X.describe(include={include}, numeric_only={numeric_only})",
                    #max_rows=max_rows,
                    dpi=dpi, float_fmt="{:.3f}"
                )

                save_table_png_pretty(
                    y_tbl, output_root, f"{tables_dir}/y_descriptive.png",
                    title=f"Y.describe(include={include}, numeric_only={numeric_only})",
                    #max_rows=max_rows,
                    dpi=dpi, float_fmt="{:.3f}"
                )

            # --- Method 2: min/max/mean/std with numeric-only option
            m_mm = methods.get("min_max_mean_std", {})
            if m_mm.get("enabled", False):
                p = m_mm.get("params", {})
                numeric_only = bool(p.get("numeric_only", True))
                cols_x = _numeric_cols(X, exclude=["id"]) if numeric_only else list(X.columns)
                cols_y = _numeric_cols(Y, exclude=["id"]) if numeric_only else list(Y.columns)

                def _mm(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
                    d = pd.DataFrame(index=cols)
                    d["min"] = df[cols].min()
                    d["max"] = df[cols].max()
                    d["mean"] = df[cols].mean()
                    d["std"] = df[cols].std()
                    return d
                #
                # save_table_png(_as_table_by_column(_mm(X, cols_x)), output_root, f"{tables_dir}/x_min_max_mean_std.png",
                #                "X min/max/mean/std", dpi=dpi)
                # save_table_png(_as_table_by_column(_mm(Y, cols_y)), output_root, f"{tables_dir}/y_min_max_mean_std.png",
                #                "Y min/max/mean/std", dpi=dpi)

                save_table_png_pretty_all(as_table_by_column_heuristic(_mm(X, cols_x)), output_root, f"{tables_dir}/x_min_max_mean_std.png",
                               "X min/max/mean/std", dpi=dpi)
                save_table_png_pretty_all(as_table_by_column_heuristic(_mm(Y, cols_y)), output_root, f"{tables_dir}/y_min_max_mean_std.png",
                               "Y min/max/mean/std", dpi=dpi)


        # --- Technique 2: schema_inspection
        si = techs.get("schema_inspection", {})
        if si.get("enabled", False):
            methods = si.get("methods", {})

            # -- Method 1: dtype analysis (count of dtypes in X and Y)
            if methods.get("dtype_analysis", {}).get("enabled", False):
                x_dtype = pd.DataFrame({"dtype": X.dtypes.astype(str)})
                y_dtype = pd.DataFrame({"dtype": Y.dtypes.astype(str)})
                # save_table_png(_as_table_by_column(x_dtype), output_root, f"{tables_dir}/x_dtype_analysis.png", "X dtypes", dpi=dpi)
                # save_table_png(_as_table_by_column(y_dtype), output_root, f"{tables_dir}/y_dtype_analysis.png", "Y dtypes", dpi=dpi)

                save_table_png_pretty_all(as_table_by_column_heuristic(x_dtype), output_root, f"{tables_dir}/x_dtype_analysis.png", "X dtypes", dpi=dpi)
                save_table_png_pretty_all(as_table_by_column_heuristic(y_dtype), output_root, f"{tables_dir}/y_dtype_analysis.png", "Y dtypes", dpi=dpi)

            # -- Method 2: cardinality count (number of unique values per column, sorted desc, top N)
            if methods.get("cardinality_count", {}).get("enabled", False):
                p = methods["cardinality_count"].get("params", {})
                max_u = int(p.get("max_unique_to_report", 50))

                x_card = pd.DataFrame({"unique": X.nunique(dropna=False)}).sort_values("unique", ascending=False).head(max_u)
                y_card = pd.DataFrame({"unique": Y.nunique(dropna=False)}).sort_values("unique", ascending=False).head(max_u)

                # save_table_png(_as_table_by_column(x_card), output_root, f"{tables_dir}/x_cardinality.png", "X cardinality", dpi=dpi)
                # save_table_png(_as_table_by_column(y_card), output_root, f"{tables_dir}/y_cardinality.png", "Y cardinality", dpi=dpi)

                save_table_png_pretty_all(as_table_by_column_heuristic(x_card), output_root, f"{tables_dir}/x_cardinality.png", "X cardinality", dpi=dpi)
                save_table_png_pretty_all(as_table_by_column_heuristic(y_card), output_root, f"{tables_dir}/y_cardinality.png", "Y cardinality", dpi=dpi)

            # -- Method 3: missing values count and ratio
            if methods.get("null_count", {}).get("enabled", False):
                x_null = pd.DataFrame({"missing_count": X.isna().sum(), "missing_ratio": (X.isna().sum()/len(X)).round(6)})
                y_null = pd.DataFrame({"missing_count": Y.isna().sum(), "missing_ratio": (Y.isna().sum()/len(Y)).round(6)})

                # save_table_png(_as_table_by_column(x_null.sort_values("missing_ratio", ascending=False)),
                #                output_root, f"{tables_dir}/x_missing.png", "X missing", dpi=dpi)
                # save_table_png(_as_table_by_column(y_null.sort_values("missing_ratio", ascending=False)),
                #                output_root, f"{tables_dir}/y_missing.png", "Y missing", dpi=dpi)

                save_table_png_pretty_all(as_table_by_column_heuristic(x_null.sort_values("missing_ratio", ascending=False)),
                               output_root, f"{tables_dir}/x_missing.png", "X missing", dpi=dpi)
                save_table_png_pretty_all(as_table_by_column_heuristic(y_null.sort_values("missing_ratio", ascending=False)),
                           output_root, f"{tables_dir}/y_missing.png", "Y missing", dpi=dpi)

    # ============================================================
    # Step 2.3 Data quality assessment
    # ============================================================
    s23 = steps.get("step_2_3_data_quality_assessment", {})
    if s23.get("enabled", False):
        methods = s23.get("methods", {})

        # percentile_analysis
        # percentiles: [ 0.01, 0.05, 0.95, 0.99, 0.999 ]
        if methods.get("percentile_analysis", {}).get("enabled", False):
            p = methods["percentile_analysis"].get("params", {})
            percentiles = p.get("percentiles", [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
            #percentiles = p.get("percentiles", [0.25, 0.5, 0.75])
            cols_x = _numeric_cols(X, exclude=["id"])
            cols_y = _numeric_cols(Y, exclude=["id"])

            # def _percentiles(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
            #     d = pd.DataFrame(index=cols)
            #     for q in percentiles:
            #         d[f"{int(q*100)}%"] = df[cols].quantile(q)
            #     return d

            def _percentiles(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
                if not cols:
                    return pd.DataFrame()
                qs = df[cols].quantile(percentiles).T  # index = cols, columns = percentiles
                qs.columns = [f"p{int(q*1000)/10:g}" for q in percentiles]  # p1, p5, p99.9, etc.
                return qs

            save_table_png_pretty_all(as_table_by_column_heuristic(_percentiles(X, cols_x)), output_root, f"{tables_dir}/x_percentiles.png",
                               "X Percentiles", dpi=dpi)
            save_table_png_pretty_all(as_table_by_column_heuristic(_percentiles(Y, cols_y)), output_root, f"{tables_dir}/y_percentiles.png",
                               "Y Percentiles", dpi=dpi)

    # ============================================================
    # Step 2.4 EDA - Histograms for X + Y (including faulty)
    # ============================================================
    s24 = steps.get("step_2_4_eda", {})
    if s24.get("enabled", False):
        methods = s24.get("methods", {})

        hist_cfg = methods.get("histograms", {})
        if hist_cfg.get("enabled", False):
            p = hist_cfg.get("params", {})
            bins = int(p.get("bins", 30))
            max_cols = int(p.get("max_columns", 20))

            # X numeric hist (exclude id)
            x_num = _numeric_cols(X, exclude=["id"])[:max_cols]
            for c in x_num:
                plt.figure()
                X[c].hist(bins=bins)
                plt.title(f"X Histogram: {c}")
                save_fig(output_root / f"{figs_dir}/hist_x_{c}.png", dpi=dpi)

            # Y: faulty as bar, trq_margin as hist (exclude id)
            if "faulty" in Y.columns:
                plt.figure()
                Y["faulty"].value_counts(dropna=False).sort_index().plot(kind="bar")
                plt.title("Y Distribution: faulty")
                save_fig(output_root / f"{figs_dir}/bar_y_faulty.png", dpi=dpi)

            y_num = [c for c in _numeric_cols(Y, exclude=["id", "faulty"])][:max_cols]
            for c in y_num:
                plt.figure()
                Y[c].hist(bins=bins)
                plt.title(f"Y Histogram: {c}")
                save_fig(output_root / f"{figs_dir}/hist_y_{c}.png", dpi=dpi)

    log.info("=== STAGE 2 END [run_stage2] ===")
    return {"X_sample": X, "Y_sample": Y}


def _as_table_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pandas describe output to 'one row per column' with:
    No | NameColumn | metrics...
    """
    t = df.copy()
    # if describe() returns metrics in index and columns=features, transpose it
    if "NameColumn" not in t.columns:
        t = t.T
    t.insert(0, "NameColumn", t.index.astype(str))
    t.insert(0, "Nr.", range(1, len(t) + 1))
    return t.reset_index(drop=True)

def as_table_by_column(desc: pd.DataFrame, original_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Convierte el output de df.describe() (stats x columns)
    a formato "por columna" (rows=features) con NameColumn.
    """
    if desc is None or desc.empty:
        return pd.DataFrame()

    # describe -> filas=estadísticas, columnas=features
    # queremos filas=features, columnas=estadísticas
    tbl = desc.T.reset_index().rename(columns={"index": "NameColumn"})

    # Mantener el orden original del dataset (muy importante para que se vea coherente)
    if original_columns is not None and len(original_columns) > 0:
        order = [c for c in original_columns if c in set(tbl["NameColumn"])]
        tbl["NameColumn"] = pd.Categorical(tbl["NameColumn"], categories=order, ordered=True)
        tbl = tbl.sort_values("NameColumn")

    tbl = tbl.reset_index(drop=True)
    return tbl


def _numeric_cols(df: pd.DataFrame, exclude: List[str] | None = None) -> List[str]:
    exclude = set(exclude or [])
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols



def safe_describe(df: pd.DataFrame, include="all", numeric_only: bool = False) -> pd.DataFrame:

    if df is None or df.shape[1] == 0:
        return pd.DataFrame()

    if numeric_only:
        df = df.select_dtypes(include="number")
        if df.shape[1] == 0:
            return pd.DataFrame()
        return df.describe()

    return df.describe(include=include)


