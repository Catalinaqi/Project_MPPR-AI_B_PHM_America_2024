# src/phm_america_2024/data/profiling_profiler_data.py
# src/phm_america_2024/data/profiling_profiler_data.py
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
from phm_america_2024.common.io_service_common import load_parquet
from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 – Column metadata
# ─────────────────────────────────────────────────────────────────────────────


def compute_column_metadata(
    df: Any, tech_cfg: dict, ctx: Any, base_dir: Any
) -> dict[str, Any]:
    params = tech_cfg.get("params", {})
    include_dtypes = params.get("include_dtypes", True)
    include_cardinality = params.get("include_cardinality", False)
    include_null_counts = params.get("include_null_counts", True)

    log.info("[compute_column_metadata] start rows=%d cols=%d", len(df), df.shape[1])

    metadata: dict[str, Any] = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": {},
    }
    for col in df.columns:
        entry: dict[str, Any] = {}
        if include_dtypes:
            entry["dtype"] = str(df[col].dtype)
        if include_null_counts:
            null_count = int(df[col].isna().sum())
            null_pct = round(null_count / len(df) * 100, 4) if len(df) > 0 else 0.0
            entry["null_count"] = null_count
            entry["null_pct"] = null_pct
        if include_cardinality:
            entry["cardinality"] = int(df[col].nunique())
        metadata["columns"][col] = entry

    log.info(
        "[compute_column_metadata] completed – columns=%d", len(metadata["columns"])
    )
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 – Descriptive statistics (basic_stats)
# ─────────────────────────────────────────────────────────────────────────────


def compute_descriptive_statistics(
    df: Any, tech_cfg: dict, ctx: Any, base_dir: Any
) -> dict[str, Any]:
    params = tech_cfg.get("params", {})
    columns = params.get("columns", None)
    metrics = params.get("metrics", None)

    log.info("[compute_descriptive_statistics] start")

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
    if metrics is None:
        metrics = ["count", "mean", "std", "min", "max"]

    metric_map: dict[str, Any] = {
        "count": lambda s: int(s.count()),
        "mean": lambda s: float(s.mean()) if s.notna().sum() > 0 else None,
        "std": lambda s: float(s.std()) if s.notna().sum() > 1 else None,
        "min": lambda s: float(s.min()) if s.notna().sum() > 0 else None,
        "max": lambda s: float(s.max()) if s.notna().sum() > 0 else None,
        "skewness": lambda s: float(s.skew()) if s.notna().sum() > 2 else None,
        "kurtosis": lambda s: float(s.kurtosis()) if s.notna().sum() > 3 else None,
        "q25": lambda s: float(s.quantile(0.25)),
        "q50": lambda s: float(s.median()),
        "q75": lambda s: float(s.quantile(0.75)),
    }

    stats: dict[str, dict[str, Any]] = {}
    for col in columns:
        if col not in df.columns:
            log.warning(
                "[compute_descriptive_statistics] column '%s' not found – skip", col
            )
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        col_stats: dict[str, Any] = {}
        for m in metrics:
            if m in metric_map:
                try:
                    col_stats[m] = metric_map[m](s)
                except Exception:
                    col_stats[m] = None
        stats[col] = col_stats

    result: dict[str, Any] = {
        "n_rows": len(df),
        "columns_profiled": list(stats.keys()),
        "metrics_computed": metrics,
        "statistics": stats,
    }
    log.info("[compute_descriptive_statistics] completed – columns=%d", len(stats))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 – Null count per column
# ─────────────────────────────────────────────────────────────────────────────


def compute_null_counts(
    df: Any, tech_cfg: dict, ctx: Any, base_dir: Any
) -> dict[str, Any]:
    params = tech_cfg.get("params", {})
    columns = params.get("columns", None)

    log.info("[compute_null_counts] start")

    if columns is None:
        columns = df.columns.tolist()

    null_counts: dict[str, dict[str, Any]] = {}
    n_total = len(df)
    for col in columns:
        if col not in df.columns:
            log.warning("[compute_null_counts] column '%s' not found – skip", col)
            continue
        n_null = int(df[col].isna().sum())
        pct_null = round(n_null / n_total * 100, 4) if n_total > 0 else 0.0
        null_counts[col] = {
            "n_null": n_null,
            "pct_null": pct_null,
        }

    result: dict[str, Any] = {
        "n_rows": n_total,
        "n_columns": len(null_counts),
        "null_counts": null_counts,
    }
    log.info(
        "[compute_null_counts] completed – columns_with_nulls=%d",
        sum(1 for v in null_counts.values() if v["n_null"] > 0),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 – Target distribution analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyze_target_distributions(
    df: Any, tech_cfg: dict, ctx: Any, base_dir: Any
) -> dict[str, Any]:
    params = tech_cfg.get("params", {})
    classification_target = params.get("classification_target", None)
    regression_target = params.get("regression_target", None)
    compute_imbalance_ratio = params.get("compute_imbalance_ratio", True)
    compute_skewness = params.get("compute_skewness", True)
    fit_distributions = params.get("fit_distributions", None)

    log.info(
        "[analyze_target_distributions] starting – classification=%s regression=%s",
        classification_target,
        regression_target,
    )

    result: dict[str, Any] = {}

    if classification_target and classification_target in df.columns:
        vc = df[classification_target].value_counts(dropna=False)
        cls_report: dict[str, Any] = {
            "target": classification_target,
            "n_classes": int(len(vc)),
            "value_counts": vc.to_dict(),
        }
        if compute_imbalance_ratio and len(vc) > 1:
            ratio = float(vc.max() / vc.min())
            cls_report["imbalance_ratio"] = round(ratio, 4)
        result["classification"] = cls_report

    if regression_target and regression_target in df.columns:
        s = pd.to_numeric(df[regression_target], errors="coerce").dropna()
        reg_report: dict[str, Any] = {
            "target": regression_target,
            "n": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        }
        if compute_skewness:
            reg_report["skewness"] = round(float(s.skew()), 4)
        result["regression"] = reg_report

    fit_results: dict[str, Any] = {}
    if fit_distributions and regression_target and regression_target in df.columns:
        from scipy import stats as sp_stats

        s = pd.to_numeric(df[regression_target], errors="coerce").dropna().values
        for dist_name in fit_distributions:
            try:
                dist = getattr(sp_stats, dist_name, None)
                if dist is None:
                    continue
                dist_params = dist.fit(s)
                ks_stat, ks_pval = sp_stats.kstest(s, dist_name, args=dist_params)
                fit_results[dist_name] = {
                    "params": [round(p, 6) for p in dist_params],
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_pvalue": round(float(ks_pval), 4),
                }
            except Exception as exc:
                log.warning(
                    "[analyze_target_distributions] fit '%s' failed – %s",
                    dist_name,
                    exc,
                )
        result["fit_results"] = fit_results

    log.info("[analyze_target_distributions] completed – keys=%s", list(result.keys()))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.3 – Zero / negative check
# ─────────────────────────────────────────────────────────────────────────────


def zero_or_negative_check(
    df: Any, tech_cfg: dict, ctx: Any, base_dir: Any
) -> dict[str, Any]:
    params = tech_cfg.get("params", {})
    check_columns = params.get("check_columns", [])
    flag_if_less_than_or_equal_to = params.get("flag_if_less_than_or_equal_to", 0.0)

    log.info(
        "[zero_or_negative_check] columns=%s threshold=%.2f",
        check_columns,
        flag_if_less_than_or_equal_to,
    )

    report: dict[str, Any] = {"threshold": flag_if_less_than_or_equal_to, "columns": {}}
    for col in check_columns:
        if col not in df.columns:
            continue
        mask = df[col] <= flag_if_less_than_or_equal_to
        n_flagged = int(mask.sum())
        pct = float(mask.mean() * 100.0) if len(df) > 0 else 0.0
        report["columns"][col] = {
            "n_flagged": n_flagged,
            "pct_flagged": round(pct, 4),
            "min_value": float(df[col].min()),
        }

    log.info(
        "[zero_or_negative_check] completed – total_columns_checked=%d",
        len(report["columns"]),
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 2.3 – Collinearity / leakage analysis
# ─────────────────────────────────────────────────────────────────────────────


def collinearity_analysis(
    df: Any, tech_cfg: dict, ctx: Any, base_dir: Any
) -> dict[str, Any]:
    params = tech_cfg.get("params", {})
    suspect_pairs = params.get("suspect_pairs", [])
    leakage_threshold = params.get("leakage_threshold", 0.95)
    secondary_threshold = params.get("secondary_threshold", None)
    target_column = params.get("target_column", None)

    log.info(
        "[collinearity_analysis] pairs=%d leakage_threshold=%.2f",
        len(suspect_pairs),
        leakage_threshold,
    )

    pairs_result: list[dict[str, Any]] = []
    for pair in suspect_pairs:
        if len(pair) != 2:
            continue
        c1, c2 = pair
        if c1 not in df.columns or c2 not in df.columns:
            continue

        corr_val = float(df[c1].corr(df[c2]))
        threshold_used = leakage_threshold
        if target_column and (c1 == target_column or c2 == target_column):
            if secondary_threshold is not None:
                threshold_used = secondary_threshold

        flagged = abs(corr_val) >= threshold_used
        pairs_result.append(
            {
                "col_1": c1,
                "col_2": c2,
                "correlation": round(corr_val, 4),
                "threshold_applied": threshold_used,
                "flagged": flagged,
                "leakage_risk": "high" if flagged else "none",
            }
        )

    report: dict[str, Any] = {
        "pairs_checked": len(pairs_result),
        "pairs": pairs_result,
        "leakage_threshold": leakage_threshold,
        "secondary_threshold": secondary_threshold,
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 – Column catalog
# ─────────────────────────────────────────────────────────────────────────────


def categorize_columns(
    df: Any, tech_cfg: dict, ctx: Any, base_dir: Any
) -> dict[str, Any]:
    params = tech_cfg.get("params", {})
    roles = params.get("roles", None)
    categorize_by_role = params.get("categorize_by_role", True)

    if not categorize_by_role:
        log.warning("[categorize_columns] categorize_by_role set to False – skipping")
        return {"n_roles": 0, "catalog": {}}

    log.info("[categorize_columns] start")

    if roles is None:
        roles = {
            "identifiers": [c for c in df.columns if c.lower() == "id"],
            "targets": [c for c in df.columns if c in ("faulty", "trq_margin")],
            "temperatures": [c for c in df.columns if c in ("oat", "mgt")],
            "power_metrics": [
                c for c in df.columns if c in ("pa", "np", "trq_measured")
            ],
            "speeds": [c for c in df.columns if c in ("ng", "ias")],
        }

    catalog: dict[str, dict[str, Any]] = {}
    for role_name, cols in roles.items():
        present = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        role_entry: dict[str, Any] = {
            "n_columns": len(present),
            "columns": present,
            "missing_from_df": missing,
        }
        if present:
            role_entry["dtypes"] = {c: str(df[c].dtype) for c in present}
        catalog[role_name] = role_entry

    return {"n_roles": len(roles), "catalog": catalog}


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 – KS test per feature
# ─────────────────────────────────────────────────────────────────────────────


def ks_test_per_feature(
    df: Any,
    tech_cfg: dict,
    ctx: Any,
    base_dir: Any,
) -> dict[str, Any]:
    params = tech_cfg.get("params", {})
    features = params.get("features", [])
    significance_level = params.get("significance_level", 0.01)
    shift_threshold = params.get("shift_threshold", 0.1)

    # LÓGICA INTELIGENTE: Si df es solo train, buscamos el parquet de validation automáticamente

    # 1. El Orquestador nos pasó "df", que sabemos que es el TRAIN. Lo tomamos como referencia.
    df_ref = df
    # 1. Leemos dinámicamente los splits a comparar desde tu YAML
    compare_splits = params.get("compare_splits", ["validation"])

    log.info(
        "[ks_test_per_feature] features=%d sig_level=%.3f shift_threshold=%.2f compares=%s",
        len(features),
        significance_level,
        shift_threshold,
        compare_splits,
    )

    # 2. LÓGICA INTELIGENTE: La función busca por su cuenta el VALIDATION en el disco duro
    per_feature: list[dict[str, Any]] = []

    # 2. Iteramos sobre CADA split que pediste en el YAML (validation, test, etc.)
    for split_name in compare_splits:
        # LÓGICA INTELIGENTE: Buscamos dinámicamente el sufijo del split en el disco
        # ej. "*_validation.parquet", "*_test.parquet"
        split_matches = list(Path(base_dir).glob(f"*_{split_name}.parquet"))

        if not split_matches:
            log.warning(
                "[ks_test_per_feature] Could not find %s parquet. Skipping this split.",
                split_name,
            )
            continue

        # Cargamos el split a la memoria RAM
        df_comp = load_parquet(str(split_matches[0]))
        log.info(
            "[ks_test_per_feature] %s split loaded dynamically for comparison.",
            split_name.capitalize(),
        )

        # 3. Comparamos todas las features contra este split
        for feat in features:
            if feat not in df_ref.columns or feat not in df_comp.columns:
                continue
            s_ref = df_ref[feat].dropna().values
            s_comp = df_comp[feat].dropna().values
            if len(s_ref) == 0 or len(s_comp) == 0:
                continue

            stat, pval = ks_2samp(s_ref, s_comp)
            flagged = (pval < significance_level) or (stat > shift_threshold)
            per_feature.append(
                {
                    "feature": feat,
                    "compared_to": split_name,  # <-- CRÍTICO: Registramos contra quién se comparó
                    "ks_statistic": round(float(stat), 4),
                    "ks_pvalue": round(float(pval), 4),
                    "significance_level": significance_level,
                    "shift_threshold": shift_threshold,
                    "flagged": bool(flagged),
                }
            )

    return {
        "features_tested_per_split": len(features),
        "splits_compared": compare_splits,
        "results": per_feature,
        "significance_level": significance_level,
        "shift_threshold": shift_threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 – GMM exploration
# ─────────────────────────────────────────────────────────────────────────────


def gmm_exploration(
    df: Any,
    tech_cfg: dict,
    ctx: Any,
    base_dir: Any,
) -> dict[str, Any]:
    params = tech_cfg.get("params", {})
    features = params.get("features", [])
    k_range = params.get("k_range", [2, 3, 4, 5, 6])
    selection_criterion = params.get("selection_criterion", "BIC")
    random_state = params.get("random_state", 7)

    from sklearn.mixture import GaussianMixture

    log.info(
        "[gmm_exploration] features=%s k_range=%s criterion=%s",
        features,
        k_range,
        selection_criterion,
    )

    available = [c for c in features if c in df.columns]
    if not available:
        return {"error": "no_valid_features"}

    X = df[available].dropna().values
    if len(X) == 0:
        return {"error": "empty_data"}

    metrics: dict[str, list] = {"k": [], "BIC": [], "AIC": []}
    for k in sorted(k_range):
        try:
            gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=5)
            gmm.fit(X)
            metrics["k"].append(k)
            metrics["BIC"].append(round(float(gmm.bic(X)), 2))
            metrics["AIC"].append(round(float(gmm.aic(X)), 2))
        except Exception as exc:
            log.warning("[gmm_exploration] k=%d failed – %s", k, exc)

    values = metrics["BIC"] if selection_criterion.upper() == "BIC" else metrics["AIC"]
    optimal_k = metrics["k"][int(np.argmin(values))] if values else None

    return {
        "features_used": available,
        "k_range_used": sorted(k_range),
        "selection_criterion": selection_criterion.upper(),
        "optimal_k": optimal_k,
        "curve": metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 – Flight regime binning
# ─────────────────────────────────────────────────────────────────────────────


def flight_regime_binning(
    df: Any,
    tech_cfg: dict,
    ctx: Any,
    base_dir: Any,
) -> dict[str, Any]:
    params = tech_cfg.get("params", {})
    column = params.get("column", "")
    bin_size = params.get("bin_size", 10.0)

    log.info("[flight_regime_binning] column='%s' bin_size=%.2f", column, bin_size)

    if column not in df.columns:
        return {"error": f"column '{column}' not found"}

    s = df[column].dropna()
    if len(s) == 0:
        return {"error": "empty_column"}

    val_min, val_max = float(s.min()), float(s.max())
    bins = np.arange(val_min, val_max + bin_size, bin_size)
    counts, edges = np.histogram(s, bins=bins)

    bin_centers = (edges[:-1] + edges[1:]) / 2
    sorted_indices = np.argsort(counts)[::-1]
    dominant_bins = [
        {
            "bin_center": round(float(bin_centers[idx]), 2),
            "count": int(counts[idx]),
            "pct_of_total": round(float(counts[idx] / len(s) * 100), 2),
        }
        for idx in sorted_indices[:2]
    ]

    return {
        "column": column,
        "bin_size": bin_size,
        "n_total": int(len(s)),
        "n_bins": len(bins) - 1,
        "bin_edges": [round(float(e), 2) for e in edges],
        "bin_counts": [int(c) for c in counts],
        "dominant_regimes": dominant_bins,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 – Feature drift summary
# ─────────────────────────────────────────────────────────────────────────────


# def feature_drift_summary(
#     df: Any,
#     tech_cfg: dict,
#     ctx: Any,
#     base_dir: Any,
# ) -> dict[str, Any]:
#     params = tech_cfg.get("params", {})
#     aggregate_by = params.get("aggregate_by", "max_ks_statistic")
#     flag_critical_shifts = params.get("flag_critical_shifts", True)
#
#     log.info("[feature_drift_summary] aggregate_by='%s'", aggregate_by)
#
#     # LÓGICA INTELIGENTE: Como ya no recibimos ks_results directamente, lo buscamos en el disco.
#     ks_results = {}
#     try:
#         ks_matches = list(Path(base_dir).glob("*.ks_test*.json"))
#         if ks_matches:
#             ks_results = json.loads(ks_matches[0].read_text(encoding="utf-8"))
#         else:
#             log.warning(
#                 "[feature_drift_summary] KS Report JSON not found to summarize."
#             )
#             return {"error": "ks_report_not_found"}
#     except Exception as e:
#         log.error("[feature_drift_summary] Failed to load KS results: %s", e)
#         return {"error": "load_failure"}
#
#     results = ks_results.get("results", [])
#     if not results:
#         return {"error": "empty_results"}
#
#     ks_values = [r["ks_statistic"] for r in results]
#     pvalues = [r["ks_pvalue"] for r in results]
#
#     summary: dict[str, Any] = {
#         "n_features_tested": len(results),
#         "max_ks_statistic": round(float(max(ks_values)), 4),
#         "mean_ks_statistic": round(float(np.mean(ks_values)), 4),
#         "min_ks_pvalue": round(float(min(pvalues)), 4),
#         "n_flagged": sum(1 for r in results if r["flagged"]),
#     }
#
#     if aggregate_by == "max_ks_statistic":
#         summary["aggregate"] = summary["max_ks_statistic"]
#     elif aggregate_by == "mean_ks_statistic":
#         summary["aggregate"] = summary["mean_ks_statistic"]
#     else:
#         summary["aggregate"] = None
#
#     if flag_critical_shifts:
#         critical = [r for r in results if r["flagged"]]
#         summary["critical_shifts"] = [
#             {
#                 "feature": r["feature"],
#                 "ks_statistic": r["ks_statistic"],
#                 "ks_pvalue": r["ks_pvalue"],
#             }
#             for r in critical
#         ]
#
#     return summary


def feature_drift_summary(
    df: Any, tech_cfg: dict, ctx: Any, base_dir: Any
) -> dict[str, Any]:
    log.info("[feature_drift_summary] Aggregating KS results...")

    # 1. Cargamos el JSON avanzado que acaba de generar el KS Test
    ks_results = {}
    try:
        ks_matches = list(Path(base_dir).glob("*.ks_report*.json"))
        if ks_matches:
            import json

            ks_results = json.loads(ks_matches[0].read_text(encoding="utf-8"))
        else:
            log.warning(
                "[feature_drift_summary] KS Report JSON not found to summarize."
            )
            return {"error": "ks_report_not_found"}
    except Exception as e:
        log.error("[feature_drift_summary] Failed to load KS results: %s", e)
        return {"error": "load_failure"}

    results = ks_results.get("results", [])
    if not results:
        return {"error": "empty_results"}

    # 2. Identificamos contra qué splits comparamos (ej. ['validation', 'test'])
    splits = list(set([r.get("compared_to", "unknown") for r in results]))

    summary: dict[str, Any] = {
        "splits_analyzed": splits,
        "overall_flagged": sum(1 for r in results if r["flagged"]),
        "split_summaries": {},
    }

    # 3. Agrupamos las estadísticas de Drift separadas por split
    for split in splits:
        split_res = [r for r in results if r.get("compared_to") == split]
        ks_vals = [r["ks_statistic"] for r in split_res]

        summary["split_summaries"][split] = {
            "n_features_tested": len(split_res),
            "max_ks_statistic": round(float(max(ks_vals)), 4),
            "mean_ks_statistic": round(float(sum(ks_vals) / len(ks_vals)), 4),
            "n_flagged": sum(1 for r in split_res if r["flagged"]),
            "critical_shifts": [
                {
                    "feature": r["feature"],
                    "ks_statistic": r["ks_statistic"],
                    "ks_pvalue": r["ks_pvalue"],
                }
                for r in split_res
                if r["flagged"]
            ],
        }

    log.info("[feature_drift_summary] completed – splits=%s", splits)
    return summary
