# src/phm_america_2024/data/profiling_profiler_data.py
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from pathlib import Path

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 – Column metadata
# ─────────────────────────────────────────────────────────────────────────────

def compute_column_metadata(
    df: pd.DataFrame,
    *,
    include_dtypes: bool = True,
    include_cardinality: bool = False,
    include_null_counts: bool = True,
) -> dict[str, Any]:
    """Extract column-level metadata: dtypes, null %, cardinality.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    include_dtypes : bool
        Include dtype per column.
    include_cardinality : bool
        Include unique value count.
    include_null_counts : bool
        Include null count and percentage.

    Returns
    -------
    dict
        Nested dict with per-column metadata.
    """
    log.info("[compute_column_metadata] start rows=%d cols=%d", len(df), df.shape[1])

    metadata: dict[str, Any] = {"n_rows": len(df), "n_columns": len(df.columns), "columns": {}}
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

    log.info("[compute_column_metadata] completed – columns=%d", len(metadata["columns"]))
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 – Descriptive statistics (basic_stats)
# ─────────────────────────────────────────────────────────────────────────────

def compute_descriptive_statistics(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Compute summary statistics for selected numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str], optional
        Columns to profile. Default: all numeric columns.
    metrics : list[str], optional
        Metrics to compute. Default: ['count', 'mean', 'std', 'min', 'max'].

    Returns
    -------
    dict
        Per-column statistics.
    """
    log.info("[compute_descriptive_statistics] start")

    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()
    if metrics is None:
        metrics = ['count', 'mean', 'std', 'min', 'max']

    # Build mapping of metric name → pandas method or custom
    metric_map: dict[str, Any] = {
        'count': lambda s: int(s.count()),
        'mean': lambda s: float(s.mean()) if s.notna().sum() > 0 else None,
        'std': lambda s: float(s.std()) if s.notna().sum() > 1 else None,
        'min': lambda s: float(s.min()) if s.notna().sum() > 0 else None,
        'max': lambda s: float(s.max()) if s.notna().sum() > 0 else None,
        'skewness': lambda s: float(s.skew()) if s.notna().sum() > 2 else None,
        'kurtosis': lambda s: float(s.kurtosis()) if s.notna().sum() > 3 else None,
        'q25': lambda s: float(s.quantile(0.25)),
        'q50': lambda s: float(s.median()),
        'q75': lambda s: float(s.quantile(0.75)),
    }

    stats: dict[str, dict[str, Any]] = {}
    for col in columns:
        if col not in df.columns:
            log.warning("[compute_descriptive_statistics] column '%s' not found – skip", col)
            continue
        s = pd.to_numeric(df[col], errors='coerce')
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
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Count null values per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str], optional
        Columns to inspect. Default: all columns.

    Returns
    -------
    dict
        Per-column null count and percentage.
    """
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
    log.info("[compute_null_counts] completed – columns_with_nulls=%d",
             sum(1 for v in null_counts.values() if v["n_null"] > 0))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 – Target distribution analysis (two targets + skewness + fit)
# ─────────────────────────────────────────────────────────────────────────────


def analyze_target_distributions(
        df: pd.DataFrame,
        *,
        classification_target: str | None = None,
        regression_target: str | None = None,
        compute_imbalance_ratio: bool = True,
        compute_skewness: bool = True,
        fit_distributions: list[str] | None = None,
) -> dict[str, Any]:
    """Profile classification and regression targets with distribution fitting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    classification_target : str, optional
        Binary target column (e.g. ``"faulty"``).
    regression_target : str, optional
        Continuous target column (e.g. ``"trq_margin"``).
    compute_imbalance_ratio : bool
        If True, compute max/min class ratio for classification target.
    compute_skewness : bool
        If True, compute skewness for regression target.
    fit_distributions : list[str], optional
        Distributions to fit (e.g. ``["norm", "skewnorm", "gamma"]``).

    Returns
    -------
    dict
        Nested dict with keys ``classification``, ``regression``, ``fit_results``.
    """
    log.info("[analyze_target_distributions] starting – classification=%s regression=%s",
             classification_target, regression_target)

    result: dict[str, Any] = {}

    # — Classification target ————————————————————————————————
    if classification_target and classification_target in df.columns:
        log.debug("[analyze_target_distributions] processing classification target='%s'",
                  classification_target)
        vc = df[classification_target].value_counts(dropna=False)
        cls_report: dict[str, Any] = {
            "target": classification_target,
            "n_classes": int(len(vc)),
            "value_counts": vc.to_dict(),
        }
        if compute_imbalance_ratio and len(vc) > 1:
            ratio = float(vc.max() / vc.min())
            cls_report["imbalance_ratio"] = round(ratio, 4)
            log.debug("[analyze_target_distributions] imbalance_ratio=%.4f", ratio)
        result["classification"] = cls_report
    else:
        log.debug("[analyze_target_distributions] classification target not found or not provided")

    # — Regression target ————————————————————————————————————
    if regression_target and regression_target in df.columns:
        log.debug("[analyze_target_distributions] processing regression target='%s'",
                  regression_target)
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
            skew_val = float(s.skew())
            reg_report["skewness"] = round(skew_val, 4)
            log.debug("[analyze_target_distributions] skewness=%.4f", skew_val)
        result["regression"] = reg_report
    else:
        log.debug("[analyze_target_distributions] regression target not found or not provided")

    # — Distribution fitting —————————————————————————————————
    fit_results: dict[str, Any] = {}
    if fit_distributions and regression_target and regression_target in df.columns:
        from scipy import stats as sp_stats
        s = pd.to_numeric(df[regression_target], errors="coerce").dropna().values
        for dist_name in fit_distributions:
            try:
                dist = getattr(sp_stats, dist_name, None)
                if dist is None:
                    log.warning("[analyze_target_distributions] unknown distribution='%s' – skip", dist_name)
                    continue
                params = dist.fit(s)
                # Kolmogorov-Smirnov test against fitted distribution
                ks_stat, ks_pval = sp_stats.kstest(s, dist_name, args=params)
                fit_results[dist_name] = {
                    "params": [round(p, 6) for p in params],
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_pvalue": round(float(ks_pval), 4),
                }
                log.debug("[analyze_target_distributions] fit '%s' ks_stat=%.4f pval=%.4f",
                          dist_name, ks_stat, ks_pval)
            except Exception as exc:
                log.warning("[analyze_target_distributions] fit '%s' failed – %s", dist_name, exc)
        result["fit_results"] = fit_results

    log.info("[analyze_target_distributions] completed – keys=%s", list(result.keys()))
    return result



# ─────────────────────────────────────────────────────────────────────────────
# 2.3 – Zero / negative check
# ─────────────────────────────────────────────────────────────────────────────
def zero_or_negative_check(
        df: pd.DataFrame,
        *,
        check_columns: list[str],
        flag_if_less_than_or_equal_to: float = 0.0,
) -> dict[str, Any]:
    """Check columns for zero-or-negative values that may cause physical invalidity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    check_columns : list[str]
        Column names to inspect.
    flag_if_less_than_or_equal_to : float
        Threshold; values ≤ this are flagged.

    Returns
    -------
    dict
        Per-column counts and percentages of flagged values.
    """
    log.info("[zero_or_negative_check] columns=%s threshold=%.2f",
             check_columns, flag_if_less_than_or_equal_to)

    report: dict[str, Any] = {"threshold": flag_if_less_than_or_equal_to, "columns": {}}
    for col in check_columns:
        if col not in df.columns:
            log.warning("[zero_or_negative_check] column '%s' not found – skip", col)
            continue
        mask = df[col] <= flag_if_less_than_or_equal_to
        n_flagged = int(mask.sum())
        pct = float(mask.mean() * 100.0) if len(df) > 0 else 0.0
        report["columns"][col] = {
            "n_flagged": n_flagged,
            "pct_flagged": round(pct, 4),
            "min_value": float(df[col].min()),
        }
        log.debug("[zero_or_negative_check] col='%s' flagged=%d (%.2f%%)", col, n_flagged, pct)

    log.info("[zero_or_negative_check] completed – total_columns_checked=%d", len(report["columns"]))
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 2.3 – Collinearity / leakage analysis
# ─────────────────────────────────────────────────────────────────────────────


def collinearity_analysis(
        df: pd.DataFrame,
        *,
        suspect_pairs: list[list[str]],
        leakage_threshold: float = 0.95,
        secondary_threshold: float | None = None,
        target_column: str | None = None,
) -> dict[str, Any]:
    """Compute pairwise correlations and flag suspect pairs above thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    suspect_pairs : list[list[str]]
        List of column-name pairs to examine.
    leakage_threshold : float
        Primary correlation threshold (default 0.95).
    secondary_threshold : float, optional
        Lower threshold applied only if ``target_column`` is involved.
    target_column : str, optional
        If provided, pairs containing this column are evaluated against
        ``secondary_threshold`` (or ``leakage_threshold`` if not set).

    Returns
    -------
    dict
        List of pairs with correlation, flags for leaks, and warnings.
    """
    log.info("[collinearity_analysis] pairs=%d leakage_threshold=%.2f secondary=%.2f target=%s",
             len(suspect_pairs), leakage_threshold, secondary_threshold or leakage_threshold, target_column)

    pairs_result: list[dict[str, Any]] = []
    for pair in suspect_pairs:
        if len(pair) != 2:
            log.warning("[collinearity_analysis] invalid pair=%s – skip", pair)
            continue
        c1, c2 = pair
        if c1 not in df.columns or c2 not in df.columns:
            log.warning("[collinearity_analysis] columns not found: %s – skip", pair)
            continue
        corr_val = float(df[c1].corr(df[c2]))
        # Determine effective threshold
        threshold_used = leakage_threshold
        if target_column and (c1 == target_column or c2 == target_column):
            if secondary_threshold is not None:
                threshold_used = secondary_threshold
        flagged = abs(corr_val) >= threshold_used
        entry = {
            "col_1": c1,
            "col_2": c2,
            "correlation": round(corr_val, 4),
            "threshold_applied": threshold_used,
            "flagged": flagged,
            "leakage_risk": "high" if flagged else "none",
        }
        pairs_result.append(entry)
        log.debug("[collinearity_analysis] pair=(%s, %s) corr=%.4f flagged=%s",
                  c1, c2, corr_val, flagged)

    report: dict[str, Any] = {
        "pairs_checked": len(pairs_result),
        "pairs": pairs_result,
        "leakage_threshold": leakage_threshold,
        "secondary_threshold": secondary_threshold,
    }
    log.info("[collinearity_analysis] completed – flagged=%d", sum(1 for p in pairs_result if p["flagged"]))
    return report



# ─────────────────────────────────────────────────────────────────────────────
# 2.4 – Column catalog (feature_inventory)
# ─────────────────────────────────────────────────────────────────────────────

def categorize_columns(
        df: pd.DataFrame,
        *,
        roles: dict[str, list[str]] | None = None,
        categorize_by_role: bool = True,  # <--- Agregado para aceptar el parámetro del YAML
        **kwargs,                         # <--- Esto absorbe cualquier otro parámetro extra sin romper
) -> dict[str, Any]:
    """Categorize columns into predefined roles.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    roles : dict, optional
        Mapping of role name → list of column names.

    Returns
    -------
    dict
        Catalog with role keys and column metadata.
    """

    # Si categorize_by_role es False, podrías optar por ignorar los roles
    # o devolver una estructura vacía si eso prefieres.
    if not categorize_by_role:
        log.warning("[categorize_columns] categorize_by_role set to False – skipping")
        return {"n_roles": 0, "catalog": {}}


    log.info("[categorize_columns] start")

    if roles is None:
        # Default heuristic: infer from dtype and column name
        roles = {
            "identifiers": [c for c in df.columns if c.lower() == "id"],
            "targets": [c for c in df.columns if c in ("faulty", "trq_margin")],
            "temperatures": [c for c in df.columns if c in ("oat", "mgt")],
            "power_metrics": [c for c in df.columns if c in ("pa", "np", "trq_measured")],
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

    result: dict[str, Any] = {
        "n_roles": len(roles),
        "catalog": catalog,
    }
    log.info("[categorize_columns] completed – roles=%d", len(roles))
    return result



# ─────────────────────────────────────────────────────────────────────────────
# 2.4 – KS test per feature (domain shift)
# ─────────────────────────────────────────────────────────────────────────────


def ks_test_per_feature(
    df_ref: pd.DataFrame,
    df_comp: pd.DataFrame,
    *,
    features: list[str],
    significance_level: float = 0.01,
    shift_threshold: float = 0.1,
) -> dict[str, Any]:
    """Run two-sample KS test for each feature between a reference and comparison split.

    Parameters
    ----------
    df_ref : pd.DataFrame
        Reference dataset (e.g. train).
    df_comp : pd.DataFrame
        Comparison dataset (e.g. validation or test).
    features : list[str]
        Numeric features to test.
    significance_level : float
        p-value threshold for statistical significance.
    shift_threshold : float
        KS statistic threshold above which a shift is considered meaningful.

    Returns
    -------
    dict
        Per-feature KS statistics, p-values, and shift flags.
    """
    log.info("[ks_test_per_feature] features=%d sig_level=%.3f shift_threshold=%.2f",
             len(features), significance_level, shift_threshold)

    from scipy.stats import ks_2samp

    per_feature: list[dict[str, Any]] = []
    for feat in features:
        if feat not in df_ref.columns or feat not in df_comp.columns:
            log.warning("[ks_test_per_feature] feature '%s' missing in one split – skip", feat)
            continue
        s_ref = df_ref[feat].dropna().values
        s_comp = df_comp[feat].dropna().values
        if len(s_ref) == 0 or len(s_comp) == 0:
            log.warning("[ks_test_per_feature] feature '%s' empty – skip", feat)
            continue
        stat, pval = ks_2samp(s_ref, s_comp)
        flagged = (pval < significance_level) or (stat > shift_threshold)
        per_feature.append({
            "feature": feat,
            "ks_statistic": round(float(stat), 4),
            "ks_pvalue": round(float(pval), 4),
            "significance_level": significance_level,
            "shift_threshold": shift_threshold,
            "flagged": bool(flagged),
        })
        log.debug("[ks_test_per_feature] '%s' ks=%.4f pval=%.4f flagged=%s",
                  feat, stat, pval, flagged)

    report: dict[str, Any] = {
        "features_tested": len(per_feature),
        "results": per_feature,
        "significance_level": significance_level,
        "shift_threshold": shift_threshold,
    }
    log.info("[ks_test_per_feature] completed – flagged=%d", sum(1 for f in per_feature if f["flagged"]))
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 – GMM exploration (BIC/AIC curve)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 2.4 – GMM exploration PNG generation ?
# ─────────────────────────────────────────────────────────────────────────────



def gmm_exploration(
    df: pd.DataFrame,
    *,
    features: list[str],
    k_range: list[int],
    selection_criterion: str = "BIC",
    random_state: int = 7,
) -> dict[str, Any]:
    """Fit Gaussian Mixture models for a range of k and return BIC/AIC values.

    The runner can use the returned dict to generate a PNG.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    features : list[str]
        Features to use for GMM.
    k_range : list[int]
        Range of components (e.g. ``[2, 3, 4, 5, 6]``).
    selection_criterion : str
        ``"BIC"`` or ``"AIC"`` to determine the optimal k.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Per-k BIC/AIC values and optimal k.
    """
    from sklearn.mixture import GaussianMixture

    log.info("[gmm_exploration] features=%s k_range=%s criterion=%s",
             features, k_range, selection_criterion)

    # Prepare data
    available = [c for c in features if c in df.columns]
    if not available:
        log.error("[gmm_exploration] no valid features from %s", features)
        return {"error": "no_valid_features", "features_requested": features}

    X = df[available].dropna().values
    if len(X) == 0:
        log.error("[gmm_exploration] empty data after dropna")
        return {"error": "empty_data"}

    metrics: dict[str, list] = {"k": [], "BIC": [], "AIC": []}
    for k in sorted(k_range):
        try:
            gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=5)
            gmm.fit(X)
            metrics["k"].append(k)
            metrics["BIC"].append(round(float(gmm.bic(X)), 2))
            metrics["AIC"].append(round(float(gmm.aic(X)), 2))
            log.debug("[gmm_exploration] k=%d BIC=%.2f AIC=%.2f", k, metrics["BIC"][-1], metrics["AIC"][-1])
        except Exception as exc:
            log.warning("[gmm_exploration] k=%d failed – %s", k, exc)

    # Determine optimal k
    if selection_criterion.upper() == "BIC":
        values = metrics["BIC"]
    else:
        values = metrics["AIC"]

    optimal_k: int | None = None
    if values:
        min_idx = int(np.argmin(values))
        optimal_k = metrics["k"][min_idx]

    report: dict[str, Any] = {
        "features_used": available,
        "k_range_used": sorted(k_range),
        "selection_criterion": selection_criterion.upper(),
        "optimal_k": optimal_k,
        "curve": metrics,
    }
    log.info("[gmm_exploration] completed – optimal_k=%s", optimal_k)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 – Flight regime binning (histogram data)
# ─────────────────────────────────────────────────────────────────────────────


def flight_regime_binning(
    df: pd.DataFrame,
    *,
    column: str,
    bin_size: float = 10.0,
) -> dict[str, Any]:
    """Bin a continuous column (e.g. OAT) and return histogram data for plotting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to bin (e.g. ``"oat"``).
    bin_size : float
        Width of bins.

    Returns
    -------
    dict
        Bin edges, counts, and dominant regime information.
    """
    log.info("[flight_regime_binning] column='%s' bin_size=%.2f", column, bin_size)

    if column not in df.columns:
        log.error("[flight_regime_binning] column '%s' not found", column)
        return {"error": f"column '{column}' not found"}

    s = df[column].dropna()
    if len(s) == 0:
        log.warning("[flight_regime_binning] column '%s' is empty", column)
        return {"error": "empty_column"}

    # Build bins from min to max
    val_min = float(s.min())
    val_max = float(s.max())
    bins = np.arange(val_min, val_max + bin_size, bin_size)
    counts, edges = np.histogram(s, bins=bins)

    # Identify dominant bins (top 2)
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

    report: dict[str, Any] = {
        "column": column,
        "bin_size": bin_size,
        "n_total": int(len(s)),
        "n_bins": len(bins) - 1,
        "bin_edges": [round(float(e), 2) for e in edges],
        "bin_counts": [int(c) for c in counts],
        "dominant_regimes": dominant_bins,
    }
    log.info("[flight_regime_binning] completed – %d bins, top regime center=%.2f",
             len(bins) - 1, dominant_bins[0]["bin_center"] if dominant_bins else None)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 – Feature drift summary (aggregate KS results)
# ─────────────────────────────────────────────────────────────────────────────


def feature_drift_summary(
    ks_results: dict[str, Any],
    *,
    aggregate_by: str = "max_ks_statistic",
    flag_critical_shifts: bool = True,
) -> dict[str, Any]:
    """Aggregate per-feature KS results into a compact summary.

    Parameters
    ----------
    ks_results : dict
        Output of ``ks_test_per_feature``.
    aggregate_by : str
        Metric to aggregate by (``"max_ks_statistic"`` or ``"mean_ks_statistic"``).
    flag_critical_shifts : bool
        If True, add a list of critically shifted features.

    Returns
    -------
    dict
        Summary with aggregate metrics and optional critical list.
    """
    log.info("[feature_drift_summary] aggregate_by='%s' flag_critical=%s",
             aggregate_by, flag_critical_shifts)

    results = ks_results.get("results", [])
    if not results:
        log.warning("[feature_drift_summary] no results to aggregate")
        return {"error": "empty_results"}

    ks_values = [r["ks_statistic"] for r in results]
    pvalues = [r["ks_pvalue"] for r in results]

    summary: dict[str, Any] = {
        "n_features_tested": len(results),
        "max_ks_statistic": round(float(max(ks_values)), 4),
        "mean_ks_statistic": round(float(np.mean(ks_values)), 4),
        "min_ks_pvalue": round(float(min(pvalues)), 4),
        "n_flagged": sum(1 for r in results if r["flagged"]),
    }

    if aggregate_by == "max_ks_statistic":
        summary["aggregate"] = summary["max_ks_statistic"]
    elif aggregate_by == "mean_ks_statistic":
        summary["aggregate"] = summary["mean_ks_statistic"]
    else:
        summary["aggregate"] = None

    if flag_critical_shifts:
        critical = [r for r in results if r["flagged"]]
        summary["critical_shifts"] = [
            {
                "feature": r["feature"],
                "ks_statistic": r["ks_statistic"],
                "ks_pvalue": r["ks_pvalue"],
            }
            for r in critical
        ]
        log.debug("[feature_drift_summary] critical_shifts=%d", len(critical))

    log.info("[feature_drift_summary] completed – n_flagged=%d", summary["n_flagged"])
    return summary