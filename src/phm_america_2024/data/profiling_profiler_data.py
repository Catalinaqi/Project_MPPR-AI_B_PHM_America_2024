# src/phm_america_2024/data/profiling_profiler_data.py
from __future__ import annotations

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Stateless DataFrame analysis helpers for CRISP-DM Phase 2.
# Pure functions: no side effects, no I/O, no RunContext dependency.
# =============================================================================

"""Pure DataFrame profiling and drift-detection helpers for the CRISP-DM pipeline."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

from phm_america_2024.configuration.enum_registry_config import ProblemType
from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level DEFAULT drift thresholds
# ---------------------------------------------------------------------------
PSI_WARN: float = 0.10
PSI_DRIFT: float = 0.20
KS_ALPHA: float = 0.05


# =============================================================================
# SECTION 1 — COLUMN SELECTION HELPERS
# =============================================================================


def numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return column names with numeric dtype."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# =============================================================================
# SECTION 2 — SCHEMA AND STATISTICS TABLES
# =============================================================================


def schema_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-column summary: dtype, null count, null %, cardinality."""
    return pd.DataFrame(
        {
            "column": list(df.columns),
            "dtype": [str(df[c].dtype) for c in df.columns],
            "n_null": [int(df[c].isna().sum()) for c in df.columns],
            "null_pct": [float(df[c].isna().mean() * 100.0) for c in df.columns],
            "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    ).sort_values(["null_pct", "n_unique"], ascending=[False, False])


def describe_table(
        df: pd.DataFrame,
        include: Any = None,
        percentiles: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Transpose df.describe() and promote index to column field."""
    kwargs: dict[str, Any] = {"include": include}
    if percentiles is not None:
        kwargs["percentiles"] = percentiles
    desc = df.describe(**kwargs).transpose()
    desc.insert(0, "column", desc.index.astype(str))
    return desc.reset_index(drop=True)


def min_max_mean_std(
        df: pd.DataFrame,
        *,
        numeric_only: bool = True,
        exclude_bigint_hashed: bool = False,
        metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compute descriptive statistics for each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_only : bool
        If True, only process numeric columns.
    exclude_bigint_hashed : bool
        If True, skip BIGINT columns with high cardinality (pseudo-categorical).
    metrics : list[str] | None
        If provided, only compute the specified metrics.
        Valid values: "count", "min", "max", "mean", "std".
        If None, computes all: min, max, mean, std.
    """
    cols = numeric_cols(df) if numeric_only else list(df.columns)

    # Filter out BIGINT pseudo-categorical columns if requested
    if exclude_bigint_hashed:
        filtered_cols = []
        for c in cols:
            n_unique = int(df[c].nunique(dropna=True))
            if n_unique > 1000:  # heuristic: high-cardinality integer → skip
                log.debug("[min_max_mean_std] excluding bigint-pseudo column: %s", c)
                continue
            filtered_cols.append(c)
        cols = filtered_cols

    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        has_data = bool(s.notna().any())
        row: dict[str, Any] = {"column": c}

        if metrics is not None:
            # Only compute requested metrics
            if "count" in metrics:
                row["count"] = int(s.notna().sum()) if has_data else 0
            if "min" in metrics:
                row["min"] = float(s.min()) if has_data else None
            if "max" in metrics:
                row["max"] = float(s.max()) if has_data else None
            if "mean" in metrics:
                row["mean"] = float(s.mean()) if has_data else None
            if "std" in metrics:
                row["std"] = float(s.std()) if has_data else None
        else:
            # Default: all classic metrics
            row["min"] = float(s.min()) if has_data else None
            row["max"] = float(s.max()) if has_data else None
            row["mean"] = float(s.mean()) if has_data else None
            row["std"] = float(s.std()) if has_data else None

        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 3 — DUPLICATE DETECTION
# =============================================================================


def duplicates_summary(
        df: pd.DataFrame, *, subset: Optional[list[str]] = None, keep: str = "first"
) -> pd.DataFrame:
    """Produce one-row DataFrame with duplicate statistics."""
    dup_mask = df.duplicated(subset=subset, keep=keep)  # type: ignore[arg-type]
    n = len(df)
    return pd.DataFrame(
        [
            {
                "rows": int(n),
                "duplicates": int(dup_mask.sum()),
                "dup_pct": float(dup_mask.mean() * 100.0) if n else 0.0,
                "subset": str(subset),
                "keep": str(keep),
            }
        ]
    )


# =============================================================================
# SECTION 4 — STATISTICAL DRIFT DETECTION (PSI + KS)
# =============================================================================


def compute_psi(expected: pd.Series, actual: pd.Series, *, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two numeric series."""
    # Step 1: Drop NaN and guard empty
    exp = expected.dropna()
    act = actual.dropna()
    if len(exp) == 0 or len(act) == 0:
        return 0.0

    # Step 2: Build shared bins
    combined_min = float(min(exp.min(), act.min()))
    combined_max = float(max(exp.max(), act.max()))
    if combined_min == combined_max:
        return 0.0
    bins = np.linspace(combined_min, combined_max, n_bins + 1)

    # Step 3: Normalised frequencies
    exp_counts, _ = np.histogram(exp, bins=bins)
    act_counts, _ = np.histogram(act, bins=bins)
    exp_pct = np.clip(exp_counts / len(exp), 1e-6, None)
    act_pct = np.clip(act_counts / len(act), 1e-6, None)

    # Step 4: PSI formula
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_ks(expected: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """Compute KS statistic and p-value."""
    exp = expected.dropna()
    act = actual.dropna()
    if len(exp) == 0 or len(act) == 0:
        return 0.0, 1.0
    result = stats.ks_2samp(exp.to_numpy(), act.to_numpy())
    return float(result.statistic), float(result.pvalue)


def build_drift_report(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        numeric_feature_cols: list[str],
        task: ProblemType,
        *,
        target_col: Optional[str] = None,
        psi_drift: float = PSI_DRIFT,
        ks_alpha: float = KS_ALPHA,
        n_bins: int,
) -> pd.DataFrame:
    """Build per-column drift summary — task-aware."""
    PT = ProblemType
    rows: list[dict] = []

    # Step 1: PSI + KS on features
    for col in numeric_feature_cols:
        psi = compute_psi(df_train[col], df_test[col], n_bins=n_bins)
        ks_stat, ks_pval = compute_ks(df_train[col], df_test[col])
        rows.append(
            {
                "column": col,
                "check_type": "feature_psi_ks",
                "psi": round(psi, 4),
                "ks_stat": round(ks_stat, 4),
                "ks_pvalue": round(ks_pval, 4),
                "chi2_stat": None,
                "chi2_pvalue": None,
                "drift_flag": psi >= psi_drift or ks_pval < ks_alpha,
            }
        )

    # Step 2: Classification target Chi-square
    if task == PT.CLASSIFICATION and target_col and target_col in df_train.columns:
        train_counts = df_train[target_col].value_counts()
        test_counts = df_test[target_col].value_counts()
        all_cats = sorted(set(train_counts.index) | set(test_counts.index))
        contingency = pd.DataFrame(
            {
                "train": [train_counts.get(c, 0) for c in all_cats],
                "test": [test_counts.get(c, 0) for c in all_cats],
            }
        )
        chi2, p_chi2, _, _ = chi2_contingency(contingency.to_numpy())
        rows.append(
            {
                "column": target_col,
                "check_type": "target_chi2",
                "psi": None,
                "ks_stat": None,
                "ks_pvalue": None,
                "chi2_stat": round(float(chi2), 4),
                "chi2_pvalue": round(float(p_chi2), 4),
                "drift_flag": float(p_chi2) < ks_alpha,
            }
        )

    # Step 3: Regression target PSI + KS
    elif task == PT.REGRESSION and target_col and target_col in df_train.columns:
        psi = compute_psi(df_train[target_col], df_test[target_col], n_bins=n_bins)
        ks_stat, ks_pval = compute_ks(df_train[target_col], df_test[target_col])
        rows.append(
            {
                "column": target_col,
                "check_type": "target_psi_ks",
                "psi": round(psi, 4),
                "ks_stat": round(ks_stat, 4),
                "ks_pvalue": round(ks_pval, 4),
                "chi2_stat": None,
                "chi2_pvalue": None,
                "drift_flag": psi >= psi_drift or ks_pval < ks_alpha,
            }
        )

    # Step 4: Timeseries window drift
    elif task == PT.TIMESERIES:
        mid = len(df_train) // 2
        df_early = df_train.iloc[:mid]
        df_late = df_train.iloc[mid:]
        for col in numeric_feature_cols:
            psi = compute_psi(df_early[col], df_late[col], n_bins=n_bins)
            ks_stat, ks_pval = compute_ks(df_early[col], df_late[col])
            rows.append(
                {
                    "column": col,
                    "check_type": "timeseries_window_psi_ks",
                    "psi": round(psi, 4),
                    "ks_stat": round(ks_stat, 4),
                    "ks_pvalue": round(ks_pval, 4),
                    "chi2_stat": None,
                    "chi2_pvalue": None,
                    "drift_flag": psi >= psi_drift or ks_pval < ks_alpha,
                }
            )

    df_result = pd.DataFrame(rows)
    if df_result.empty:
        return df_result
    return df_result.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


# =============================================================================
# SECTION 5 — NEW PHASE 2 TECHNIQUES (2.1, 2.2, 2.3, 2.4)
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 2.1 — Data Acquisition
# ─────────────────────────────────────────────────────────────────────────────


def hierarchy_profiling_report(
        df: pd.DataFrame,
        *,
        hierarchy_levels: list[str],
        compute_ratios: bool = True,
        expected_ratios: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Analyze Evidence→Alert→Incident hierarchy."""
    # Step 1: Count unique per level
    counts = {lvl: int(df[lvl].nunique(dropna=True)) for lvl in hierarchy_levels if lvl in df.columns}

    # Step 2: Compute ratios
    ratios: dict[str, float] = {}
    if compute_ratios and len(hierarchy_levels) >= 2:
        for i in range(len(hierarchy_levels) - 1):
            parent = hierarchy_levels[i + 1]
            child = hierarchy_levels[i]
            if parent in counts and child in counts and counts[parent] > 0:
                ratios[f"{child}_per_{parent}"] = round(counts[child] / counts[parent], 2)

    # Step 3: Compare to expected
    deviations = {}
    if expected_ratios:
        for key, expected in expected_ratios.items():
            if key in ratios:
                deviations[f"{key}_deviation"] = round(ratios[key] - expected, 2)

    log.debug("[hierarchy_profiling_report] counts=%s ratios=%s", counts, ratios)
    return {"counts": counts, "ratios": ratios, "deviations": deviations}


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 — Data Description
# ─────────────────────────────────────────────────────────────────────────────


def column_metadata_report(
        df: pd.DataFrame,
        *,
        include_cardinality: bool = True,
        include_dtypes: bool = True,
        cardinality_threshold: int = 1000,
        detect_bigint_pseudo_categorical: bool = True,
) -> pd.DataFrame:
    """Generate column metadata with cardinality classification.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    include_cardinality : bool
        Whether to classify cardinality as low/medium/high.
    include_dtypes : bool
        Whether to include detailed dtype string in the output.
    cardinality_threshold : int
        Threshold distinguishing medium from high cardinality.
    detect_bigint_pseudo_categorical : bool
        Whether to flag BIGINT columns that may be pseudo-categorical.
    """
    rows = []
    for col in df.columns:
        n_unique = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)

        # Step 1: Classify cardinality
        if include_cardinality:
            if n_unique <= 20:
                cardinality_class = "low"
            elif n_unique <= cardinality_threshold:
                cardinality_class = "medium"
            else:
                cardinality_class = "high"
        else:
            cardinality_class = None

        # Step 2: Detect BIGINT pseudo-categorical
        is_bigint_pseudo = False
        if detect_bigint_pseudo_categorical and dtype_str.startswith("int") and n_unique > cardinality_threshold:
            is_bigint_pseudo = True

        row = {
            "column": col,
            "n_unique": n_unique,
            "cardinality_class": cardinality_class,
            "is_bigint_pseudo_categorical": is_bigint_pseudo,
        }

        # Step 3: Include dtype info only if requested
        if include_dtypes:
            row["dtype"] = dtype_str

        rows.append(row)

    return pd.DataFrame(rows)


def schema_comparison_report(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        *,
        check_column_names: bool = True,
        check_dtypes: bool = True,
        check_column_order: bool = False,
        strict_mode: bool = False,
        report_missing_in_train: bool = True,
        report_missing_in_test: bool = True,
) -> dict[str, Any]:
    """Compare schemas between train and test.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training DataFrame.
    df_test : pd.DataFrame
        Test/validation DataFrame.
    check_column_names : bool
        Whether to compare column name sets.
    check_dtypes : bool
        Whether to compare dtypes for common columns.
    check_column_order : bool
        Whether to verify exact column order match.
    strict_mode : bool
        If True, treat missing_in_train as incompatibility.
    report_missing_in_train : bool
        Whether to include missing_in_train list in output.
    report_missing_in_test : bool
        Whether to include missing_in_test list in output.
    """
    # Step 1: Column name comparison
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)
    missing_in_test = sorted(train_cols - test_cols) if check_column_names else []
    missing_in_train = sorted(test_cols - train_cols) if check_column_names else []

    # Step 2: Dtype comparison
    dtype_mismatches = []
    if check_dtypes:
        common_cols = train_cols & test_cols
        for col in common_cols:
            if str(df_train[col].dtype) != str(df_test[col].dtype):
                dtype_mismatches.append(
                    {
                        "column": col,
                        "train_dtype": str(df_train[col].dtype),
                        "test_dtype": str(df_test[col].dtype),
                    }
                )

    # Step 3: Check column order
    column_order_match = True
    if check_column_order:
        column_order_match = list(df_train.columns) == list(df_test.columns)

    # Step 4: Build report
    is_compatible = len(missing_in_test) == 0 and len(dtype_mismatches) == 0 and column_order_match
    if strict_mode:
        is_compatible = is_compatible and len(missing_in_train) == 0

    report: dict[str, Any] = {
        "is_compatible": is_compatible,
        "dtype_mismatches": dtype_mismatches,
        "column_order_match": column_order_match if check_column_order else None,
    }

    if report_missing_in_test:
        report["missing_in_test"] = missing_in_test
    if report_missing_in_train:
        report["missing_in_train"] = missing_in_train

    log.debug("[schema_comparison] compatible=%s missing_in_test=%d", is_compatible, len(missing_in_test))
    return report


def multi_value_parser(
        df: pd.DataFrame,
        *,
        columns: list[str],
        delimiter: str = ",",
        max_values_per_row: int = 10,
        min_frequency: float = 0.001,
        report_top_n: int = 20,
) -> dict[str, Any]:
    """Parse comma-separated values in columns like MitreTechniques, Roles."""
    results = {}

    for col in columns:
        if col not in df.columns:
            continue

        # Step 1: Split and flatten
        all_values: list[str] = []
        for cell in df[col].dropna():
            parts = str(cell).split(delimiter)[:max_values_per_row]
            all_values.extend([p.strip() for p in parts if p.strip()])

        # Step 2: Frequency analysis
        value_counts = pd.Series(all_values).value_counts()
        total = len(all_values)
        filtered = value_counts[value_counts / total >= min_frequency]

        results[col] = {
            "total_values": total,
            "unique_values": len(value_counts),
            "top_values": filtered.head(report_top_n).to_dict(),
        }
        log.debug("[multi_value_parser] col=%s total=%d unique=%d", col, total, len(value_counts))

    return results


def cardinality_profiler(
        df: pd.DataFrame,
        *,
        target_columns: dict[str, list[str]],
        report_top_n: int = 20,
        flag_if_cardinality_exceeds: int = 10000,
) -> dict[str, Any]:
    """Classify columns by cardinality and flag high-cardinality columns."""
    results = {}

    for category, cols in target_columns.items():
        category_results = []
        for col in cols:
            if col not in df.columns:
                continue

            n_unique = int(df[col].nunique(dropna=True))
            value_counts = df[col].value_counts().head(report_top_n)

            category_results.append(
                {
                    "column": col,
                    "n_unique": n_unique,
                    "exceeds_threshold": n_unique > flag_if_cardinality_exceeds,
                    "top_values": value_counts.to_dict(),
                }
            )

        results[category] = category_results
        log.debug("[cardinality_profiler] category=%s cols=%d", category, len(category_results))

    return results


def target_distribution_report(
        df: pd.DataFrame,
        *,
        target_column: str,
        detect_none_values: bool = True,
        compute_imbalance_ratio: bool = True,
        report_value_counts: bool = True,
) -> dict[str, Any]:
    """Analyze target distribution with 'None' detection.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_column : str
        Name of the target column.
    detect_none_values : bool
        Whether to detect 'None' string values.
    compute_imbalance_ratio : bool
        Whether to compute the imbalance ratio.
    report_value_counts : bool
        Whether to include value counts in the output dictionary.
    """
    if target_column not in df.columns:
        return {}

    # Step 1: Value counts
    value_counts = df[target_column].value_counts(dropna=False)

    # Step 2: Detect 'None' string values
    has_none_string = False
    none_count = 0
    if detect_none_values:
        none_mask = df[target_column].astype(str).str.lower() == "none"
        none_count = int(none_mask.sum())
        has_none_string = none_count > 0

    # Step 3: Imbalance ratio
    imbalance_ratio = None
    if compute_imbalance_ratio and len(value_counts) > 1:
        imbalance_ratio = round(float(value_counts.max() / value_counts.min()), 2)

    log.debug(
        "[target_distribution_report] target=%s classes=%d none_count=%d",
        target_column,
        len(value_counts),
        none_count,
    )

    report: dict[str, Any] = {
        "target_column": target_column,
        "has_none_string": has_none_string,
        "none_count": none_count,
        "imbalance_ratio": imbalance_ratio,
    }

    if report_value_counts:
        report["value_counts"] = value_counts.to_dict()

    return report


def detect_id_columns(
        df: pd.DataFrame, *, id_patterns: list[str], uniqueness_min: float = 0.95
) -> list[str]:
    """Detect identifier columns with uniqueness > threshold."""
    id_cols = []

    for col in df.columns:
        # Step 1: Check pattern match
        matches_pattern = any(
            col.endswith(pattern.replace("*", "")) or col == pattern for pattern in id_patterns
        )

        # Step 2: Check uniqueness
        uniqueness = df[col].nunique(dropna=True) / len(df) if len(df) > 0 else 0.0

        if matches_pattern and uniqueness >= uniqueness_min:
            id_cols.append(col)

    log.debug("[detect_id_columns] found=%d cols=%s", len(id_cols), id_cols)
    return id_cols


def entity_conditional_sparsity(
        df: pd.DataFrame,
        *,
        entity_column: str,
        conditional_columns: dict[str, list[str]],
        report_null_percentages: bool = True,
) -> dict[str, Any]:
    """Detect EntityType-dependent NULL patterns."""
    if entity_column not in df.columns:
        return {}

    results = {}

    for entity_type, cols in conditional_columns.items():
        entity_mask = df[entity_column] == entity_type
        n_rows = int(entity_mask.sum())

        if n_rows == 0:
            continue

        col_nulls = {}
        if report_null_percentages:
            for col in cols:
                if col in df.columns:
                    null_pct = round(float(df.loc[entity_mask, col].isna().mean() * 100), 2)
                    col_nulls[col] = null_pct

        results[entity_type] = {"n_rows": n_rows, "null_percentages": col_nulls}
        log.debug("[entity_conditional_sparsity] entity=%s rows=%d", entity_type, n_rows)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.3 — Data Quality Verification
# ─────────────────────────────────────────────────────────────────────────────


def completeness_report(
        df: pd.DataFrame,
        *,
        include_patterns: bool = True,
        show_top_columns: int = 50,
        distinguish_null_by_design: bool = True,
) -> dict[str, Any]:
    """Generate completeness report with null patterns."""
    # Step 1: Null counts per column
    null_df = pd.DataFrame(
        {
            "column": list(df.columns),
            "n_null": [int(df[c].isna().sum()) for c in df.columns],
            "null_pct": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        }
    ).sort_values("null_pct", ascending=False)

    # Step 2: Overall stats
    total_nulls = int(null_df["n_null"].sum())
    total_cells = len(df) * len(df.columns)
    overall_null_pct = round(100 * total_nulls / total_cells, 2) if total_cells > 0 else 0.0

    log.debug("[completeness_report] total_nulls=%d overall_pct=%.2f", total_nulls, overall_null_pct)

    return {
        "total_nulls": total_nulls,
        "overall_null_pct": overall_null_pct,
        "top_columns": null_df.head(show_top_columns).to_dict(orient="records"),
    }


def detect_sentinel_values(
        df: pd.DataFrame, *, sentinel_values: list[int | float], check_columns: list[str]
) -> dict[str, Any]:
    """Detect sentinel values like -1, 999, 9999."""
    results = {}

    for col in check_columns:
        if col not in df.columns:
            continue

        col_results = {}
        for sentinel in sentinel_values:
            count = int((df[col] == sentinel).sum())
            if count > 0:
                col_results[str(sentinel)] = count

        if col_results:
            results[col] = col_results

    log.debug("[detect_sentinel_values] cols_with_sentinels=%d", len(results))
    return results


def crosstab_leakage_analysis(
        df: pd.DataFrame,
        *,
        target_column: str,
        suspect_columns: list[str],
        normalize: str = "index",
        leakage_threshold: float = 0.95,
) -> dict[str, Any]:
    """Detect leakage via crosstab correlation > threshold."""
    if target_column not in df.columns:
        return {}

    leakage_suspects = []

    for col in suspect_columns:
        if col not in df.columns or col == target_column:
            continue

        # Step 1: Build crosstab
        ct = pd.crosstab(df[col], df[target_column], normalize=normalize)

        # Step 2: Check if any column has correlation > threshold
        max_corr = float(ct.max().max()) if not ct.empty else 0.0

        if max_corr >= leakage_threshold:
            leakage_suspects.append({"column": col, "max_correlation": round(max_corr, 4)})

    log.debug("[crosstab_leakage_analysis] suspects=%d", len(leakage_suspects))
    return {"leakage_suspects": leakage_suspects}


def post_triage_detector(
        df: pd.DataFrame, *, high_missingness_threshold: float = 0.9, suspect_columns: list[str]
) -> list[str]:
    """Detect features likely generated after incident triage."""
    post_triage_cols = []

    for col in suspect_columns:
        if col not in df.columns:
            continue

        null_rate = float(df[col].isna().mean())
        if null_rate >= high_missingness_threshold:
            post_triage_cols.append(col)

    log.debug("[post_triage_detector] found=%d", len(post_triage_cols))
    return post_triage_cols


def timestamp_range_validator(
        df: pd.DataFrame,
        *,
        timestamp_column: str,
        expected_min_days: int,
        detect_timezone_issues: bool = True,
) -> dict[str, Any]:
    """Verify observation window duration."""
    if timestamp_column not in df.columns:
        return {}

    # Step 1: Convert to datetime
    ts = pd.to_datetime(df[timestamp_column], errors="coerce")
    ts_clean = ts.dropna()

    if len(ts_clean) == 0:
        return {"valid": False, "reason": "no_valid_timestamps"}

    # Step 2: Compute range
    min_ts = ts_clean.min()
    max_ts = ts_clean.max()
    actual_days = (max_ts - min_ts).days

    # Step 3: Validation
    is_valid = actual_days >= expected_min_days

    log.debug(
        "[timestamp_range_validator] actual_days=%d expected=%d valid=%s",
        actual_days,
        expected_min_days,
        is_valid,
    )

    return {
        "valid": is_valid,
        "actual_days": actual_days,
        "expected_min_days": expected_min_days,
        "min_timestamp": str(min_ts),
        "max_timestamp": str(max_ts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 — Exploratory Analysis
# ─────────────────────────────────────────────────────────────────────────────


def column_catalog_by_roles(
        df: pd.DataFrame,
        *,
        roles: dict[str, list[str]],
        include_created_in_phase_2: bool = False,
        categorize_by_role: bool = True,
) -> dict[str, Any]:
    """Categorize columns by their analytical role.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    roles : dict[str, list[str]]
        Dictionary mapping role names to lists of column names.
    include_created_in_phase_2 : bool
        Whether to include columns created in Phase 2 (not yet available).
    categorize_by_role : bool
        If True, includes an 'uncategorized' list for columns not matched
        to any defined role.
    """
    catalog = {}

    for role_name, col_list in roles.items():
        present_cols = [c for c in col_list if c in df.columns]
        catalog[role_name] = present_cols

    # Report uncategorized columns if requested
    if categorize_by_role:
        all_categorized: set[str] = set()
        for col_list in roles.values():
            all_categorized.update(col_list)
        uncategorized = [c for c in df.columns if c not in all_categorized]
        catalog["uncategorized"] = uncategorized
        log.debug("[column_catalog_by_roles] uncategorized=%d", len(uncategorized))

    log.debug("[column_catalog_by_roles] roles=%d", len(catalog))
    return catalog


##################################


# =============================================================================
# SECTION 6 — NEW PHYSICAL & STATISTICAL CHECKS (Phase 2, steps 2.2-2.4)
# =============================================================================


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