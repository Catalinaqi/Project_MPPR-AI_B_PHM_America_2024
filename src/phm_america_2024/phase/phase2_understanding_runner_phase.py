# src/phm_america_2024/phase/phase2_understanding_runner_phase.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from phm_america_2024.common.context_facade_common import RunContext
from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.dict_facade_common import enabled, dget, ensure_native_types
from phm_america_2024.data.load_loader_data import load_parquet as load_pq
from phm_america_2024.configuration.enum_registry_config import StepsPhase, ReadMode
from phm_america_2024.configuration.read_strategy_repository_config import (
    ReadStrategyContract,
    DataSourceConfig,
)

from phm_america_2024.data.csv_loader_data import (
    load_by_strategy,
    load_with_origin,
    load_train_only,
)
from phm_america_2024.data.persist_persister_data import save_parquet, save_json
from phm_america_2024.data.profiling_profiler_data import (
    column_metadata_report,
    min_max_mean_std,
    analyze_target_distributions,
    zero_or_negative_check,
    collinearity_analysis,
    column_catalog_by_roles,
    ks_test_per_feature,
    feature_drift_summary,
    gmm_exploration,
    flight_regime_binning,
)

from phm_america_2024.registry.generator_registry_registry import write_output_artifacts
from phm_america_2024.common.path_service_common import resolve_path

# ── Side‑effect: register artifact generators ───────────────────────────
import phm_america_2024.registry.phase2_generator_registry  # noqa: F401

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_phase2(
    ctx: RunContext,
    steps_filter: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run Phase 2 sub‑steps and return a summary dictionary.

    Step 1: Extract phase configuration from context.
    Step 2: If steps_filter is provided, only steps whose enum value is in
            the list are executed; otherwise all defined steps run.
    Step 3: For each enabled step, resolve input DataFrames, execute
            techniques, and register the summary.

    Parameters
    ----------
    ctx : RunContext
        Immutable run context carrying the merged configuration and
        artifact registry.
    steps_filter : list[str], optional
        List of step enum values to execute (e.g.
        ``["step_2_1_data_acquisition"]``).  If ``None``, all steps
        defined in YAML are run.

    Returns
    -------
    dict
        Summary with keys ``status``, ``n_artifacts``, ``errors``,
        ``warnings_logged``.
    """
    # Step 1: Extract phase configuration from context.
    phase_cfg = ctx.config.phases.phase2_data_understanding
    common_cfg = ctx.config.common_base_config
    compression = common_cfg.output_policy.compression

    log.info("[Phase 2] starting – objective='%s'", phase_cfg.objective)
    log.debug("[Phase 2] steps_filter=%s", steps_filter)

    # ── Log all critical warnings from YAML ──────────────────────────────────
    _log_critical_warnings(phase_cfg)

    # ── Execute steps in order ──────────────────────────────────────────────
    step_order = [
        StepsPhase.STEP_2_1,
        StepsPhase.STEP_2_2,
        StepsPhase.STEP_2_3,
        StepsPhase.STEP_2_4,
    ]

    total_artifacts = 0
    errors: list[str] = []

    for step_key in step_order:
        # Step 2: Skip step if not in filter (when filter is provided)
        if steps_filter is not None and step_key.value not in steps_filter:
            log.debug("[Phase 2] step '%s' not in steps_filter – skip", step_key.value)
            continue

        step_cfg = phase_cfg.steps.get(step_key.value)
        if step_cfg is None:
            log.warning("[Phase 2] step '%s' not defined in YAML – skipping", step_key.value)
            continue

        if not enabled(step_cfg):
            log.debug("[Phase 2] step '%s' disabled – skip", step_key.value)
            continue

        log.info("[Phase 2] executing step='%s' – %s", step_key.value, step_cfg.description)

        # Step 3: Determine input DataFrame(s) for this step
        try:
            input_dfs = _resolve_step_input(ctx, step_key, step_cfg, phase_cfg, compression)
        except RuntimeError as exc:
            log.error("[Phase 2] step '%s' input resolution failed – %s", step_key.value, exc)
            ctx.collect_error(str(exc))
            errors.append(str(exc))
            continue

        # Run the step
        try:
            n_out = _run_step(ctx, step_key, step_cfg, input_dfs, phase_cfg, common_cfg)
            total_artifacts += n_out
        except Exception as exc:
            log.error("[Phase 2] step '%s' execution failed – %s", step_key.value, exc)
            ctx.collect_error(str(exc))
            errors.append(str(exc))

    # ── Final summary ────────────────────────────────────────────────────────
    summary = {
        "status": "completed_with_errors" if errors else "completed",
        "n_artifacts_written": total_artifacts,
        "errors": errors,
        "warnings_logged": len(phase_cfg.get("critical_warnings", {})),
    }
    ctx.register_phase_result("phase2", summary)
    log.info("[Phase 2] completed – status=%s artifacts=%d errors=%d",
             summary["status"], total_artifacts, len(errors))
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
# ... (rest of the file remains identical)

# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log_critical_warnings(phase_cfg: dict[str, Any]) -> None:
    """Emit all critical_warnings defined in the YAML phase block."""
    warnings = phase_cfg.get("critical_warnings", {})
    if not warnings:
        log.debug("[Phase 2] no critical warnings defined")
        return

    for key, warning in warnings.items():
        severity = warning.get("severity", "INFO").upper()
        msg = f"[Phase 2] {severity}: {key} – {warning.get('description', 'no description')}"
        if severity == "CRITICAL":
            log.warning(msg)
        elif severity == "ALERT":
            log.warning(msg)
        elif severity == "PENDING":
            log.warning(msg)
        else:
            log.info(msg)


def _resolve_step_input(
    ctx: RunContext,
    step_key: StepsPhase,
    step_cfg: dict[str, Any],
    phase_cfg: dict[str, Any],
    compression: str,
) -> dict[str, pd.DataFrame]:
    """Return a dict of resolved DataFrames needed by the step.

    For step 2.1 the input is loaded directly from raw CSV files.
    For subsequent steps the input is loaded from the parquet artifact
    of the preceding step (``sample_x_y_train_parquet``).
    Additional splits (test, validation) are loaded when required
    by a technique's ``split`` field.
    """
    if step_key == StepsPhase.STEP_2_1:
        # ── Load raw CSVs and merge X + Y ──────────────────────────────────
        return _load_raw_csvs(ctx, phase_cfg, compression)
    else:
        # ── Load from previous step's parquet artifacts ─────────────────────
        # Determine which artifacts are referenced by the step's techniques.
        needed_artifacts = _collect_split_artifacts(step_cfg, phase_cfg)
        if not needed_artifacts:
            # Default: load the main train parquet from step 2.1
            needed_artifacts = {"sample_x_y_train_parquet"}
        return _load_artifacts(ctx, needed_artifacts)


def _collect_split_artifacts(step_cfg: dict[str, Any], phase_cfg: dict[str, Any]) -> set[str]:
    """Collect all distinct artifact keys referenced by any technique's ``split`` field."""
    artifacts: set[str] = set()
    for method_cfg in step_cfg.get("methods", {}).values():
        for tech_cfg in method_cfg.get("techniques", {}).values():
            split_list = tech_cfg.get("split")
            if split_list:
                for ref in split_list:
                    # ref e.g. "${sample_x_y_train_parquet}"
                    if isinstance(ref, str) and ref.startswith("${") and ref.endswith("}"):
                        key = ref[2:-1]
                        artifacts.add(key)
    log.debug("[_collect_split_artifacts] found artifact keys: %s", artifacts)
    return artifacts


def _load_artifacts(ctx: RunContext, artifact_keys: set[str]) -> dict[str, pd.DataFrame]:
    """Load parquet artifacts from context and return a name → DataFrame map."""


    result = {}
    for key in artifact_keys:
        if key not in ctx.artifacts:
            raise RuntimeError(f"[_load_artifacts] artifact '{key}' not registered – run step 2.1 first")
        path = ctx.artifacts[key]
        log.debug("[_load_artifacts] loading artifact '%s' from %s", key, path)
        df = load_pq(str(path))
        result[key] = df
    return result


def _load_raw_csvs(
    ctx: RunContext,
    phase_cfg: dict[str, Any],
    compression: str,
) -> dict[str, pd.DataFrame]:
    """Load X_train, Y_train, merge, and also load test/validation (no Y).

    Returns a dict with keys ``"x_y_train"``, ``"x_test"``, ``"x_val"``.
    Also saves the three parquet artifacts via write_output_artifacts.
    """
    log.info("[Step 2.1] loading raw CSV files")

    dataset_input = phase_cfg.dataset_input
    read_strategy_raw = phase_cfg.read_strategy

    # Build ReadStrategyContract from YAML read_strategy block
    strategy = ReadStrategyContract.from_dict(read_strategy_raw)

    # Resolve CSV paths (already injected by ConfigBuilder)
    x_train_path = resolve_path(dataset_input.x_train_path)
    y_train_path = resolve_path(dataset_input.y_train_path)
    x_test_path = resolve_path(dataset_input.x_test_path)
    x_val_path = resolve_path(dataset_input.x_validation_path)

    # Load X_train and Y_train using the strategy (sampled)
    log.debug("[Step 2.1] loading X_train from %s", x_train_path)
    df_x, _, _ = load_by_strategy(str(x_train_path), csv_params=None, strategy=strategy)
    log.info("[Step 2.1] X_train loaded: %s", df_x.shape)

    log.debug("[Step 2.1] loading Y_train from %s", y_train_path)
    df_y, _, _ = load_by_strategy(str(y_train_path), csv_params=None, strategy=strategy)
    log.info("[Step 2.1] Y_train loaded: %s", df_y.shape)

    # Merge on join_key
    join_key = strategy.join_key or "id"
    df_merged = df_x.merge(df_y, on=join_key, how="inner")
    log.info("[Step 2.1] merged train shape: %s", df_merged.shape)

    # Load test and validation (full, since small)
    log.debug("[Step 2.1] loading X_test from %s", x_test_path)
    df_test = pd.read_csv(x_test_path)
    log.info("[Step 2.1] X_test loaded: %s", df_test.shape)

    log.debug("[Step 2.1] loading X_validation from %s", x_val_path)
    df_val = pd.read_csv(x_val_path)
    log.info("[Step 2.1] X_validation loaded: %s", df_val.shape)

    # Save the three parquet artifacts via the generator registry
    # write_output_artifacts(
    #     ctx,
    #     step_key=StepsPhase.STEP_2_1.value,
    #     step_cfg=phase_cfg.steps[StepsPhase.STEP_2_1.value],
    #     df_merged=df_merged,
    #     df_test=df_test,
    #     df_val=df_val,
    # )

    # Return dict for in‑memory use if needed (techniques of step 2.1 itself)
    return {
        "df_merged": df_merged,
        "df_test":   df_test,
        "df_val":    df_val
    }


def _run_step(
    ctx: RunContext,
    step_key: StepsPhase,
    step_cfg: dict[str, Any],
    input_dfs: dict[str, pd.DataFrame],
    phase_cfg: dict[str, Any],
    common_cfg: dict[str, Any],
) -> int:
    """Execute all techniques of one step and write outputs.

    Returns number of artifact files written.
    """
    compression = common_cfg.output_policy.compression
    phase_dir = ctx.phase2_dir
    n_written = 0

    # Parse methods and techniques from YAML
    methods = step_cfg.get("methods", {})

    for method_name, method_cfg in methods.items():
        if not enabled(method_cfg):
            log.debug("[Phase 2] method '%s' disabled – skip", method_name)
            continue
        for tech_name, tech_cfg in method_cfg.get("techniques", {}).items():
            if not enabled(tech_cfg):
                log.debug("[Phase 2] technique '%s' disabled – skip", tech_name)
                continue

            # Determine which dataframes this technique needs
            tech_split = tech_cfg.get("split", [])
            if not tech_split:
                # Default: use the main train dataframe (first available)
                tech_df = _select_default_df(input_dfs)
            else:
                # Resolve split references to actual dataframes
                tech_df = _resolve_technique_df(tech_split, input_dfs)

            # Parameters from YAML (already native types after OmegaConf)
            params = ensure_native_types(tech_cfg.get("params", {}))

            # Dispatch to appropriate profiling function
            log.debug("[Phase 2] executing technique='%s' with params=%s", tech_name, params)
            result = _dispatch_technique(tech_name, tech_df, params)

            # Determine output path
            output_rel = tech_cfg.get("output")
            if not output_rel:
                log.warning("[Phase 2] technique '%s' has no output path – report not saved", tech_name)
                continue

            output_path = phase_dir / output_rel
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save result (JSON or PNG based on extension)
            _save_technique_output(result, output_path, compression)
            n_written += 1
            log.info("[Phase 2] saved technique output: %s", output_path)

    # After techniques, call write_output_artifacts for the step's defined artifacts
    # (only step 2.1 actually has output_artifacts in YAML)
    if step_cfg.get("output_artifacts"):
        write_output_artifacts(ctx, step_key.value, step_cfg, **input_dfs)

    return n_written


def _select_default_df(input_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the first available DataFrame (used when no split specified)."""
    for name, df in input_dfs.items():
        log.debug("[_select_default_df] using '%s' (shape=%s)", name, df.shape)
        return df
    raise RuntimeError("[_select_default_df] no input DataFrame available")


def _resolve_technique_df(
    split_refs: list[str],
    input_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Convert split references to a single DataFrame or a dict of them.

    If only one reference, return that DataFrame directly.
    If multiple, return a dict keyed by artifact name.
    """
    resolved: dict[str, pd.DataFrame] = {}
    for ref in split_refs:
        # Remove ${} wrapper
        if isinstance(ref, str) and ref.startswith("${") and ref.endswith("}"):
            key = ref[2:-1]
        else:
            key = ref
        if key not in input_dfs:
            raise RuntimeError(f"[_resolve_technique_df] artifact '{key}' not available")
        resolved[key] = input_dfs[key]

    if len(resolved) == 1:
        return next(iter(resolved.values()))
    return resolved


def _dispatch_technique(
    tech_name: str,
    df_input: pd.DataFrame | dict[str, pd.DataFrame],
    params: dict[str, Any],
) -> Any:
    """Call the appropriate profiling function based on technique name.

    The mapping is defined here to keep the runner decoupled from
    profiling internals.  Each case extracts the required arguments
    from ``params`` (which come from the YAML configuration).
    """
    # ── Step 2.2 techniques ──────────────────────────────────────────────────
    if tech_name == "load_and_merge":
        log.info("[_dispatch_technique] Executing profiling for load_and_merge")
        # Generamos un diccionario con metadatos reales del DataFrame combinado
        return {
            "status": "success",
            "rows": int(df_input.shape[0]),
            "columns": int(df_input.shape[1]),
            "column_names": list(df_input.columns),
            "null_counts": df_input.isnull().sum().to_dict(),
            "dtypes": {k: str(v) for k, v in df_input.dtypes.items()}
        }
    if tech_name == "column_metadata":
        return column_metadata_report(df_input, **params)

    if tech_name == "basic_stats":
        return min_max_mean_std(df_input, **params)

    if tech_name == "null_count_per_column":
        cols = params.get("columns", list(df_input.columns))
        return {col: int(df_input[col].isna().sum()) for col in cols if col in df_input.columns}

    if tech_name == "distribution_analysis":
        return analyze_target_distributions(df_input, **params)

    # ── Step 2.3 techniques ──────────────────────────────────────────────────
    if tech_name == "zero_or_negative_check":
        return zero_or_negative_check(df_input, **params)

    if tech_name == "collinearity_analysis":
        return collinearity_analysis(df_input, **params)

    # ── Step 2.4 techniques ──────────────────────────────────────────────────
    if tech_name == "column_catalog":
        return column_catalog_by_roles(df_input, **params)

    if tech_name == "ks_test_per_feature":
        # Expects df_input to be a dict with keys train, test, validation
        if not isinstance(df_input, dict):
            raise TypeError("[ks_test_per_feature] requires a dict of DataFrames (train, test, validation)")
        # Determine which splits to compare
        compare_splits = params.get("compare_splits", ["validation", "test"])
        ref_df = df_input.get("sample_x_y_train_parquet")
        if ref_df is None:
            # Fallback: use first available
            ref_df = next(iter(df_input.values()))
        results = []
        for comp_key in compare_splits:
            comp_df = df_input.get(f"sample_x_{comp_key}_parquet") or df_input.get(comp_key)
            if comp_df is None:
                log.warning("[ks_test_per_feature] split '%s' not available – skip", comp_key)
                continue
            res = ks_test_per_feature(ref_df, comp_df, **params)
            results.append({comp_key: res})
        return results

    if tech_name == "gmm_exploration":
        return gmm_exploration(df_input, **params)

    if tech_name == "flight_regime_binning":
        return flight_regime_binning(df_input, **params)

    if tech_name == "feature_drift_summary":
        # This technique expects the ks_report as input, not a DataFrame.
        # The runner should have run ks_test_per_feature first and stored its
        # output in memory?  For MVP we compute it inline here by calling
        # ks_test_per_feature again (simplest).
        log.warning("[feature_drift_summary] currently requires pre‑computed ks results – computing inline")
        # Not implemented in this MVP; skip.
        return {"error": "not_implemented_directly"}

    # ── Fallback ─────────────────────────────────────────────────────────────
    log.warning("[_dispatch_technique] unknown technique '%s' – returning empty dict", tech_name)
    return {}


def _save_technique_output(
    data: Any,
    path: Path,
    compression: str,
) -> None:
    """Save technique output to disk as JSON or PNG based on file extension."""
    ext = path.suffix.lower()

    if ext == ".json":
        save_json(data, str(path), indent=2)

    elif ext == ".png":
        # If data is a matplotlib figure, save it; otherwise assume it's a dict
        # with 'figure' key.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if isinstance(data, plt.Figure):
            data.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close(data)
        elif isinstance(data, dict) and "figure" in data:
            data["figure"].savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close(data["figure"])
        else:
            log.warning("[_save_technique_output] cannot save technique output as PNG – data type=%s", type(data).__name__)

    elif ext == ".parquet":
        if isinstance(data, pd.DataFrame):
            save_parquet(data, str(path), compression=compression)
        else:
            log.warning("[_save_technique_output] expected DataFrame for parquet output, got %s", type(data).__name__)

    else:
        log.warning("[_save_technique_output] unsupported extension '%s' – saving as JSON", ext)
        save_json(data, str(path), indent=2)