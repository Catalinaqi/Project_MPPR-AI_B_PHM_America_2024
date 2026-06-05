# src/phm_america_2024/data/csv_loader_data.py
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import pandas as pd

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.path_service_common import resolve_path
from phm_america_2024.domain.enum_registry_domain import ReadMode
from phm_america_2024.data.read_strategy_repository_data import ReadStrategyContract

log = get_logger(__name__)


def _filter_read_csv_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    # Step 1: Build whitelist from pandas signature
    valid_params = set(inspect.signature(pd.read_csv).parameters.keys())

    # Step 2: Filter and log any skipped keys
    filtered = {k: v for k, v in params.items() if k in valid_params}
    skipped = set(params.keys()) - set(filtered.keys())
    if skipped:
        log.debug("[_filter_read_csv_kwargs] skipping non-pandas keys: %s", skipped)
    return filtered


def _load_csv_head(
        path: str,
        *,
        csv_params: Optional[Dict[str, Any]],
        sample_rows: int,
) -> pd.DataFrame:
    # Step 1: Resolve path to absolute location
    resolved = resolve_path(path)
    log.debug("[_load_csv_head] resolved path=%s sample_rows=%d", resolved, sample_rows)

    # Step 2: Merge csv_params with defaults
    params: Dict[str, Any] = dict(csv_params or {})

    # Step 3: Filter to valid pd.read_csv kwargs
    params_filtered = _filter_read_csv_kwargs(params)

    # Step 4: Load first N rows
    try:
        log.info("[_load_csv_head] loading first %d rows from path=%s", sample_rows, resolved)
        df: pd.DataFrame = pd.read_csv(resolved, nrows=sample_rows, **params_filtered)
    except Exception:
        log.exception("[_load_csv_head] unexpected error path=%s", resolved)
        raise

    # Step 5: Log result dimensions
    log.info("[_load_csv_head] loaded rows=%d cols=%d path=%s", len(df), df.shape[1], resolved)
    return df


def _load_csv_random_sample(
        path: str,
        *,
        csv_params: Optional[Dict[str, Any]],
        sample_rows: int,
        chunksize: int,
        random_state: int,
) -> pd.DataFrame:
    # Step 1: Resolve path to absolute location
    resolved = resolve_path(path)

    # Step 2: Merge csv_params with defaults
    params: Dict[str, Any] = dict(csv_params or {})

    # Step 3: Validate file existence
    if not resolved.exists():
        log.error("[_load_csv_random_sample] file not found path=%s", resolved)
        raise FileNotFoundError(f"CSV file not found: {resolved}")

    # Step 4: Filter to valid pd.read_csv kwargs
    params_filtered = _filter_read_csv_kwargs(params)

    # Step 5: Iterate chunks and collect a proportional sub-sample
    samples: list[pd.DataFrame] = []
    rows_seen: int = 0
    log.info("[_load_csv_random_sample] starting chunked sampling path=%s", resolved)

    try:
        for chunk in pd.read_csv(resolved, chunksize=chunksize, **params_filtered):
            rows_seen += len(chunk)
            n: int = min(len(chunk), sample_rows)
            samples.append(chunk.sample(n=n, random_state=random_state))
    except Exception:
        log.exception("[_load_csv_random_sample] error during chunked read path=%s", resolved)
        raise

    # Step 6: Concatenate all chunk sub-samples
    df: pd.DataFrame = pd.concat(samples, ignore_index=True)

    # Step 7: Down-sample to exactly sample_rows if over-sampled
    if len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=random_state).reset_index(drop=True)

    # Step 8: Log final result dimensions
    log.info("[_load_csv_random_sample] done rows=%d cols=%d path=%s", len(df), df.shape[1], resolved)
    return df


def _load_csv_stratified_sample(
        path: str,
        *,
        csv_params: Optional[Dict[str, Any]],
        strategy: ReadStrategyContract,
) -> pd.DataFrame:
    # Step 1: Resolve path and validate existence
    resolved = resolve_path(path)
    params: Dict[str, Any] = dict(csv_params or {})

    if not resolved.exists():
        log.error("[_load_csv_stratified_sample] file not found path=%s", resolved)
        raise FileNotFoundError(f"CSV file not found: {resolved}")

    # Step 2: Filter to valid pd.read_csv kwargs
    params_filtered = _filter_read_csv_kwargs(params)

    # Step 3: Pre-load label registry cleanly using the label_path from the contract (ZERO HARDCODING)
    y_labels: Optional[pd.DataFrame] = None
    if strategy.label_path and strategy.stratify_column:
        y_path = resolve_path(strategy.label_path)
        if y_path.exists():
            log.info("[_load_csv_stratified_sample] Pre-loading labels from %s", y_path)
            y_labels = pd.read_csv(y_path, usecols=["id", strategy.stratify_column])
            y_labels.set_index("id", inplace=True)

    def _inject_strata(chunk_df: pd.DataFrame) -> pd.DataFrame:
        if strategy.stratify_column in chunk_df.columns:
            return chunk_df
        if y_labels is not None and "id" in chunk_df.columns:
            return chunk_df.join(y_labels, on="id", how="inner")
        return chunk_df

    # Step 4: Open chunked reader and read first chunk for distribution estimation
    reader = pd.read_csv(resolved, chunksize=strategy.chunksize, **params_filtered)
    try:
        first_chunk: pd.DataFrame = next(reader)
        first_chunk = _inject_strata(first_chunk)
    except StopIteration:
        log.warning("[_load_csv_stratified_sample] CSV file is empty path=%s", resolved)
        return pd.DataFrame()

    if strategy.stratify_column not in first_chunk.columns:
        log.error("[_load_csv_stratified_sample] stratify_column='%s' missing.", strategy.stratify_column)
        raise KeyError(f"stratify_column='{strategy.stratify_column}' not found.")

    # Step 5: Compute per-stratum target counts
    stratum_counts: pd.Series = first_chunk[strategy.stratify_column].value_counts()
    total_in_scan: int = len(first_chunk)

    per_stratum_target: dict[Any, int] = {}
    for stratum_label, count_in_scan in stratum_counts.items():
        proportion: float = count_in_scan / total_in_scan
        per_stratum_target[stratum_label] = max(1, round(strategy.sample_rows * proportion))

    # Step 6: Initialise collection state and sample first chunk
    collected: dict[Any, list[pd.DataFrame]] = {s: [] for s in stratum_counts.index}
    total_collected: int = 0
    target_total: int = int(sum(per_stratum_target.values()))

    for stratum_label, group in first_chunk.groupby(strategy.stratify_column):
        n_take: int = min(len(group), per_stratum_target.get(stratum_label, 1))
        collected[stratum_label].append(group.sample(n=n_take, random_state=strategy.random_state))
        total_collected += n_take

    # Step 7: Process remaining chunks with early stopping
    chunk_index: int = 0
    for chunk in reader:
        chunk_index += 1
        chunk = _inject_strata(chunk)

        for stratum_label, group in chunk.groupby(strategy.stratify_column):
            already: int = sum(len(df_sub) for df_sub in collected.get(stratum_label, []))
            still_needed: int = per_stratum_target.get(stratum_label, 1) - already
            if still_needed <= 0:
                continue
            n_take = min(len(group), still_needed)
            collected[stratum_label].append(group.sample(n=n_take, random_state=strategy.random_state + chunk_index))
            total_collected += n_take

        if total_collected >= target_total:
            break

    # Step 8: Concatenate all collected sub-samples
    collected_dfs: list[pd.DataFrame] = []
    for lst in collected.values():
        collected_dfs.extend(lst)

    if not collected_dfs:
        return pd.DataFrame()
    df: pd.DataFrame = pd.concat(collected_dfs, ignore_index=True)

    # Step 9: Final stratified down-sample to exact target rows
    if len(df) > strategy.sample_rows:
        df = (
            df.groupby(strategy.stratify_column, group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), max(1, round(strategy.sample_rows * len(x) / len(df)))), random_state=strategy.random_state))
            .reset_index(drop=True)
        )

    # Step 10: Drop injected column if it was not originally in the file features
    if y_labels is not None and strategy.stratify_column in df.columns:
        df = df.drop(columns=[strategy.stratify_column], errors="ignore")

    log.info("[_load_csv_stratified_sample] done rows=%d path=%s", len(df), resolved)
    return df


def _load_csv_chunks(
        path: str,
        *,
        csv_params: Optional[Dict[str, Any]],
        chunksize: int,
) -> Generator[pd.DataFrame, None, None]:
    # Step 1: Resolve path to absolute location
    resolved = resolve_path(path)

    # Step 2: Merge csv_params with defaults
    params: Dict[str, Any] = dict(csv_params or {})

    # Step 3: Validate file existence
    if not resolved.exists():
        log.error("[_load_csv_chunks] file not found path=%s", resolved)
        raise FileNotFoundError(f"CSV file not found: {resolved}")

    # Step 4: Initialise pandas TextFileReader
    try:
        params_filtered = _filter_read_csv_kwargs(params)
        reader = pd.read_csv(resolved, chunksize=chunksize, **params_filtered)
    except Exception:
        log.exception("[_load_csv_chunks] failed to initialise reader path=%s", resolved)
        raise

    # Step 5: Yield chunks lazily
    chunk_index: int = 0
    for chunk in reader:
        yield chunk
        chunk_index += 1


def _load_csv_full(
        path: str,
        *,
        csv_params: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    # Step 1: Resolve path to absolute location
    resolved = resolve_path(path)
    params: Dict[str, Any] = dict(csv_params or {})

    # Step 2: Validate file existence
    if not resolved.exists():
        log.error("[_load_csv_full] file not found path=%s", resolved)
        raise FileNotFoundError(f"CSV file not found: {resolved}")

    # Step 3: Load entire file into memory
    try:
        params_filtered = _filter_read_csv_kwargs(params)
        df: pd.DataFrame = pd.read_csv(resolved, **params_filtered)
    except Exception:
        log.exception("[_load_csv_full] read_csv failed path=%s", resolved)
        raise

    log.info("[_load_csv_full] loaded rows=%d cols=%d path=%s", len(df), df.shape[1], resolved)
    return df


def load_by_strategy(
        path: str | Path,
        *,
        csv_params: Optional[Dict[str, Any]] = None,
        strategy: ReadStrategyContract,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[Generator[pd.DataFrame, None, None]],
    ReadStrategyContract,
]:
    log.info("[load_by_strategy] path=%s mode=%s", path, strategy.mode.value)

    # Step 1: Dispatch to stratified sample primitive
    if strategy.mode == ReadMode.SAMPLE and strategy.sample_method == "stratified":
        df = _load_csv_stratified_sample(str(path), csv_params=csv_params, strategy=strategy)
        return df, None, strategy

    # Step 2: Dispatch to random sample primitive
    if strategy.mode == ReadMode.SAMPLE and strategy.sample_method == "random":
        df = _load_csv_random_sample(str(path), csv_params=csv_params, sample_rows=strategy.sample_rows, chunksize=strategy.chunksize, random_state=strategy.random_state)
        return df, None, strategy

    # Step 3: Dispatch to head sample primitive
    if strategy.mode == ReadMode.SAMPLE and strategy.sample_method in {"head", "tail"}:
        df = _load_csv_head(str(path), csv_params=csv_params, sample_rows=strategy.sample_rows)
        return df, None, strategy

    # Step 4: Dispatch to chunked generator primitive
    if strategy.mode == ReadMode.CHUNKED:
        generator = _load_csv_chunks(str(path), csv_params=csv_params, chunksize=strategy.chunksize)
        return None, generator, strategy

    # Step 5: Dispatch to full load primitive
    if strategy.mode == ReadMode.FULL:
        df = _load_csv_full(str(path), csv_params=csv_params)
        return df, None, strategy

    log.error("[load_by_strategy] unrecognised mode=%s", strategy.mode)
    raise ValueError(f"Unrecognised ReadMode: {strategy.mode}")