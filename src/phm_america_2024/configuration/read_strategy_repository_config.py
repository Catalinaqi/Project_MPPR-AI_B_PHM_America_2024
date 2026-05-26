# src/phm_america_2024/configuration/read_strategy_repository_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.configuration.enum_registry_config import ReadMode, normalize_read_mode

log = get_logger(__name__)


@dataclass(frozen=True)
class ReadStrategyContract:
    """Reading strategy parameters derived from configuration."""

    mode: ReadMode
    sample_rows: int
    chunksize: int
    random_state: int
    sample_method: str
    stratify_column: str
    join_key: Optional[str]
    label_path: Optional[str]
    input_source: Optional[str]
    input_source_full: Optional[str]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReadStrategyContract":
        """Build contract from a raw configuration dictionary strictly."""
        # Validar que el diccionario de estrategia no sea nulo o vacío antes de procesar
        if not raw:
            log.error("[ReadStrategyContract.from_dict] Configuration block 'read_strategy' is empty or missing.")
            raise ValueError("The 'read_strategy' configuration block must be populated and cannot be empty.")

        # Step 1: Log raw keys for traceability before any parsing
        log.debug("[ReadStrategyContract.from_dict] raw keys=%s", list(raw.keys()))

        # Step 2: Extract and parse fields strictly checking for their existence
        try:
            mode: ReadMode = normalize_read_mode(raw["mode"])
            sample_rows: int = int(raw["sample_rows"])
            chunksize: int = int(raw["chunksize"])
            random_state: int = int(raw["random_state"])
            sample_method: str = str(raw["sample_method"])
            stratify_column: str = str(raw["stratify_column"])
        except KeyError as err:
            log.error("[ReadStrategyContract.from_dict] Missing mandatory configuration key: %s", err)
            raise ValueError(f"ReadStrategy configuration missing required parameter: {err}") from err
        except (ValueError, TypeError) as err:
            log.error("[ReadStrategyContract.from_dict] Data type casting failed: %s", err)
            raise ValueError(f"Invalid data type or format in configuration values: {err}") from err

        # Step 3: Extract genuinely optional pipeline keys (No defaults assigned)
        join_key: Optional[str] = raw.get("join_key")
        label_path: Optional[str] = raw.get("label_path")
        input_source: Optional[str] = raw.get("input_source")
        input_source_full: Optional[str] = raw.get("input_source_full")

        # Step 4: Log fully resolved values to aid debugging
        log.info(
            "[ReadStrategyContract.from_dict] resolved — "
            "mode=%s sample_rows=%d chunksize=%d join_key=%s stratify=%s",
            mode.value, sample_rows, chunksize, join_key, stratify_column
        )

        # Step 5: Build and return the immutable contract
        return cls(
            mode=mode,
            sample_rows=sample_rows,
            chunksize=chunksize,
            random_state=random_state,
            sample_method=sample_method,
            stratify_column=stratify_column,
            join_key=join_key,
            label_path=label_path,
            input_source=input_source,
            input_source_full=input_source_full,
        )


@dataclass(frozen=True)
class DataSourceConfig:
    """Dataset physical locations and parsing parameters."""

    x_train_path: str
    y_train_path: str
    x_test_path: str
    x_validation_path: str
    join_key: str
    csv_params: Dict[str, Any]
    strategy: ReadStrategyContract

    @classmethod
    def from_dict(cls, dataset_input: dict[str, Any]) -> "DataSourceConfig":
        """Build data source config explicitly mapping PHM entities."""
        # Validar la existencia de la raíz de la configuración de datos
        if not dataset_input:
            log.error("[DataSourceConfig.from_dict] The dataset_input dictionary is empty.")
            raise ValueError("The 'dataset_input' configuration dictionary cannot be empty.")

        # Step 1: Log raw keys for traceability
        log.debug("[DataSourceConfig.from_dict] raw keys=%s", list(dataset_input.keys()))

        # Step 2: Extract mandatory paths for PHM dataset
        try:
            paths = dataset_input["paths"]
            x_train_path: str = str(paths["train"]["x_train"])
            y_train_path: str = str(paths["train"]["y_train"])
            x_test_path: str = str(paths["test"]["x_test"])
            x_validation_path: str = str(paths["validation"]["x_validation"])
            join_key: str = str(paths["train"]["join_key"])
        except KeyError as err:
            log.error("[DataSourceConfig.from_dict] Missing mandatory path config: %s", err)
            raise ValueError(f"YAML configuration missing key: {err}") from err

        # Step 3: Extract CSV parsing parameters safely from structure
        try:
            csv_params: Dict[str, Any] = dataset_input["csv_params"]
        except KeyError as err:
            log.error("[DataSourceConfig.from_dict] Missing mandatory 'csv_params' block: %s", err)
            raise ValueError(f"Configuration missing key: {err}") from err

        # Step 4: Extract the sub-dict and enforce strict validation downstream
        try:
            read_strategy_dict: dict[str, Any] = dataset_input["read_strategy"]
        except KeyError as err:
            log.error("[DataSourceConfig.from_dict] Missing mandatory 'read_strategy' block: %s", err)
            raise ValueError(f"Configuration missing key: {err}") from err

        # Step 5: CALL ReadStrategyContract.from_dict() — build nested reading strategy
        strategy: ReadStrategyContract = ReadStrategyContract.from_dict(read_strategy_dict)

        # Step 6: Log final resolved configuration
        log.info(
            "[DataSourceConfig.from_dict] resolved — x_train=%s y_train=%s join_key=%s",
            x_train_path, y_train_path, join_key
        )

        # Step 7: Build and return the immutable configuration
        return cls(
            x_train_path=x_train_path,
            y_train_path=y_train_path,
            x_test_path=x_test_path,
            x_validation_path=x_validation_path,
            join_key=join_key,
            csv_params=csv_params,
            strategy=strategy,
        )