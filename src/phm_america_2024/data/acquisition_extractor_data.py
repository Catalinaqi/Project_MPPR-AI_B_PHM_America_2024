# src/phm_america_2024/data/acquisition_extractor_data.py
from typing import Any
from omegaconf import OmegaConf
from phm_america_2024.configuration.yml_repository_config import YmlRepository
from phm_america_2024.data.read_strategy_repository_data import DataSourceConfig
from phm_america_2024.data.utils.load_loader_data import (
    load_train_merged,
    load_test,
    load_validation,
)
from phm_america_2024.domain.enum_registry_domain import StepOutputArtifact
from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def load_and_merge(
    df: Any,
    tech_cfg: dict,
    ctx: Any,
    base_dir: Any,
) -> tuple[Any, dict[str, Any]]:
    """Técnica que ingesta los CSVs crudos y los convierte en DataFrames."""
    log.info("[load_and_merge_technique] Iniciando ingesta de datos...")

    # 1. Extraemos la config de la FASE (AQUÍ VIVE EL READ STRATEGY)
    step_cfg = ctx.config.phases.phase2_data_understanding

    # Step 2: CALL get_step_config() — extract pipeline configuration
    raw_step_cfg = (
        OmegaConf.to_container(step_cfg, resolve=True)
        if OmegaConf.is_config(step_cfg)
        else step_cfg
    )

    # Step 3: CALL get_dataset_config() — resolve raw dataset settings
    dataset_key = getattr(ctx, "dataset_key", "phm2024")
    dataset_real_cfg = YmlRepository.get_dataset_by_key(dataset_key)
    raw_dataset_cfg = (
        OmegaConf.to_container(dataset_real_cfg, resolve=True)
        if OmegaConf.is_config(dataset_real_cfg)
        else dataset_real_cfg
    )

    # Step 4: CALL parse_yaml_paths() — extract paths and CSV parameters
    yaml_paths = raw_dataset_cfg.get("paths", {})
    yaml_csv_params = raw_dataset_cfg.get(
        "csv_params", {"sep": ",", "encoding": "utf-8", "decimal": "."}
    )

    # Step 5: CALL map_dataset_paths() — normalize YAML keys for DataSourceConfig
    train_node = yaml_paths.get("train", {})
    test_node = yaml_paths.get("test", {})
    val_node = yaml_paths.get("validation", {})

    normalized_paths = {
        "train": {
            "x_train": train_node.get("X_TRAIN_FEATURE") or train_node.get("x_train"),
            "y_train": train_node.get("Y_TRAIN_TARGET") or train_node.get("y_train"),
            "join_key": train_node.get("join_key", "id"),
        },
        "test": {"x_test": test_node.get("X_TEST_FEATURE") or test_node.get("x_test")},
        "validation": {
            "x_validation": val_node.get("X_VALIDATION_FEATURE")
            or val_node.get("x_validation")
        },
    }

    # Step 6: CALL build_payload() — create nested DataSourceConfig dictionary
    nested_payload = {
        "paths": normalized_paths,
        "csv_params": yaml_csv_params,
        "read_strategy": raw_step_cfg.get("read_strategy", {}),
    }
    log.debug("[_execute_data_acquisition] Formatted target payload ready for factory.")

    # Step 7: CALL instantiate_source_config() — initialize valid data source configuration
    source_config = DataSourceConfig.from_dict(nested_payload)

    # Step 8: CALL load_dataframes() — stream datasets into memory
    log.info("[_execute_data_acquisition] Loading dataframes from disk into memory...")
    df_train, _ = load_train_merged(source_config)
    df_test, _ = load_test(source_config)
    df_val, _ = load_validation(source_config)

    merged_metadata = {
        "join_key": normalized_paths["train"].get("join_key", "id"),
        "rows": len(df_train),
        "status": "merged_successfully",
        "infer_datetime": False,
    }

    # Step 9: CALL prepare_registry_payload() — map artifacts for persistence
    context_payload = {
        StepOutputArtifact.sample_x_y_train_parquet.value: df_train,
        StepOutputArtifact.sample_x_test_parquet.value: df_test,
        StepOutputArtifact.sample_x_validation_parquet.value: df_val,
        # "load_and_merge_json": merged_metadata
    }

    # Step 10: CALL audit_registry_payload() — log dispatch metadata
    log.info(
        "[_execute_data_acquisition] Dispatching to registry. Keys: %s",
        list(context_payload.keys()),
    )
    log.debug(
        "[_execute_data_acquisition] Target key existence: %s",
        StepOutputArtifact.load_and_merge_json.value in context_payload,
    )
    for key, df in context_payload.items():
        log.debug(
            "[_execute_data_acquisition] Payload '%s' shape: %s",
            key,
            getattr(df, "shape", "N/A"),
        )
    log.info("DEBUG: Llaves en el payload: %s", list(context_payload.keys()))
    for k in context_payload.keys():
        log.info("DEBUG: Intentando procesar llave: %s", k)

    # Retornamos df_train como el DataFrame principal, y el diccionario de artefactos
    return df_train, context_payload
