# src/phm_america_2024/domain/enum_registry_domain.py
from __future__ import annotations

"""
=============================================================================
Why this module exists
-----------------------------------------------------------------------------
Central registry of "configuration enums" and their normalization helpers.
YAML files contain raw strings — this module converts them into controlled,
typed enum values so the rest of the pipeline never operates on bare strings.

Covered pipeline types: clustering · classification · regression · timeseries
Covered options:        Option A (drift-aware) · Option B (train-only)

Enum inventory:
-----------------------------------------------------------------------------
  ProblemType          — ML task family (clustering/classification/…)
  CsvSourceType        — data source format (csv only for this project)
  ReadMode             — CSV loading strategy (full/sample/chunked)
  LogLevel             — logging verbosity (DEBUG → … → CRITICAL)
  FeatureSelectionMode — feature selection strategy (auto/include/exclude)

Program flow:
-----------------------------------------------------------------------------
- YAML configuration (string values)
    -> load_loader_config.load_and_resolve()  -> raw dict
    -> validate_validator_config.validate_config_dict() (uses normalize_*)
    -> schema_dto_config.ProjectConfig.from_dict()      (uses normalize_*)
    -> typed DTOs with enum fields

Design principles:
-----------------------------------------------------------------------------
- All enums inherit (str, Enum) so instances compare equal to their string
  equivalents (e.g. ProblemType.CLUSTERING == "clustering" → True).
  This lets callers pass either a ProblemType or the plain string "clustering"
  without adapter code, and allows direct YAML-value comparison.

- __str__ returns self.value (the YAML-compatible string) so enum instances
  format correctly in log messages, f-strings, and Path concatenations.
  No log.debug inside __str__ — that causes log noise and recursive logging
  on every format call (every log.info("task=%s", task) would trigger a
  log.debug inside task.__str__()).

- normalize_*() functions are the single entry point for string → enum
  conversion. They are called by both the validator and the DTO factory so
  the conversion logic is never duplicated.

Design patterns
-----------------------------------------------------------------------------
- GoF -> Gang of Four: none.
- Enterprise/Architectural:
  - Typed Configuration Boundary: invalid strings are rejected at the
      boundary (normalize_*) so internal code only ever sees valid enums.
  - Defensive Parsing (fail-fast): normalize_* raises ValueError immediately
      on unrecognised values — misconfiguration surfaces at startup.
=============================================================================
"""


# =============================================================================
# SECTION 1 – Standard-library imports
# =============================================================================
from enum import Enum

# =============================================================================
# SECTION 2 – Third-party imports
# =============================================================================
# (none required – all functionality uses stdlib logging + optional extensions)

# =============================================================================
# SECTION 3 – Internal imports
# =============================================================================
# from phm_america_2024.common.logging_adapter_common import get_logger

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Level logger
# ──────────────────────────────────────────────────────────────────────────────
# log = get_logger(__name__)

# =============================================================================
# SECTION 1 — ENUMS
# =============================================================================

# =============================================================================
# SECTION 5 — Constants
# ============================================================================

# =============================================================================
# SECTION 6 — Type variable
# =============================================================================
# (none required – )

# =============================================================================
# SECTION 7 — Class
# =============================================================================


class LogLevel(str, Enum):
    """Logging verbosity level for a pipeline run.

    Inherits from ``str`` so that instances compare equal to their string
    values, enabling direct YAML comparison and use with
    ``logging.getLevelName()``.  Maps 1-to-1 with Python's five standard
    logging levels.

    The YAML key ``runtime.log_level`` must be one of these values.

    Attributes
    ----------
    DEBUG : str
        Most verbose — emits all operations.  Use for troubleshooting.
    INFO : str
        Major milestones only — stage start/end, key metrics.
    WARNING : str
        Non-blocking issues that may affect results.
    ERROR : str
        Blocking failures that prevent stage completion.
    CRITICAL : str
        Unrecoverable failures that abort the entire pipeline run.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        """Return the YAML-compatible string value (e.g. ``"DEBUG"``).

        Returns
        -------
        str
            The enum member's string value.
        """
        return self.value


class ProblemType(str, Enum):
    """
    Top-level ML problem family — drives pipeline branch selection.

    Inherits from ``str`` so ``ProblemType.CLUSTERING == "clustering"`` is
    ``True``, enabling direct comparison with YAML string values.

    Attributes
    ----------
    CLUSTERING : str
        Unsupervised grouping — no target column required.
    CLASSIFICATION : str
        Supervised discrete-label prediction — requires ``target_col``.
    REGRESSION : str
        Supervised continuous-value prediction — requires ``target_col``.
    TIMESERIES : str
        Temporal sequence modelling — requires ``time_col``.
    """

    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIMESERIES = "timeseries"

    def __str__(self) -> str:
        """Return the YAML-compatible string value."""
        return self.value


class ReadMode(str, Enum):
    """
    CSV loading strategy for large files (train ~2 GB, test ~1 GB).

    Inherits from ``str`` for direct YAML string comparison.
    The YAML ``read_strategy.mode`` key maps to one of these values.

    Attributes
    ----------
    FULL : str
        Load the entire CSV into memory.  Only safe for small files or
        environments with sufficient RAM.
    SAMPLE : str
        Load a random / head / tail subset of rows.  Used in Stages 2–4
        to keep memory bounded while preserving statistical properties.
    CHUNKED : str
        Process the CSV in fixed-size iteration chunks.  Used in Stage 5
        for full test-set evaluation without loading 1 GB at once.
    """

    FULL = "full"
    SAMPLE = "sample"
    CHUNKED = "chunked"

    def __str__(self) -> str:
        """Return the YAML-compatible string value."""
        return self.value


class Phase(str, Enum):
    """
    CRISP-DM stage directory names under ``out/runs/{task}/{dataset}/{run_id}/``.

    Inherits from ``str`` so ``PhaseDir.PHASE2 == "phase2_data_understanding"`` is
    ``True``, enabling direct use in ``Path`` expressions without calling
    ``.value`` explicitly.

    Single source of truth for stage directory names — the string literal
    ``"phase2_data_understanding"`` never appears outside this definition.

    Attributes
    ----------
    MODELS : str
        Serialised model artefacts (joblib, pickle) — not a CRISP-DM stage.
    STAGE2 : str
        Stage 2 — Data Understanding outputs.
    STAGE3 : str
        Stage 3 — Data Preparation outputs.
    STAGE4 : str
        Stage 4 — Modelling outputs.
    STAGE5 : str
        Stage 5 — Evaluation outputs.
    """

    PHASE2 = "phase2_data_understanding"
    PHASE3 = "phase3_data_preparation"
    PHASE4 = "phase4_data_modeling"
    PHASE5 = "phase5_evaluation_and_interpretation"
    PHASE6 = "phase6_deployment"

    def __str__(self) -> str:
        return self.value


class StepOutputArtifact(str, Enum):
    """Keys for output_artifacts in Phase 2 steps YAML configuration."""

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2 – Data Understanding
    # ──────────────────────────────────────────────────────────────────────────
    # Step 2.1
    load_and_merge_json = "load_and_merge_x_y"

    sample_x_y_train_parquet = "sample_x_y_train_parquet"  # sample_x_y_train_parquet
    sample_x_test_parquet = "sample_x_test_parquet"
    sample_x_validation_parquet = "sample_x_validation_parquet"

    # Step 2.2
    column_metadata_json = "column_metadata"
    sensor_stats_json = "basic_stats"
    null_count_json = "null_count_per_column"
    target_distribution_json = "distribution_analysis"

    # Step 2.3
    zero_or_negative_check_json = "zero_or_negative_check"
    collinearity_json = "collinearity_analysis"

    # Step 2.4
    column_catalog_json = "column_catalog"
    ks_report_json = "ks_test_per_feature"
    gmm_curve_png = "gmm_exploration"
    drift_summary_json = "feature_drift_summary"

    flight_regimes_png = "flight_regime_binning"

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3 – Data Preparation (from regression_pipeline_config.yml)
    # ──────────────────────────────────────────────────────────────────────────
    # Step 3.1 – Data Selection
    selected_regression_train_parquet = "selected_regression_train_parquet"
    # Step 3.2 – Data Cleaning
    cleaned_regression_train_parquet = "cleaned_regression_train_parquet"
    # Step 3.3 – Data Transformation
    engineered_regression_train_parquet = "engineered_regression_train_parquet"
    transformed_regression_train_parquet = "transformed_regression_train_parquet"
    fitted_scaler_regression_artifact = "fitted_scaler_regression_artifact"
    # Step 3.5 – Data Formatting
    engineered_train_split = "engineered_train_split"
    engineered_val_split = "engineered_val_split"
    engineered_test_split = "engineered_test_split"
    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3 – Data Preparation (from classification_pipeline_config.yml)
    # ──────────────────────────────────────────────────────────────────────────
    # Step 3.1 – Data Selection
    selected_classification_train_parquet = "selected_classification_train_parquet"
    # Step 3.2 – Data Cleaning
    cleaned_classification_train_parquet = "cleaned_classification_train_parquet"
    engineered_classification_train_parquet = "engineered_classification_train_parquet"
    # Step 3.3 – Data Transformation
    transformed_classification_train_parquet = (
        "transformed_classification_train_parquet"
    )
    fitted_scaler_bin = "fitted_scaler_bin"

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 4 – Modeling (from regression_pipeline_config.yml)
    # ──────────────────────────────────────────────────────────────────────────
    # Step 4.2 – Model Training
    trained_ngboost_model = "trained_model"
    # Step 4.4 – Model Evaluation
    best_regression_model_metadata = "best_regression_model_metadata"

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 4 – Modeling (from classification_pipeline_config.yml)
    # ──────────────────────────────────────────────────────────────────────────
    # Step 4.2 – Model Training
    trained_model = "trained_model"
    fitted_isotonic_calibrator = "fitted_isotonic_calibrator"  # <-- NUEVO
    # Step 4.4 – Model Evaluation
    best_classification_model_metadata = "best_classification_metadata"

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 5 – Evaluation & Interpretation (from regression_pipeline_config.yml)──
    # ──────────────────────────────────────────────────────────────────────────
    # step_5_1
    fi_importance_plot = "fi_importance_plot"
    fi_permutation_plot = "fi_permutation_plot"
    # step_5_2
    evaluation_summary_json = "evaluation_summary_json"
    eval_calibration_plot = "eval_calibration_plot"
    eval_degradation_plot = "eval_degradation_plot"
    # step_5_4
    deployment_sign_off = "deployment_sign_off"


class StepsPhase(str, Enum):  # noqa: D101
    STEP_2_1 = "step_2_1_data_acquisition"
    STEP_2_2 = "step_2_2_data_description"
    STEP_2_3 = "step_2_3_data_quality_assessment"
    STEP_2_4 = "step_2_4_data_exploration"
    STEP_3_1 = "step_3_1_data_selection"
    STEP_3_2 = "step_3_2_data_cleaning"
    STEP_3_3 = "step_3_3_data_transformation"
    STEP_3_4 = "step_3_4_data_integration"
    STEP_3_5 = "step_3_5_data_formatting"
    STEP_4_1 = "step_4_1_algorithm_selection"
    STEP_4_2 = "step_4_2_model_training"
    STEP_4_4 = "step_4_4_model_evaluation"
    STEP_5_1 = "step_5_1_interpretation"
    STEP_5_2 = "step_5_2_business_evaluation"
    STEP_5_3 = "step_5_3_process_audit"
    STEP_5_4 = "step_5_4_decision_making"
    STEP_6_1 = "step_6_1_academic_scoring"
    STEP_6_2 = "step_6_2_package_deliverables"

    def __str__(self) -> str:
        return self.value


# =============================================================================
# SECTION 8 — Private functions
# =============================================================================
# (none required – )


# =============================================================================
# SECTION 9 — Public functions
# ============================================================================
def normalize_problem_type(
    value: str | ProblemType,
) -> ProblemType:
    """
    Normalize a raw string or existing enum into a ``ProblemType`` member.

    Called by the validator and the DTO factory so conversion logic is
    never duplicated.  Accepts the already-enum case cheaply (no string
    operations).

    Parameters
    ----------
    value : str | ProblemType
        Raw YAML string (e.g. ``"clustering"``) or an existing
        ``ProblemType`` member.

    Returns
    -------
    ProblemType
        The corresponding enum member.

    Raises
    ------
    ValueError
        If *value* is not a recognised ``ProblemType`` string.
    """
    # Step 1: Pass-through if already a valid enum — avoids redundant work.
    if isinstance(value, ProblemType):
        # log.debug("[normalize_problem_type] already enum value=%s", value.value)
        return value

    # Step 2: Normalise to lowercase stripped string to tolerate YAML casing.
    normalised: str = (value or "").strip().lower()

    # Step 3: Parse into enum — raises ValueError on unrecognised values.
    try:
        result = ProblemType(normalised)
    except ValueError:
        valid = [m.value for m in ProblemType]
        # log.error("[normalize_problem_type] invalid value=%r valid=%s", value, valid)
        raise ValueError(f"Unknown ProblemType={value!r}. Valid values: {valid}")

    # Step 4: Log resolved value and return.
    # log.debug("[normalize_problem_type] resolved value=%r -> %s", value, result.value)
    return result


def normalize_read_mode(value: str | ReadMode) -> ReadMode:
    """
    Normalize a raw string or existing enum into a ``ReadMode`` member.

    Parameters
    ----------
    value : str | ReadMode
        Raw YAML string (e.g. ``"sample"``) or an existing ``ReadMode``
        member.

    Returns
    -------
    ReadMode
        The corresponding enum member.

    Raises
    ------
    ValueError
        If *value* is not a recognised ``ReadMode`` string.
    """
    # Step 1: Pass-through if already a valid enum — avoids redundant work.
    if isinstance(value, ReadMode):
        # log.debug("[normalize_read_mode] already enum value=%s", value.value)
        return value

    # Step 2: Normalise to lowercase stripped string to tolerate YAML casing.
    normalised: str = (value or "").strip().lower()

    # Step 3: Parse into enum — raises ValueError on unrecognised values.
    try:
        result = ReadMode(normalised)
    except ValueError:
        valid = [m.value for m in ReadMode]
        # log.error("[normalize_read_mode] invalid value=%r valid=%s", value, valid)
        raise ValueError(f"Unknown ReadMode={value!r}. Valid values: {valid}")

    # Step 4: Log resolved value and return.
    # log.debug("[normalize_read_mode] resolved value=%r -> %s", value, result.value)
    return result


def normalize_log_level(
    value: str | LogLevel,
) -> LogLevel:
    """Normalise a raw string or existing enum into a ``LogLevel`` member.

    Accepts the YAML string exactly as written (any case) and returns the
    canonical ``LogLevel`` enum.  Raises ``ValueError`` immediately if the
    value is unrecognised — no silent fallback.

    Parameters
    ----------
    value : str | LogLevel
        Raw YAML string (e.g. ``"debug"``, ``"INFO"``) or an existing
        ``LogLevel`` member.

    Returns
    -------
    LogLevel
        The corresponding enum member.

    Raises
    ------
    ValueError
        If *value* does not match any ``LogLevel`` member after normalisation.
        The error message lists all valid values.
    """
    # Step 1: Pass-through if already a valid enum — avoids redundant work.
    if isinstance(value, LogLevel):
        return value

    # Step 2: Strip whitespace and convert to uppercase — YAML authors may
    #         write lowercase or mixed-case (e.g. "debug", "Debug").
    normalised: str = (value or "").strip().upper()

    # Step 3: Parse into enum — raises ValueError on unrecognised values
    #         with a message that lists all accepted members.
    try:
        return LogLevel(normalised)
    except ValueError:
        valid = [m.value for m in LogLevel]
        raise ValueError(
            f"[normalize_log_level] Unknown LogLevel={value!r}. "
            f"Valid values: {valid}. "
            f"Check runtime.log_level in the pipeline YAML."
        )
