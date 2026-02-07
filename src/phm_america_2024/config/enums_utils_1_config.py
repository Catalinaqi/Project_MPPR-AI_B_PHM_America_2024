# src/phm_america_2024/config/enums_utils_1_config.py
from __future__ import annotations

from enum import Enum, auto

from phm_america_2024.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Central place for "configuration enums" and normalization helpers.
# - YAML is text -> we must convert strings into controlled enums.
# - Keeps parsing/normalization logic consistent across the codebase.
#
# Program flow:
# - load_loader_config.load_and_resolve() returns raw dicts
# - schema_dto_config.ProjectConfig.from_dict() converts dicts -> typed DTOs
# - normalize_* helpers are called during DTO building and validation
#
# Design patterns
# - GoF: none
# - Enterprise/Architectural:
#   - "Typed configuration" boundary (prevents invalid config values)
#   - Defensive parsing (fail-fast)
# =============================================================================


class ProblemType(str, Enum):
    """Top-level ML problem family used to select pipeline behavior."""
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIMESERIES = "timeseries"

    def __str__(self) -> str:
        #log.debug("[Enum] ProblemType resolved: %s", self.name)
        return self.value


class CsvSourceType(str, Enum):
    """Data source type (your scenario is CSV-only)."""
    CSV = "csv"

    def __str__(self) -> str:
        #log.debug("[Enum] CsvSourceType resolved: %s", self.name)
        return self.value


class ReadMode(str, Enum):
    """
    How we read CSVs, especially large ones.
    - full: load entire CSV into memory
    - sample: read a sample (rows or fraction)
    - chunked: process in chunks (stream-like)
    """
    FULL = "full"
    SAMPLE = "sample"
    CHUNKED = "chunked"

    def __str__(self) -> str:
        #log.debug("[Enum] ReadMode resolved: %s", self.name)
        return self.value


class LogLevel(str, Enum):
    """Log verbosity for a run."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

    def __str__(self) -> str:
        #log.debug("[Enum] LogLevel resolved: %s", self.name)
        return self.value


class FeatureSelectionMode(Enum):
    """
    Strategy for selecting input features for modeling.

    AUTO:
      - Uses all columns except ID, target, and time.
      - Recommended default for most datasets.

    INCLUDE:
      - Uses only explicitly listed columns (feature_config.include).

    EXCLUDE:
      - Uses all columns except those in feature_config.exclude.

    Use INCLUDE/EXCLUDE when you need strict manual control.
    """
    AUTO = auto()
    INCLUDE = auto()
    EXCLUDE = auto()

    def __str__(self) -> str:
        #log.debug("[Enum] FeatureSelectionMode resolved: %s", self.name)
        return self.name


def normalize_problem_type(value: str | ProblemType) -> ProblemType:
    """
    Normalize an incoming string/enum into ProblemType.

    Why:
    - Notebook may pass strings
    - YAML contains strings
    - internal code should use a stable enum
    """
    if isinstance(value, ProblemType):
        log.debug("normalize_problem_type: already enum=%s", value.value)
        return value

    v = (value or "").strip().lower()
    try:
        out = ProblemType(v)
        log.debug("normalize_problem_type: parsed value=%s -> %s", value, out.value)
        return out
    except Exception as e:
        log.error("normalize_problem_type: invalid value=%r error=%s", value, e)
        raise

def normalize_read_mode(value: str | ReadMode) -> ReadMode:
    """
    Normalize an incoming string/enum into ReadMode.

    Why:
    - Notebook may pass strings
    - YAML contains strings
    - internal code should use a stable enum
    """
    if isinstance(value, ReadMode):
        log.debug("normalize_read_mode: already enum=%s", value.value)
        return value

    v = (value or "").strip().lower()
    try:
        out = ReadMode(v)
        log.debug("normalize_read_mode: parsed value=%s -> %s", value, out.value)
        return out
    except Exception as e:
        log.error("normalize_read_mode: invalid value=%r error=%s", value, e)
        raise

def normalize_log_level(value: str | LogLevel) -> LogLevel:
    """
    Normalize an incoming string/enum into LogLevel.

    Why:
    - Notebook may pass strings
    - YAML contains strings
    - internal code should use a stable enum
    """
    if isinstance(value, LogLevel):
        log.debug("normalize_log_level: already enum=%s", value.value)
        return value

    v = (value or "").strip().upper()
    try:
        out = LogLevel(v)
        log.debug("normalize_log_level: parsed value=%s -> %s", value, out.value)
        return out
    except Exception as e:
        log.error("normalize_log_level: invalid value=%r error=%s", value, e)
        raise