from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from crispdm.config.load_log_loader_config import load_logging_config
# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Logging is a cross-cutting concern (observability). We centralize logging
# configuration here so the rest of the codebase never deals with handlers,
# formats, or file creation logic.
#
# Program flow expectation:
# - Facade (preview_facade_api.py / run_facade_api.py) calls init_logging() once.
# - Every other module calls get_logger(__name__) and uses log.debug/info/...
#
# Design patterns
# - GoF: none (Python logging is a library facility).
# - Enterprise/Architectural:
#   - Cross-cutting concern / Observability
#   - "Singleton-ish initialization" (configure handlers once per run)
#   - Facade orchestrates logging init (not each module)
# =============================================================================

_CONFIGURED_FOR: Optional[Path] = None

  cfg = load_log_loader_config("logging_config.yml")

  if cfg.get("logger_namespace"):
     _LOGGER_NAMESPACE = cfg["logger_namespace"]
  if cfg.get("logs_dir"):
     _LOGS_DIR_NAME = cfg["logs_dir"]
  if cfg.get("default_level"):
     _DEFAULT_LOG_LEVEL = cfg["default_level"]

def _safe_name(name: str) -> str:
    """
    Convert a run/pipeline name into a filesystem-safe token.
    Keeps: letters, digits, '_' and '-'.

    :parameter
        name: Original run/pipeline name
    """
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9_\-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "run"


def build_log_file(output_root: Path | str,
                   run_name: str,
                   timestamp: Optional[str] = None) -> Path:
    """
    Build the log file path for a single execution.

    Important:
    - This function does NOT create the file.
    - The file is created only when init_logging() is called.

    Example:
      out/logs/classification_preview_20260126_154512.log

    :parameter
        output_root: Root output directory (e.g., "out/")
        run_name: Name of the run/pipeline (e.g., "classification_preview")
        timestamp: Optional timestamp string (YYYYMMDD_HHMMSS). If None, uses current time.
    """
    out_root = Path(output_root)
    logs_dir = out_root / _LOGS_DIR_NAME
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"{_safe_name(run_name)}_{ts}.log"


def init_logging(log_file: Path, level: str = _DEFAULT_LOG_LEVEL) -> logging.Logger:
    """
    Initialize crispdm logging for the current run.

    Behavior:
    - Creates out/logs/ if missing
    - Writes logs to both console and file
    - Avoids duplicated handlers if called multiple times

    Expected call site:
    - ONLY in Facade layer (preview_facade_api / run_facade_api)
    """
    global _CONFIGURED_FOR

    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger(_LOGGER_NAMESPACE)

    # If already configured for the same file, do nothing (idempotent init).
    if _CONFIGURED_FOR == log_file and root.handlers:
        return root

    # If configuring a new run in the same process, clear previous handlers.
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    root.propagate = False

    # Console formatter: compact, readable while debugging notebooks.
    console_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # File formatter: includes filename:line for precise debugging.
    file_formatter = logging.Formatter(
        fmt="%(asctime)s\t%(levelname)s\t%(name)s\t%(filename)s:%(lineno)d\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(root.level)
    console.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(root.level)
    file_handler.setFormatter(file_formatter)

    root.addHandler(console)
    root.addHandler(file_handler)

    _CONFIGURED_FOR = log_file
    root.debug("Logging initialized. log_file=%s level=%s", log_file, level.upper())
    return root


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a namespaced logger. init_logging() should be called beforehand by the Facade.
    """
    return logging.getLogger(f"{_LOGGER_NAMESPACE}.{module_name}")
