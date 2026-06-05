# src/phm_america_2024/data/download_extractor_data.py

from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.path_service_common import resolve_path

# Initialize logger
logger = get_logger(__name__)


# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Implements local data availability validation and integrity checks for the
# PHM North America 2024 Conference Data Challenge dataset.
#
# Since the dataset must be obtained via manual interaction from the official
# PHM Society data repository interface, this module serves as a gatekeeper.
# It verifies that all required assets have been correctly placed into the
# project's internal directory structure (data/raw/) before allowing subsequent
# pipeline execution stages to proceed.
#
# Key Features:
# - Centralized path resolution integrated with path_service_common.
# - Idempotent validation check to minimize pipeline overhead.
# - Explicit error containment: Halts execution with precise instructions
#   if local raw files are missing.
#
# Program flow:
# -----------------------------------------------------------------------------
# - download_phm_2024_dataset(force)
#   -> Resolve absolute target paths for X_train, Y_train, X_validation, and X_test.
#   -> IF all target files exist AND force=False: Log info and exit.
#   -> ELSE: Emit explicit warning and error instructions for manual download.
#   -> Raise FileNotFoundError to secure down-stream components.
# =============================================================================
def download_phm_2024_dataset(
        force: bool = False,
) -> None:
    """
    Validate the local availability of the PHM 2024 Challenge dataset in data/raw/.

    The local filesystem is the absolute check — if the target files exist, the
    validation succeeds and skips execution. If files are missing, it throws an
    exception instructing the operator to perform the manual web download steps.

    Parameters
    ----------
    force : bool, default=False
            If True, forces the validation gatekeeper to trigger the warning
            and error messaging paths regardless of current file status.

    Returns
    -------
    None
            Execution flows normally if validation passes.

    Raises
    ------
    FileNotFoundError
        If any expected file is missing from the designated local raw directories.
    Exception
        For critical unexpected system or path runtime issues.
    """
    try:
        logger.debug("Resolving internal raw repository asset destinations...")

        # 1. Resolve target pipeline destinations using the centralized path service
        x_train_dest = resolve_path("data/raw/train/X_train.csv")
        y_train_dest = resolve_path("data/raw/train/Y_train.csv")
        x_val_dest = resolve_path("data/raw/validation/X_validation.csv")
        x_test_dest = resolve_path("data/raw/test/X_test.csv")

        # 2. Idempotency check
        all_destinations = [x_train_dest, y_train_dest, x_val_dest, x_test_dest]
        if not force and all(dest.exists() for dest in all_destinations):
            logger.info("Dataset already exists in target directories. Skipping ....")
            return

        # 3. Handle missing assets path via explicit logging and error raising
        logger.warning("Missing dataset components detected in local repository workspace.")

        # Identify specific missing files for cleaner debugging outputs
        for dest in all_destinations:
            if not dest.exists():
                logger.error("CRITICAL ASSET MISSING: File not found at '%s'", dest)

        logger.error(
            "MANUAL ACTION REQUIRED: You must manually download the dataset from "
            "data.phmsociety.org by clicking the green links and place all CSV files "
            "inside your local data/raw directory structure."
        )

        raise FileNotFoundError(
            "PHM 2024 dataset files are missing. Please complete the manual download steps."
        )

    except FileNotFoundError as fnf_err:
        # Expected manual interaction branch: re-raise directly to keep clean console frames
        raise fnf_err
    except Exception:
        # Rule TRY400: Capture unexpected systemic context automatically
        logger.exception("Critical error during data acquisition validation")
        raise