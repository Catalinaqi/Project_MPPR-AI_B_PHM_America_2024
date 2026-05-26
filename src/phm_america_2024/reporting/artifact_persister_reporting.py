# src/phm_america_2024/reporting/artifact_persister_reporting.py
from __future__ import annotations

# ---------------------------------------------------------------------------
# SECTION 1 – Standard-library imports
# ---------------------------------------------------------------------------
import json
import pickle
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# SECTION 2 – Third-party imports
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# SECTION 3 – Internal imports
# ---------------------------------------------------------------------------
from phm_america_2024.configuration.enum_registry_config import Phase, StepsPhase
from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.path_service_common import resolve_path

# ---------------------------------------------------------------------------
# SECTION 4 – Module-level logger
# ---------------------------------------------------------------------------
log = get_logger(__name__)


def save_table_png(
        df: pd.DataFrame,
        *,
        out_path: Path,
        title: Optional[str] = None,
        max_rows: int = 30,
        dpi: int,
) -> Path:
    """Save a pandas DataFrame as a PNG image using a Matplotlib table.

    Keeps the project *"every artefact visible as PNG"* convention so that
    run results are inspectable without a notebook or CSV viewer.

    Parameters
    ----------
    df:
        DataFrame to render.  Must not be ``None``.
    out_path:
        Destination ``.png`` file path.
    title:
        Optional title rendered above the table.
    max_rows:
        Maximum number of rows to include; surplus rows are silently dropped.
    dpi:
        Dots-per-inch resolution for the saved image.  No default is
        provided; callers must pass an explicit value to prevent silent
        resolution mismatches across pipeline runs.

    Returns
    -------
    Path
        The path that was written.

    Raises
    ------
    ValueError
        If *df* is ``None``.
    """
    # Step 2: Resolve destination path to absolute location.
    resolved: Path = resolve_path(out_path)

    # Step 3: Create parent directories if they do not exist.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    log.debug("[save_parquet] ensured parent dir=%s", resolved.parent)

    # -----

    # Step 1 – Guard: reject None DataFrame input explicitly
    if df is None:
        raise ValueError("df must not be None")

    # Step 2 – Truncate to max_rows to keep the PNG readable
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)

    # Step 3 – Create figure and axis; hide the default axis frame
    fig, ax = plt.subplots(figsize=(12, 0.4 * (len(df2) + 2)))
    ax.axis("off")

    # Step 4 – Optionally render a title above the table
    if title:
        ax.set_title(title)

    # Step 5 – Render the DataFrame as a Matplotlib table widget
    tbl = ax.table(
        cellText=df2.values,
        colLabels=list(df2.columns),
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)

    # Step 6 – Apply tight layout, write PNG at caller-specified resolution
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    # Step 7 – Emit info-level confirmation
    log.info("[artifacts] table png saved: %s", out_path)
    return out_path


def save_figure(
        fig: plt.Figure,
        *,
        out_path: Path,
        dpi: int,
) -> Path:
    """Save a Matplotlib figure and close it to avoid memory leaks.

    Parameters
    ----------
    fig:
        Matplotlib ``Figure`` instance to persist.
    out_path:
        Destination ``.png`` file path.
    dpi:
        Dots-per-inch resolution for the saved image.  No default is
        provided; callers must pass an explicit value.

    Returns
    -------
    Path
        The path that was written.
    """
    # Step 1 – Apply tight layout to minimise surrounding whitespace
    fig.tight_layout()

    # Step 2 – Render and write the figure at caller-specified resolution
    fig.savefig(out_path, dpi=dpi)

    # Step 3 – Release figure from memory (critical inside notebook loops)
    plt.close(fig)

    # Step 4 – Emit info-level confirmation
    log.info("[artifacts] figure saved: %s", out_path)
    return out_path

