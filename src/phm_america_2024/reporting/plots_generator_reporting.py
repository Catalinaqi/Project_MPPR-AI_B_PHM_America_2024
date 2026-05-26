# src/phm_america_2024/reporting/plots_generator_reporting.py
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
import numpy as np


# ---------------------------------------------------------------------------
# SECTION 3 – Internal imports
# ---------------------------------------------------------------------------
from phm_america_2024.common.logging_adapter_common import get_logger
from phm_america_2024.common.path_service_common import resolve_path

# ---------------------------------------------------------------------------
# SECTION 4 – Module-level logger
# ---------------------------------------------------------------------------
log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Plot-factory utilities producing Matplotlib Figure objects in consistent
# visual style across all modelling tracks.
# =============================================================================

def plot_gmm_analysis(report: dict[str, Any], *, title: str = "GMM Model Selection") -> plt.Figure:
    """Return GMM BIC/AIC curves figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    curve = report["curve"]
    ax.plot(curve["k"], curve["BIC"], label="BIC", marker='o')
    ax.plot(curve["k"], curve["AIC"], label="AIC", marker='o')

    ax.set_title(f"{title} (Optimal k={report.get('optimal_k')})")
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    log.debug("[plots] plot_gmm_analysis optimal_k=%s", report.get("optimal_k"))
    return fig

def plot_flight_regime_binning(
        data: pd.Series,
        *,
        title: str,
        bins: int,
        plot_type: str = "hist"
) -> plt.Figure:
    """Return histogram or KDE plot showing flight regime binning distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if plot_type == "hist":
        ax.hist(data.dropna(), bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    elif plot_type == "kde":
        data.dropna().plot.kde(ax=ax, color='blue')
    else:
        log.warning("[plots] unknown plot_type='%s', defaulting to 'hist'", plot_type)
        ax.hist(data.dropna(), bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle='--', alpha=0.6)

    return fig

# ?
def plot_gmm_curve(
        gmm_result: dict[str, Any],
        output_path: Path,
) -> None:
    """Generate and save a BIC/AIC curve PNG from GMM exploration results.

    Parameters
    ----------
    gmm_result : dict
        Output of ``gmm_exploration`` containing ``curve`` key.
    output_path : Path
        Destination for the PNG file.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    log.info("[plot_gmm_curve] generating PNG for path=%s", output_path)

    curve = gmm_result.get("curve")
    if not curve or "k" not in curve:
        log.error("[plot_gmm_curve] missing curve data in gmm_result")
        return

    k_vals = curve["k"]
    bic = curve.get("BIC", [])
    aic = curve.get("AIC", [])

    fig, ax = plt.subplots(figsize=(8, 5))
    if bic:
        ax.plot(k_vals, bic, marker='o', label='BIC')
    if aic:
        ax.plot(k_vals, aic, marker='s', label='AIC')
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Criterion value")
    ax.set_title("GMM BIC/AIC curve")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Mark optimal k if available
    optimal_k = gmm_result.get("optimal_k")
    if optimal_k and optimal_k in k_vals:
        ax.axvline(x=optimal_k, color='red', linestyle=':', alpha=0.7, label=f"Optimal k={optimal_k}")
        ax.legend()

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    size_kb = output_path.stat().st_size / 1024
    log.info("[plot_gmm_curve] saved size_kb=%.2f path=%s", size_kb, output_path)
