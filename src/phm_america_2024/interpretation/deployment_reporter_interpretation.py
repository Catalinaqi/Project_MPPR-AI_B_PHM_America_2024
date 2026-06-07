# src/phm_america_2024/phase/deployment_reporter_interpretation.py
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from phm_america_2024.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def final_sign_off(
    df: pd.DataFrame,
    tech_cfg: dict[str, Any],
    ctx: Any,
    output_dir: Path,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Generate deployment sign‑off certificate based on technique configuration.

    Args:
        df: Unused (signature consistency).
        tech_cfg: Technique configuration dict (e.g. from YAML final_sign_off).
        ctx: Should contain:
            - ``ctx.evaluation_metrics`` (dict, optional): summary of evaluation results.
        output_dir: Directory to write output files.

    Returns:
        Unmodified df and a dict with key ``deployment_sign_off`` containing
        the sign‑off certificate data.
    """
    log.debug("[final_sign_off] entry")

    params = tech_cfg.get("params", {})
    required_review = params.get("required_review", True)
    review_roles = params.get("review_roles", [])
    output_file = tech_cfg.get("output", "5.4.decision.sign_off_trace.json")

    sign_off = {
        "pipeline_version": getattr(ctx, "pipeline_version", "2.1"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "required_review": required_review,
        "review_roles": review_roles,
        "reviewer": getattr(ctx, "reviewer_name", "System Auto‑Approval (MVP)"),
        "evaluation_metrics_summary": getattr(ctx, "evaluation_metrics", {}),
        "decision": "APPROVED_FOR_DEPLOYMENT",
        "signature": "MVP_INTERNAL",
    }

    out_path = output_dir / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sign_off, indent=2), encoding="utf-8")
    log.info("Sign‑off written to %s", out_path)

    log.debug("[final_sign_off] completed")
    return df, {"deployment_sign_off": sign_off}
