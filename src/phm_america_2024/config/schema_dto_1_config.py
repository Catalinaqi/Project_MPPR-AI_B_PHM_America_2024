# src/phm_america_2024/config/schema_dto_1_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from phm_america_2024.core.logging_utils_core import get_logger
from phm_america_2024.config.enums_utils_1_config import (
    ProblemType, normalize_problem_type, LogLevel, normalize_log_level)

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# This is the "schema.py" layer:
# - Converts resolved YAML dicts into typed Python objects (DTOs)
# - Provides a stable configuration contract for the rest of the pipeline
#
# Program flow:
# - load_loader_config.load_and_resolve() -> dict
# - ProjectConfig.from_dict(resolved_dict) -> ProjectConfig (typed)
# - stages/pipelines consume ProjectConfig, never raw YAML
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - DTO / Schema layer
#   - Configuration Object (single source of truth)
# =============================================================================



@dataclass(frozen=True)
class ProjectConfig:
    """
    Root configuration object used by the entire program.
    """
    version: str
    pipeline: PipelineMeta
    runtime: RuntimeConfig
    stages: StagesConfig

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProjectConfig":
        """
        Build a typed ProjectConfig from a resolved YAML dictionary.

        Why strict parsing?
        - It catches config drift early (typos, missing sections).
        - It keeps the rest of the codebase stable.
        """

        log.debug("[from_dict] ProjectConfig.from_dict: top_keys=%s", list(d.keys()))

        version = str(d.get("version", "1.0"))
        pipeline = d.get("pipeline", {})
        runtime = d.get("runtime", {})
        stages = d.get("stages", {})

        if not isinstance(pipeline, dict):
            raise ValueError("pipeline must be a dict")
        if not isinstance(runtime, dict):
            raise ValueError("runtime must be a dict")
        if not isinstance(stages, dict):
            raise ValueError("stages must be a dict")

        ################################# pipeline #################################
        pipeline_meta = PipelineMeta(
            name=str(pipeline.get("name", "")),
            task=normalize_problem_type(pipeline.get("task", "")),
            objective=str(pipeline.get("objective", "") or ""),
            variables=pipeline.get("variables", {}) if isinstance(pipeline.get("variables", {}), dict) else {},
        )

        ################################# runtime #################################
        runtime_cfg = RuntimeConfig(
            random_seed=int(runtime.get("random_seed", 42)),
            output_root=Path(str(runtime.get("output_root") or "out")),
            overwrite_artifacts=bool(runtime.get("overwrite_artifacts", True)),
            #log_level=normalize_log_level(runtime.get("log_level", LogLevel.DEBUG.value)),
            #if isinstance(runtime.get("log_level", LogLevel.DEBUG.value), str) else LogLevel.DEBUG,
        )

        ################################# stages ##################################
        # ---------------- Stage 2 ----------------
        s2 = stages.get("stage2_understanding", {})
        if not isinstance(s2, dict):
            raise ValueError("stages.stage2_understanding must be a dict")

        stage2_cfg = Stage2Config(
            enabled=bool(s2.get("enabled", True)),
            objective=str(s2.get("objective", "") or ""),
            dataset_input=s2.get("dataset_input", {}) if isinstance(s2.get("dataset_input", {}), dict) else {},
            output_policy=s2.get("output_policy", {}) if isinstance(s2.get("output_policy", {}), dict) else {},
            steps=s2.get("steps", {}) if isinstance(s2.get("steps", {}), dict) else {},
        )

        # ---------------- Stage 3 ----------------
        s3 = stages.get("stage3_preparation", {})
        if not isinstance(s3, dict):
            raise ValueError("stages.stage3_preparation must be a dict")

        stage3_cfg = Stage3Config(
            enabled=bool(s3.get("enabled", True)),
            objective=str(s3.get("objective", "") or ""),
            output_policy=s3.get("output_policy", {}) if isinstance(s3.get("output_policy", {}), dict) else {},
            steps=s3.get("steps", {}) if isinstance(s3.get("steps", {}), dict) else {},
        )
        # ---------------- Stage 4 ----------------
        s4 = stages.get("stage4_modeling", {})
        if not isinstance(s4, dict):
            raise ValueError("stages.stage4_modeling must be a dict")
        stage4_cfg = Stage4Config(
            enabled=bool(s4.get("enabled", True)),
            objective=str(s4.get("objective", "") or ""),
            output_policy=s4.get("output_policy", {}) if isinstance(s4.get("output_policy", {}), dict) else {},
            steps=s4.get("steps", {}) if isinstance(s4.get("steps", {}), dict) else {},
        )
        # ---------------- Stage 5 ----------------
        s5 = stages.get("stage5_evaluation_and_interpretation", {})
        if not isinstance(s5, dict):
            raise ValueError("stages.stage5_evaluation_and_interpretation must be a dict")
        stage5_cfg = Stage5Config(
            enabled=bool(s5.get("enabled", True)),
            objective=str(s5.get("objective", "") or ""),
            output_policy=s5.get("output_policy", {}) if isinstance(s5.get("output_policy", {}), dict) else {},
            steps=s5.get("steps", {}) if isinstance(s5.get("steps", {}), dict) else {},
        )

        cfg = ProjectConfig(
            version=version,
            pipeline=pipeline_meta,
            runtime=runtime_cfg,
            stages=StagesConfig(
                stage2_understanding=stage2_cfg,
                stage3_preparation=stage3_cfg,
                stage4_modeling=stage4_cfg,
                stage5_evaluation_and_interpretation=stage5_cfg,
            ),
        )

        log.info(
            "ProjectConfig.from_dict: done pipeline=%s task=%s output_root=%s log_level=%s",
            cfg.pipeline.name,
            cfg.pipeline.task.value,
            cfg.runtime.output_root,
            #cfg.runtime.log_level.value,
        )

        log.info("ProjectConfig.from_dict: done pipeline=%s task=%s output_root=%s",
                 cfg.pipeline.name,
                 cfg.pipeline.task.value,
                 cfg.runtime.output_root)

        return cfg


@dataclass(frozen=True)
class PipelineMeta:
    """
    Describes the pipeline as a product: name, task, objective, and variables.
    """
    name: str
    task: ProblemType
    objective: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Global runtime settings for the whole run (not "theory CRISP", but execution ops).
    """
    random_seed: int = 42
    output_root: Path = Path("out")
    overwrite_artifacts: bool = True
    #log_level: LogLevel = LogLevel.DEBUG

@dataclass(frozen=True)
class StagesConfig:
    """
    Container for stage configs. We start with Stage2 and extend later.
    """
    stage2_understanding: Stage2Config
    stage3_preparation: Stage3Config
    stage4_modeling: Stage4Config
    stage5_evaluation_and_interpretation: Stage5Config

@dataclass(frozen=True)
class Stage2Config:
    """
    Stage 2 (Data Understanding) configuration.
    Stage 2 MUST be report-only (no data modification).
    """
    enabled: bool
    objective: str
    dataset_input: Dict[str, Any] = field(default_factory=dict)
    output_policy: Dict[str, Any] = field(default_factory=dict)
    steps: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Stage3Config:
    """
    Stage 3 (Data Preparation) configuration.
    Stage 3 MAY modify data (cleaning, feature engineering, etc.).
    """
    enabled: bool
    objective: str
    output_policy: Dict[str, Any] = field(default_factory=dict)
    steps: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Stage4Config:
    """
    Stage 4 (Modeling) configuration.
    Stage 4 MAY modify data (model training, evaluation, etc.).
    """
    enabled: bool
    objective: str
    output_policy: Dict[str, Any] = field(default_factory=dict)
    steps: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Stage5Config:
    """
    Stage 5 Evaluation + interpretation configuration.
    Stage 5 MAY modify data (model deployment, monitoring, etc.).
    """
    enabled: bool
    objective: str
    output_policy: Dict[str, Any] = field(default_factory=dict)
    steps: Dict[str, Any] = field(default_factory=dict)
