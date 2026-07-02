<div align="center">

# PHM America 2024: Preventive Maintenance for Robotics & Intelligent Automation

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Poetry](https://img.shields.io/badge/Poetry-60A5FA?style=for-the-badge&logo=poetry&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=for-the-badge&logo=duckdb&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![PyArrow](https://img.shields.io/badge/PyArrow-DC3545?style=for-the-badge&logo=apache&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![YAML](https://img.shields.io/badge/YAML-CB171E?style=for-the-badge&logo=yaml&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-262626?style=for-the-badge&logo=python&logoColor=white)
![MyPy](https://img.shields.io/badge/MyPy-2A6DB2?style=for-the-badge&logo=python&logoColor=white)

</div>

**Production-grade Machine Learning pipeline** designed to handle complex aero-thermodynamic data, transitioning from raw telemetric CSVs to probabilistic maintenance diagnostics for helicopter turbine engines.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Layer Responsibilities](#3-layer-responsibilities)
4. [Project Structure](#4-project-structure)
5. [Technologies & Tools](#5-technologies--tools)
6. [Key Design Principles](#6-key-design-principles)
7. [Quick Start](#7-quick-start)
8. [Reproducibility & Auditability](#8-reproducibility--auditability)
9. [Roadmap](#9-roadmap)

---

## 1. Project Overview

### 1.1 PHM Context and Preventive Maintenance

**Prognostics and Health Management (PHM)** is an engineering discipline that monitors, diagnoses, and predicts the health state of mechanical, electronic, or structural systems. The goal is to transition from corrective maintenance (repair after failure) or fixed-interval preventive maintenance to **predictive maintenance** based on actual asset conditions, reducing operational costs, increasing availability, and improving safety.

In the context of **helicopter turbine engines**, PHM is critical: an in-flight failure can have catastrophic consequences. Continuous monitoring of parameters such as gas temperature, compressor speed, torque output, and their comparison with theoretical values enables early detection of degradation or anomalies.

PHM is structured around three fundamental phases:

- **Diagnostics** – Detection and identification of a fault that has already occurred. In this project, it corresponds to the **binary classification** of engine state (nominal 0 vs faulty 1).
- **Prognostics** – Prediction of the future evolution of degradation and estimation of the remaining useful life (RUL). Here, **probabilistic regression of the torque margin** provides a quantitative measure of current degradation that can be projected forward.
- **Health Management** – Operational decisions based on diagnostic and prognostic information: scheduling maintenance interventions, adjusting operating regimes to slow degradation, or stopping the asset for safety.

The **Torque Margin** is the key indicator for quantifying degradation:

$$ \text{Torque Margin} = \text{Torque Target} - \text{Torque Measured} $$

Where **Torque Target** is the theoretical torque a healthy engine should deliver under given environmental conditions, and **Torque Measured** is the actual torque measured by sensors. A negative or excessively deviated margin indicates an imminent fault.

### 1.2 The PHM North America 2024 Challenge

This project participates in the **PHM2024 Conference Data Challenge** (https://data.phmsociety.org/phm2024-conference-data-challenge/), which requires developing a Machine Learning model capable of:

- **Classifying** engine health state (nominal 0 vs faulty 1);
- **Estimating** the continuous torque margin value with uncertainty quantification (probability distribution).

The dataset contains observations from **7 engines** of the same type:

- **4** for training (shuffled, without temporal or engine identifiers);
- **3** for testing/validation (cross-asset generalization).

| Partition | Rows | Percentage | Targets Available |
|-----------|------|------------|-------------------|
| Training | 742,625 | 94.54% | `faulty`, `trq_margin` |
| Validation | 21,436 | 2.73% | Features only |
| Test | 21,436 | 2.73% | Features only |
| **Total** | **785,497** | **100%** | — |

### 1.3 Architecture Overview

The solution adopts a **hybrid cascade architecture** combining a probabilistic NGBoost regressor with a calibrated LightGBM classifier, supported by physics-based feature engineering and a robust validation strategy for cross-asset generalization:

```
Input → Audit & Data Prep → Feature Engineering → NGBoost (μ, σ²) → μ, σ² as features → LightGBM → Isotonic Calibration → Output: faulty + PDF
```

- **NGBoost** (Natural Gradient Boosting) with Normal distribution outputs native mean μ and variance σ² per sample.
- **LightGBM** performs binary diagnostics, enhanced with NGBoost predictions as cascade features.
- **Isotonic Regression** calibrates raw probabilities for reliable confidence estimates.
- **Validation** uses GMM clustering (k=4) + GroupKFold (k=5) to simulate cross-engine generalization.
- **Total compute time**: under 30 seconds on CPU, no GPU required.

---

## 2. System Architecture

The project implements a clean, modular architecture fully aligned with the **CRISP-DM** standard.

### Pipeline Flow Diagram

```text
                    ┌─────────────────────────────────────────┐
                    │         API / FACADE LAYER              │
                    │     (execution_facade_api.py)           │
                    └──────────────┬──────────────────────────┘
                                   │ Context Initialization
                    ┌──────────────▼──────────────────────────┐
                    │      CONFIGURATION & VALIDATION         │
            ┌───────┼─────────────────────────────────────────┼───────┐
            │       │ YAML Profiles ──► Pydantic DTOs         │       │
            │       └──────────────┬──────────────────────────┘       │
            │                      │ Strict Validation Passed         │
            │       ┌──────────────▼──────────────────────────┐       │
            │       │    CRISP-DM PIPELINE ORCHESTRATION      │       │
            │       │ ┌────────┐  ┌────────┐  ┌────────┐      │       │
            │ Data  │ │Phase 2 │  │Phase 3 │  │Phase 4 │      │ Model │
            │ Ingest│ │  EDA   │─►│ Prep   │─►│ Model  │─►... │ Output│
            │ DuckDB│ └────────┘  └────────┘  └────────┘      │       │
            │       └──────────────┬──────────────────────────┘       │
            │                      │ Deterministic Execution          │
            └──────────────────────┼──────────────────────────────────┘
                    ┌──────────────▼──────────────────────────┐
                    │        REPORTING & PERSISTENCE          │
                    │   (Parquet, JSON Traces, PNG Plots)     │
                    └─────────────────────────────────────────┘
```

---

## 3. Layer Responsibilities

Each layer maintains a strict contract, enabling independent phase execution without forcing unnecessary re-runs.

| Layer | Responsibility | Key Components |
|-------|----------------|----------------|
| API | Single entry point. Orchestrates phases & verifies dependencies. | `execution_facade_api.py` |
| Configuration | YAML loading, merging, and strict DTO validation. | `build_factory_config.py`, `yml_repository_config.py`, `pipeline_task_dto_config.py` |
| Data | Data ingestion, DuckDB ETL, profiling, and strategy-based reading. | `acquisition_extractor_data.py`, `profiling_profiler_data.py`, `read_strategy_repository_data.py`, `download_extractor_data.py`, `utils/csv_loader_data.py`, `utils/load_loader_data.py` |
| Domain | Enumerations and registry types for pipeline configurations. | `enum_registry_domain.py` |
| Feature | Cleaning, formatting, transformation, and selection of features. | `cleaning_transformer_feature.py`, `formatting_transformer_feature.py`, `transformation_transformer_feature.py`, `selection_selector_feature.py` |
| Model | Algorithm selection, training, and evaluation for regression and classification. | `regression_trainer_model.py`, `regression_evaluator_model.py`, `regression_algorithm_selector_model.py`, `classification_trainer_model.py`, `classification_evaluator_model.py`, `classification_algorithm_selector_model.py`, `utils/model_registry_config.py` |
| Phase | Independent CRISP-DM runners. Never re-executes completed work. | `phase2_understanding_runner_phase.py`, `phase3_preparation_runner_phase.py`, `phase4_modeling_runner_phase.py`, `phase5_evaluation_and_interpretation_phase.py`, `phase6_deployment_phase.py` |
| Pipeline | Problem-specific orchestrators linking phases. | `classification_runner_pipeline.py`, `regression_runner_pipeline.py`, `utils/context_facade_common.py` |
| Registry | Dynamic phase component registries for modular extension. | `generator_registry_registry.py`, `phase2_generator_registry.py`, `phase3_generator_registry.py`, `phase4_generator_registry.py`, `phase5_generator_registry.py` |
| Interpretation | Pipeline auditing, cluster interpretation, deployment reporting, business alignment. | `pipeline_auditor_interpretation.py`, `cluster_interpreter_interpretation.py`, `deployment_reporter_interpretation.py`, `business_alignment_evaluator_interpretation.py` |
| Reporting | Artifact persistence, plot generation for EDA and model evaluation. | `artifact_persister_reporting.py`, `plots_generator_reporting.py` |
| Deployment | Academic scoring and final package deliverable generation. | `academic_scoring_deployment.py`, `package_deliverable_deployment.py` |
| Common | Cross-cutting utilities: paths, I/O, logging, context management. | `path_service_common.py`, `io_service_common.py`, `logging_adapter_common.py` |
| API Entry | Main entry point for the entire application. | `main.py` |

---

## 4. Project Structure

```
Project_MPPR-ALB_PHM_America_2024/
│
├── config/                         # Strict YAML configurations & rules
│   ├── dataset/
│   │   └── dataset_config.yml      # Dataset loading parameters
│   ├── pipeline/
│   │   ├── active_profile.yml      # Active execution profile (dev/prod)
│   │   ├── base_pipeline_config.yml
│   │   ├── regression_pipeline_config.yml
│   │   └── classification_pipeline_config.yml
│   └── rules/
│       ├── dataset_schema.yml       # Column schema definitions
│       └── ranges_quality_rules_config.yml  # Physics-based quality rules
│
├── data/raw/                       # Original challenge CSV files
│   ├── train/
│   │   ├── X_train.csv
│   │   └── Y_train.csv
│   ├── validation/
│   │   └── X_validation.csv
│   └── test/
│       └── X_test.csv
│
├── notebooks/                      # Jupyter notebooks for analysis & prototyping
│   ├── 00_pre_analysis-part-0.ipynb
│   ├── 00_pre_analysis-part-1.ipynb
│   ├── 00_pre_analysis-part-2.ipynb
│   ├── phase2_analysis.ipynb
│   ├── phase3_analysis.ipynb
│   ├── phase3_class_analysis.ipynb
│   ├── phase4_analysis.ipynb
│   ├── phase4_class_analysis.ipynb
│   ├── phase5_analysis.ipynb
│   ├── phase6_analysis.ipynb
│   └── dev/
│       ├── phase3_analysis-dev.ipynb
│       └── phase4_analysis-dev.ipynb
│
├── src/phm_america_2024/           # Production ML source code
│   ├── __init__.py
│   ├── main.py                     # Application entry point
│   │
│   ├── api/
│   │   └── execution_facade_api.py # Orchestration facade
│   │
│   ├── common/
│   │   ├── io_service_common.py
│   │   ├── path_service_common.py
│   │   └── logging_adapter_common.py
│   │
│   ├── configuration/
│   │   ├── build_factory_config.py
│   │   ├── pipeline_task_dto_config.py
│   │   └── yml_repository_config.py
│   │
│   ├── data/
│   │   ├── acquisition_extractor_data.py
│   │   ├── download_extractor_data.py
│   │   ├── profiling_profiler_data.py
│   │   ├── read_strategy_repository_data.py
│   │   └── utils/
│   │       ├── csv_loader_data.py
│   │       └── load_loader_data.py
│   │
│   ├── deployment/
│   │   ├── academic_scoring_deployment.py
│   │   └── package_deliverable_deployment.py
│   │
│   ├── domain/
│   │   └── enum_registry_domain.py
│   │
│   ├── feature/
│   │   ├── cleaning_transformer_feature.py
│   │   ├── formatting_transformer_feature.py
│   │   ├── selection_selector_feature.py
│   │   └── transformation_transformer_feature.py
│   │
│   ├── interpretation/
│   │   ├── business_alignment_evaluator_interpretation.py
│   │   ├── cluster_interpreter_interpretation.py
│   │   ├── deployment_reporter_interpretation.py
│   │   └── pipeline_auditor_interpretation.py
│   │
│   ├── model/
│   │   ├── classification_algorithm_selector_model.py
│   │   ├── classification_evaluator_model.py
│   │   ├── classification_trainer_model.py
│   │   ├── regression_algorithm_selector_model.py
│   │   ├── regression_evaluator_model.py
│   │   ├── regression_trainer_model.py
│   │   └── utils/
│   │       └── model_registry_config.py
│   │
│   ├── phase/
│   │   ├── phase2_understanding_runner_phase.py
│   │   ├── phase3_preparation_runner_phase.py
│   │   ├── phase4_modeling_runner_phase.py
│   │   ├── phase5_evaluation_and_interpretation_phase.py
│   │   └── phase6_deployment_phase.py
│   │
│   ├── pipeline/
│   │   ├── classification_runner_pipeline.py
│   │   ├── regression_runner_pipeline.py
│   │   └── utils/
│   │       └── context_facade_common.py
│   │
│   ├── registry/
│   │   ├── generator_registry_registry.py
│   │   ├── phase2_generator_registry.py
│   │   ├── phase3_generator_registry.py
│   │   ├── phase4_generator_registry.py
│   │   └── phase5_generator_registry.py
│   │
│   └── reporting/
│       ├── artifact_persister_reporting.py
│       └── plots_generator_reporting.py
│
├── outputs/                        # Deterministic, versioned runtime snapshots
│   ├── logs/                       # Execution logs (.log)
│   └── runs/                       # Run artifacts organized by task & timestamp
│       ├── regression/
│       │   └── phm2024/
│       │       ├── <timestamp>/
│       │       └── ...
│       └── classification/
│           └── phm2024/
│               ├── <timestamp>/
│               └── ...
│
├── docs/                           # Multilingual documentation
│   ├── ES/                         # Spanish
│   ├── IT/                         # Italian
│   │   ├── funzionale/             # Functional documentation
│   │   │   ├── paper/              # Paper analysis (7 competition papers)
│   │   │   │   ├── general_paper/
│   │   │   │   └── models_paper/
│   │   │   └── ...
│   │   └── tecnico/                # Technical documentation
│   └── RELAZIONE/                  # Final report (Italian)
│       ├── 1.contesto.md
│       ├── 2.0.metodologia.md
│       ├── 2.1.phase2.md
│       ├── 2.2.phase3.md
│       ├── 2.3.phase4.md
│       ├── 2.4.phase5.md
│       ├── 2.5.phase6.md
│       └── 3.risultati.md
│
├── pyproject.toml                  # Poetry dependencies (PEP 621 compliant)
├── poetry.lock
├── poetry.toml
├── .gitignore
└── README.md                       # This file
```

---

## 5. Technologies & Tools

Built on a modern, high-performance Python 3.10 stack managed by Poetry 2.0+.

| Category | Tools & Libraries | Purpose |
|----------|-------------------|---------|
| **Core & Infra** | `python = "3.10"`, `poetry-core >= 2.0` | Environment & dependency management |
| **Data Engine** | `duckdb (1.5+)`, `pandas`, `pyarrow` | High-performance I/O and data manipulation |
| **Configuration** | `pydantic (2.13+)`, `omegaconf`, `pyyaml` | Hierarchical YAML merging and strict typing |
| **Modeling** | `scikit-learn (1.7+)`, `ngboost (0.5+)` | ML algorithms and Probabilistic Regression |
| **Artifacts & Viz** | `joblib`, `matplotlib` | Deterministic model versioning & plotting |
| **QA & Linting** | `ruff`, `mypy`, `deptry` | Strict code quality, static typing & dependency tree validation |

---

## 6. Key Design Principles

1. **Non-Monolithic Execution**: Each phase checks the `RunContext` for existing artifacts. If present, it loads them. If missing, it raises a `RuntimeError` prompting the prerequisite phase.

2. **Fail-Fast Configuration**: YAML → OmegaConf → Pydantic DTO. Any typo or missing field aborts at `init_run_facade_api()`, preventing expensive runtime crashes mid-training.

3. **Immutable Context**: `RunContext` is a frozen dataclass. DataFrames act as optional in-memory caches, but the single source of truth remains the `.parquet` files.

4. **Model/Config Decoupling**: Algorithms are mapped via a dynamic registry. Switching configurations requires zero code changes.

5. **Deterministic Execution**: All random seeds are fixed and documented. The entire pipeline is reproducible across runs given the same configuration.

6. **Physics-Based Validation**: Sensor readings are validated against thermodynamic constraints. Impossible values (e.g., negative torque, cold engine in flight) are flagged or removed before modeling.

---

## 7. Quick Start

### Prerequisites

- Python 3.10
- [Poetry 2.0+](https://python-poetry.org/docs/#installation) installed

### Installation & Execution

```bash
# 1. Clone the repository
git clone https://github.com/Catalinaqi/project-manutenzione_preventiva_per_la_robotica.git
cd project-manutenzione_preventiva_per_la_robotica

# 2. Install dependencies via Poetry (Core + Dev tools)
poetry install

# 3. Configure your active profile (dev vs prod)
# Edit config/pipeline/active_profile.yml to set sample_rows or full execution

# 4. Run the pipeline via the API Facade
poetry run python src/phm_america_2024/api/execution_facade_api.py
```

### Code Quality Checks (Dev)

```bash
poetry run ruff check .
poetry run mypy src/
poetry run deptry src/
```

---

## 8. Reproducibility & Auditability

Every run produces a self-contained snapshot under `outputs/runs/<task>/<timestamp>/` containing:

- **Execution logs** (idempotent console + file handlers).
- **Intermediate artifacts** (Parquet datasets, Pickle models, scaling parameters).
- **Automated Reports** (JSON traces and PNG visualizations for every step, making results inspectable without code).
- **Model metadata** (best fold selection, hyperparameters, evaluation metrics).
- **Deployment packages** (final submission ZIP for the PHM2024 competition).

The outputs directory structure follows the CRISP-DM phases:

```
outputs/runs/regression/phm2024/<timestamp>/
├── phase2_data_understanding/
│   ├── *.parquet          # Acquired datasets
│   ├── *.json             # Data descriptions, quality assessments, drift reports
│   └── *.png              # EDA visualizations (flight regimes, GMM exploration)
├── phase3_data_preparation/
│   ├── *.parquet          # Cleaned, selected, transformed, formatted datasets
│   ├── *.pkl              # Fitted scalers
│   └── *.json             # Transformation logs, engineering formulas
├── phase4_data_modeling/
│   ├── *.pkl              # Trained models (NGBoost, LightGBM, Isotonic calibrator)
│   └── *.json             # Cross-validation traces, evaluation rankings, best model meta
├── phase5_evaluation_and_interpretation/
│   ├── *.json             # Permutation importance, degradation traces, leakage audit
│   ├── *.png              # Feature importance, calibration curves, degradation comparison
│   └── *.json             # Final sign-off certificate
└── phase6_deployment/
    ├── *.parquet          # Final academic predictions
    └── *.zip              # Deliverable package
```
---

## 9. Roadmap

* [x] **Phase 1-2**: Architecture Setup, Data Ingestion & EDA.
* [x] **Phase 3**: Data Preparation (Robust Scaling & Feature Engineering).
* [x] **Phase 4**: Modeling & Calibration (Probabilistic Regression via NGBoost).
* [x] **Phase 5**: Interpretation & Evaluation Metrics.
* [x] **Phase 6**: Final Deployment & Challenge Submission.
* [ ] **Refactoring**: Containerization (Docker) & CI/CD Pipelines.
* [ ] **MLflow Integration**: Experiment tracking & model registry
* [ ] **Unit & Integration Testing**: Comprehensive test coverage

---

**Built with ❤️ by [Catalinaqi**](https://www.google.com/search?q=https://github.com/Catalinaqi) *Software Engineering*