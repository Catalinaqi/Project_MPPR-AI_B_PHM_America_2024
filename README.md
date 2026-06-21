# PHM America 2024: Preventive Maintenance for Robotics & Intelligent Automation

<div align="center">

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


**Production-grade Machine Learning pipeline** designed to handle complex aero-thermodynamic data, transitioning from raw telemetric CSVs to probabilistic maintenance diagnostics.

---

## **Table of Contents**

- [Overview & Journey](#-overview--journey)
- [System Architecture](#-system-architecture)
- [Layer Responsibilities](#-layer-responsibilities)
- [Technologies & Tools](#-technologies--tools)
- [Key Design Principles](#-key-design-principles)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)

---

## **Overview & Journey**

### **From Software Engineering to Data Science**
This repository represents my professional transition from **Software Engineering** to **Data Science & Machine Learning**. 

Rather than building experimental models in isolated Jupyter notebooks, this project focuses on engineering a **production-ready ML system**. It addresses typical Data Science pitfalls—such as data leakage, configuration drift, and non-deterministic execution—by applying rigorous Software Engineering best practices:
* **Strict DTO Validation** (Fail-fast configuration via Pydantic)
* **Immutable Run Contexts** (State management)
* **Centralized Artifact Management** (Reproducibility via DuckDB & PyArrow)
* **Static Typing & Linting** (MyPy & Ruff integration)

---

## **System Architecture**

The project implements a clean, modular architecture fully aligned with the **CRISP-DM** standard.

### **Pipeline Flow Diagram**

```text
                    ┌─────────────────────────────────────────┐
                    │            API / FACADE LAYER           │
                    │        (execution_facade_api.py)        │
                    └──────────────┬──────────────────────────┘
                                   │ Context Initialization
                    ┌──────────────▼──────────────────────────┐
                    │       CONFIGURATION & VALIDATION        │
            ┌───────┼─────────────────────────────────────────┼───────┐
            │       │ YAML Profiles ──► Pydantic DTOs         │       │
            │       └──────────────┬──────────────────────────┘       │
            │                      │ Strict Validation Passed         │
            │       ┌──────────────▼──────────────────────────┐       │
            │       │     CRISP-DM PIPELINE ORCHESTRATION     │       │
            │       │ ┌────────┐  ┌────────┐  ┌────────┐      │       │
            │ Data  │ │Phase 2 │  │Phase 3 │  │Phase 4 │      │ Model │
            │ Ingest│ │  EDA   │─►│ Prep   │─►│ Model  │─►... │ Output│
            │ DuckDB│ └────────┘  └────────┘  └────────┘      │       │
            │       └──────────────┬──────────────────────────┘       │
            │                      │ Deterministic Execution          │
            └──────────────────────┼──────────────────────────────────┘
                    ┌──────────────▼──────────────────────────┐
                    │         REPORTING & PERSISTENCE         │
                    │   (Parquet, JSON Traces, PNG Plots)     │
                    └─────────────────────────────────────────┘

```

---

## **Layer Responsibilities**

Each layer maintains a strict contract, enabling independent phase execution without forcing unnecessary re-runs.

| Layer | Responsibility | Key Components |
| --- | --- | --- |
| `🔌 API` | Single entry point. Orchestrates phases & verifies dependencies. | `execution_facade_api.py` |
| `⚙️ Configuration` | YAML loading, merging, and strict DTO validation. | Pydantic, OmegaConf, PyYAML |
| `💾 Data` | Data ingestion, chunking, and automatic profiling. | DuckDB, Pandas, PyArrow |
| `🔄 Phase` | Independent CRISP-DM runners. Never re-executes completed work. | Phase 2–5 Runners |
| `🚀 Pipeline` | Problem-specific orchestrators. | Classification / Regression Runners |
| `📊 Reporting` | Artifact persistence and automated EDA generation. | Joblib, Matplotlib |

---

## 🛠️ **Technologies & Tools**

Built on a modern, high-performance Python 3.10 stack managed by Poetry 2.0+.

| Category | Tools & Libraries | Purpose |
| --- | --- | --- |
| **Core & Infra** | `python = "3.10"`, `poetry-core >= 2.0` | Environment & dependency management |
| **Data Engine** | `duckdb (1.5+)`, `pandas`, `pyarrow` | High-performance I/O and data manipulation |
| **Configuration** | `pydantic (2.13+)`, `omegaconf`, `pyyaml` | Hierarchical YAML merging and strict typing |
| **Modeling** | `scikit-learn (1.7+)`, `ngboost (0.5+)` | ML algorithms and Probabilistic Regression |
| **Artifacts & Viz** | `joblib`, `matplotlib` | Deterministic model versioning & plotting |
| **QA & Linting** | `ruff`, `mypy`, `deptry` | Strict code quality, static typing & dependency tree validation |

---

## **Key Design Principles**

1. **Non-Monolithic Execution**: Each phase checks the `RunContext` for existing artifacts. If present, it loads them. If missing, it raises a `RuntimeError` prompting the prerequisite phase.
2. **Fail-Fast Configuration**: YAML → OmegaConf → Pydantic DTO. Any typo or missing field aborts at `init_run_facade_api()`, preventing expensive runtime crashes mid-training.
3. **Immutable Context**: `RunContext` is a frozen dataclass. DataFrames act as optional in-memory caches, but the single source of truth remains the `.parquet` files.
4. **Model/Config Decoupling**: Algorithms are mapped via a dynamic registry. Switching configurations requires zero code changes.

---

## **Quick Start**

### **Prerequisites**

* Python 3.10
* [Poetry 2.0+](https://www.google.com/search?q=https://python-poetry.org/docs/%23installation) installed

### **Installation & Execution**

```bash
# 1. Clone the repository
git clone [https://github.com/Catalinaqi/project-manutenzione_preventiva_per_la_robotica.git](https://github.com/Catalinaqi/project-manutenzione_preventiva_per_la_robotica.git)
cd project-manutenzione_preventiva_per_la_robotica

# 2. Install dependencies via Poetry (Core + Dev tools)
poetry install

# 3. Configure your active profile (dev vs prod)
# Edit config/pipeline/active_profile.yml to set sample_rows or full execution

# 4. Run the pipeline via the API Facade
poetry run python src/phm_america_2024/api/execution_facade_api.py

```

### **Code Quality Checks (Dev)**

```bash
poetry run ruff check .
poetry run mypy src/
poetry run deptry src/

```

---

## **Project Structure**

```text
Project_MPPR-ALB_PHM_America_2024/
├── config/             # Strict YAML configurations & rules
├── data/raw/           # Original challenge CSV files
├── src/phm_america_2024/
│   ├── api/            # Orchestration facade
│   ├── common/         # Immutable context & cross-cutting utilities
│   ├── configuration/  # Pydantic validation & registries
│   ├── data/           # DuckDB ETL & profiling
│   ├── phase/          # CRISP-DM runners (Phase 2-5)
│   └── reporting/      # Artifact persistence & plotting
├── outputs/            # Deterministic, versioned runtime snapshots
│   └── runs/           # Execution logs, models, and Parquet data
├── pyproject.toml      # Poetry dependencies (PEP 621 compliant)
└── README.md           # Documentation

```

---

## **Reproducibility & Auditability**

Every run produces a self-contained snapshot under `outputs/runs/<task>/<timestamp>/` containing:

* **Execution logs** (idempotent console + file handlers).
* **Intermediate artifacts** (Parquet datasets, Pickle models, scaling parameters).
* **Automated Reports** (JSON traces and PNG visualizations for every step, making results inspectable without code).

---

## **Roadmap**

* [x] **Phase 1-2**: Architecture Setup, Data Ingestion & EDA.
* [x] **Phase 3**: Data Preparation (Robust Scaling & Feature Engineering).
* [x] **Phase 4**: Modeling & Calibration (Probabilistic Regression via NGBoost).
* [x] **Phase 5**: Interpretation & Evaluation Metrics.
* [x] **Phase 6**: Final Deployment & Challenge Submission.
* [ ] **Refactoring**: Containerization (Docker) & CI/CD Pipelines.

---

**Built with ❤️ by [Catalinaqi**](https://www.google.com/search?q=https://github.com/Catalinaqi) *Applying Software Engineering rigor to Data Science & Intelligent Automation.*

