# 🚀 PHM America 2024: Prognostics and Health Management

This repository contains the architecture and implementation for the **PHM America 2024 Challenge**. It is built as a highly modular, CRISP-DM compliant pipeline designed to handle complex aero-thermodynamic data, transitioning from raw telemetric CSVs to probabilistic maintenance diagnostics.

### 👨‍💻 About this project

This project represents my professional transition from **Software Engineering** to **Data Science & Machine Learning**. It reflects my focus on building **production-grade ML systems**: 
rather than just building models in isolated notebooks, I have designed a robust, layer-based architecture that emphasizes **reproducibility, traceability, and fail-fast configuration**.

The design addresses the typical pitfalls of DS projects—such as data leakage, configuration drift, and non-deterministic execution—by applying software engineering best practices: 
**strict DTO validation, immutable run contexts, and centralized artifact management.**

---

## 🏗️ Layer-Based Architecture

The project follows a clean, modular architecture. Each layer maintains a strict contract, enabling independent phase execution without forcing unnecessary re-runs.

| Layer | Responsibility |
| --- | --- |
| `🔌 API` | Single entry point. Initializes context, orchestrates phases, and verifies dependencies. |
| `⚙️ Configuration` | YAML loading, merging, and strict validation via Pydantic DTOs. |
| `💾 Data` | Ingestion, persistence, and automatic profiling of raw/intermediate datasets. |
| `🔄 Phase` | Independent CRISP-DM Phase 2–5 runners. Each phase is artifact-aware. |
| `🚀 Pipeline` | Problem-specific orchestrators (Classification vs. Regression). |
| ` Reporting` | Artifact persistence, EDA, and auto-generated PNG visualizations. |

---

## 📂 Project Directory Structure

```text
Project_MPPR-ALB_PHM_America_2024/
├── config/             # Strict YAML configurations & safety thresholds
├── data/raw/           # Original challenge CSV files
├── src/phm_america_2024/
│   ├── api/            # Orchestration facade
│   ├── common/         # Cross-cutting infrastructure (logging, paths, context)
│   ├── configuration/  # Parsing, validation & model registry
│   ├── data/           # ETL, CSV loading & profiling
│   ├── phase/          # CRISP-DM runners (Phase 2-5)
│   └── reporting/      # Artifact persistence & plotting utilities
└── outputs/            # Deterministic, versioned runtime snapshots

```

---

## 🛠️ Key Design Principles

* **Non-Monolithic Execution**: Phases check for existing artifacts. If missing, the pipeline prompts for the prerequisite phase; otherwise, it resumes from the last known state.
* **Fail-Fast Configuration**: YAML files are validated against **Pydantic DTOs** at the `init` stage, preventing runtime errors mid-execution.
* **Deterministic Traceability**: Artifact names are hashed and versioned (e.g., `2.1.data_acquisition.prod_300000_stratified_train.parquet`), ensuring every model can be mapped back to its exact training data.
* **Model/Config Decoupling**: Algorithms are mapped via `ModelRegistry`. Switching from NGBoost to LightGBM is a configuration change, not a code change.

---

## 📦 Reproducibility & Auditability

Every execution produces a self-contained snapshot under `outputs/runs/`, containing:

1. **Full execution logs** (namespaced for debugging).
2. **Intermediate artifacts** (Parquet datasets, trained models, scaling parameters).
3. **Automated Reports** (JSON traces and PNG visualizations for every step, ensuring the results are inspectable without re-running the pipeline).

---

## ⚙️ Environment Setup

* **Dependency Management**: Powered by [Poetry](https://python-poetry.org/).
* **Setup**: `poetry install`
* **Execution profiles**: `config/pipeline/active_profile.yml` allows switching between `dev` (small sample) and `prod` (full-scale) execution environments.

---

### 📝 Roadmap

* [x] **Phase 1-2**: Data ingestion & EDA.
* [x] **Phase 3**: Data preparation (Robust Scaling & Physical KPI Engineering).
* [x] **Phase 4**: Modeling (Probabilistic Regression).
* [x] **Phase 5**: Interpretation & Evaluation.
* [ ] **Phase 6**: Final Deployment & Challenge Submission.

---

*Built with ❤️ for PHM America 2024.*