

---

# 📘 Project Structure & Architecture

## 🏗️ Layer-Based Architecture
The project follows a clean, modular architecture aligned with **CRISP-DM**. Each layer maintains a strict contract, enabling independent phase execution without forcing unnecessary re-runs.

| Layer | Responsibility | Key Files |
|-------|----------------|-----------|
| `🔌 API` | Single entry point. Initializes context, orchestrates phases, and verifies dependencies. | `execution_facade_api.py` |
| ` Common` | Cross-cutting utilities: immutable context, logging, path resolution, infrastructure enums. | `context_facade_common.py`, `logging_adapter_common.py`, `path_service_common.py`, `enum_registry_common.py` |
| `⚙️ Configuration` | YAML loading, merging, and strict validation. Model registries and read strategy contracts. | `build_factory_config.py`, `pipeline_task_dto_config.py`, `model_registry_config.py`, `read_strategy_repository_config.py`, `yaml_repository_config.py` |
| `💾 Data` | Data ingestion, persistence, and automatic profiling of raw/intermediate datasets. | `csv_loader_data.py`, `load_loader_data.py`, `persist_persister_data.py`, `profiling_profiler_data.py` |
| `🔄 Phase` | Independent CRISP-DM Phase 2–5 runners. Each reads previous artifacts; never re-executes completed work. | `phase2_...py` through `phase5_...py` |
| `🚀 Pipeline` | Problem-specific orchestrators (classification vs. regression). | `classification_runner_pipeline.py`, `regression_runner_pipeline.py` |
| ` Reporting` | Artifact persistence and EDA/visualization generation. | `artifact_persister_reporting.py`, `plots_generator_reporting.py` |

---

##  Project Directory Structure & Responsibilities

```
Project_MPPR-ALB_PHM_America_2024/
├── config/
│   ├── dataset/
│   │   └── dataset_config.yml          # 📦 Dataset metadata: CSV paths, read parameters, download URLs
│   ├── pipeline/
│   │   ├── active_profile.yml          # 🔀 Active profile (dev/prod): controls sample_rows, read mode, artifact naming
│   │   ├── base_pipeline_config.yml    # 🧱 Base CRISP-DM config: schema, physical warnings, validation strategies, domain shift checks
│   │   ├── classification_pipeline_config.yml # 🏷️ Overrides for classification (Phases 3–5): LightGBM, calibration, metrics
│   │   └── regression_pipeline_config.yml     # 📉 Overrides for regression (Phases 3–5): NGBoost, PDFs, target clipping
│   └── rules/                          # 📏 Physical/thermodynamic rules & safety thresholds (optional)
├── data/
│   └── raw/                            #  Raw CSVs (X_train, Y_train, X_validation, X_test)
├── docs/                               # 📑 Technical documentation & generated reports
├── notebooks/                          #  Jupyter notebooks for rapid EDA & load testing
├── src/phm_america_2024/
│   ├── api/
│   │   └── execution_facade_api.py     # 🚪 Single entry point: creates RunContext, configures logging, exposes run_phase2()→run_phase5()
│   ├── common/                         # 🔧 Cross-cutting infrastructure
│   │   ├── context_facade_common.py    # 🧠 RunContext: immutable state, artifact tracking, phase dirs, run summary
│   │   ├── dict_facade_common.py       # 🛠️ Safe dictionary manipulation helpers
│   │   ├── enum_registry_common.py     # 🏷️ Infrastructure enums (LogLevel) + anti-corruption normalization
│   │   ├── logging_adapter_common.py   # 📝 Namespaced logger, idempotent run configuration, file/console handlers
│   │   └── path_service_common.py      # 📍 Project root detection (pyproject.toml/.git) & absolute path resolution
│   ├── configuration/                  # ⚙️ Parsing, validation & registries
│   │   ├── build_factory_config.py     # 🏭 Merges base + specific + dataset + profile → valid BuiltConfig
│   │   ├── enum_registry_config.py     #  Domain enums: ProblemType, PhaseDir, ReadMode, StepsPhase
│   │   ├── model_registry_config.py    # 🤖 String-to-class mapping (sklearn/xgboost) + dynamic technique loading via importlib
│   │   ├── pipeline_task_dto_config.py # ✅ Pydantic DTOs: strict YAML structure validation before execution
│   │   ├── read_strategy_repository_config.py # 📖 Immutable contracts: ReadStrategyContract, DataSourceConfig
│   │   └── yaml_repository_config.py   #  YAML loading with OmegaConf, caching, active profile resolution
│   ├── data/                           # 💾 Ingestion & persistence
│   │   ├── csv_loader_data.py          # 📥 CSV loading with DuckDB/Pandas, chunking, dtype optimization
│   │   ├── download_extractor_data.py  #  Automatic PHM2024 dataset download & extraction
│   │   ├── load_loader_data.py         # 🔄 Load orchestrator: applies sample/chunked/full per contract
│   │   ├── persist_persister_data.py   # 💾 Persists DataFrames as Parquet/CSV with compression & metadata
│   │   └── profiling_profiler_data.py  # 🔍 Automatic data profiling: nulls, outliers, distributions, cardinality
│   ├── feature/                        # 🔬 Feature engineering: physical KPIs, interactions, transformations
│   ├── interpretation/                 # 📊 SHAP, feature importance, contribution analysis
│   ├── model/                          # 🧩 Training wrappers, early stopping, imbalance handling
│   ├── phase/                          # 🔄 Independent CRISP-DM phase runners
│   │   ├── phase2_understanding_runner_phase.py     # 📊 EDA, domain shift detection, physical limit checks
│   │   ├── phase3_preparation_runner_phase.py       # 🧹 Preprocessing, RobustScaler, Kelvin shift, null handling
│   │   ├── phase4_modeling_runner_phase.py          # 🤖 Modeling (NGBoost/LightGBM), isotonic calibration, CV
│   │   └── phase5_evaluation_and_interpretation_phase.py # 📈 Evaluation on validation/test, official metrics, reporting
│   ├── pipeline/                       # 🚀 Problem-specific orchestrators
│   │   ├── classification_runner_pipeline.py   # ️ Full binary classification workflow
│   │   └── regression_runner_pipeline.py         #  Full probabilistic regression workflow
│   ── reporting/                      # 📑 Persistence & visualization
│       ├── artifact_persister_reporting.py   # 📦 Persists JSON/Parquet with config hash, timestamp, run_id
│       └── plots_generator_reporting.py      # 📈 Generates PNGs: histograms, correlation, reliability diagrams, Q-Q plots
├── outputs/                            # 📂 Runtime: logs, data, models, reports (generated during execution)
├── .continuer.yml                      # 🐳 Environment/container configuration
├── .gitignore
├── poetry.lock / pyproject.toml        # 📦 Dependency management & reproducible environment
└── README.md                           # 📘 Project documentation
```

---

##  Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Non-Monolithic Execution** | Each phase checks `ctx.artifacts`. If present, it loads them. If missing, it raises a `RuntimeError` prompting execution of the prerequisite phase. Completed phases are never re-executed. |
| **Immutable Context** | `RunContext` is a frozen `dataclass`. DataFrames are optional in-memory caches; the source of truth is the `.parquet` files in `outputs/`. |
| **Fail-Fast Configuration** | YAML → OmegaConf → Pydantic DTO → `ReadStrategyContract`. Any typo or missing field aborts at `init_run_facade_api()`, not mid-run. |
| **Deterministic Traceability** | Artifact names encode profile, `sample_rows`, and method: `2.1.data_acquisition.prod_300000_stratified_train.parquet`. |
| **Model/Config Decoupling** | `ModelRegistry` maps YAML strings to actual classes. Switching algorithms only requires editing YAML, not phase code. |

---

## ️ Configuration & Environment

- **Dependency management**: `poetry install` (see `pyproject.toml`)
- **Execution profiles**: `config/pipeline/active_profile.yml` (`dev` → 7k rows, `prod` → 300k rows)
- **Notebook variables**: Injected via `notebook_vars={}` in `init_run_facade_api()`
- **Logs**: `outputs/logs/run_{task}_{dataset}_{run_id}.log` (console + file, idempotent per run)

---

## 📦 Output Structure

Every run produces a self-contained, reproducible snapshot under `outputs/`:

```
outputs/
└── runs/
    ── <task>/                          # Problem type (e.g., classification, regression)
        └── <dataset_key>/               # Dataset identifier
            └── <timestamp>/             # Run timestamp (YYYYMMDD_HHMMSS)
                ├── logs/                # Full execution log
                ├── phase2_data_understanding/
                ├── phase3_data_preparation/
                ├── phase4_data_modeling/
                └── phase5_evaluation_and_interpretation/
```

Each phase directory contains its specific artifacts (JSON reports, Parquet datasets, trained models, and PNG visualizations), ensuring full reproducibility and auditability.

--- 

