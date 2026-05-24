Aquí tienes una estructura completa y lista para copiar en tu `README.md`. Está 
organizada por capas, con la responsabilidad exacta de cada archivo y una explicación clara de cómo interactúan sin crear dependencias monolíticas.

---

# 📘 Project Structure & Architecture

## 🏗️ Arquitectura por Capas
El proyecto sigue una arquitectura limpia y modular alineada con **CRISP-DM**. 
Cada capa tiene un contrato claro, permitiendo ejecutar fases de forma independiente 
sin forzar re-ejecuciones innecesarias.

| Capa | Responsabilidad | Archivos Clave |
|------|----------------|----------------|
| `🔌 API` | Punto de entrada único. Inicializa contexto, orquesta fases y verifica dependencias. | `execution_facade_api.py` |
| ` Common` | Utilidades transversales: contexto inmutable, logging, resolución de paths, enums. | `context_facade_*.py`, `logging_adapter_*.py`, `path_service_*.py`, `enum_registry_*.py` |
| `⚙️ Configuration` | Carga, fusión y validación estricta de YAMLs. Registros de modelos y estrategias de lectura. | `build_factory_config.py`, `pipeline_task_dto_config.py`, `model_registry_config.py`, `read_strategy_repository_config.py`, `yaml_repository_config.py` |
| ` Data` | Ingesta, persistencia y profiling automático de datos crudos/intermedios. | `csv_loader_data.py`, `load_loader_data.py`, `persist_persister_data.py`, `profiling_profiler_data.py` |
| `🔄 Phase` | Ejecutores independientes de Fases 2–5 de CRISP-DM. Cada uno lee artifacts previos, no re-ejecuta. | `phase2_...py` a `phase5_...py` |
| `🚀 Pipeline` | Orquestadores específicos por tipo de problema (clasificación vs regresión). | `classification_runner_pipeline.py`, `regression_runner_pipeline.py` |
| ` Reporting` | Persistencia de artifacts y generación de visualizaciones/EDA. | `artifact_persister_reporting.py`, `plots_generator_reporting.py` |

---

## 📂 Estructura del Proyecto y Responsabilidades

```
Project_MPPR-ALB_PHM_America_2024/
├── config/
│   ├── dataset/
│   │   └── dataset_config.yml          # 📦 Metadatos del dataset: rutas CSV, parámetros de lectura, URLs de descarga
│   ├── pipeline/
│   │   ├── active_profile.yml          # 🔀 Perfil activo (dev/prod): controla sample_rows, modo de lectura y nombres de artifacts
│   │   ├── base_pipeline_config.yml    # 🧱 Config base CRISP-DM: schema, warnings físicos, estrategias de validación, domain shift checks
│   │   ├── classification_pipeline_config.yml # 🏷️ Overrides para clasificación (Fases 3–5): LightGBM, calibración, métricas
│   │   └── regression_pipeline_config.yml     # 📉 Overrides para regresión (Fases 3–5): NGBoost, PDFs, clipping de targets
│   └── rules/                          # 📏 Reglas físicas/termodinámicas y umbrales de seguridad (opcional)
├── data/
│   └── raw/                            #  CSVs crudos (X_train, Y_train, X_validation, X_test)
├── docs/                               # 
├── notebooks/                          # 🧪 Jupyter notebooks para EDA rápido y pruebas de lectura
├── src/phm_america_2024/
│   ├── api/
│   │   └── execution_facade_api.py     # 🚪 Entry point único: crea RunContext, configura logging, expone run_phase2()→run_phase5()
│   ├── common/                         # ️ Infraestructura transversal
│   │   ├── context_facade_common.py    # 🧠 RunContext: estado inmutable, tracking de artifacts, dirs por fase, summary
│   │   ├── dict_facade_common.py       # 🔧 Helpers para manipulación segura de diccionarios
│   │   ├── enum_registry_common.py     # 🏷️ Enums de infraestructura (LogLevel) + normalización anti-corruption
│   │   ├── logging_adapter_common.py   # 📝 Logger namespaced, configuración idempotente por run, handlers file/console
│   │   └── path_service_common.py      # 📍 Detección de project root (pyproject.toml/.git) y resolución absoluta de paths
│   ├── configuration/                  # ⚙️ Parsing, validación y registros
│   │   ├── build_factory_config.py     # 🏭 Fusiona base + specific + dataset + profile → BuiltConfig válido
│   │   ├── enum_registry_config.py     # 📊 Enums de dominio: ProblemType, PhaseDir, ReadMode, StepsPhase
│   │   ├── model_registry_config.py    # 🤖 Mapeo string→clase (sklearn/xgboost) + carga dinámica de técnicas vía importlib
│   │   ├── pipeline_task_dto_config.py # ✅ Pydantic DTOs: validación estricta de estructura YAML antes de ejecución
│   │   ├── read_strategy_repository_config.py # 📖 Contratos inmutables: ReadStrategyContract, DataSourceConfig
│   │   └── yaml_repository_config.py   # 📄 Carga de YAMLs con OmegaConf, cacheo, resolución de perfil activo
│   ├── data/                           # 💾 Ingesta y persistencia
│   │   ├── csv_loader_data.py          # 📥 Lectura de CSVs con DuckDB/Pandas, chunking, dtype optimization
│   │   ├── download_extractor_data.py  # 🌐 Descarga automática del dataset PHM2024 y extracción
│   │   ├── load_loader_data.py         # 🔄 Orquestador de carga: aplica sample/chunked/full según contrato
│   │   ├── persist_persister_data.py   # 💾 Guarda DataFrames como Parquet/CSV con compresión y metadatos
│   │   ── profiling_profiler_data.py  # 🔍 Profiling automático: nulls, outliers, distribuciones, cardinalidad
│   ├── feature/                        # 🔬 Feature engineering: KPIs físicos, interacciones, transformaciones
│   ├── interpretation/                 # 📊 SHAP, feature importance, análisis de contribuciones
│   ├── model/                          # 🧩 Wrappers de entrenamiento, early stopping, manejo de desbalance
│   ├── phase/                          # 🔄 Ejecutores independientes de CRISP-DM
│   │   ├── phase2_understanding_runner_phase.py     # 📊 EDA, detección de domain shift, chequeo de límites físicos
│   │   ├── phase3_preparation_runner_phase.py       # 🧹 Preprocessing, RobustScaler, Kelvin shift, manejo de nulls
│   │   ├── phase4_modeling_runner_phase.py          # 🤖 Entrenamiento (NGBoost/LightGBM), calibración isotonica, CV
│   │   └── phase5_evaluation_and_interpretation_phase.py #  Evaluación en val/test, métricas oficiales, reportes
│   ├── pipeline/                       # 🚀 Orquestadores por tipo de problema
│   │   ├── classification_runner_pipeline.py   # 🏷️ Workflow completo de clasificación binaria
│   │   └── regression_runner_pipeline.py         # 📉 Workflow completo de regresión probabilística
│   └── reporting/                      # 📑 Persistencia y visualización
│       ├── artifact_persister_reporting.py   # 📦 Guarda JSON/Parquet con hash de config, timestamp, run_id
│       └── plots_generator_reporting.py      # 📈 Genera PNGs: histograms, correlation, reliability diagrams, Q-Q
├── outputs/                            # 📂 Runtime: logs, data, models, reports (generado en ejecución)
├── .continuer.yml                      # 🐳 Configuración de entorno/container
├── .gitignore
├── poetry.lock / pyproject.toml        # 📦 Gestión de dependencias y entorno reproducible
└── README.md                           # 📘 Documentación del proyecto
```

---

## 🔑 Principios de Diseño Clave

| Principio | Implementación |
|-----------|----------------|
| **Ejecución No Monolítica** | Cada fase verifica `ctx.artifacts`. Si existen, los carga. Si no, lanza `RuntimeError` pidiendo ejecutar la fase previa. Nunca re-ejecuta fases completadas. |
| **Contexto Inmutable** | `RunContext` es una `dataclass` congelada. Los DataFrames son opcionales en memoria; la fuente de verdad son los `.parquet` en `outputs/`. |
| **Fail-Fast en Configuración** | YAML → OmegaConf → Pydantic DTO → `ReadStrategyContract`. Cualquier typo o campo faltante aborta en `init_run_facade_api()`, no mid-run. |
| **Trazabilidad Determinística** | Nombres de artifacts codifican perfil, sample_rows y método: `2.1.data_acquisition.prod_300000_stratified_train.parquet`. |
| **Desacople Modelo/Config** | `ModelRegistry` mapea strings YAML a clases reales. Cambiar algoritmo solo requiere editar YAML, no tocar código de fase. |

---

---

##  Configuración y Entorno

- **Gestión de dependencias**: `poetry install` (ver `pyproject.toml`)
- **Perfiles de ejecución**: `config/pipeline/active_profile.yml` (`dev` → 7k rows, `prod` → 300k rows)
- **Variables de notebook**: Se inyectan vía `notebook_vars={}` en `init_run_facade_api()`
- **Logs**: `outputs/logs/run_{task}_{dataset}_{run_id}.log` (console + file, idempotente por run)

---

## Output Structure

Every run produces a self-contained, reproducible snapshot under `out/`:

```
outputs/runs/<task>/<dataset_key>/<timestamp>/
├── logs/                           # Full execution log
├── runs/                           # Full execution runs
    ├── task/                           # type problem: example: regresion
        ├── dataset_key/                           # 
            ├── timestamp/                           # 
                ├── phase2_data_understanding/
                ├── phase3_data_preparation/
                ├── phase4_data_modeling/
                └── phase5_evaluation_and_interpretation/
```
