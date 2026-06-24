# Strumenti di Machine Learning Utilizzati

## Dipendenze del Progetto

Il progetto si basa su un ecosistema di librerie Python specializzate per il PHM (*Prognostics and Health Management*). Di seguito viene fornita una descrizione dettagliata di ciascuna dipendenza, del suo ruolo nel progetto e delle motivazioni alla base della scelta.

### Dipendenze Principali

---

#### `pandas` (>=2.3.3, <3.0.0)

**Libreria per la manipolazione e l'analisi dei dati.**

Fornisce strutture dati flessibili come i DataFrame, indispensabili per:

- Caricare i dataset di training, validazione e test (formato CSV/Parquet).
- Eseguire operazioni di pulizia e trasformazione (filtraggio, rimozione outlier, gestione valori nulli).
- Creare nuove feature ingegnerizzate (KPI termodinamici, interazioni quadratiche).
- Aggregare e analizzare statistiche descrittive durante la fase di *Data Understanding*.

La versione 2.3.x garantisce compatibilità con le strutture dati di DuckDB e PyArrow per l'esecuzione di query efficienti su grandi volumi di dati.

---

#### `matplotlib` (>=3.10.7, <4.0.0)

**Libreria per la visualizzazione grafica.**

Utilizzata per:

- Tracciare istogrammi e boxplot delle distribuzioni delle variabili durante l'analisi esplorativa.
- Visualizzare il *Torque Margin* nel tempo e identificare pattern di degrado.
- Generare il *Reliability Diagram* per valutare la calibrazione delle probabilità del classificatore.
- Creare grafici di confronto tra valori reali e predetti (scatter plot, residui).

---

#### `scikit-learn` (>=1.7.2, <2.0.0)

**Libreria di machine learning classico.**

Costituisce la spina dorsale di molte componenti del progetto:

- **`PolynomialFeatures`**: espansione polinomiale di 2° grado per il feature engineering.
- **`CalibratedClassifierCV`**: calibrazione isotonica delle probabilità del classificatore LightGBM.
- **`GroupKFold`**: validazione incrociata che rispetta i gruppi (cluster GMM).
- **`StandardScaler`**: normalizzazione delle feature per modelli sensibili alla scala (es. regressione lineare).
- **`BrierScoreLoss`**, **`log_loss`**, **`roc_auc_score`**: metriche per la valutazione della classificazione.
- **`GaussianMixture`**: clustering dei regimi operativi come proxy dell'engine ID.

---

#### `pyyaml` (>=6.0.3, <7.0.0)

**Libreria per la lettura/scrittura di file YAML.**

Utilizzata per caricare i file di configurazione del progetto:

- `config/rules/ranges_quality_rules_config.yml`: definisce gli intervalli di validità fisica delle variabili.
- `config/dataset_schema.yml`: definisce lo schema e i metadati del dataset.

Fornisce un'interfaccia semplice e dichiarativa per separare la configurazione dal codice.

---

#### `joblib` (>=1.5.3, <2.0.0)

**Libreria per il salvataggio e il caricamento di modelli.**

Consente di:

- Serializzare i modelli addestrati (LightGBM, NGBoost, scaler, GMM) su disco.
- Caricare rapidamente i modelli pre-addestrati per la fase di inferenza.
- Eseguire il parallelismo leggero per operazioni di *batch processing*.

---

#### `duckdb` (>=1.5.3, <2.0.0)

**Database SQL embedded ottimizzato per l'analisi.**

Utilizzato per:

- Eseguire query SQL direttamente su file CSV/Parquet senza caricarli integralmente in memoria.
- Calcolare statistiche descrittive (COUNT, MIN, MAX, MEDIAN, quantili) sulle colonne del dataset.
- Eseguire aggregazioni complesse durante la fase di *Data Ingestion & Understanding*.

DuckDB è ideale per dataset di dimensioni medio-grandi (oltre 700.000 righe) grazie alla sua velocità e al basso footprint di memoria.

---

#### `omegaconf` (>=2.3.0, <3.0.0)

**Libreria per la gestione della configurazione gerarchica.**

Fornisce:

- Un sistema di configurazione fortemente tipizzato per parametri di modello, percorsi e iperparametri.
- Validazione automatica della struttura della configurazione.
- Supporto per l'override dei parametri da riga di comando.

Separa la logica sperimentale dal codice, facilitando la riproducibilità.

---

#### `pydantic` (>=2.13.4, <3.0.0)

**Libreria per la validazione dei dati tramite modelli Python.**

Utilizzata per:

- Definire schemi di dati fortemente tipizzati per i DataFrame (campi, tipi, vincoli).
- Validare i dati in ingresso e in uscita dai modelli.
- Garantire che le predizioni e le metriche rispettino i formati attesi.

---

#### `pyarrow` (>=24.0.0, <25.0.0)

**Libreria per la gestione efficiente di dati in formato colonnare.**

Fornisce:

- Lettura/scrittura ad alte prestazioni di file Parquet.
- Integrazione nativa con DuckDB per query su dati Arrow.
- Ottimizzazione della memoria per il trasferimento di grandi dataset.

---

#### `ngboost` (>=0.5.10, <0.6.0)

**Libreria per il boosting probabilistico.**

Costituisce il **regressore probabilistico principale** del progetto. NGBoost:

- Produce nativamente i parametri di una distribuzione di probabilità (media μ e varianza σ² per una distribuzione Normale).
- Fornisce incertezza **eteroschedastica** (dipende dalla regione dei dati).
- Ha un costo computazionale paragonabile a LightGBM/XGBoost (< 30 secondi su CPU).

La scelta di NGBoost è motivata dall'analisi comparativa di 7 paper della PHM Challenge, dove nessun approccio classico (BayesianRidge, GPR) o moderno (MLP probabilistico) offriva un output probabilistico nativo con un costo computazionale così basso.

---

#### `lightgbm` (>=4.6.0, <5.0.0)

**Libreria per il gradient boosting ad alte prestazioni.**

Costituisce il **classificatore principale** del progetto. LightGBM:

- È l'algoritmo più veloce tra i boosting su CPU grazie all'utilizzo di alberi con crescita per foglia (*leaf-wise*).
- Gestisce in modo efficiente dati con squilibrio di classe (`is_unbalance=True`).
- Supporta la calibrazione post-hoc tramite `CalibratedClassifierCV`.

Nel Paper 3 della competizione (1° classificato), LightGBM ha dimostrato di raggiungere score superiori a 0.99 se combinato con regole fisiche e k-NN.

---

### Dipendenze di Sviluppo

#### `python-dotenv` (>=1.2.2)

Carica variabili d'ambiente da file `.env` per la gestione sicura di percorsi e credenziali.

#### `ruff` (>=0.15.16)

Linter e formatter Python estremamente veloce (scritto in Rust). Sostituisce Flake8, isort e Black.

#### `mypy` (>=2.1.0)

Controllo statico dei tipi per individuare errori di tipo prima dell'esecuzione.

#### `deptry` (>=0.25.1)

Rileva dipendenze mancanti o non utilizzate nel progetto, mantenendo pulito l'ambiente.

---

## Riepilogo delle Scelte

| Categoria | Libreria | Ruolo nel Progetto |
|-----------|----------|--------------------|
| **Manipolazione dati** | `pandas`, `pyarrow` | Caricamento, pulizia, trasformazione |
| **Database analitico** | `duckdb` | Query SQL veloci per analisi esplorativa |
| **Configurazione** | `pyyaml`, `omegaconf`, `pydantic` | Gestione parametri e validazione dati |
| **Visualizzazione** | `matplotlib` | Grafici diagnostici e di valutazione |
| **Machine Learning classico** | `scikit-learn` | Feature engineering, validazione, metriche |
| **Regressore probabilistico** | `ngboost` | Stima di μ e σ² del Torque Margin |
| **Classificatore** | `lightgbm` | Diagnosi binaria sano/guasto |
| **Serializzazione** | `joblib` | Salvataggio e caricamento modelli |
| **Sviluppo** | `ruff`, `mypy`, `deptry` | Qualità del codice e type safety |

---

## Note sulla Versione

Le versioni sono state bloccate per garantire riproducibilità. L'ambiente Python è fissato a `>=3.10,<3.11` per garantire compatibilità con le versioni più recenti di DuckDB e PyArrow.

La scelta di non utilizzare TensorFlow o PyTorch è deliberata: il costo computazionale delle reti neurali non è giustificato per un dataset tabulare di ~750.000 righe, dove approcci come LightGBM e NGBoost offrono prestazioni competitive con tempi di addestramento di secondi su CPU.