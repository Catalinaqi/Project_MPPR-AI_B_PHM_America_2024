
---

# 📑 Indice

- [1. Output richiesti](#1-output-richiesti)
- [2. Come si svolge il lavoro (Pipeline metodologica)](#2-come-si-svolge-il-lavoro-pipeline-metodologica)
  - [2.1. Comprensione del dominio (Domain Understanding)](#21-comprensione-del-dominio-domain-understanding)
  - [2.2. Data Ingestion & Understanding](#22-data-ingestion--understanding)
  - [2.3. Estrazione e Data Preparation](#23-estrazione-e-data-preparation)
  - [2.4. Feature Engineering](#24-feature-engineering)
  - [2.5. Modellazione (Development)](#25-modellazione-development)
  - [2.6. Testing e Valutazione (Scoring)](#26-testing-e-valutazione-scoring)

---

## 1. Output richiesti (Cosa deve produrre il modello)

Il modello deve produrre i seguenti risultati finali:

* **Classificazione binaria:** Dello stato di salute + metrica di confidenza (0-1).
* **PDF (Probability Density Function) del torque margin:** Una regressione probabilistica. Il modello non deve restituire un numero singolo, ma i parametri di una distribuzione statistica (es. media e varianza).


---

## 2. Come si svolge il lavoro (Pipeline metodologica)
Per raggiungere l'obiettivo, il progetto seguirà un flusso di lavoro strutturato (ispirato allo standard CRISP-DM per il Data Mining), articolato nei seguenti step fondamentali:

### 2.1. Comprensione del dominio (Domain Understanding)
Analisi approfondita della documentazione tecnica sul sito ufficiale PHM per comprendere la fisica dei motori a turbina e la natura delle variabili operative.

### 2.2. Data Ingestion & Understanding
Importazione del dataset in ambiente Python (con ausilio di strumenti come DuckDB per l'analisi esplorativa) per comprendere le distribuzioni e gestire il formato shuffled dei dati.

### 2.3. Estrazione e Data Preparation
Pulizia dei dati, isolamento dei set di addestramento e validazione, e pre-processamento delle serie.

### 2.4. Feature Engineering
Calcolo e definizione di nuove feature (es. termini polinomiali, interazioni termodinamiche e filtri basati su regole fisiche) ottimizzate per i task di classificazione e regressione.

### 2.5. Modellazione (Development)
Sviluppo e addestramento dei due moduli principali:

Modulo di diagnostica: Classificatore.
Modulo di stima del degrado: Regressore probabilistico per il torque margin.

### 2.6. Testing e Valutazione (Scoring)
Validazione incrociata dei modelli e calcolo rigoroso della metrica di scoring ufficiale della competizione, la quale valuta simultaneamente la correttezza della predizione e l'affidabilità della misura di confidenza associata.