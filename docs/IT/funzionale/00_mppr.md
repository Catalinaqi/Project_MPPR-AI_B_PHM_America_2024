

---

# 📑 Indice

- [1. Obiettivo](#1-obiettivo)
- [2. Dataset e Variabili](#2-dataset-e-variabili)
  - [2.1. Struttura del Dataset](#21-struttura-del-dataset)
  - [2.2. Variabili Sensoristiche (Features)](#22-variabili-sensoristiche-features)
  - [2.3. Descrizione del Dataset](#23-descrizione-del-dataset)
- [3. Definizione di Stato di Salute (Health) e Target](#3-definizione-di-stato-di-salute-health-e-target)
  - [3.1. Dal punto di vista fisico](#31-dal-punto-di-vista-fisico)
  - [3.2. Nel contesto del Machine Learning](#32-nel-contesto-del-machine-learning)

---

# Tema del progetto: PHM America 2024 – Health monitoring motori a turbina
# corso di laurea magistrale: manutenzione preventiva per la robotica e l'automazione intelligente
https://data.phmsociety.org/phm2024-conference-data-challenge/

## 1. Obiettivo

Sviluppare un modello di Machine Learning per stimare lo stato di salute di un motore a turbina di elicottero affrontando due task paralleli:

* **Classificazione:** Rilevamento dello stato del motore (stato nominale 0 vs guasto 1).
* **Regressione:** Stima quantitativa del degrado tramite il torque margin.

> **Nota:** 
> 1. In entrambi i casi, il modello deve fornire una misura rigorosa di confidenza/incertezza associata alla predizione.

---

## 2. Dataset e Variabili

### 2.1. Struttura del Dataset

Il dataset contiene dati operativi provenienti da 7 motori dello stesso tipo:

* **4** utilizzati per il training.
  Le osservazioni sono state mescolate (shuffled) e private 
  dell'identificatore temporale e del motore specifico.
* **3** trattenuti per il testing/validazione (valutazione della generalizzazione cross-asset).
  **CROSS ASSET GENERALIZATION:** il modello deve essere in grado di generalizzare a 
  motori non visti durante l'addestramento, ciò implica che deve apprendere pattern 
  fisici globali anziché particolarità di un motore specifico.

> **Nota critica del dataset:**
> Dati sotto restrizioni di incertezza.
> Il dataset dispone di variabili strumentate critiche.

### 2.2. Variabili Sensoristiche (Features)

Per ogni osservazione sono presenti le seguenti misure sensoristiche (features):

* **Temperatura aria esterna** (`oat`)
* **Temperatura media dei gas** (`mgt`)
* **Potenza disponibile** (`pa`)
* **Velocità indicata** (`ias`)
* **Potenza netta** (`np`)
* **Velocità del compressore** (`ng`)
* **Coppia fornita / misurata in tempo reale** (`Torque Measured`)

### 2.3. Descrizione del Dataset

**x_train**
- dati di input -> feature: x
- dati dei sensori
- **Shuffled**: dati mescolati
- non si può identificare a quale motore corrisponde ogni campione
- non si può identificare l'ordine temporale dei campioni
- Senza engine ID: gli ID dei motori sono stati eliminati

**y_train**
- dati dell'output atteso -> target: y
- dati delle risposte corrette
- dati specifici ed etichettati quando erano in guasto e quando no
- contiene esattamente i 2 obiettivi che il modello deve imparare a predire in futuro
  - `faulty`: label binario
  - `trq_margin`: target di regressione

**x_test**
- dati di feature: x

**x_validation**
- dati di feature: x

---

## 3. Definizione di Stato di Salute (Health) e Target

### 3.1. Dal punto di vista fisico

Il degrado si quantifica tramite il **Torque Margin**, definito come la differenza tra la Coppia Desiderata (Torque Target teorico) e la Coppia Fornita (Torque Measured), normalizzata in percentuale.

### 3.2. Nel contesto del Machine Learning

Il dataset fornisce due target specifici per l'addestramento:

* Un'**etichetta binaria** di stato (faulty/nominal).
* Il **valore continuo** del torque margin.

---
