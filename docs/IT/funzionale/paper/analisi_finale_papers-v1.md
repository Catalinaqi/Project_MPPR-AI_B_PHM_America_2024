
---

# 📑 Indice

- [Quadro Comparativo degli 8 Paper](#quadro-comparativo-degli-8-paper)
- [5. Tabella riassuntiva della tua proposta vs. paper](#5-tabella-riassuntiva-della-tua-proposta-vs-paper)
- [8. Confronto Architetturale](#8-confronto-architetturale)
  - [8.1. Analisi Critica](#81-analisi-critica)
- [Il Vincitore Indiscutibile: L'Approccio Ibrido "Frankenstein Efficiente"](#il-vincitore-indiscutibile-lapproccio-ibrido-frankenstein-efficiente)
- [Tabella Comparativa Consolidata (Paper 1-8)](#tabella-comparativa-consolidata-paper-1-8)
- [ALGORITMI CLASSICI E MODERNI: UNA SINTESI](#algoritmi-classici-e-moderni-una-sintesi)
- [Osservazioni finali](#osservazioni-finali)

---

### Quadro Comparativo degli 8 Paper

| # Paper | Approccio Filosofico Generale | Algoritmo Regressione *(Torque Margin)* | Algoritmo Classificazione *(Stato di Salute)* | Strategia di Incertezza / Post-processing | Costo Computazionale |
| --- | --- | --- | --- | --- | --- |
| **1** | **Classico e Spiegabile** | Regressione Polinomiale (3° ordine) | Regressione Logistica (Segmentata per `np/ng`) | Campionamento dei residui empirici in training | **Ultra Basso** (Millisecondi) |
| **2** | **Architettura a Cascata** | Bagging di Regressioni Lineari Polinomiali | **Random Forest** (Usa la predizione di torque come input) | Ensemble per media / Validazione Group-K-Fold | **Basso - Medio** (Secondi) |
| **3 e 4** | **Ibrido Moderno (Fisica + ML)** | Regressione Polinomiale (2° ordine) | **LightGBM** (Ottimizzato con Optuna) | **k-NN** per continuità temporale + **Filtro di Regole Fisiche** | **Medio** (Per l'ottimizzazione Optuna) |
| **5** | **Attenzione e Adattamento** | Regressione Bayesiana | **Multi-Head Attention** (Deep Learning) + XGBoost | Segmentazione con **GMM** + Adattamento di Dominio (MMD/GRL) | **Alto** (Richiede GPU) |
| **6** | **Boosting di Alta Efficienza** | Regressione Lineare Interattiva e Quadratica | **AdaBoost** (200 alberi di decisione semplici) | AutoML con ottimizzazione ASHA (Scarto di Reti Neurali) | **Basso** (Molto veloce) |
| **7** | **Stacking Avanzato** | **Gaussian Process Regression (GPR)** | **Stacking Ensemble** (CNN, MLP, XGBoost, AdaBoost) | Meta-modello finale basato su **Reggressione Logistica** | **Molto Alto** (GPR scala in $O(N^3)$) |
| **8** | **Ingegneria Fisica (KPI)** | Random Forest Regressor | Random Forest Classifier | Variabili di input elaborate con termodinamica (Efficienza) | **Medio** (Calcolo standard di alberi) |



## 5. Tabella riassuntiva della tua proposta vs. paper

| Dimensione | Tua soluzione | Paper migliore nel singolo aspetto | Tua valutazione |
|------------|---------------|-------------------------------------|-----------------|
| **Regressione probabilistica** | NGBoost (Normale) | Paper 7 (GPR) per calibrazione, ma troppo costoso | ✅ **Migliore** di Paper 1, 2, 6, 8 |
| **Classificazione** | LightGBM + cal. isotonica | Paper 5 (Stacking) per precisione, ma troppo complesso | ✅ **Migliore** di Paper 1, 2, 3/4, 8 |
| **Feature engineering** | KPI fisici + interazioni | Paper 8 (KPI termodinamici) | ✅ **Pari al meglio** |
| **Validazione** | Group‑K‑Fold con GMM | Paper 2 (Group‑K‑Fold) | ✅ **Pari al meglio** |
| **Costo computazionale** | Molto basso (CPU, secondi) | Paper 1 e 6 (ultra low) | ✅ **Tra i più bassi** |
| **Output probabilistico** | PDF nativa + prob. calibrate | Paper 5 e 7 (bayesiano) | ✅ **Molto buono** |


## 8. Confronto Architetturale

| Componente | Soluzione Originale | Piano A (NGBoost) | Piano B (BayesianRidge) |
| :--- | :--- | :--- | :--- |
| **Regressione probabilistica** | BayesianRidge + PolynomialFeatures(degree=2, interaction_only=False) | NGBoost(dist=Normal) | BayesianRidge + PolynomialFeatures(degree=2) |
| **Output regressione** | y_pred_mean, y_pred_std → normale | μ e σ² nativi | μ e σ² nativi |
| **Classificazione** | LGBMClassifier(is_unbalance=True, n_estimators=200, max_depth=7) | LightGBM + calibrazione isotonica | LightGBM + calibrazione isotonica |
| **Calibrazione** | CalibratedClassifierCV(method='isotonic', cv=5) | CalibratedClassifierCV(method='isotonic', cv=GroupKFold) | CalibratedClassifierCV(method='isotonic', cv=GroupKFold) |
| **Feature engineering** | Polinomiale degree=2 sulle features originali | KPI fisici (mgt/oat, ng², np/(ng·oat)) + interazioni quadratiche | KPI fisici (mgt/oat, ng², np/(ng·oat)) + interazioni quadratiche |
| **Cascata** | μ e σ² come feature del classificatore | μ e σ² come feature del classificatore | μ e σ² come feature del classificatore |
| **Regole fisiche** | Safety override (mgt < soglia, torque margin > soglia) | Post-processing opzionale con soglie da training set | Post-processing opzionale con soglie da training set |
| **Validazione** | K-Fold cv=5 standard | Group-K-Fold (k=5) + GMM (k=4) | Group-K-Fold (k=5) + GMM (k=4) |
| **Metriche regressione** | NLL | NLL, PICP, MPIW, RMSE (secondario) | NLL, PICP, MPIW, RMSE (secondario) |
| **Metriche classificazione** | Brier Score, Log-loss, reliability diagram | Brier Score, ECE, AUC-ROC, reliability diagram | Brier Score, ECE, AUC-ROC, reliability diagram |
| **Costo stimato** | < 5 secondi CPU | < 30 secondi CPU | < 5 secondi CPU |
| **Rischio principale** | cv=5 standard ottimista con dati shuffled, max_depth=7 non giustificato | Convergenza lenta, varianza instabile | Non-linearità catturate solo via feature manuali |
| **Trigger per cambio piano** | — | Varianza collassa o diverge in 20 minuti | — |

### 8.1. Analisi Critica
Le due differenze più importanti tra la soluzione originale e i due piani migliorati sono:

1.  **Strategia di Validazione:** Mentre il K-Fold standard risulta ottimista a causa dei dati *shuffled*, l'approccio con **Group-K-Fold e GMM** risulta metodologicamente più rigoroso e onesto.
2.  **Feature Engineering:** L'utilizzo di **KPI fisici** (es. mgt/oat, ng², np/(ng·oat)) apporta una maggiore quantità di segnale interpretabile rispetto all'espansione polinomiale pura applicata alle variabili originali.

---

### Il Vincitore Indiscutibile: L'Approccio Ibrido "Frankenstein Efficiente"

Per il tuo corso di **Manutenzione Preventiva**, l'opzione migliore non è copiare un singolo paper ciecamente, ma fare una fusione intelligente del meglio dei **Paper 2, 3/4 e 6**. 
Questo ti dà un modello con precisione da competizione, ma che gira in meno di 10 secondi sul tuo laptop senza stressare la CPU.

#### La Ricetta Vincente per il tuo Progetto:

1. **Per la Regressione (Margine di Coppia): Regressione Lineare Quadratica con Interazioni (Paper 6)**
* *Perché:* Il Paper 6 ha dimostrato che aggiungendo interazioni (es. $mgt \times oat$), una semplice regressione lineare raggiunge un $R^2 \approx 1.0$. Non sprecare cicli di CPU in Reti Neurali o GPR (Paper 7) per questo; la fisica del motore è altamente prevedibile con polinomi di secondo grado.

2. **Per la Classificazione (Guasti): LightGBM con iperparametri fissi (Paper 3 e 4)**
* *Perché:* LightGBM è incredibilmente più veloce e leggero di Random Forest (Paper 2) o XGBoost (Paper 5). Invece di perdere tempo eseguendo Optuna per ore, usa i parametri standard impostando `is_unbalance=True` per gestire lo squilibrio dei guasti.

3. **Per la Struttura: Connessione a Cascata (Paper 2)**
* *Perché:* Addestri prima la regressione lineare, calcoli il margine di coppia previsto e **lo inietti come una nuova colonna** nel LightGBM. Fisicamente, la perdita di coppia spiega il guasto, quindi questo passaggio logico dà una precisione elevata al classificatore.

4. **Per la Post-Valutazione: Filtro di Regole Fisiche (Paper 3 e 4)**
* *Perché:* Costo computazionale zero. Se il LightGBM dubita di un guasto ma i sensori indicano che il motore è freddo (`mgt` basso) o il margine di coppia è perfetto, una semplice regola `IF/ELSE` in Python corregge il tiro.

Questa combinazione è **semplice da codificare, matematicamente impeccabile per difenderti dai tuoi professori e ultra-potente nei risultati**. Tutta questa logica è già stata strutturata nel file `config.yml` che abbiamo progettato in precedenza. Pronti per lanciare le prime query in DuckDB e vedere come si comportano queste variabili?

---

# Tabella Comparativa Consolidata (Paper 1-8)

| Paper | Algoritmi Principali | Semplicità | Affidabilità | Precisione | Costo Computazionale |
| --- | --- | --- | --- | --- | --- |
| **1** | Reg. Polinomiale ($3^{\circ}$) + Reg. Logistica (split `np/ng`) + Campionamento residui | ⭐⭐⭐⭐⭐ Molto Alta | ⭐⭐⭐⭐ Alta (fisica) | ⭐⭐⭐ Media (lineare) | ⭐⭐⭐⭐⭐ Minimo (CPU, <1 min) |
| **2** | **Bagged Reg. Lineare + Random Forest (Cascata)** | ⭐⭐⭐⭐ Alta | ⭐⭐⭐⭐⭐ Molto Alta | ⭐⭐⭐⭐ Alta | ⭐⭐⭐⭐ Basso (scikit-learn) |
| **3/4** | Reg. Polinomiale ($2^{\circ}$) + LightGBM + Optuna + $k$-NN + Regole | ⭐⭐⭐ Media | ⭐⭐⭐⭐⭐ Molto Alta | ⭐⭐⭐⭐⭐ Molto Alta | ⭐⭐⭐⭐ Basso-Mod. (Optuna in dev) |
| **5** | GMM + XGBoost + Attention DL + MMD/GRL + Reg. Bayesiana | ⭐ Bassa | ⭐⭐⭐ Med-Alta | ⭐⭐⭐⭐⭐ Molto Alta | ⭐ Alto (DL + GPU) |
| **6** | Reg. Lineare (quad./interazione) + AdaBoost (200 alberi) + ASHA | ⭐⭐⭐⭐ Alta | ⭐⭐⭐⭐⭐ Molto Alta | ⭐⭐⭐⭐ Alta | ⭐⭐⭐⭐⭐ Minimo (Classici puri) |
| **7** | GPR (Matérn) + [CNN, MLP, XGBoost, AdaBoost] $\rightarrow$ Reg. Logistica | ⭐ Bassa | ⭐⭐⭐ Media (leakage) | ⭐⭐⭐⭐⭐ Molto Alta | ⭐ Alto ($O(N^3)$ per GPR) |
| **8** | MLP Probabilistico (TFP/NLL) + MLP Classificazione + Baseline CDF | ⭐⭐ Media | ⭐⭐⭐ Media (sovradattamento) | ⭐⭐⭐⭐ Alta | ⭐⭐⭐ Moderato-Alto (GPU / TFP) |



---

# ALGORITMI CLASSICI E MODERNI: UNA SINTESI

## Regressione (stima del torque margin / torque target)

### Algoritmi Classici
| Algoritmo | Paper di Riferimento |
|-----------|----------------------|
| **Reg. polinomiale multivariata** (3° ordine + interazioni) | Paper 1 |
| **Reg. polinomiale di 2° ordine** | Paper 3, 4 |
| **Reg. lineare interattiva e quadratica** | Paper 6 |
| **Bagging di regressioni lineari** (con polinomiali) | Paper 2 |
| **Bayesian Regression** (ritenuta classica per approccio probabilistico lineare) | Paper 5 |
| **Gaussian Process Regression (GPR)** (classico/avanzato) | Paper 7 |
| **Campionamento residui empirici** (empirical error sampling) | Paper 1 |

### Algoritmi Moderni
| Algoritmo | Paper di Riferimento |
|-----------|----------------------|
| **MLP probabilistica** (TensorFlow Probability, output media+std) | Paper 8 |
| **Multi‑task ANN** (due output: classificazione + regressione) | Paper 2 |
| **NGBoost** (non citato nei paper ma proposto nella tua soluzione) | Tuo suggerimento |
| **Deep Learning per regressione** (citato ma non adottato) | Paper 2, 8 |

---

## 2. Classificazione (stato di salute nominale / faulty)

### Algoritmi Classici
| Algoritmo | Paper di Riferimento |
|-----------|----------------------|
| **Regressione Logistica** (con split su `np/ng`, loss asimmetrica) | Paper 1, 3, 4 |
| **Random Forest** | Paper 2 |
| **AdaBoost** (200 alberi) | Paper 6, 7 |
| **Decision Trees / Bagged Trees** | Paper 6 |
| **k‑NN** (come filtro temporale) | Paper 3, 4 |
| **Classificazione basata su CDF** (confronto distribuzioni normali) | Paper 8 |
| **GMM** (clustering, usato come pre‑processing) | Paper 5 |

### Algoritmi Moderni
| Algoritmo | Paper di Riferimento |
|-----------|----------------------|
| **LightGBM** (con tuning Optuna) | Paper 3, 4 |
| **XGBoost** | Paper 5, 7 |
| **Stacking Ensemble** (CNN, MLP, XGBoost, AdaBoost → Logistic Regression) | Paper 7 |
| **Multi‑head Attention + Domain Adaptation** (MMD, GRL) | Paper 5 |
| **MLP (Deep / Wide)** per classificazione | Paper 8 |
| **CNN** (per estrazione pattern locali) | Paper 7 |

---

## Riepilogo schematico

| Categoria | Regressione | Classificazione |
|-----------|-------------|-----------------|
| **Classici** | Reg. polinomiale (2° e 3° ordine), Reg. lineare interattiva, Bagging di regressioni lineari, Bayesian Regression, GPR | Reg. Logistica, Random Forest, AdaBoost, k‑NN, CDF‑based, GMM, Decision Trees |
| **Moderni** | MLP probabilistica, Multi‑task ANN, NGBoost (proposto) | LightGBM, XGBoost, Stacking Ensemble, Multi‑head Attention, MLP, CNN |

---

## Osservazioni finali

- **Algoritmo classico più ricorrente** per regressione: **Reg. polinomiale** (Paper 1, 3, 4, 6).
- **Algoritmo classico più ricorrente** per classificazione: **Regressione Logistica** (Paper 1, 3, 4) e **AdaBoost** (Paper 6, 7).
- **Algoritmo moderno più performante** per classificazione (nei paper): **LightGBM** (Paper 3, 4) e **XGBoost** (Paper 5, 7).
- **NGBoost** (non presente nei paper) è un **moderno** per regressione probabilistica, con costo simile a LightGBM ma con output nativo di media e varianza.

Questa classificazione ti dà una mappa chiara per giustificare la scelta finale ibrida che hai proposto: **NGBoost (moderno) per regressione** e **LightGBM (moderno) + calibrazione per classificazione**, con supporto di feature engineering classico e validazione robusta.