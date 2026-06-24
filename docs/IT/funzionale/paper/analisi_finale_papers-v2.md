# PHM North America 2024 — Helicopter Turbine Engine Health Monitoring
## Análisis Comparativo de los 7 Papers de la Data Challenge

---

## 📑 Índice

- [Descripción del Problema](#descripción-del-problema)
- [Dataset](#dataset)
- [Métricas de Evaluación](#métricas-de-evaluación)
- [Cuadro Comparativo de los 7 Papers](#cuadro-comparativo-de-los-7-papers)
- [Análisis Individual por Paper](#análisis-individual-por-paper)
- [Comparativa de Algoritmos](#comparativa-de-algoritmos)
- [Tabla Comparativa Consolidada](#tabla-comparativa-consolidada)
- [Feature Engineering: Resumen Cruzado](#feature-engineering-resumen-cruzado)
- [Tu Propuesta vs. los Papers](#tu-propuesta-vs-los-papers)
- [Comparativa Arquitectural (Solución Propuesta)](#comparativa-arquitectural-solución-propuesta)
- [El Enfoque Híbrido Recomendado](#el-enfoque-híbrido-recomendado)
- [Algoritmos: Clasificación Clásico vs. Moderno](#algoritmos-clasificación-clásico-vs-moderno)
- [Observaciones Finales](#observaciones-finales)

---

## Descripción del Problema

La 2024 PHM Society North America Data Challenge propone estimar la salud de motores de turbina de helicóptero a través de dos tareas simultáneas:

**Tarea 1 — Regresión probabilística:** Estimar el *torque margin* (margen de par) como una función de densidad de probabilidad (PDF).

**Tarea 2 — Clasificación binaria:** Determinar si el motor es nominal (0) o defectuoso (1), con una métrica de confianza asociada.

La fórmula central del problema es:

```
Torque_Margin (%) = 100 × (Torque_measured − Torque_target) / Torque_target
```

Un margen de par bajo de forma persistente indica alta probabilidad de fallo del motor. La penalización por **falsos negativos de alta confianza** (predicción de motor sano cuando es defectuoso) es especialmente severa.

---

## Dataset

| Partición | Observaciones | Motores |
|-----------|--------------|---------|
| Training  | 742 624–742 625 | 4 de 7 |
| Test      | 21 436 | 1 de los 3 restantes |
| Validation | 21 436 | 1 de los 3 restantes |

**Variables de entrada (7 sensores):**

| Variable | Descripción |
|----------|-------------|
| `oat` | Outside Air Temperature (°C) |
| `mgt` | Mean Gas Temperature (°C) |
| `pa` | Pressure Altitude (feet) |
| `ias` | Indicated Airspeed (knots) |
| `np` | Net Power (%) |
| `ng` | Compressor Speed (%) |
| `trq_measured` | Measured Torque (%) |

**Variables objetivo:**
- `trq_target` — coppia obiettivo (calculada)
- `trq_margin` — margen de par (calculado)
- `faulty` — etiqueta de salud binaria (0 = sano, 1 = defectuoso)

---

## Métricas de Evaluación

**Puntuación de clasificación** (`score_ci`):
```
score_ci = ci              si ŷᵢ = yᵢ  (predicción correcta)
           −ci             si ŷᵢ = 1, yᵢ = 0  (falso positivo)
           −4cᵢ − cᵢ      si ŷᵢ = 0, yᵢ = 1  (falso negativo — penalidad máxima)
```

**Puntuación de regresión** (`score_ri`): intersección del valor real con la PDF predicha (normalizada para que el máximo de la PDF sea 1).

**Puntuación final:** media de todas las puntuaciones de regresión y clasificación.

---

## Cuadro Comparativo de los 7 Papers

| # | Título resumido | Ranking | Score Test | Score Validación | Regresión | Clasificación | Costo Computacional |
|---|-----------------|---------|-----------|-----------------|-----------|---------------|---------------------|
| **1** | Simple Probabilistic Approach (NTNU/DNV) | 🥈 2.º | 0.984 | — | Regresión Polinomial 3.º orden | Regresión Logística segmentada | ⚡ Mínimo |
| **2** | Ensemble Learning & ANN (PUC Paraná) | — | 0.9557 | 0.8870 | Bagged Reg. Lineal Polinomial | Random Forest en cascada | 🔵 Bajo–Medio |
| **3** | Torque Margin & Density Altitude (Mitsubishi) | 🥇 1.º | > 0.99 | > 0.99 | Reg. Polinomial 2.º orden | LightGBM + k-NN + Reglas | 🟡 Medio |
| **4** | Multi-Head Attention (Ajou University) | 🥉 3.º | 0.9858 / 0.9918 | 0.918 | Regresión Bayesiana (GNLLLoss) | XGBoost + Multi-Head Attention | 🔴 Alto (GPU) |
| **5** | AdaBoost Ensemble (MathWorks) | — | 0.9867 | — | Reg. Lineal interactiva/cuadrática | AdaBoost (Decision Trees) | ⚡ Mínimo |
| **6** | Stacking Ensemble GPR (anónimo) | — | — | — | GPR con Matérn Kernel | CNN + MLP + XGBoost + AdaBoost → Logistic Regression | 🔴 Muy Alto |
| **7** | Probabilistic Neural Network (Mad SoftMax) | — | — | — | MLP probabilístico (TFP/NLL) | Wide (Shallow) MLP + Softmax | 🟡 Moderado |

---

## Análisis Individual por Paper

### Paper 1 — Simple Yet Robust Probabilistic Approach
**Institución:** NTNU / DNV Group, Noruega · **Ranking: 2.º lugar**

**Idea central:** Demostrar que modelos simples bien diseñados superan a los complejos en generalización.

**Metodología en dos etapas:**
1. Regressione Polinomiale de 3.er orden para predecir `torque_target` → derivar el margen de par.
2. Regresión Logística binaria (dos modelos separados según `np/ng` > 1 o < 1) con función de pérdida asimétrica personalizada para penalizar falsos negativos.

**Feature engineering clave:** ratio `np/ng` como indicador de eficiencia del motor.

**Estrategia probabilística:** Muestreo empírico de residuos de entrenamiento; selección entre 4 distribuciones (Uniforme, Beta, Normal, Cauchy) mediante esquema de reglas. Todas normalizadas a PDF_max = 1.

**Función de pérdida personalizada:**
```
L_custom = (1−y)(1−ŷ) − (1−y)ŷ + yŷ − y(4(1−ŷ)¹¹ + (1−ŷ))
```

**Resultados:**
- Score test: **0.984** · Score regresión training: 0.999 · Score clasificación training: 0.867

---

### Paper 2 — Design Science: Ensemble Learning & ANN
**Institución:** PUC Paraná, Brasil

**Idea central:** Ciclo de desarrollo iterativo Design Science Research con cuantificación de incertidumbre.

**Solución final:** Ensemble en cascada:
1. **Bagged Linear Regression** (100 instancias) con features polinomiales de 2.º grado → predicción del margen de par.
2. **Random Forest** (100 árboles) para detección de fallos, usando el margen de par predicho como feature adicional.

**Comparativa explorada:** También se entrenó una ANN multi-tarea (PyTorch, ReLU, binary cross-entropy + negative log-likelihood), pero el ensemble clásico resultó más robusto.

**Validación:** K-fold + Group-fold cross-validation con Mini-batch K-means para agrupar por motor.

**Distribución de salida:** Gaussiana con desviación estándar mínima de 0.4.

**Resultados:**
- Score test: **0.9557** · Score validación: **0.8870**

---

### Paper 3 — Torque Margin & Density Altitude (Mitsubishi Electric)
**Institución:** Mitsubishi Electric Corporation, Japón · **Ranking: 1.er lugar**

**Idea central:** Combinar ML basado en datos con procesamiento basado en conocimiento del dominio; introducir la **altitud de densidad** como variable clave no presente en los datos originales.

**Variable derivada:**
```
da = 1.2376 × pa + 118.8 × oat − 1782
```

**Algoritmo híbrido de clasificación:**
1. **LightGBM** optimizado con Optuna (5-fold CV) sobre features incluyendo `da` y `trq_margin`.
2. **k-NN** (k=5) para capturar continuidad temporal entre puntos contiguos.
3. **Reglas físicas** para corregir predicciones según umbrales de altitud de densidad y margen de par:
    - Si `da < 500` Y `trq_margin < −0.01×da + 2.5` → **defectuoso**
    - Si `da ≥ 500` Y `trq_margin ≥ −0.01×da + 2.5` → **sano**

**Progresión de puntuaciones (test set):**

| Modelo | Score |
|--------|-------|
| Solo LightGBM | 0.7973 |
| + variables `da`, `trq_margin` | 0.8636 |
| + k-NN | 0.8641 |
| + reglas físicas | 0.9016 |
| + confianza fija = 1.0 | **0.9990** |

**Resultados:** Score test y validación > **0.99** — 1.er lugar.

---

### Paper 4 — Intelligent Fault Diagnosis with Multi-Head Attention
**Institución:** Ajou University, Corea del Sur · **Ranking: 3.er lugar**

**Idea central:** Usar densidad del aire (modelo ISA) como base de ingeniería de features; combinar XGBoost con un clasificador de atención multi-cabeza y adaptación de dominio.

**Cálculo de densidad del aire (modelo ISA):**
```
p/p₀ = (1 − a×h / T₀)^(g/(R×a))
ρ = P / (Rs × T)
```

**Regresión:** Regresión Bayesiana con Gaussian Negative Log-Likelihood Loss:
```
GNLLLoss = ½ × ((y − ŷ)² / σ̂² + log(σ̂²))
```

**Clasificación — ensemble de dos modelos:**
1. **XGBoost** (robusto en training, menos generalizable a nuevos dominios).
2. **Clasificador Multi-Head Attention** con técnicas de adaptación de dominio:
    - Maximum Mean Discrepancy (MMD)
    - Gradient Reversal Layer (GRL)

**Segmentación operativa:** Gaussian Mixture Model (GMM) de 2 componentes sobre `ng` y `np` ajustados por densidad de aire → foco en Cluster B presente en todos los datasets.

**Análisis de mapas de atención:** En motores defectuosos la atención se concentra en `trq_measured`, `mgt`, `np`; en motores sanos se distribuye uniformemente.

**Resultados:**
- Score test: **0.9858** (clasificación) / **0.9918** (regresión) · Score validación: 0.918

---

### Paper 5 — AdaBoost Ensemble (MathWorks)
**Institución:** MathWorks · **Herramienta:** MATLAB

**Idea central:** Flujo de trabajo sistemático con AutoML + feature engineering exhaustivo + AdaBoost para clasificación.

**Preprocesamiento:**
- Eliminación de 59 600 filas duplicadas (distribución 60-40 → 67-33).
- Balanceo con *Upsample Downweight* para llegar a 50-50.
- Expansión de 7 a 242 features (términos cuadráticos + interacciones lineales).
- Reducción a 18 features finales mediante *random noise probing* + `predictorImportance`.

**Features de ingeniería clave:**
```
ΔT_relative = (mgt − oat) / mgt    # degradación relativa de temperatura
ω = np / Torque_measured            # velocidad angular
```

**Modelo de regresión:** Regresión lineal secuencial con términos cuadráticos → R² = 1.0, 99% de residuos dentro de ±0.1%.

**PDF de salida:** Distribución **Uniforme** con flat-top (banda de 1.0%) → puntuación perfecta en regresión.

**Selección de clasificador:** AutoML con ASHA (*Asynchronous Successive Halving Algorithm*) → identificó ensembles de árboles como óptimos. Modelo final: **AdaBoost** (decision trees).

**Resultados:**
- Score inicial test: 0.9686 → Score final con AdaBoost: **0.9867**

---

### Paper 6 — Stacking Ensemble with GPR
**Institución:** no especificada

**Idea central:** Ensemble de stacking profundo para clasificación, con GPR para regresión probabilística de alta calidad usando estrategia *space-filling*.

**Regresión:** Gaussian Process Regression (GPR) con **Matérn Kernel** (ν = 3/2):
```
k_{3/2}(r) = (1 + √3·r/ℓ) × exp(−√3·r/ℓ)
```

**Estrategia space-filling (Max-min):**
```
x_new = argmax_x  min_{xᵢ ∈ X_train} ‖x − xᵢ‖₂
```
20 000 puntos seleccionados estratégicamente (2 000 iniciales + 18 000 secuenciales).

**Arquitectura de clasificación (stacking de 4 + 1):**

| Nivel | Modelo | Parámetros clave |
|-------|--------|-----------------|
| Base 1 | CNN 1D | 2 conv layers (32, 64 filtros), kernel 3 |
| Base 2 | MLP | 3 hidden layers (50, 100, 50), ReLU |
| Base 3 | XGBoost | lr=0.1, 500 estimadores, depth=5 |
| Base 4 | AdaBoost | 300 estimadores, depth=4, SAMME |
| Meta | Logistic Regression | C=0.9, liblinear |

**Selección de training set:** ~190 000 puntos filtrados por distancia euclidea mínima → mejor generalización.

**Limitación principal:** GPR escala en O(N³) — inviable sobre el dataset completo, necesita subconjunto.

---

### Paper 7 — Probabilistic Neural Network (Mad SoftMax)
**Institución:** Mad SoftMax team · **Framework:** TensorFlow + TensorFlow Probability

**Idea central:** Sistema integrado de redes neuronales; aprender `trq_target` como distribución para luego calcular `trq_margin` y usarlo como feature en clasificación.

**Modelo de regresión — MLP probabilístico:**
```
6 inputs → 256 nodos (sigmoid) → 256 nodos (sigmoid) → 2 nodos (linear) → IndependentNormal layer
```
Loss: Negative Log-Likelihood · Optimizer: Adam (β₁=0.9, β₂=0.99) · lr: 10⁻⁴ → 10⁻⁷

**Tres enfoques de clasificación explorados:**

1. **Método estadístico CDF-based** (sin ML): clasifica comparando `F_n(t)` vs `S_f(t)` — demasiada ambigüedad en la zona de solapamiento.
2. **Deep MLP:** 8 inputs → 3 × 20 nodos (ReLU) → 1 nodo (sigmoid).
3. **Wide (Shallow) MLP — modelo final:** 7 inputs → 32 nodos (sigmoid) → 2 nodos (softmax). Sorprendentemente pequeño y con el mejor rendimiento en test.

**Fórmulas de conversión de distribuciones:**
```
μ̂_mar = 100 × (trq_meas − μ̂_tgt) / μ̂_tgt
σ̂_mar = 100 × σ̂_tgt / μ̂_tgt
```

**Hallazgo clave:** El modelo Wide MLP tenía mayor loss en training pero mejor generalización en test → selección basada en test score, no en training loss.

---

## Comparativa de Algoritmos

### Regresión (estimación del torque margin / torque target)

**Algoritmos clásicos:**

| Algoritmo | Papers de referencia |
|-----------|----------------------|
| Regresión polinomial multivariada (3.er orden + interacciones) | Paper 1 |
| Regresión polinomial de 2.º orden | Paper 3 |
| Regresión lineal interactiva y cuadrática | Paper 5 |
| Bagging de regresiones lineales (con polinomiales) | Paper 2 |
| Regresión Bayesiana (BayesianRidge / GNLLLoss) | Paper 4 |
| Gaussian Process Regression (GPR) | Paper 6 |
| Muestreo de residuos empíricos | Paper 1 |

**Algoritmos modernos:**

| Algoritmo | Papers de referencia |
|-----------|----------------------|
| MLP probabilístico (TensorFlow Probability, output μ+σ) | Paper 7 |
| ANN multi-tarea (clasificación + regresión simultáneas) | Paper 2 |
| NGBoost (propuesto, no en los papers) | Propuesta propia |
| Deep Learning para regresión (explorado, no adoptado) | Papers 2, 7 |

---

### Clasificación (estado de salud nominal / defectuoso)

**Algoritmos clásicos:**

| Algoritmo | Papers de referencia |
|-----------|----------------------|
| Regresión Logística (split `np/ng`, loss asimétrica) | Paper 1 |
| Random Forest (en cascada) | Paper 2 |
| AdaBoost (200–300 árboles) | Papers 5, 6 |
| Decision Trees / Bagged Trees | Paper 5 |
| k-NN (como filtro de continuidad temporal) | Paper 3 |
| Clasificación basada en CDF (distribuciones normales) | Paper 7 |
| GMM (clustering como preprocesamiento) | Paper 4 |
| Reglas físicas (if/else sobre `da` y `trq_margin`) | Paper 3 |

**Algoritmos modernos:**

| Algoritmo | Papers de referencia |
|-----------|----------------------|
| LightGBM (con tuning Optuna) | Paper 3 |
| XGBoost | Papers 4, 6 |
| Stacking Ensemble (CNN + MLP + XGBoost + AdaBoost → LR) | Paper 6 |
| Multi-Head Attention + Domain Adaptation (MMD, GRL) | Paper 4 |
| MLP (Deep / Wide) para clasificación | Paper 7 |
| CNN 1D (extracción de patrones locales) | Paper 6 |

---

## Tabla Comparativa Consolidada

| Paper | Algoritmos principales | Simplicidad | Fiabilidad | Precisión | Costo computacional |
|-------|------------------------|-------------|------------|-----------|---------------------|
| **1** | Reg. Polinomial (3.º) + Reg. Logística (split `np/ng`) + muestreo residuos | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ Mínimo |
| **2** | Bagged Reg. Lineal + Random Forest (cascada) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ Bajo |
| **3** | Reg. Polinomial (2.º) + LightGBM + Optuna + k-NN + Reglas físicas | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ Bajo–Mod. |
| **4** | Reg. Bayesiana + XGBoost + Multi-Head Attention + MMD/GRL + GMM | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ Alto (GPU) |
| **5** | Reg. Lineal (cuad./interacción) + AdaBoost (200 árboles) + ASHA | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ Mínimo |
| **6** | GPR (Matérn) + [CNN + MLP + XGBoost + AdaBoost] → Reg. Logística | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ Muy Alto O(N³) |
| **7** | MLP Probabilístico (TFP/NLL) + Wide MLP (softmax) + Baseline CDF | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ Moderado |

---

## Feature Engineering: Resumen Cruzado

| Feature derivada | Descripción | Papers que la usan |
|------------------|-------------|-------------------|
| `np/ng` | Ratio potencia neta / velocidad compresora (eficiencia) | 1 |
| `da` (altitud de densidad) | `1.2376×pa + 118.8×oat − 1782` | 3 |
| `ρ` (densidad del aire) | Modelo ISA: `P/(Rs×T)` | 4 |
| Variables ajustadas por `ρ` | `np/ρ`, `ng/ρ`, `mgt/ρ`, `trq/ρ` | 4 |
| `ΔT_relative` | `(mgt − oat)/mgt` — degradación térmica relativa | 5 |
| `ω` (velocidad angular) | `np / trq_measured` | 5 |
| Términos cuadráticos `x²` | Para todos los predictores originales | 5, 6 |
| Interacciones lineales `xᵢ×xⱼ` | Hasta 242 features expandidas | 5 |
| `trq_margin` predicho | Como feature del clasificador (cascada) | 1, 2, 3, 4, 5, 6, 7 |

---

## Tu Propuesta vs. los Papers

| Dimensión | Tu solución | Mejor paper en ese aspecto | Evaluación |
|-----------|-------------|---------------------------|------------|
| **Regresión probabilística** | NGBoost (distribución Normal nativa) | Paper 6 (GPR) — mejor calibración pero O(N³) inviable | ✅ Superior a Papers 1, 2, 5, 7 |
| **Clasificación** | LightGBM + calibración isotónica | Paper 4 (Stacking) — más preciso pero mucho más complejo | ✅ Superior a Papers 1, 2, 5, 7 |
| **Feature engineering** | KPIs físicos + interacciones cuadráticas | Paper 5 (KPIs termodinámicos + 242 features) | ✅ Al nivel del mejor |
| **Validación** | Group-K-Fold con GMM (k=4) | Paper 2 (Group-K-Fold) | ✅ Al nivel del mejor |
| **Costo computacional** | Muy bajo (CPU, segundos) | Papers 1 y 5 (ultra bajo) | ✅ Entre los más bajos |
| **Output probabilístico** | PDF nativa + probabilidades calibradas | Papers 4, 6 (bayesiano) | ✅ Muy bueno |

---

## Comparativa Arquitectural (Solución Propuesta)

| Componente | Solución Original | Plan A (NGBoost) | Plan B (BayesianRidge) |
|:-----------|:-----------------|:-----------------|:----------------------|
| **Regresión probabilística** | BayesianRidge + PolynomialFeatures(degree=2) | NGBoost(dist=Normal) | BayesianRidge + PolynomialFeatures(degree=2) |
| **Output regresión** | `y_pred_mean`, `y_pred_std` → Normal | μ y σ² nativos | μ y σ² nativos |
| **Clasificación** | LGBMClassifier(is_unbalance=True, n_estimators=200, max_depth=7) | LightGBM + calibración isotónica | LightGBM + calibración isotónica |
| **Calibración** | CalibratedClassifierCV(method='isotonic', cv=5) | CalibratedClassifierCV(method='isotonic', cv=GroupKFold) | CalibratedClassifierCV(method='isotonic', cv=GroupKFold) |
| **Feature engineering** | Polinomial degree=2 sobre features originales | KPIs físicos (`mgt/oat`, `ng²`, `np/(ng·oat)`) + interacciones cuadráticas | KPIs físicos + interacciones cuadráticas |
| **Cascada** | μ y σ² como features del clasificador | μ y σ² como features del clasificador | μ y σ² como features del clasificador |
| **Reglas físicas** | Safety override (umbral `mgt`, umbral `trq_margin`) | Post-processing opcional con umbrales del training set | Post-processing opcional con umbrales del training set |
| **Validación** | K-Fold cv=5 estándar | Group-K-Fold (k=5) + GMM (k=4) | Group-K-Fold (k=5) + GMM (k=4) |
| **Métricas regresión** | NLL | NLL, PICP, MPIW, RMSE (secundario) | NLL, PICP, MPIW, RMSE (secundario) |
| **Métricas clasificación** | Brier Score, Log-loss, reliability diagram | Brier Score, ECE, AUC-ROC, reliability diagram | Brier Score, ECE, AUC-ROC, reliability diagram |
| **Costo estimado** | < 5 s CPU | < 30 s CPU | < 5 s CPU |
| **Riesgo principal** | cv=5 estándar optimista con datos shuffled | Convergencia lenta, varianza inestable | No linealidades capturadas solo vía features manuales |

### Análisis crítico de las diferencias clave

Las dos diferencias más importantes entre la solución original y los planes mejorados son:

**Estrategia de validación:** El K-Fold estándar resulta optimista con datos shuffled. El enfoque con Group-K-Fold + GMM es metodológicamente más riguroso y honesto, porque respeta la estructura de identidad de los motores.

**Feature engineering:** Los KPIs físicos (`mgt/oat`, `ng²`, `np/(ng·oat)`) aportan señal interpretable y alineada con la termodinámica del motor, superando la expansión polinomial pura sobre variables crudas.

---

## El Enfoque Híbrido Recomendado

Para un proyecto de **Mantenimiento Predictivo**, la mejor opción no es copiar un paper al pie de la letra, sino una fusión inteligente de lo mejor de los **Papers 2, 3 y 5**. El resultado es un modelo con precisión de competición que corre en menos de 10 segundos en CPU.

### La receta ganadora

**1. Regresión — Regresión Lineal Cuadrática con Interacciones (Paper 5)**

El Paper 5 demostró que añadiendo interacciones (`mgt×oat`, etc.) una regresión lineal alcanza R² ≈ 1.0. No tiene sentido gastar ciclos de CPU en GPR (Paper 6) o redes neuronales para esta tarea; la física del motor es altamente predecible con polinomios de segundo grado.

**2. Clasificación — LightGBM con hiperparámetros fijos (Paper 3)**

LightGBM es mucho más rápido y ligero que Random Forest (Paper 2) o XGBoost (Paper 4). En lugar de correr Optuna durante horas, usar parámetros estándar con `is_unbalance=True` para gestionar el desbalance de clases.

**3. Estructura — Conexión en cascada (Paper 2)**

Entrenar primero la regresión, calcular el margen de par predicho e **inyectarlo como nueva columna** en LightGBM. Físicamente, la pérdida de par explica el fallo, por lo que este paso lógico eleva la precisión del clasificador.

**4. Post-procesamiento — Filtro de reglas físicas (Paper 3)**

Costo computacional cero. Si LightGBM duda en un fallo pero los sensores indican motor frío (`mgt` bajo) o margen de par perfecto, una regla `IF/ELSE` simple en Python corrige la predicción.

---

## Algoritmos: Clasificación Clásico vs. Moderno

| Categoría | Regresión | Clasificación |
|-----------|-----------|---------------|
| **Clásico** | Reg. polinomial (2.º y 3.er orden), Reg. lineal interactiva, Bagging de regs. lineales, Regresión Bayesiana, GPR | Reg. Logística, Random Forest, AdaBoost, k-NN, CDF-based, GMM, Decision Trees, Reglas físicas |
| **Moderno** | MLP probabilístico, ANN multi-tarea, NGBoost (propuesto) | LightGBM, XGBoost, Stacking Ensemble, Multi-Head Attention, MLP (Deep/Wide), CNN 1D |

---

## Observaciones Finales

**Algoritmo clásico más recurrente para regresión:** Regresión polinomial (Papers 1, 2, 3, 5) — la física del motor es lo suficientemente regular como para ser capturada por polinomios de bajo orden.

**Algoritmo clásico más recurrente para clasificación:** Regresión Logística (Papers 1, 3) y AdaBoost (Papers 5, 6).

**Algoritmo moderno más eficaz para clasificación:** LightGBM (Paper 3 — 1.er lugar) y XGBoost (Papers 4, 6).

**Hallazgo transversal:** En todos los papers ganadores, el *torque margin predicho* se usa como feature del clasificador (arquitectura en cascada). Ningún paper de alto rendimiento trata regresión y clasificación como tareas completamente independientes.

**NGBoost** (no presente en los papers) es un algoritmo moderno para regresión probabilística con costo similar a LightGBM pero con output nativo de media y varianza — una alternativa válida al BayesianRidge para quien busca calibración sin el coste computacional del GPR.

---

*Basado en los 7 papers de la PHM North America 2024 Conference Data Challenge.*