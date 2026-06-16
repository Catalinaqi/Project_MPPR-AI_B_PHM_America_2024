
---
# Models and Algorithms Used

## 1. Classical Algorithms

### 1.1 Second-order Polynomial Regression

* **Type:** Classical regression algorithm.
* **Objective and Use:** Used to estimate the target torque (`trq_target_pred`) from the engine's physical variables. This value is fundamental to subsequently compute the torque margin.
* **Particularity:** Does not include density altitude as a direct explanatory variable; instead, this factor is linearly compensated through interactions with other system variables.

### 1.2 k-NN (\*k-Nearest Neighbors\*)

* **Type:** Classical classification/regression algorithm.
* **Objective and Use:** Applied in the classification stage to adjust the temporal continuity of the time series and smooth predictions about the engine health state.
* **Particularity:** Acts as a temporal proximity filter to prevent abrupt or noisy changes in the diagnosis.

### 1.3 Logistic Regression

* **Type:** Classical classification algorithm.
* **Objective and Use:** Evaluated and tested during the experimentation phase among the set of classification algorithms, although it was not ultimately selected for the production model.

---

## 2. Modern Algorithms

### 2.1 LightGBM (\*Gradient Boosting Decision Trees\*)

* **Type:** Modern algorithm based on decision trees and gradient boosting.
* **Objective and Use:** The **main component** of the system for binary classification of the engine health state (`healthy` or `faulty`).
* **Particularity:** Hyperparameter tuning and optimization were performed automatically using the **Optuna** library.

### 2.2 Deep Learning

* **Type:** Modern algorithm based on neural networks.
* **Objective and Use:** Deep learning architectures were experimented with for the classification task. However, they were discarded for the final solution because LightGBM demonstrated clearly superior performance.

---

## 3. Domain-Knowledge-Based Processing

### 3.1 Rule-Based Processing

* **Type:** Classical analytical (non-ML) approach that complements the Machine Learning models.
* **Objective and Use:** Logical and engineering-specific rules were designed to correct the outputs of predictive models.
* **Particularity:** Focuses particularly on adjusting the real physical relationship between density altitude and torque margin, acting as a safety filter over the AI.

---

## 4. Summary of the Hybrid Solution

The proposed solution in this work stands out as a **hybrid algorithm** structured as follows:

1. **Data-driven Classification and Regression:** Combines the predictive potential of modern Machine Learning techniques (`LightGBM` tuned with `Optuna`, deep learning experiments) and traditional models (`polynomial regression`, `k-NN`, `logistic regression`).
2. **Business-rule Correction:** Applies an expert layer of rule-based processing to ensure model outputs respect the real physical behavior and engine constraints under critical operating conditions.
3. **Technology Mapping:**
* **Classical Models:** Polynomial regression, k-NN, logistic regression, and rule systems.
* **Modern Models:** LightGBM and Deep Learning.