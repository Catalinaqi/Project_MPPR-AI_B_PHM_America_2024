
---
# Models and Algorithms Used

## 1. Architectural Approach (Hybrid Model)

The system is designed under a hybrid architecture that merges two worlds:

1. **Data-driven Models:** Machine Learning algorithms responsible for finding complex patterns in the engine's physical variables.
2. **Domain Knowledge:** An expert layer that applies physical rules and operational constraints to ensure the viability of the diagnosis.

---

## 2. Classical Algorithms

### 2.1 Second-Order Polynomial Regression

* **Type:** Classical regression algorithm.
* **Objective and Use:** Responsible for estimating the target torque (`trq_target_pred`) based on the system's physical variables. This value is the cornerstone for subsequently computing the torque margin.

### 2.2 k-NN (\*k-Nearest Neighbors\*)

* **Type:** Classical classification and smoothing algorithm.
* **Objective and Use:** Used in the classification stage with a dual purpose: assign state labels and adjust temporal continuity of the data series, avoiding spurious alarms caused by noise.

### 2.3 Logistic Regression

* **Type:** Classical classification algorithm.
* **Objective and Use:** Evaluated during the experimentation phase as a binary classifier. Served as a performance validation tool, although it was not selected for final deployment.

---

## 3. Modern Algorithms

### 3.1 LightGBM (\*Gradient Boosting Decision Tree\*)

* **Type:** Modern supervised learning algorithm based on gradient-boosted decision trees.
* **Objective and Use:** Acts as the **main classification engine** to determine the engine health state.
* **Optimization:** All hyperparameter tuning was delegated to the Bayesian optimization framework **Optuna**.

### 3.2 Deep Learning

* **Type:** Modern algorithm based on neural networks.
* **Objective and Use:** Developed and tested as an advanced alternative for the classification task. Ultimately discarded in production because LightGBM provided clearly superior performance metrics.

---

## 4. Domain-Knowledge-Based Processing

### 4.1 Rule-Based Processing (Rule Systems)

* **Type:** Deterministic analytical (non-ML) approach.
* **Objective and Use:** Acts as a supervisory safety filter. Applies strict physical rules and operational thresholds on the torque margin and density altitude.
* **Impact:** Its main function is to intercept and correct Machine Learning model outputs in scenarios where pure predictions contradict thermodynamic laws or the engine's physical behavior.