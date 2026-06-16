
---

# Models and Algorithms Used

## 1. Classical Models (Implemented in this work)

### 1.1 Multivariable Polynomial Regression (\*Polynomial Regression\*)

* **Objective:** Predict the target torque ($T_{\text{target}}$) from the sensor variables (`mgt`, `oat`, `ias`, `pa`, `np/ng`).
* **Characteristics:**
  * Includes polynomial terms up to the third order and interaction terms between variables.
  * Produces a probabilistic distribution of the prediction through **empirical error sampling** (\*empirical error sampling\*), using training residuals to add uncertainty.

* **Justification:** Considered a classical approach because polynomial regression and error sampling are traditional techniques in statistics and Machine Learning.

### 1.2 Logistic Regression (\*Logistic Regression\*)

* **Objective:** Perform binary classification of engine health state (`nominal` or `faulty`).
* **Characteristics:**
  * Two independent models are developed depending on whether the `np/ng` ratio is less than or greater than 1, which allows linear separation of nominal and faulty cases.
  * Incorporates a loss function optimized to **strongly penalize false negatives**, reducing safety risk.

---

## 2. Modern Algorithms (Mentioned but NOT used)

### 2.1 Convolutional Neural Networks (CNN)

* **Context:** The text refers to previous works where CNNs were used to estimate capacity factors in wind farms and isolate faults in marine systems.
* **Decision:** In this work they were discarded in favor of simpler and more explainable models.

### 2.2 Advanced Probabilistic Models

* **Context:** Mentioned as theoretical alternatives are Gaussian Processes, Bayesian Neural Networks, and Ensemble Methods (bootstrap aggregation).
* **Decision:** Not implemented. The work aims for simplicity and robustness using empirical sampling and direct rules for selecting distributions.

---

## 3. Executive Summary

1. **Classic Approach:** The project relies on polynomial and logistic regression due to their robustness, transparency, and generalization ability, avoiding the overfitting risk of complex models.
2. **Rejection of Modern Methods:** Modern algorithms are analyzed as state-of-the-art alternatives but are not implemented in the final code.
3. **Probabilistic Output:** Despite the simplicity of the models, the final approach is probabilistic, generating output distributions and confidence estimates using simple and transparent methods.