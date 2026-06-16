
---

# Models and Algorithms Used

## 1. Classical Models

### 1.1 Gaussian Mixture Model (GMM)

* **Type:** Classical unsupervised learning algorithm (clustering).
* **Objective and Use:** Used to segment the system data into different regimes and engine operating states, allowing identification of behavior patterns without prior labels.

### 1.2 XGBoost (Extreme Gradient Boosting)

* **Type:** Classical supervised learning algorithm based on decision trees (boosting).
* **Objective and Use:** Employed as the base classifier for engine fault diagnosis. It is a robust, high-performance technique for handling structured tabular data.

---

## 2. Modern Models

### 2.1 Bayesian Regression

* **Type:** Advanced probabilistic approach for regression.
* **Objective and Use:** Implemented to predict the engine torque margin.
* **Particularity:** Unlike traditional regression, this approach models intrinsic prediction uncertainty, returning statistical confidence intervals that are critical for safety in aviation applications.

### 2.2 Deep Learning with Multi-Head Attention

* **Type:** Modern deep learning algorithm based on attention architectures.
* **Objective and Use:** Designed as an advanced model for fault diagnosis, capable of capturing complex long-range interactions among physical variables.
* **Robustness (Domain Adaptation):** To ensure the model performs correctly on previously unseen test data, transfer learning techniques were incorporated using Maximum Mean Discrepancy (MMD) and a Gradient Reversal Layer (GRL).

---

## 3. Ensemble Strategy (Model Ensemble)

To maximize accuracy and mitigate false diagnoses, the final solution does not rely on a single algorithm but on a **voting architecture**:

1. **Fusion of Criteria:** Outputs from the classical classifier `XGBoost` and the modern `Deep Learning with Multi-Head Attention` model are combined in parallel.
2. **Decision Logic:** If both agree, the failure probabilities are averaged. In case of disagreement, the architecture is programmed to automatically select the prediction with the highest statistical confidence.