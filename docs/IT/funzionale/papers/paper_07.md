
---

# Models and Algorithms Used

## 1. Regression for Torque Margin Estimation

### 1.1 Gaussian Process Regression (GPR)

* **Type:** Classical/advanced algorithm (statistical and probabilistic method).
* **Configuration:** Implements a Matérn kernel combined with a space-filling sampling strategy (Max-min criterion).
* **Objective and Use:** Responsible for estimating the target torque used to compute the torque margin. The output of this regression is subsequently injected as a key predictive feature into the classification pipeline.

---

## 2. Engine Health Classification (Stacking Architecture)

Anomaly detection (healthy vs faulty engine) is solved using a stacking ensemble divided into two levels:

### 2.1 Base Models (Level 0)

#### A) Convolutional Neural Network (CNN)

* **Type:** Modern Deep Learning algorithm.
* **Use:** Deep neural network adapted to extract structural correlations and local latent relationship patterns within sensor data.

#### B) Multi-Layer Perceptron (MLP)

* **Type:** Modern Deep Learning algorithm.
* **Use:** Standard feedforward multi-layer network aimed at modeling and learning high-complexity nonlinear interactions.

#### C) XGBoost (Extreme Gradient Boosting)

* **Type:** Modern tree-based algorithm.
* **Use:** Gradient-optimized ensemble designed to maximize computational speed and accuracy on structured tabular data.

#### D) AdaBoost (Adaptive Boosting)

* **Type:** Classical boosting algorithm.
* **Use:** Traditional method that uses decision stumps to iteratively reweight misclassified samples.

### 2.2 Meta-Model (Level 1)

#### A) Logistic Regression

* **Type:** Classical linear algorithm.
* **Use:** Serves as the final stacking classifier. Receives the predicted probabilities from the four base models (CNN, MLP, XGBoost and AdaBoost) and performs an optimal linear combination to output the final binary diagnosis.

---

## 3. Technology Mapping and Summary

The methodological design emphasizes a strategic balance between statistical simplicity and deep learning capacity:

* **Classical Models:** Gaussian Process Regression (GPR), AdaBoost and Logistic Regression.
* **Modern Models:** Convolutional Neural Networks (CNN), Multi-Layer Perceptron (MLP) and XGBoost.

> **Ensemble Strategy:** The stacking architecture mitigates individual model bias. By combining the diverse nature of classical, statistical and Deep Learning models, the final system increases overall robustness and diagnostic accuracy under critical engine conditions.