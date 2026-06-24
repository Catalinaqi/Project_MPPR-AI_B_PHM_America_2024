
---

# Models and Algorithms Used

## 1. Classical Models

### 1.1 Linear Regression with Interaction and Quadratic Terms

* **Type:** Classical regression algorithm.
* **Objective and Use:** Trained to predict the design torque and compute the torque margin from the engine variables.
* **Particularity:** This sequential approach was selected because it outperformed feed\-forward neural networks, achieving an almost perfect fit with a coefficient of determination \(R^2 \approx 1.0\) and an extremely low root mean squared error (RMSE).

### 1.2 Decision Trees and Bagged Trees

* **Type:** Classical supervised learning models based on decision trees and bootstrap aggregations.
* **Objective and Use:** Used in the initial stages of the project to address classification and fault diagnosis tasks.
* **Evolution:** Although the bagged trees architecture provided strong initial results, the solution was later optimized by migrating to a boosting approach.

---

## 2. Modern Models

### 2.1 AdaBoost (Adaptive Boosting) over Decision Trees

* **Type:** Modern ensemble algorithm based on adaptive boosting.
* **Objective and Use:** Became the **final selected model** for engine classification and fault detection, outperforming the bagged trees model.
* **Functioning:** Iteratively increases the weight of examples that were misclassified in the previous round, penalizing errors to force better generalization.
* **Configuration:** Final training was configured with a strict limit of 200 trees to mitigate the risk of overfitting, achieving an error rate below 2%.

### 2.2 Automated Machine Learning with ASHA Optimization

* **Type:** Modern AutoML framework combined with the Asynchronous Successive Halving Algorithm (ASHA).
* **Objective and Use:** Employed as an advanced strategy for search space exploration, allowing efficient and parallel testing of multiple model families and their hyperparameters.
* **Result:** This automated approach scientifically identified that tree\-based ensembles were the optimal technology for the nature of this dataset.

### 2.3 Deep Learning

* **Type:** Modern algorithms based on deep, wide neural network architectures.
* **Objective and Use:** Deep models were evaluated using cloud compute and GPU acceleration.
* **Decision:** They were **completely discarded** for the final solution because they exhibited "overconfidence" in their wrong predictions and showed poor generalization on the test set.