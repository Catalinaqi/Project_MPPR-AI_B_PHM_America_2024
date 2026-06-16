
---

# Models and Algorithms Used

## 1. Classical Models (Ensemble Learning)

### 1.1 Random Forest

- Objective: Used for both classification (fault detection) and regression (prediction of the torque margin).
- Characteristics: An ensemble model based on multiple decision trees. Noted for robustness, versatility, and widespread use in traditional Machine Learning problems.

### 1.2 Bagging (Bootstrap Aggregating)

- Objective: Employed specifically for the regression task.
- Characteristics: Consists of training a set of multiple linear regression models on different random samples drawn from the original dataset, then averaging their results to reduce variance.

### 1.3 Cascaded Ensemble Model (Cascaded Model)

- Objective: Defines the specific architecture designed to solve the project's central problem.
- Characteristics: Sequentially connects linear regression with bagging and the random forest. The torque margin prediction (output of the regression) is automatically injected as an additional feature into the classification model to refine fault detection.

---

## 2. Modern Models (Deep Learning)

### 2.1 Multi-task Artificial Neural Network (ANN)

- Objective: An artificial neural network designed and trained to solve classification and regression tasks simultaneously.
- Characteristics:
  - Architecture: Input layer, dense hidden layers (64 units with ReLU activation), and two independent output branches.
  - Outputs: One output for binary classification and another for probabilistic regression (predicting the mean and standard deviation of the torque margin).
  - Optimization: Trained using a weighted sum of loss functions: binary cross-entropy for classification and negative log-likelihood for regression and uncertainty estimation.

### 2.2 Uncertainty-aware Deep Learning

- Context: Advanced deep learning techniques focused on uncertainty quantification, such as Monte Carlo dropout or direct prediction of output distribution functions.
- Decision: Discussed in the text but not directly implemented in the final model architecture.

---

## 3. Key Observations

1. Winning Architecture: The final selected model was the Cascaded Ensemble (Bagged Linear Regression with polynomial features \u2192 Random Forest for classification), leveraging the first model's output as a predictor for the second.
2. Classical Superiority: In this particular scenario, classical ensemble learning methods demonstrated better performance, consistency, and robustness compared to the modern ANN.
3. Validation Strategy: Success relied heavily on rigorous data processing (polynomial feature generation and cleaning) combined with strong validation techniques (K-fold and Group-fold) to ensure generalization and proper uncertainty estimation.