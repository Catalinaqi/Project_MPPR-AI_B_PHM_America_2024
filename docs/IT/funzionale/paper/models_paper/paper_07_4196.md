
---

# Models and Algorithms Used

## 1. Modern Models (Neural Networks)

### 1.1 Probabilistic Neural Network (MLP for Probabilistic Regression)

* **Type:** Multi-Layer Perceptron with stochastic output.
* **Objective and Use:** Estimate the distribution of the target torque (`trq tgt`) to compute the torque margin (`trq mar`).
* **Implementation:** Built with **TensorFlow** and **TensorFlow Probability**. Uses dense layers and an output layer configured to model a Normal distribution (parametric prediction of mean and standard deviation).
* **Configuration:** Optimized with the `Adam` algorithm and a Negative Log-Likelihood loss.
* **Variables:**
  * *Inputs:* Engine environmental and operational parameters (`oat`, `mgt`, `pa`, `ias`, `np`, `ng`).
  * *Output:* Continuous probabilistic distribution of the target torque.

### 1.2 Neural Networks for Classification

* **Type:** Multiple MLP architectures for binary classification (`healthy` / `faulty`).
* **Implementation:** Built on **Keras (TensorFlow)**, evaluating two design variants:
  * **Deep Architecture:** Multiple consecutive hidden layers with ReLU activations and a single output neuron with Sigmoid activation.
  * **Wide Architecture:** Fewer hidden layers but a large number of neurons per layer.
  * *Multiclass Variant:* An alternative configuration using Softmax activation was evaluated for categorized faults.
* **Configuration:** Optimized with `Adam` and losses based on cross-entropy (binary and categorical).
* **Variables:**
  * *Inputs:* Estimated torque margin, measured torque, and environmental variables.
  * *Output:* Categorical probability of engine health state accompanied by a statistical confidence metric.

---

## 2. Classical Methods

### 2.1 Distribution-Based Classification

* **Type:** Traditional statistical-analytical approach (non-ML).
* **Objective and Use:** Implemented as the baseline solution for initial classification by contrasting torque margin distributions between nominal engines and engines with known faults.
* **Implementation:** Direct mathematical computation of Cumulative Distribution Functions (CDFs) and survival functions.
* **Output:** Binary classification and confidence calculation determined by the relative position of the torque margin with respect to reference distribution curves.

---

## 3. Algorithm Comparison Matrix

| Model | Algorithm / Features | Approach |
| --- | --- | --- |
| **Probabilistic MLP** | Neural network, dense layers, Adam optimizer, negative log-likelihood loss. | **Modern** |
| **MLP Classification** | Deep and wide neural networks, ReLU/Sigmoid activations, cross-entropy. | **Modern** |
| **Classical Statistical** | Modeling with CDFs, survival functions and comparison of normal distributions. | **Classical** |

---

## 4. Performance Summary

1. **Dominance of Deep Learning:** The final solution centers on Multi-Layer Perceptrons (MLP) for both regression and classification due to their flexibility and accuracy using TensorFlow and Keras.
2. **Baseline Evolution:** Although the classical statistical method based on comparing normal curves provided a transparent theoretical baseline, it was widely outperformed in performance metrics by the generalization capability of neural networks.