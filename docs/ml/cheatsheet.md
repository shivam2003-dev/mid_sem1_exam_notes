# Machine Learning Cheat Sheet

Quick reference guide for all important formulas, concepts, and algorithms.

---

## 📐 Key Formulas

### Linear Regression

**Hypothesis Function**:
\[
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n = \theta^T x
\]

**Cost Function (MSE)**:
\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
\]

**Gradient Descent Update**:
\[
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}
\]

**Normal Equation**:
\[
\theta = (X^T X)^{-1} X^T y
\]

---

### Logistic Regression

**Sigmoid Function**:
\[
g(z) = \frac{1}{1 + e^{-z}}
\]

**Hypothesis**:
\[
h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
\]

**Cost Function (Cross-Entropy)**:
\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))]
\]

**Decision Boundary**: \(\theta^T x = 0\)

---

### Regularization

**Regularized Cost (Linear Regression)**:
\[
J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2 \right]
\]

**Regularized Cost (Logistic Regression)**:
\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
\]

**Regularized Gradient Update**:
\[
\theta_j := \theta_j \left(1 - \alpha \frac{\lambda}{m}\right) - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}
\]

---

### Evaluation Metrics

**Accuracy**:
\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

**Precision**:
\[
\text{Precision} = \frac{TP}{TP + FP}
\]

**Recall (Sensitivity)**:
\[
\text{Recall} = \frac{TP}{TP + FN} = \text{TPR}
\]

**Specificity**:
\[
\text{Specificity} = \frac{TN}{TN + FP} = \text{TNR}
\]

**F1-Score**:
\[
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
\]

**False Positive Rate**:
\[
\text{FPR} = \frac{FP}{FP + TN}
\]

**False Negative Rate**:
\[
\text{FNR} = \frac{FN}{FN + TP}
\]

---

### Decision Trees

**Entropy**:
\[
H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
\]

**Gini Impurity**:
\[
\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2
\]

**Information Gain**:
\[
\text{IG}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
\]

**Information Gain Ratio**:
\[
\text{IGR}(S, A) = \frac{\text{IG}(S, A)}{\text{SplitInfo}(S, A)}
\]

**Split Information**:
\[
\text{SplitInfo}(S, A) = -\sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)
\]

---

### K-Means Clustering

**Cost Function (Within-cluster sum of squares)**:
\[
J = \sum_{i=1}^{m} \sum_{k=1}^{K} w_{ik} ||x^{(i)} - \mu_k||^2
\]

**Centroid Update**:
\[
\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x
\]

**Euclidean Distance**:
\[
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
\]

---

### Principal Component Analysis (PCA)

**Covariance Matrix**:
\[
\Sigma = \frac{1}{m} X^T X
\]

**Variance Explained**:
\[
\text{Variance Explained} = \frac{\lambda_i}{\sum_{j=1}^{n} \lambda_j}
\]

**Cumulative Variance**:
\[
\text{Cumulative} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{n} \lambda_j}
\]

---

### Association Rules

**Support**:
\[
\text{Support}(A) = \frac{\text{Count}(A)}{N}
\]

**Confidence**:
\[
\text{Confidence}(A \to B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)} = P(B|A)
\]

**Lift**:
\[
\text{Lift}(A \to B) = \frac{\text{Confidence}(A \to B)}{\text{Support}(B)} = \frac{P(B|A)}{P(B)}
\]

---

## 🎯 Quick Reference

### Confusion Matrix

```
                Predicted
              Positive  Negative
Actual Positive    TP       FN
       Negative    FP       TN
```

### ROC Curve

- **X-axis**: False Positive Rate (FPR)
- **Y-axis**: True Positive Rate (TPR) = Recall
- **AUC**: Area under the curve (higher is better)
- **Perfect Classifier**: (0, 1) - top-left corner
- **Random Classifier**: Diagonal line (AUC = 0.5)

### Decision Tree Algorithms

| Algorithm | Impurity Measure | Features | Pruning |
|-----------|----------------|----------|---------|
| **ID3** | Entropy | Categorical only | No |
| **C4.5** | Information Gain Ratio | Categorical + Continuous | Yes |
| **CART** | Gini (classification) / MSE (regression) | Both | Yes |

### Learning Rate Guidelines

- **Too Small**: Slow convergence
- **Too Large**: May overshoot, may not converge
- **Good Range**: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0

### Regularization Parameter (λ)

- **Large λ**: Strong regularization → Simpler model (may underfit)
- **Small λ**: Weak regularization → Complex model (may overfit)
- **λ = 0**: No regularization

---

## 📝 Important Properties

### Entropy Properties
- Range: $[0, \log_2(c)]$
- Pure node: $H(S) = 0$
- Maximum (binary): $H(S) = 1$ when $p_1 = p_2 = 0.5$

### Gini Properties
- Range: $[0, 1 - \frac{1}{c}]$
- Pure node: $\text{Gini}(S) = 0$
- Maximum (binary): $\text{Gini}(S) = 0.5$ when $p_1 = p_2 = 0.5$

### Sigmoid Function
- Range: $(0, 1)$
- $g(0) = 0.5$
- As $z \to +\infty$, $g(z) \to 1$
- As $z \to -\infty$, $g(z) \to 0$

---

## 🔍 Algorithm Selection Guide

### When to Use What?

**Linear Regression**:
- ✅ Predicting continuous values
- ✅ Linear relationship between features and target
- ✅ Interpretable coefficients

**Logistic Regression**:
- ✅ Binary classification
- ✅ Need probability estimates
- ✅ Interpretable decision boundary

**Decision Trees**:
- ✅ Need interpretable model
- ✅ Mixed data types (categorical + numerical)
- ✅ Non-linear relationships

**K-Means**:
- ✅ Unsupervised clustering
- ✅ Known number of clusters
- ✅ Spherical clusters

**PCA**:
- ✅ Dimensionality reduction
- ✅ Data visualization
- ✅ Noise reduction

---

## ⚠️ Common Mistakes to Avoid

1. **Forgetting to standardize features** before gradient descent or PCA
2. **Not regularizing $\theta_0$** (bias term) in regularization
3. **Using accuracy for imbalanced datasets** (use F1-Score or AUC instead)
4. **Choosing K in K-Means** without domain knowledge or elbow method
5. **Not handling $\log(0)$** in entropy calculations (define as 0)
6. **Confusing Information Gain with Information Gain Ratio**
7. **Using MSE for logistic regression** (use cross-entropy instead)

---

## 💡 Exam Tips

1. **Memorize key formulas**: Entropy, Gini, Information Gain, Precision, Recall, F1-Score
2. **Understand when to use each metric**: Precision vs Recall tradeoff
3. **Know algorithm differences**: ID3 vs C4.5 vs CART
4. **Practice gradient descent iterations**: Show step-by-step calculations
5. **Understand regularization**: Effect of $\lambda$ on model complexity
6. **ROC Curve interpretation**: Higher AUC = Better classifier
7. **Decision tree construction**: Always show entropy/IG calculations

---

**Print this page for quick reference during exam preparation!** 📄

