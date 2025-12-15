# Module 2: Supervised Learning

## Overview

Supervised learning uses labeled training data to learn a function that maps inputs to outputs. This module covers two fundamental algorithms: **Linear Regression** and **Logistic Regression**.

---

## Linear Regression

### Introduction

**Linear Regression** is used to predict continuous numerical values. It assumes a linear relationship between input features and the target variable.

### Simple Linear Regression

**Model**: $y = \theta_0 + \theta_1 x$

Where:
- $y$ = predicted output (dependent variable)
- $x$ = input feature (independent variable)
- $\theta_0$ = y-intercept (bias term)
- $\theta_1$ = slope (weight)

### Multiple Linear Regression

**Model**: $y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n$

**Vectorized Form**: $y = \mathbf{\theta}^T \mathbf{x}$

Where:
- $\mathbf{\theta} = [\theta_0, \theta_1, \theta_2, \ldots, \theta_n]^T$ (parameters)
- $\mathbf{x} = [1, x_1, x_2, \ldots, x_n]^T$ (features with bias term)

### Cost Function (Mean Squared Error)

The cost function measures how far off our predictions are from actual values.

**For m training examples**:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Where:
- $h_\theta(x^{(i)}) = \theta^T x^{(i)}$ (prediction for example i)
- $y^{(i)}$ = actual value for example i
- $m$ = number of training examples

**Why $\frac{1}{2}$?**: Makes derivative cleaner (the 2 cancels out)

### Gradient Descent Algorithm

Gradient descent minimizes the cost function by iteratively updating parameters.

**Algorithm**:
1. Initialize parameters $\theta$ (usually to zeros or small random values)
2. Repeat until convergence:
   - Update all parameters simultaneously:
   
   $$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

**Update Rule**:

$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

**For $\theta_0$ (bias term)**:
$$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$$

**For $\theta_j$ (j > 0)**:
$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

**Parameters**:
- $\alpha$ (alpha) = learning rate (step size)
  - Too small: Slow convergence
  - Too large: May overshoot minimum, may not converge
- Number of iterations

**Vectorized Update**:
$$\theta := \theta - \alpha \frac{1}{m} X^T (X\theta - y)$$

### Learning Rate Selection

**Good Learning Rate**:
- Cost decreases smoothly
- Reaches minimum efficiently

**Too Small**:
- Very slow convergence
- May take many iterations

**Too Large**:
- Cost may increase
- May overshoot minimum
- May diverge (fail to converge)

**Rule of Thumb**: Try values like 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0

### Normal Equation (Alternative to Gradient Descent)

Closed-form solution (no iteration needed):

$$\theta = (X^T X)^{-1} X^T y$$

**When to use**:
- ✅ Small number of features (< 1000)
- ✅ Fast for small datasets
- ❌ Slow for large datasets (matrix inversion is O(n³))
- ❌ Doesn't work if $X^T X$ is not invertible

**Advantages of Gradient Descent**:
- Works well with large datasets
- More flexible (can use with other algorithms)

---

## Logistic Regression

### Introduction

**Logistic Regression** is used for binary classification (two classes: 0 and 1). Despite the name "regression," it's a classification algorithm.

### Hypothesis Function

**Sigmoid Function** (also called Logistic Function):

$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

Where $g(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.

**Properties of Sigmoid**:
- Output range: (0, 1)
- $g(0) = 0.5$
- As $z \to +\infty$, $g(z) \to 1$
- As $z \to -\infty$, $g(z) \to 0$
- S-shaped curve

**Interpretation**:
- $h_\theta(x)$ = probability that $y = 1$ given $x$
- $P(y = 1 | x; \theta) = h_\theta(x)$
- $P(y = 0 | x; \theta) = 1 - h_\theta(x)$

### Decision Boundary

**Classification Rule**:
- If $h_\theta(x) \geq 0.5$, predict $y = 1$
- If $h_\theta(x) < 0.5$, predict $y = 0$

Since $g(z) \geq 0.5$ when $z \geq 0$:
- Predict $y = 1$ if $\theta^T x \geq 0$
- Predict $y = 0$ if $\theta^T x < 0$

**Decision Boundary**: The line (or curve) where $\theta^T x = 0$

### Cost Function

**Why not use MSE?**
- MSE would give non-convex cost function
- Many local minima
- Gradient descent may not find global minimum

**Logistic Regression Cost Function**:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))]$$

**For single training example**:
$$Cost(h_\theta(x), y) = \begin{cases}
-\log(h_\theta(x)) & \text{if } y = 1 \\
-\log(1 - h_\theta(x)) & \text{if } y = 0
\end{cases}$$

**Intuition**:
- If $y = 1$: Cost is large when $h_\theta(x) \to 0$, cost is 0 when $h_\theta(x) \to 1$
- If $y = 0$: Cost is large when $h_\theta(x) \to 1$, cost is 0 when $h_\theta(x) \to 0$

### Gradient Descent for Logistic Regression

**Update Rule** (same form as linear regression!):

$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

**Vectorized**:
$$\theta := \theta - \alpha \frac{1}{m} X^T (g(X\theta) - y)$$

Where $g$ is the sigmoid function applied element-wise.

### Multiclass Classification (One-vs-All)

**Approach**:
1. Train $K$ separate logistic regression classifiers
2. For each class $k$, treat it as positive class and all others as negative
3. For prediction, choose class with highest $h_\theta^{(k)}(x)$

**Algorithm**:
- For each class $k = 1, 2, \ldots, K$:
  - Train classifier $h_\theta^{(k)}(x)$ to predict $y = k$ vs $y \neq k$
- To predict new example:
  - Compute $h_\theta^{(k)}(x)$ for all $k$
  - Choose class with maximum value

---

## Regularization

### Problem of Overfitting

**Overfitting**: Model fits training data too well but doesn't generalize to new data.

**Solutions**:
1. Reduce number of features
2. Regularization (keep all features but reduce magnitude)

### Regularized Cost Function

**Linear Regression with Regularization**:

$$J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2 \right]$$

**Logistic Regression with Regularization**:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

**Note**: Don't regularize $\theta_0$ (bias term)

### Regularized Gradient Descent

**For $j = 0$** (bias term, no regularization):
$$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$$

**For $j \geq 1$** (with regularization):
$$\theta_j := \theta_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} + \frac{\lambda}{m} \theta_j \right]$$

Can be rewritten as:
$$\theta_j := \theta_j \left(1 - \alpha \frac{\lambda}{m}\right) - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

**Regularization Parameter $\lambda$**:
- Large $\lambda$: Strong regularization, simpler model (may underfit)
- Small $\lambda$: Weak regularization, complex model (may overfit)
- $\lambda = 0$: No regularization

---

## Key Formulas Summary

### Linear Regression
- **Hypothesis**: $h_\theta(x) = \theta^T x$
- **Cost**: $J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$
- **Gradient**: $\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$

### Logistic Regression
- **Hypothesis**: $h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$
- **Cost**: $J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))]$
- **Gradient**: Same form as linear regression!

### Regularization
- **Regularized Cost**: Add $\frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$
- **Regularized Update**: Add $\frac{\lambda}{m} \theta_j$ to gradient

---

## Important Points to Remember

✅ **Linear Regression**: Predicts continuous values, uses MSE cost function

✅ **Logistic Regression**: Binary classification, uses sigmoid function, cross-entropy cost

✅ **Gradient Descent**: Iterative optimization, requires learning rate

✅ **Regularization**: Prevents overfitting by penalizing large parameters

✅ **Feature Scaling**: Important for gradient descent convergence

✅ **Bias Term**: $\theta_0$ is usually not regularized

---

**Previous**: [Module 1 - Introduction](module1-introduction.md) | **Next**: [Module 3 - Classification & Evaluation](module3-classification-evaluation.md)

