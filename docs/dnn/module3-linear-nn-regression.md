# Module 3: Linear Neural Networks for Regression

## Overview

This module covers linear neural networks used for regression tasks, including forward propagation, backpropagation, and gradient computation.

---

## Linear Neural Network for Regression

### Architecture

**Structure**:
- **Input Layer**: $n$ input features
- **Output Layer**: 1 neuron (continuous output)
- **No Hidden Layers**: Direct mapping from input to output

**Mathematical Model**:

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b
$$

Where:
- $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$ (input vector)
- $\mathbf{w} = [w_1, w_2, \ldots, w_n]^T$ (weight vector)
- $b$ = bias term
- $\hat{y}$ = predicted output (continuous value)

### Difference from Perceptron

- **Perceptron**: Binary classification, step activation
- **Linear NN Regression**: Continuous output, linear activation (identity function)

---

## Forward Propagation

### Process

**Step 1**: Compute weighted sum

$$
z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b
$$

**Step 2**: Apply activation function (for regression, often identity)

$$
\hat{y} = f(z) = z \quad \text{(Linear/Identity activation)}
$$

**Vectorized Form** (for batch of $m$ examples):

$$
\mathbf{Z} = \mathbf{X} \mathbf{w} + \mathbf{b}
$$

$$
\hat{\mathbf{Y}} = \mathbf{Z}
$$

Where:
- $\mathbf{X}$ = $m \times n$ input matrix
- $\mathbf{w}$ = $n \times 1$ weight vector
- $\mathbf{b}$ = $m \times 1$ bias vector (all elements = $b$)
- $\hat{\mathbf{Y}}$ = $m \times 1$ output vector

---

## Loss Function

### Mean Squared Error (MSE)

**For single example**:

$$
L(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2
$$

**For $m$ training examples**:

$$
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
$$

$$
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (\mathbf{w}^T \mathbf{x}^{(i)} + b - y^{(i)})^2
$$

**Why $\frac{1}{2}$?**: Makes derivative cleaner (the 2 cancels out)

```{admonition} Important
:class: note
The factor $\frac{1}{2}$ doesn't change the optimal solution, but simplifies gradient calculations.

```

---

## Backpropagation Algorithm

### Goal

Minimize the loss function by computing gradients and updating weights.

### Gradient Computation

**For weight $w_j$**:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}
$$

**For bias $b$**:

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$

### Derivation

**Chain Rule Application**:

$$
\frac{\partial J}{\partial w_j} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w_j}
$$

**Step 1**: $\frac{\partial J}{\partial \hat{y}} = \hat{y} - y$

**Step 2**: $\frac{\partial \hat{y}}{\partial z} = 1$ (linear activation)

**Step 3**: $\frac{\partial z}{\partial w_j} = x_j$

**Combined**:

$$
\frac{\partial J}{\partial w_j} = (\hat{y} - y) \cdot 1 \cdot x_j = (\hat{y} - y) \cdot x_j
$$

**For $m$ examples**:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}
$$

---

## Gradient Descent Update

### Update Rules

**Weight Update**:

$$
w_j := w_j - \alpha \frac{\partial J}{\partial w_j}
$$

$$
w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}
$$

**Bias Update**:

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

$$
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$

**Vectorized Update**:

$$
\mathbf{w} := \mathbf{w} - \alpha \frac{1}{m} \mathbf{X}^T (\hat{\mathbf{Y}} - \mathbf{Y})
$$

$$
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$

Where:
- $\alpha$ = learning rate
- $\mathbf{Y}$ = target output vector

---

## Complete Training Algorithm

### Steps

1. **Initialize**: Set weights and bias to small random values (or zeros)

2. **Forward Propagation**:
   - Compute predictions: $\hat{y}^{(i)} = \mathbf{w}^T \mathbf{x}^{(i)} + b$ for all examples

3. **Compute Loss**:
   - $J = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$

4. **Backward Propagation**:
   - Compute gradients: $\frac{\partial J}{\partial w_j}$ and $\frac{\partial J}{\partial b}$

5. **Update Parameters**:
   - $w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$
   - $b := b - \alpha \frac{\partial J}{\partial b}$

6. **Repeat**: Steps 2-5 until convergence

---

## Numerical Example

### Given Data

| $x_1$ | $x_2$ | $y$ |
|-------|-------|-----|
| 1     | 2     | 5   |
| 2     | 3     | 8   |
| 3     | 1     | 7   |

### Step-by-Step Calculation

**Initialization**: $w_1 = 0.5$, $w_2 = 0.3$, $b = 0.1$, $\alpha = 0.1$

**Forward Pass** (Example 1: $x_1=1, x_2=2, y=5$):

$$
\hat{y} = 0.5 \cdot 1 + 0.3 \cdot 2 + 0.1 = 0.5 + 0.6 + 0.1 = 1.2
$$

**Error**: $1.2 - 5 = -3.8$

**Gradients**:
- $\frac{\partial J}{\partial w_1} = -3.8 \cdot 1 = -3.8$
- $\frac{\partial J}{\partial w_2} = -3.8 \cdot 2 = -7.6$
- $\frac{\partial J}{\partial b} = -3.8$

**Updates**:
- $w_1 := 0.5 - 0.1 \cdot (-3.8) = 0.5 + 0.38 = 0.88$
- $w_2 := 0.3 - 0.1 \cdot (-7.6) = 0.3 + 0.76 = 1.06$
- $b := 0.1 - 0.1 \cdot (-3.8) = 0.1 + 0.38 = 0.48$

**Repeat for all examples, then iterate until convergence.**

---

## Key Formulas Summary

### Forward Propagation

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b
$$

### Loss Function

$$
J = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
$$

### Gradients

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$

### Updates

$$
w_j := w_j - \alpha \frac{\partial J}{\partial w_j}
$$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

---

## Important Points to Remember

✅ **Linear NN Regression**: Direct mapping from input to continuous output

✅ **Forward Propagation**: Compute prediction using $\hat{y} = \mathbf{w}^T \mathbf{x} + b$

✅ **Loss Function**: Mean Squared Error (MSE)

✅ **Backpropagation**: Compute gradients using chain rule

✅ **Gradient Descent**: Update weights to minimize loss

✅ **Learning Rate**: Controls step size in weight updates

---

**Previous**: [Module 2 - ANN & Perceptron](module2-ann-perceptron.md) | **Next**: [Module 4 - Linear NN Classification](module4-linear-nn-classification.md)

