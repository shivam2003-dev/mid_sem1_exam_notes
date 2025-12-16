# Module 2: ANN & Perceptron

## Overview

This module covers Artificial Neural Networks (ANN) and the fundamental Perceptron model, including its learning algorithm and limitations.

---

## Artificial Neural Networks (ANN)

### Definition

**Artificial Neural Network (ANN)** is a computational model inspired by biological neural networks. It consists of interconnected nodes (neurons) organized in layers.

### Basic Structure

**Components**:
- **Input Layer**: Receives input data
- **Hidden Layers**: Process information (optional)
- **Output Layer**: Produces final output
- **Connections**: Weighted links between neurons

**Architecture**:

```
Input Layer    Hidden Layer    Output Layer
    x₁ ──────→     h₁    ──────→    y₁
    x₂ ──────→     h₂    ──────→    y₂
    x₃ ──────→     h₃
```

---

## Perceptron Model

### Single Perceptron

**Definition**: A single-layer neural network with one output neuron.

**Mathematical Model**:

$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w}^T \mathbf{x} + b)
$$

Where:
- $x_i$ = input $i$
- $w_i$ = weight for input $i$
- $b$ = bias term
- $f$ = activation function
- $y$ = output

### Activation Function

**Step Function (Binary)**:

$$
f(z) = \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
$$

**Sign Function (Bipolar)**:

$$
f(z) = \begin{cases}
+1 & \text{if } z \geq 0 \\
-1 & \text{if } z < 0
\end{cases}
$$

### Decision Boundary

For a perceptron with step function:

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

This defines a **hyperplane** that separates classes.

**For 2D case**:
$$
w_1 x_1 + w_2 x_2 + b = 0
$$

This is a **line** separating the two classes.

---

## Perceptron Learning Algorithm

### Algorithm Steps

**Given**:
- Training examples: $\{(\mathbf{x}^{(1)}, y^{(1)}), (\mathbf{x}^{(2)}, y^{(2)}), \ldots, (\mathbf{x}^{(m)}, y^{(m)})\}$
- Learning rate: $\alpha$ (typically 0.1 or 1.0)

**Steps**:

1. **Initialize**: Set weights and bias to small random values (or zeros)
   - $\mathbf{w} = [w_1, w_2, \ldots, w_n]^T$
   - $b = 0$ (or small random value)

2. **For each training example** $(\mathbf{x}^{(i)}, y^{(i)})$:
   
   a. **Compute output**:
   $$
   z^{(i)} = \mathbf{w}^T \mathbf{x}^{(i)} + b
$$
   $$
   \hat{y}^{(i)} = f(z^{(i)})
$$
   
   b. **Update weights if error**:
   $$
   \text{If } \hat{y}^{(i)} \neq y^{(i)} \text{ then:}
$$
   $$
   w_j := w_j + \alpha \cdot (y^{(i)} - \hat{y}^{(i)}) \cdot x_j^{(i)}
$$
   $$
   b := b + \alpha \cdot (y^{(i)} - \hat{y}^{(i)})
$$

3. **Repeat**: Until all examples are classified correctly (or max iterations)

### Update Rule (Vectorized)

**For binary classification** ($y \in \{0, 1\}$):

$$
\mathbf{w} := \mathbf{w} + \alpha \cdot (y^{(i)} - \hat{y}^{(i)}) \cdot \mathbf{x}^{(i)}
$$

$$
b := b + \alpha \cdot (y^{(i)} - \hat{y}^{(i)})
$$

**For bipolar classification** ($y \in \{-1, +1\}$):

$$
\mathbf{w} := \mathbf{w} + \alpha \cdot y^{(i)} \cdot \mathbf{x}^{(i)} \quad \text{(if misclassified)}
$$

$$
b := b + \alpha \cdot y^{(i)}
$$

!!! note "Key Point"
    The perceptron only updates weights when there's a misclassification. If the prediction is correct, no update occurs.

!!! tip "Learning Rate"
    A smaller learning rate ($\alpha = 0.1$) provides smoother convergence, while a larger rate ($\alpha = 1.0$) may converge faster but could overshoot.

---

## Perceptron Convergence Theorem

### Statement

**If the training data is linearly separable**, the perceptron learning algorithm will converge to a solution in a finite number of steps.

**Conditions**:
- Data must be linearly separable
- Learning rate $\alpha > 0$
- Weights initialized to zeros or small values

**Implications**:
- Guaranteed to find separating hyperplane if one exists
- Number of updates is bounded
- Convergence is guaranteed (not just probable)

!!! warning "Important"
    The perceptron will **NOT converge** if the data is **not linearly separable**. It will keep updating weights indefinitely.

---

## Limitations of Perceptron

### 1. Linearly Separable Data Only

**Problem**: Perceptron can only learn linearly separable patterns.

**Example - XOR Problem**:

| $x_1$ | $x_2$ | $x_1$ XOR $x_2$ |
|-------|-------|-----------------|
| 0     | 0     | 0               |
| 0     | 1     | 1               |
| 1     | 0     | 1               |
| 1     | 1     | 0               |

**Visualization**:
```
x₂
1 |  ●     ○
  |    ✗
0 |  ○     ●
  └─────────── x₁
    0    1
```

**No single line can separate the classes!**

**Solution**: Need multi-layer networks (MLP)

### 2. Binary Classification Only

- Single perceptron can only classify into 2 classes
- For multi-class, need multiple perceptrons or softmax

### 3. No Probabilistic Output

- Output is binary (0 or 1)
- Cannot provide confidence/probability
- Need sigmoid/softmax for probabilities

### 4. Sensitive to Feature Scaling

- Features should be normalized
- Large input values can cause issues

---

## Multi-Layer Perceptron (MLP)

### Solution to XOR Problem

**Architecture**:
- **Input Layer**: 2 neurons ($x_1$, $x_2$)
- **Hidden Layer**: 2 neurons (with non-linear activation)
- **Output Layer**: 1 neuron

**Key**: Hidden layer allows learning non-linear decision boundaries.

**XOR Solution with MLP**:

$$
h_1 = f(w_{11}x_1 + w_{12}x_2 + b_1)
$$

$$
h_2 = f(w_{21}x_1 + w_{22}x_2 + b_2)
$$

$$
y = f(w_1 h_1 + w_2 h_2 + b)
$$

With appropriate weights, this can solve XOR!

---

## Perceptron Example

### Problem

Classify points as class 1 or class 0:

- $(1, 1)$ → Class 1
- $(2, 2)$ → Class 1
- $(0, 0)$ → Class 0
- $(1, 0)$ → Class 0

### Solution

**Initialization**:
- $w_1 = 0$, $w_2 = 0$, $b = 0$
- $\alpha = 1.0$
- Activation: Step function

**Iteration 1**:
- Input: $(1, 1)$, Target: $1$
- $z = 0 \cdot 1 + 0 \cdot 1 + 0 = 0$
- $\hat{y} = f(0) = 1$ ✓ (Correct, no update)

**Iteration 2**:
- Input: $(2, 2)$, Target: $1$
- $z = 0 \cdot 2 + 0 \cdot 2 + 0 = 0$
- $\hat{y} = 1$ ✓ (Correct, no update)

**Iteration 3**:
- Input: $(0, 0)$, Target: $0$
- $z = 0$, $\hat{y} = 1$ ✗ (Wrong!)
- Update: $w_1 = 0 + 1 \cdot (0 - 1) \cdot 0 = 0$
- Update: $w_2 = 0 + 1 \cdot (0 - 1) \cdot 0 = 0$
- Update: $b = 0 + 1 \cdot (0 - 1) = -1$

**After updates**: $w_1 = 0$, $w_2 = 0$, $b = -1$

**Continue iterations until convergence...**

---

## Key Formulas Summary

### Perceptron Output

$$
y = f(\mathbf{w}^T \mathbf{x} + b)
$$

### Weight Update (Binary)

$$
w_j := w_j + \alpha \cdot (y^{(i)} - \hat{y}^{(i)}) \cdot x_j^{(i)}
$$

$$
b := b + \alpha \cdot (y^{(i)} - \hat{y}^{(i)})
$$

### Weight Update (Bipolar)

$$
\mathbf{w} := \mathbf{w} + \alpha \cdot y^{(i)} \cdot \mathbf{x}^{(i)} \quad \text{(if misclassified)}
$$

### Decision Boundary

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

---

## Important Points to Remember

✅ **Perceptron**: Single-layer neural network for binary classification

✅ **Learning Algorithm**: Updates weights only on misclassification

✅ **Convergence**: Guaranteed if data is linearly separable

✅ **Limitation**: Cannot solve non-linearly separable problems (e.g., XOR)

✅ **Solution**: Multi-layer networks (MLP) can solve XOR

✅ **Activation**: Step function for binary, sign function for bipolar

---

**Previous**: [Module 1 - Introduction](module1-introduction.md) | **Next**: [Module 3 - Linear NN Regression](module3-linear-nn-regression.md)

