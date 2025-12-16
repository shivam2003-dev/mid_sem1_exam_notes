# Module 4: Linear Neural Networks for Classification

## Overview

This module covers linear neural networks for classification tasks, including activation functions, loss functions, and multi-class classification.

---

## Linear Neural Network for Classification

### Architecture

**Structure**:
- **Input Layer**: $n$ input features
- **Output Layer**: 
  - **Binary**: 1 neuron with sigmoid activation
  - **Multi-class**: $K$ neurons with softmax activation

**Key Difference from Regression**:
- Uses **non-linear activation function** (sigmoid/softmax)
- Output represents **probability** of class membership
- Uses **cross-entropy loss** instead of MSE

---

## Activation Functions

### 1. Sigmoid Function

**Formula**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}
$$

**Properties**:
- Range: $(0, 1)$
- $\sigma(0) = 0.5$
- As $z \to +\infty$, $\sigma(z) \to 1$
- As $z \to -\infty$, $\sigma(z) \to 0$
- S-shaped curve (sigmoid curve)

**Derivative**:

$$
\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))
$$

**Use**: Binary classification output layer

```{admonition} Key Point
:class: note
The sigmoid function squashes any real number into the range (0, 1), making it perfect for representing probabilities.

```

### 2. Tanh Function (Hyperbolic Tangent)

**Formula**:

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{2}{1 + e^{-2z}} - 1
$$

**Properties**:
- Range: $(-1, 1)$
- $\tanh(0) = 0$
- As $z \to +\infty$, $\tanh(z) \to 1$
- As $z \to -\infty$, $\tanh(z) \to -1$
- Zero-centered (unlike sigmoid)

**Derivative**:

$$
\frac{d\tanh}{dz} = 1 - \tanh^2(z)
$$

**Use**: Hidden layers (better than sigmoid for hidden layers)

### 3. ReLU (Rectified Linear Unit)

**Formula**:

$$
\text{ReLU}(z) = \max(0, z) = \begin{cases}
z & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

**Properties**:
- Range: $[0, +\infty)$
- Non-linear but piecewise linear
- Computationally efficient
- Solves vanishing gradient problem

**Derivative**:

$$
\frac{d\text{ReLU}}{dz} = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

**Use**: Hidden layers (most common in modern deep learning)

```{admonition} Dead ReLU Problem
:class: warning
If a ReLU neuron outputs 0 for all inputs, it becomes "dead" and never activates. Use Leaky ReLU or initialization techniques to prevent this.

```

### 4. Softmax Function

**Formula** (for $K$ classes):

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Properties**:
- Outputs probability distribution: $\sum_{i=1}^{K} \text{softmax}(z_i) = 1$
- Each output is in range $(0, 1)$
- Largest $z_i$ gets highest probability

**Use**: Multi-class classification output layer

**Example** (3 classes):
- $z = [2, 1, 0.1]$
- $\text{softmax}(z) = [0.659, 0.242, 0.099]$
- Class 1 has highest probability (65.9%)

---

## Binary Classification

### Architecture

**Single Output Neuron** with sigmoid activation:

$$
\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

**Interpretation**: $\hat{y} = P(y = 1 | \mathbf{x})$

### Loss Function: Binary Cross-Entropy

**For single example**:

$$
L(\hat{y}, y) = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
$$

**For $m$ examples**:

$$
J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]
$$

**Intuition**:
- If $y = 1$: Loss is large when $\hat{y} \to 0$, loss is 0 when $\hat{y} \to 1$
- If $y = 0$: Loss is large when $\hat{y} \to 1$, loss is 0 when $\hat{y} \to 0$

### Gradient Computation

**For weight $w_j$**:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}
$$

**For bias $b$**:

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$

**Derivation** (using chain rule):

$$
\frac{\partial J}{\partial w_j} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w_j}
$$

- $\frac{\partial J}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$
- $\frac{\partial \hat{y}}{\partial z} = \hat{y}(1-\hat{y})$ (sigmoid derivative)
- $\frac{\partial z}{\partial w_j} = x_j$

**Combined**:

$$
\frac{\partial J}{\partial w_j} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) \cdot x_j = (\hat{y} - y) \cdot x_j
$$

```{admonition} Important
:class: tip
Notice that the gradient for binary cross-entropy with sigmoid has the same form as MSE with linear activation! This is a beautiful property.

```

---

## Multi-Class Classification

### Architecture

**$K$ Output Neurons** with softmax activation:

$$
\hat{y}_k = \text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

Where $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$ for class $k$.

**Output**: Probability distribution over $K$ classes

$$
\hat{\mathbf{y}} = [\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_K]^T
$$

with $\sum_{k=1}^{K} \hat{y}_k = 1$

### Loss Function: Categorical Cross-Entropy

**For single example** (one-hot encoded target):

$$
L(\hat{\mathbf{y}}, \mathbf{y}) = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)
$$

Since only one $y_k = 1$ (true class), this simplifies to:

$$
L(\hat{\mathbf{y}}, \mathbf{y}) = -\log(\hat{y}_{\text{true class}})
$$

**For $m$ examples**:

$$
J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{y}_k^{(i)})
$$

### Gradient Computation

**For weight $w_{jk}$** (weight from input $j$ to output $k$):

$$
\frac{\partial J}{\partial w_{jk}} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_k^{(i)} - y_k^{(i)}) \cdot x_j^{(i)}
$$

**For bias $b_k$**:

$$
\frac{\partial J}{\partial b_k} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_k^{(i)} - y_k^{(i)})
$$

```{admonition} Key Insight
:class: note
The gradient for softmax + cross-entropy has the same elegant form: prediction error times input!

```

---

## Decision Boundary

### Binary Classification

**Decision Rule**: Predict class 1 if $\hat{y} \geq 0.5$, else predict class 0.

Since $\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b)$:

$$
\sigma(\mathbf{w}^T \mathbf{x} + b) \geq 0.5
$$

$$
\mathbf{w}^T \mathbf{x} + b \geq 0
$$

**Decision Boundary**: $\mathbf{w}^T \mathbf{x} + b = 0$ (linear boundary)

### Multi-Class Classification

**Decision Rule**: Predict class with highest probability

$$
\text{Predicted Class} = \arg\max_k \hat{y}_k
$$

**Decision Boundaries**: Linear boundaries between classes (for linear NN)

---

## Key Formulas Summary

### Binary Classification

**Output**:
$$
\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b)
$$

**Loss**:
$$
J = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]
$$

**Gradient**:
$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}
$$

### Multi-Class Classification

**Output**:
$$
\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Loss**:
$$
J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{y}_k^{(i)})
$$

**Gradient**:
$$
\frac{\partial J}{\partial w_{jk}} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_k^{(i)} - y_k^{(i)}) \cdot x_j^{(i)}
$$

### Activation Function Derivatives

**Sigmoid**:
$$
\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))
$$

**Tanh**:
$$
\frac{d\tanh}{dz} = 1 - \tanh^2(z)
$$

**ReLU**:
$$
\frac{d\text{ReLU}}{dz} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}
$$

---

## Important Points to Remember

✅ **Sigmoid**: Binary classification output, range (0, 1)

✅ **Softmax**: Multi-class classification output, probability distribution

✅ **ReLU**: Best for hidden layers, solves vanishing gradient

✅ **Cross-Entropy Loss**: Used for classification (not MSE!)

✅ **Gradient Form**: Same elegant form $(\hat{y} - y) \cdot x$ for both binary and multi-class

✅ **Decision Boundary**: Linear for linear neural networks

---

**Previous**: [Module 3 - Linear NN Regression](module3-linear-nn-regression.md) | **Next**: [Module 5 - Deep Feedforward Neural Networks](module5-dfnn.md)

