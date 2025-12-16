# 2024 EndSem Regular DNN Paper - Complete Solutions

## Question 1: Deep Network Backpropagation

### Problem Statement

Given a 3-layer network:
- Layer 1: 2 inputs → 3 hidden (ReLU)
- Layer 2: 3 hidden → 2 hidden (ReLU)  
- Layer 3: 2 hidden → 1 output (sigmoid)

**Weights**:
\[
\mathbf{W}^{[1]} = \begin{bmatrix} 1 & 2 \\ -1 & 1 \\ 0 & 1 \end{bmatrix}, \quad \mathbf{b}^{[1]} = \begin{bmatrix} 0 \\ 1 \\ -1 \end{bmatrix}
\]

\[
\mathbf{W}^{[2]} = \begin{bmatrix} 1 & -1 & 0 \\ 0 & 1 & -1 \end{bmatrix}, \quad \mathbf{b}^{[2]} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
\]

\[
\mathbf{W}^{[3]} = \begin{bmatrix} 2 & -1 \end{bmatrix}, \quad b^{[3]} = 0
\]

**Input**: $\mathbf{x} = [1, 1]^T$, **Target**: $y = 1$

**a)** Perform complete forward propagation.

**b)** Calculate binary cross-entropy loss.

**c)** Perform backpropagation to compute all gradients.

---

### Solution

#### Part (a): Forward Propagation

**Layer 1**:

\[
\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]} = \begin{bmatrix} 1 & 2 \\ -1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \\ -1 \end{bmatrix}
\]

\[
\mathbf{z}^{[1]} = \begin{bmatrix} 3 \\ 0 \\ 1 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 3 \\ 1 \\ 0 \end{bmatrix}
\]

\[
\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]}) = \begin{bmatrix} 3 \\ 1 \\ 0 \end{bmatrix}
\]

**Layer 2**:

\[
\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]} = \begin{bmatrix} 1 & -1 & 0 \\ 0 & 1 & -1 \end{bmatrix} \begin{bmatrix} 3 \\ 1 \\ 0 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix}
\]

\[
\mathbf{z}^{[2]} = \begin{bmatrix} 2 \\ 1 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}
\]

\[
\mathbf{a}^{[2]} = \text{ReLU}(\mathbf{z}^{[2]}) = \begin{bmatrix} 3 \\ 1 \end{bmatrix}
\]

**Layer 3**:

\[
z^{[3]} = \mathbf{W}^{[3]} \mathbf{a}^{[2]} + b^{[3]} = \begin{bmatrix} 2 & -1 \end{bmatrix} \begin{bmatrix} 3 \\ 1 \end{bmatrix} + 0 = 5
\]

\[
\hat{y} = \sigma(z^{[3]}) = \sigma(5) = \frac{1}{1 + e^{-5}} = 0.993
\]

**Answer**: $\hat{y} = 0.993$

---

#### Part (b): Loss Calculation

**Binary Cross-Entropy Loss**:

\[
J = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
\]

\[
J = -[1 \cdot \log(0.993) + 0 \cdot \log(0.007)] = -\log(0.993) = 0.007
\]

**Answer**: Loss $J = 0.007$

---

#### Part (c): Backpropagation

**Layer 3 Gradients**:

\[
\frac{\partial J}{\partial z^{[3]}} = \hat{y} - y = 0.993 - 1 = -0.007
\]

\[
\frac{\partial J}{\partial \mathbf{W}^{[3]}} = \frac{\partial J}{\partial z^{[3]}} \cdot (\mathbf{a}^{[2]})^T = -0.007 \cdot \begin{bmatrix} 3 & 1 \end{bmatrix} = \begin{bmatrix} -0.021 & -0.007 \end{bmatrix}
\]

\[
\frac{\partial J}{\partial b^{[3]}} = -0.007
\]

**Layer 2 Gradients**:

\[
\frac{\partial J}{\partial \mathbf{z}^{[2]}} = (\mathbf{W}^{[3]})^T \frac{\partial J}{\partial z^{[3]}} \odot \text{ReLU}'(\mathbf{z}^{[2]})
\]

\[
= \begin{bmatrix} 2 \\ -1 \end{bmatrix} \cdot (-0.007) \odot \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} -0.014 \\ 0.007 \end{bmatrix}
\]

\[
\frac{\partial J}{\partial \mathbf{W}^{[2]}} = \frac{\partial J}{\partial \mathbf{z}^{[2]}} (\mathbf{a}^{[1]})^T = \begin{bmatrix} -0.014 \\ 0.007 \end{bmatrix} \begin{bmatrix} 3 & 1 & 0 \end{bmatrix} = \begin{bmatrix} -0.042 & -0.014 & 0 \\ 0.021 & 0.007 & 0 \end{bmatrix}
\]

\[
\frac{\partial J}{\partial \mathbf{b}^{[2]}} = \begin{bmatrix} -0.014 \\ 0.007 \end{bmatrix}
\]

**Layer 1 Gradients**:

\[
\frac{\partial J}{\partial \mathbf{z}^{[1]}} = (\mathbf{W}^{[2]})^T \frac{\partial J}{\partial \mathbf{z}^{[2]}} \odot \text{ReLU}'(\mathbf{z}^{[1]})
\]

\[
= \begin{bmatrix} 1 & 0 \\ -1 & 1 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} -0.014 \\ 0.007 \end{bmatrix} \odot \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} -0.014 \\ 0.021 \\ 0 \end{bmatrix}
\]

\[
\frac{\partial J}{\partial \mathbf{W}^{[1]}} = \frac{\partial J}{\partial \mathbf{z}^{[1]}} \mathbf{x}^T = \begin{bmatrix} -0.014 \\ 0.021 \\ 0 \end{bmatrix} \begin{bmatrix} 1 & 1 \end{bmatrix} = \begin{bmatrix} -0.014 & -0.014 \\ 0.021 & 0.021 \\ 0 & 0 \end{bmatrix}
\]

\[
\frac{\partial J}{\partial \mathbf{b}^{[1]}} = \begin{bmatrix} -0.014 \\ 0.021 \\ 0 \end{bmatrix}
\]

**Answer**: All gradients computed as shown above.

---

## Question 2: CNN Architecture

### Problem Statement

Design a CNN for 28×28 grayscale image classification with 10 classes.

**Requirements**:
- First conv layer: 32 filters, 5×5
- Pooling: 2×2 max pooling
- Second conv layer: 64 filters, 3×3
- Pooling: 2×2 max pooling
- Fully connected: 128 neurons
- Output: 10 classes

**a)** Calculate the size of feature maps after each layer.

**b)** Calculate total number of parameters.

---

### Solution

#### Part (a): Feature Map Sizes

**Input**: 28×28×1

**Conv Layer 1** (32 filters, 5×5, stride=1, padding=0):
\[
\text{Output} = \frac{28 - 5 + 2 \times 0}{1} + 1 = 24
\]
**Size**: 24×24×32

**Max Pooling 1** (2×2, stride=2):
\[
\text{Output} = \frac{24 - 2 + 2 \times 0}{2} + 1 = 12
\]
**Size**: 12×12×32

**Conv Layer 2** (64 filters, 3×3, stride=1, padding=0):
\[
\text{Output} = \frac{12 - 3 + 2 \times 0}{1} + 1 = 10
\]
**Size**: 10×10×64

**Max Pooling 2** (2×2, stride=2):
\[
\text{Output} = \frac{10 - 2 + 2 \times 0}{2} + 1 = 5
\]
**Size**: 5×5×64

**Flatten**: 5 × 5 × 64 = 1600

**FC Layer**: 128 neurons

**Output**: 10 neurons

**Answer**: Feature map sizes calculated as shown above.

---

#### Part (b): Parameter Count

**Conv Layer 1**:
- Filters: 32 × (5×5×1 + 1 bias) = 32 × 26 = 832

**Conv Layer 2**:
- Filters: 64 × (3×3×32 + 1 bias) = 64 × 289 = 18,496

**FC Layer**:
- Weights: 1600 × 128 = 204,800
- Bias: 128
- Total: 204,928

**Output Layer**:
- Weights: 128 × 10 = 1,280
- Bias: 10
- Total: 1,290

**Total Parameters**:
\[
832 + 18,496 + 204,928 + 1,290 = 225,546
\]

**Answer**: Total parameters = **225,546**

---

## Summary

This paper covered:
1. ✅ Deep Network Forward and Backward Propagation
2. ✅ CNN Architecture Design and Calculations
3. ✅ Parameter Counting in CNNs

**Key Takeaways**:
- Practice forward/backward propagation step-by-step
- Understand CNN layer size calculations
- Know how to count parameters in each layer type

---

**Good luck with your exam!** 🎯

