# Deep Neural Networks Cheat Sheet

Quick reference guide for all important formulas, concepts, and algorithms in DNN.

---

## 📐 Key Formulas

### Perceptron

**Output**:
\[
y = f(\mathbf{w}^T \mathbf{x} + b)
\]

**Weight Update (Binary)**:
\[
w_j := w_j + \alpha \cdot (y^{(i)} - \hat{y}^{(i)}) \cdot x_j^{(i)}
\]

**Weight Update (Bipolar)**:
\[
\mathbf{w} := \mathbf{w} + \alpha \cdot y^{(i)} \cdot \mathbf{x}^{(i)} \quad \text{(if misclassified)}
\]

**Decision Boundary**:
\[
\mathbf{w}^T \mathbf{x} + b = 0
\]

---

### Linear NN Regression

**Forward Propagation**:
\[
\hat{y} = \mathbf{w}^T \mathbf{x} + b
\]

**Loss Function (MSE)**:
\[
J = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
\]

**Gradients**:
\[
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}
\]

\[
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
\]

---

### Linear NN Classification

**Binary Classification (Sigmoid)**:
\[
\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
\]

**Loss (Binary Cross-Entropy)**:
\[
J = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]
\]

**Multi-Class (Softmax)**:
\[
\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
\]

**Loss (Categorical Cross-Entropy)**:
\[
J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{y}_k^{(i)})
\]

---

### Activation Functions

**Sigmoid**:
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

**Sigmoid Derivative**:
\[
\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))
\]

**Tanh**:
\[
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
\]

**Tanh Derivative**:
\[
\frac{d\tanh}{dz} = 1 - \tanh^2(z)
\]

**ReLU**:
\[
\text{ReLU}(z) = \max(0, z)
\]

**ReLU Derivative**:
\[
\frac{d\text{ReLU}}{dz} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}
\]

**Softmax**:
\[
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
\]

---

### Deep Feedforward Networks

**Forward Propagation (Layer $l$)**:
\[
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
\]

\[
\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})
\]

**Backpropagation**:

**Output Layer**:
\[
\frac{\partial J}{\partial \mathbf{z}^{[L]}} = \hat{\mathbf{y}} - \mathbf{y}
\]

**Hidden Layers**:
\[
\frac{\partial J}{\partial \mathbf{z}^{[l]}} = (\mathbf{W}^{[l+1]})^T \frac{\partial J}{\partial \mathbf{z}^{[l+1]}} \odot g'^{[l]}(\mathbf{z}^{[l]})
\]

**Weights**:
\[
\frac{\partial J}{\partial \mathbf{W}^{[l]}} = \frac{\partial J}{\partial \mathbf{z}^{[l]}} (\mathbf{a}^{[l-1]})^T
\]

**Bias**:
\[
\frac{\partial J}{\partial \mathbf{b}^{[l]}} = \frac{\partial J}{\partial \mathbf{z}^{[l]}}
\]

**L2 Regularization**:
\[
J_{\text{reg}} = J + \frac{\lambda}{2m} \sum_{l=1}^{L} ||\mathbf{W}^{[l]}||_F^2
\]

---

### Convolutional Neural Networks

**Convolution Operation**:
\[
\text{Output}(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \text{Input}(i+m, j+n) \cdot \text{Filter}(m, n) + b
\]

**Output Size**:
\[
\text{Output} = \left\lfloor \frac{\text{Input} - \text{Filter} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1
\]

**Max Pooling**:
\[
\text{Output}(i, j) = \max_{m,n \in \text{window}} \text{Input}(i+m, j+n)
\]

**Average Pooling**:
\[
\text{Output}(i, j) = \frac{1}{k^2} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \text{Input}(i+m, j+n)
\]

---

## 🎯 Quick Reference

### Perceptron Convergence

- **Converges**: If data is linearly separable
- **Doesn't Converge**: If data is not linearly separable (e.g., XOR)

### Activation Function Selection

| Layer Type | Recommended Activation |
|------------|----------------------|
| **Hidden Layers** | ReLU (most common) |
| **Binary Output** | Sigmoid |
| **Multi-class Output** | Softmax |
| **Regression Output** | Linear (identity) |

### Gradient Problems

**Vanishing Gradient**:
- **Cause**: Small activation derivatives (sigmoid, tanh)
- **Solution**: Use ReLU, proper initialization, batch norm

**Exploding Gradient**:
- **Cause**: Large weights, many layers
- **Solution**: Gradient clipping, proper initialization

### Weight Initialization

**He Initialization** (for ReLU):
\[
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]}}\right)
\]

**Xavier Initialization** (for tanh/sigmoid):
\[
W_{ij} \sim \mathcal{N}\left(0, \frac{1}{n^{[l-1]}}\right)
\]

---

## ⚠️ Common Mistakes to Avoid

1. **Using MSE for classification** (use cross-entropy instead)
2. **Using sigmoid in hidden layers** (use ReLU)
3. **Initializing all weights to zero** (breaks symmetry)
4. **Forgetting bias term** in calculations
5. **Wrong gradient sign** (should be $-\alpha$ for gradient descent)
6. **Not handling ReLU derivative** at $z=0$ (define as 0)
7. **Confusing convolution with correlation** in backpropagation

---

## 💡 Exam Tips

1. **Memorize activation derivatives**: Especially sigmoid and ReLU
2. **Practice backpropagation**: Show chain rule step-by-step
3. **Know perceptron algorithm**: Step-by-step weight updates
4. **Understand gradient flow**: How gradients propagate backward
5. **CNN calculations**: Practice convolution and pooling operations
6. **Know when to use what**: ReLU vs sigmoid, MSE vs cross-entropy

---

**Print this page for quick reference during exam preparation!** 📄

