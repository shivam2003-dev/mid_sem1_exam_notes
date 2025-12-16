# Module 5: Deep Feedforward Neural Networks (DFNN)

## Overview

This module covers deep feedforward neural networks (multi-layer perceptrons), including forward propagation, backpropagation through multiple layers, and techniques to handle gradient problems.

---

## Deep Feedforward Neural Networks

### Definition

**Deep Feedforward Neural Network (DFNN)** is a multi-layer neural network where information flows in one direction (forward) from input to output through multiple hidden layers.

**Also called**:
- Multi-Layer Perceptron (MLP)
- Deep Neural Network (DNN)
- Fully Connected Network

### Architecture

**Structure**:

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Hidden Layer L → Output Layer
    x₁            h₁¹              h₁²                        h₁^L          y₁
    x₂            h₂¹              h₂²                        h₂^L          y₂
    ...           ...               ...                        ...
    xₙ            hₘ¹              hₘ²                        hₘ^L
```

**Key Components**:
- **Input Layer**: $n$ neurons (features)
- **Hidden Layers**: $L$ layers with varying number of neurons
- **Output Layer**: $K$ neurons (for $K$ classes) or 1 neuron (for regression)

---

## Forward Propagation

### Single Layer Forward Pass

**For layer $l$**:

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})
$$

Where:
- $\mathbf{W}^{[l]}$ = weight matrix for layer $l$ (size: $n^{[l]} \times n^{[l-1]}$)
- $\mathbf{a}^{[l-1]}$ = activations from previous layer
- $\mathbf{b}^{[l]}$ = bias vector for layer $l$
- $g^{[l]}$ = activation function for layer $l$
- $\mathbf{z}^{[l]}$ = pre-activation (linear combination)
- $\mathbf{a}^{[l]}$ = post-activation (output of layer $l$)

**Notation**:
- $n^{[l]}$ = number of neurons in layer $l$
- $\mathbf{a}^{[0]} = \mathbf{x}$ (input)
- $\mathbf{a}^{[L]} = \hat{\mathbf{y}}$ (output)

### Complete Forward Propagation

**For a network with $L$ layers**:

1. **Input**: $\mathbf{a}^{[0]} = \mathbf{x}$

2. **For each layer $l = 1, 2, \ldots, L$**:
   $$
   \mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$
   $$
   \mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})
$$

3. **Output**: $\hat{\mathbf{y}} = \mathbf{a}^{[L]}$

### Example: 3-Layer Network

**Architecture**: Input (2) → Hidden (3) → Output (1)

**Layer 1 (Hidden)**:
$$
\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}
$$
$$
\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]})
$$

**Layer 2 (Output)**:
$$
\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}
$$
$$
\hat{y} = \sigma(\mathbf{z}^{[2]}) \quad \text{(for binary classification)}
$$

---

## Backpropagation Algorithm

### Goal

Compute gradients for all layers to update weights using gradient descent.

### Chain Rule for Multiple Layers

**Key Insight**: Gradients flow backward from output to input.

### Backpropagation Steps

**Given**: Loss function $J$ and network output $\hat{\mathbf{y}}$

**Step 1**: Compute output layer gradient

$$
\frac{\partial J}{\partial \mathbf{a}^{[L]}} = \frac{\partial J}{\partial \hat{\mathbf{y}}}
$$

**Step 2**: For each layer $l = L, L-1, \ldots, 1$ (backward):

a. **Gradient w.r.t. pre-activation**:
$$
\frac{\partial J}{\partial \mathbf{z}^{[l]}} = \frac{\partial J}{\partial \mathbf{a}^{[l]}} \odot g'^{[l]}(\mathbf{z}^{[l]})
$$

Where $\odot$ is element-wise multiplication.

b. **Gradient w.r.t. weights**:
$$
\frac{\partial J}{\partial \mathbf{W}^{[l]}} = \frac{\partial J}{\partial \mathbf{z}^{[l]}} (\mathbf{a}^{[l-1]})^T
$$

c. **Gradient w.r.t. bias**:
$$
\frac{\partial J}{\partial \mathbf{b}^{[l]}} = \frac{\partial J}{\partial \mathbf{z}^{[l]}}
$$

d. **Gradient w.r.t. previous layer activations**:
$$
\frac{\partial J}{\partial \mathbf{a}^{[l-1]}} = (\mathbf{W}^{[l]})^T \frac{\partial J}{\partial \mathbf{z}^{[l]}}
$$

### Detailed Formulas

**For output layer $L$** (binary classification with sigmoid):

$$
\frac{\partial J}{\partial \mathbf{z}^{[L]}} = \mathbf{a}^{[L]} - \mathbf{y} = \hat{\mathbf{y}} - \mathbf{y}
$$

**For hidden layer $l$**:

$$
\frac{\partial J}{\partial \mathbf{z}^{[l]}} = (\mathbf{W}^{[l+1]})^T \frac{\partial J}{\partial \mathbf{z}^{[l+1]}} \odot g'^{[l]}(\mathbf{z}^{[l]})
$$

**Weight updates**:

$$
\mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \alpha \frac{\partial J}{\partial \mathbf{W}^{[l]}}
$$

$$
\mathbf{b}^{[l]} := \mathbf{b}^{[l]} - \alpha \frac{\partial J}{\partial \mathbf{b}^{[l]}}
$$

!!! note "Key Point"
    Backpropagation uses the chain rule to compute gradients layer by layer, starting from the output and working backward to the input.

---

## Vanishing Gradient Problem

### Problem

In deep networks, gradients can become **extremely small** as they propagate backward through many layers.

**Cause**: When activation function derivatives are small (e.g., sigmoid: $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$), repeated multiplication makes gradients vanish.

**Effect**: Early layers learn very slowly or not at all.

### Example

**5-layer network with sigmoid**:

$$
\frac{\partial J}{\partial \mathbf{W}^{[1]}} = \frac{\partial J}{\partial \mathbf{z}^{[5]}} \cdot \sigma'(\mathbf{z}^{[5]}) \cdot \sigma'(\mathbf{z}^{[4]}) \cdot \sigma'(\mathbf{z}^{[3]}) \cdot \sigma'(\mathbf{z}^{[2]}) \cdot \sigma'(\mathbf{z}^{[1]})
$$

If each $\sigma'(z) \approx 0.25$, then:

$$
\frac{\partial J}{\partial \mathbf{W}^{[1]}} \approx \text{(small)} \times 0.25^4 = \text{(very small)}
$$

### Solutions

1. **Use ReLU**: Derivative is 1 (when active), prevents vanishing
2. **Residual Connections**: Skip connections (ResNet)
3. **Batch Normalization**: Normalize activations
4. **Proper Initialization**: Xavier/He initialization
5. **Gradient Clipping**: Prevent exploding gradients

---

## Exploding Gradient Problem

### Problem

Gradients can become **extremely large**, causing unstable training.

**Cause**: Large weights or many layers with large derivatives.

**Effect**: Weights update too much, training diverges.

### Solutions

1. **Gradient Clipping**: Limit gradient magnitude
   $$
   \text{if } ||\mathbf{g}|| > \text{threshold}: \mathbf{g} = \mathbf{g} \cdot \frac{\text{threshold}}{||\mathbf{g}||}
$$

2. **Weight Initialization**: Start with small weights
3. **Batch Normalization**: Stabilize activations
4. **Lower Learning Rate**: Smaller steps

---

## Regularization Techniques

### 1. L2 Regularization (Weight Decay)

**Modified Loss Function**:

$$
J_{\text{reg}} = J + \frac{\lambda}{2m} \sum_{l=1}^{L} ||\mathbf{W}^{[l]}||_F^2
$$

Where $||\mathbf{W}^{[l]}||_F^2$ is Frobenius norm (sum of squares of all elements).

**Weight Update**:

$$
\mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \alpha \left(\frac{\partial J}{\partial \mathbf{W}^{[l]}} + \frac{\lambda}{m} \mathbf{W}^{[l]}\right)
$$

**Effect**: Penalizes large weights, prevents overfitting.

### 2. Dropout

**During Training**:
- Randomly set some neurons to 0 with probability $p$ (dropout rate)
- Only keep neurons with probability $(1-p)$

**During Testing**:
- Use all neurons
- Scale activations by $(1-p)$

**Effect**: Prevents co-adaptation, reduces overfitting.

### 3. Early Stopping

- Monitor validation loss
- Stop training when validation loss starts increasing
- Prevents overfitting

### 4. Batch Normalization

**Normalize activations**:

$$
\hat{z}^{[l]} = \frac{z^{[l]} - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
\tilde{z}^{[l]} = \gamma \hat{z}^{[l]} + \beta
$$

**Benefits**:
- Faster training
- Less sensitive to initialization
- Acts as regularization

---

## Weight Initialization

### Poor Initialization

**Problem**: If all weights are same (e.g., all zeros), all neurons learn same thing (symmetry breaking problem).

### Good Initialization Strategies

**1. Xavier/Glorot Initialization** (for tanh/sigmoid):

$$
W_{ij} \sim \mathcal{N}\left(0, \frac{1}{n^{[l-1]}}\right)
$$

or

$$
W_{ij} \sim \mathcal{U}\left(-\frac{\sqrt{6}}{\sqrt{n^{[l-1]} + n^{[l]}}}, \frac{\sqrt{6}}{\sqrt{n^{[l-1]} + n^{[l]}}}\right)
$$

**2. He Initialization** (for ReLU):

$$
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]}}\right)
$$

!!! recommendation "Best Practice"
    Use He initialization for ReLU networks and Xavier initialization for tanh/sigmoid networks.

---

## Key Formulas Summary

### Forward Propagation

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})
$$

### Backpropagation

**Output Layer**:
$$
\frac{\partial J}{\partial \mathbf{z}^{[L]}} = \hat{\mathbf{y}} - \mathbf{y}
$$

**Hidden Layers**:
$$
\frac{\partial J}{\partial \mathbf{z}^{[l]}} = (\mathbf{W}^{[l+1]})^T \frac{\partial J}{\partial \mathbf{z}^{[l+1]}} \odot g'^{[l]}(\mathbf{z}^{[l]})
$$

**Weights**:
$$
\frac{\partial J}{\partial \mathbf{W}^{[l]}} = \frac{\partial J}{\partial \mathbf{z}^{[l]}} (\mathbf{a}^{[l-1]})^T
$$

**Bias**:
$$
\frac{\partial J}{\partial \mathbf{b}^{[l]}} = \frac{\partial J}{\partial \mathbf{z}^{[l]}}
$$

### Regularization

**L2 Regularization**:
$$
J_{\text{reg}} = J + \frac{\lambda}{2m} \sum_{l=1}^{L} ||\mathbf{W}^{[l]}||_F^2
$$

---

## Important Points to Remember

✅ **DFNN**: Multi-layer network with forward and backward propagation

✅ **Forward Pass**: Compute activations layer by layer

✅ **Backpropagation**: Compute gradients using chain rule, backward through layers

✅ **Vanishing Gradient**: Use ReLU, proper initialization, batch norm

✅ **Exploding Gradient**: Use gradient clipping, proper initialization

✅ **Regularization**: L2, dropout, early stopping, batch normalization

✅ **Initialization**: He for ReLU, Xavier for tanh/sigmoid

---

**Previous**: [Module 4 - Linear NN Classification](module4-linear-nn-classification.md) | **Next**: [Module 6 - Convolutional Neural Networks](module6-cnn.md)

