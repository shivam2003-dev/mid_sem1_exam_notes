# 2024 MidSem Regular DNN Paper - Complete Solutions

## Question 1: Perceptron Learning Algorithm

### Problem Statement

Given training data for binary classification:

| x₁ | x₂ | y  |
|----|----|-----|
| 1  | 1  | 1   |
| 2  | 2  | 1   |
| 0  | 0  | 0   |
| 1  | 0  | 0   |

**a)** Initialize perceptron with $w_1 = 0$, $w_2 = 0$, $b = 0$, learning rate $\alpha = 1.0$.

**b)** Perform 2 iterations of the perceptron learning algorithm.

**c)** What is the decision boundary after 2 iterations?

---

### Solution

#### Part (a): Initialization

**Given**:
- $w_1 = 0$, $w_2 = 0$, $b = 0$
- $\alpha = 1.0$
- Activation: Step function ($f(z) = 1$ if $z \geq 0$, else $0$)

---

#### Part (b): Perceptron Learning - Iteration 1

**Example 1**: $(x_1=1, x_2=1, y=1)$

**Forward Pass**:
\[
z = w_1 \cdot x_1 + w_2 \cdot x_2 + b = 0 \cdot 1 + 0 \cdot 1 + 0 = 0
\]
\[
\hat{y} = f(0) = 1
\]

**Check**: $\hat{y} = 1$, $y = 1$ → **Correct** (no update)

---

**Example 2**: $(x_1=2, x_2=2, y=1)$

**Forward Pass**:
\[
z = 0 \cdot 2 + 0 \cdot 2 + 0 = 0
\]
\[
\hat{y} = f(0) = 1
\]

**Check**: $\hat{y} = 1$, $y = 1$ → **Correct** (no update)

---

**Example 3**: $(x_1=0, x_2=0, y=0)$

**Forward Pass**:
\[
z = 0 \cdot 0 + 0 \cdot 0 + 0 = 0
\]
\[
\hat{y} = f(0) = 1
\]

**Check**: $\hat{y} = 1$, $y = 0$ → **Wrong!** (update needed)

**Update Weights**:
\[
w_1 := w_1 + \alpha \cdot (y - \hat{y}) \cdot x_1 = 0 + 1 \cdot (0 - 1) \cdot 0 = 0
\]
\[
w_2 := w_2 + \alpha \cdot (y - \hat{y}) \cdot x_2 = 0 + 1 \cdot (0 - 1) \cdot 0 = 0
\]
\[
b := b + \alpha \cdot (y - \hat{y}) = 0 + 1 \cdot (0 - 1) = -1
\]

**After Example 3**: $w_1 = 0$, $w_2 = 0$, $b = -1$

---

**Example 4**: $(x_1=1, x_2=0, y=0)$

**Forward Pass**:
\[
z = 0 \cdot 1 + 0 \cdot 0 + (-1) = -1
\]
\[
\hat{y} = f(-1) = 0
\]

**Check**: $\hat{y} = 0$, $y = 0$ → **Correct** (no update)

**After Iteration 1**: $w_1 = 0$, $w_2 = 0$, $b = -1$

---

#### Iteration 2

**Example 1**: $(x_1=1, x_2=1, y=1)$

**Forward Pass**:
\[
z = 0 \cdot 1 + 0 \cdot 1 + (-1) = -1
\]
\[
\hat{y} = f(-1) = 0
\]

**Check**: $\hat{y} = 0$, $y = 1$ → **Wrong!** (update needed)

**Update**:
\[
w_1 := 0 + 1 \cdot (1 - 0) \cdot 1 = 1
\]
\[
w_2 := 0 + 1 \cdot (1 - 0) \cdot 1 = 1
\]
\[
b := -1 + 1 \cdot (1 - 0) = 0
\]

**After Example 1**: $w_1 = 1$, $w_2 = 1$, $b = 0$

---

**Example 2**: $(x_1=2, x_2=2, y=1)$

**Forward Pass**:
\[
z = 1 \cdot 2 + 1 \cdot 2 + 0 = 4
\]
\[
\hat{y} = f(4) = 1
\]

**Check**: **Correct** (no update)

---

**Example 3**: $(x_1=0, x_2=0, y=0)$

**Forward Pass**:
\[
z = 1 \cdot 0 + 1 \cdot 0 + 0 = 0
\]
\[
\hat{y} = f(0) = 1
\]

**Check**: $\hat{y} = 1$, $y = 0$ → **Wrong!**

**Update**:
\[
w_1 := 1 + 1 \cdot (0 - 1) \cdot 0 = 1
\]
\[
w_2 := 1 + 1 \cdot (0 - 1) \cdot 0 = 1
\]
\[
b := 0 + 1 \cdot (0 - 1) = -1
\]

**After Example 3**: $w_1 = 1$, $w_2 = 1$, $b = -1$

---

**Example 4**: $(x_1=1, x_2=0, y=0)$

**Forward Pass**:
\[
z = 1 \cdot 1 + 1 \cdot 0 + (-1) = 0
\]
\[
\hat{y} = f(0) = 1
\]

**Check**: $\hat{y} = 1$, $y = 0$ → **Wrong!**

**Update**:
\[
w_1 := 1 + 1 \cdot (0 - 1) \cdot 1 = 0
\]
\[
w_2 := 1 + 1 \cdot (0 - 1) \cdot 0 = 1
\]
\[
b := -1 + 1 \cdot (0 - 1) = -2
\]

**After Iteration 2**: $w_1 = 0$, $w_2 = 1$, $b = -2$

---

#### Part (c): Decision Boundary

**After 2 iterations**: $w_1 = 0$, $w_2 = 1$, $b = -2$

**Decision Boundary Equation**:

\[
w_1 x_1 + w_2 x_2 + b = 0
\]

\[
0 \cdot x_1 + 1 \cdot x_2 - 2 = 0
\]

\[
x_2 = 2
\]

**Answer**: Decision boundary is the horizontal line $x_2 = 2$

**Interpretation**: 
- Points with $x_2 \geq 2$ → Class 1
- Points with $x_2 < 2$ → Class 0

---

## Question 2: Forward and Backward Propagation

### Problem Statement

Given a 2-layer neural network:

- **Layer 1**: 2 inputs, 3 hidden neurons, ReLU activation
- **Layer 2**: 3 inputs (from layer 1), 1 output, linear activation

**Weights**:
\[
\mathbf{W}^{[1]} = \begin{bmatrix} 1 & 2 \\ -1 & 1 \\ 0 & 1 \end{bmatrix}, \quad \mathbf{b}^{[1]} = \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix}
\]

\[
\mathbf{W}^{[2]} = \begin{bmatrix} 1 & -1 & 2 \end{bmatrix}, \quad b^{[2]} = 0
\]

**Input**: $\mathbf{x} = [1, 2]^T$, **Target**: $y = 5$

**a)** Perform forward propagation.

**b)** Calculate the loss (MSE).

**c)** Compute gradients for $\mathbf{W}^{[2]}$ and $b^{[2]}$.

---

### Solution

#### Part (a): Forward Propagation

**Layer 1**:

\[
\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]} = \begin{bmatrix} 1 & 2 \\ -1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix}
\]

\[
\mathbf{z}^{[1]} = \begin{bmatrix} 1 \cdot 1 + 2 \cdot 2 \\ -1 \cdot 1 + 1 \cdot 2 \\ 0 \cdot 1 + 1 \cdot 2 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix} = \begin{bmatrix} 5 \\ 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix} = \begin{bmatrix} 6 \\ 1 \\ 1 \end{bmatrix}
\]

**ReLU Activation**:
\[
\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]}) = \begin{bmatrix} \max(0, 6) \\ \max(0, 1) \\ \max(0, 1) \end{bmatrix} = \begin{bmatrix} 6 \\ 1 \\ 1 \end{bmatrix}
\]

**Layer 2**:

\[
z^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + b^{[2]} = \begin{bmatrix} 1 & -1 & 2 \end{bmatrix} \begin{bmatrix} 6 \\ 1 \\ 1 \end{bmatrix} + 0
\]

\[
z^{[2]} = 1 \cdot 6 + (-1) \cdot 1 + 2 \cdot 1 = 6 - 1 + 2 = 7
\]

**Linear Activation**:
\[
\hat{y} = z^{[2]} = 7
\]

**Answer**: $\hat{y} = 7$

---

#### Part (b): Loss Calculation

**Mean Squared Error**:

\[
J = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(7 - 5)^2 = \frac{1}{2} \cdot 4 = 2
\]

**Answer**: Loss $J = 2$

---

#### Part (c): Gradient Computation

**Gradient w.r.t. $z^{[2]}$**:

\[
\frac{\partial J}{\partial z^{[2]}} = \hat{y} - y = 7 - 5 = 2
\]

**Gradient w.r.t. $\mathbf{W}^{[2]}$**:

\[
\frac{\partial J}{\partial \mathbf{W}^{[2]}} = \frac{\partial J}{\partial z^{[2]}} \cdot (\mathbf{a}^{[1]})^T = 2 \cdot \begin{bmatrix} 6 \\ 1 \\ 1 \end{bmatrix}^T = \begin{bmatrix} 12 & 2 & 2 \end{bmatrix}
\]

**Gradient w.r.t. $b^{[2]}$**:

\[
\frac{\partial J}{\partial b^{[2]}} = \frac{\partial J}{\partial z^{[2]}} = 2
\]

**Answer**:
- $\frac{\partial J}{\partial \mathbf{W}^{[2]}} = [12, 2, 2]$
- $\frac{\partial J}{\partial b^{[2]}} = 2$

---

## Question 3: Activation Functions

### Problem Statement

**a)** Calculate the output of a neuron with:
- Input: $z = 2$
- Activation: Sigmoid function

**b)** Calculate the derivative of sigmoid at $z = 2$.

**c)** Why is ReLU preferred over sigmoid for hidden layers?

---

### Solution

#### Part (a): Sigmoid Output

**Sigmoid Function**:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

**For $z = 2$**:

\[
\sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.1353} = \frac{1}{1.1353} = 0.881
\]

**Answer**: $\sigma(2) = 0.881$

---

#### Part (b): Sigmoid Derivative

**Derivative Formula**:

\[
\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))
\]

**At $z = 2$**:

\[
\frac{d\sigma}{dz}\bigg|_{z=2} = \sigma(2)(1 - \sigma(2)) = 0.881 \times (1 - 0.881) = 0.881 \times 0.119 = 0.105
\]

**Answer**: Derivative = $0.105$

---

#### Part (c): ReLU vs Sigmoid for Hidden Layers

**ReLU Advantages**:

1. **Solves Vanishing Gradient**:
   - ReLU derivative = 1 (when active)
   - Sigmoid derivative ≤ 0.25 (always small)
   - In deep networks, sigmoid gradients vanish quickly

2. **Computational Efficiency**:
   - ReLU: Simple max operation
   - Sigmoid: Requires exponential computation

3. **Sparsity**:
   - ReLU creates sparse representations (many zeros)
   - Can be beneficial for learning

4. **Faster Convergence**:
   - ReLU networks train faster
   - Less prone to saturation

**Sigmoid Issues**:
- Vanishing gradients in deep networks
- Saturation (outputs near 0 or 1)
- Not zero-centered

**Answer**: ReLU is preferred because it prevents vanishing gradients, is computationally efficient, and enables faster training in deep networks.

---

## Question 4: Backpropagation in Deep Network

### Problem Statement

Given a 3-layer network with:
- Layer 1: 2 inputs → 3 hidden (ReLU)
- Layer 2: 3 hidden → 2 hidden (ReLU)
- Layer 3: 2 hidden → 1 output (sigmoid)

**Given**:
- $\frac{\partial J}{\partial z^{[3]}} = 0.5$
- $\mathbf{W}^{[3]} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$
- $\mathbf{a}^{[2]} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$
- $\mathbf{z}^{[2]} = \begin{bmatrix} 3 \\ -1 \end{bmatrix}$

**Calculate**:
**a)** $\frac{\partial J}{\partial \mathbf{W}^{[3]}}$
**b)** $\frac{\partial J}{\partial \mathbf{z}^{[2]}}$

---

### Solution

#### Part (a): Gradient w.r.t. $\mathbf{W}^{[3]}$

**Formula**:

\[
\frac{\partial J}{\partial \mathbf{W}^{[3]}} = \frac{\partial J}{\partial z^{[3]}} \cdot (\mathbf{a}^{[2]})^T
\]

**Calculation**:

\[
\frac{\partial J}{\partial \mathbf{W}^{[3]}} = 0.5 \cdot \begin{bmatrix} 2 \\ 1 \end{bmatrix}^T = 0.5 \cdot \begin{bmatrix} 2 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0.5 \end{bmatrix}
\]

**Answer**: $\frac{\partial J}{\partial \mathbf{W}^{[3]}} = [1, 0.5]$

---

#### Part (b): Gradient w.r.t. $\mathbf{z}^{[2]}$

**Formula**:

\[
\frac{\partial J}{\partial \mathbf{z}^{[2]}} = (\mathbf{W}^{[3]})^T \frac{\partial J}{\partial z^{[3]}} \odot g'^{[2]}(\mathbf{z}^{[2]})
\]

**Step 1**: Compute $(\mathbf{W}^{[3]})^T \frac{\partial J}{\partial z^{[3]}}$:

\[
(\mathbf{W}^{[3]})^T \frac{\partial J}{\partial z^{[3]}} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}^T \cdot 0.5 = \begin{bmatrix} 1 & -1 \end{bmatrix} \cdot 0.5 = \begin{bmatrix} 0.5 & -0.5 \end{bmatrix}
\]

Wait, this should be a column vector. Let me recalculate:

\[
(\mathbf{W}^{[3]})^T \frac{\partial J}{\partial z^{[3]}} = \begin{bmatrix} 1 \\ -1 \end{bmatrix} \cdot 0.5 = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}
\]

**Step 2**: Compute ReLU derivative $g'^{[2]}(\mathbf{z}^{[2]})$:

\[
g'^{[2]}(\mathbf{z}^{[2]}) = \begin{bmatrix} \text{ReLU}'(3) \\ \text{ReLU}'(-1) \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
\]

**Step 3**: Element-wise multiplication:

\[
\frac{\partial J}{\partial \mathbf{z}^{[2]}} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} \odot \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0 \end{bmatrix}
\]

**Answer**: $\frac{\partial J}{\partial \mathbf{z}^{[2]}} = [0.5, 0]^T$

!!! note "Key Point"
    Notice that the gradient for the second neuron is 0 because ReLU is inactive (input was negative). This is the "dead ReLU" problem.

---

## Question 5: CNN Convolution Operation

### Problem Statement

Given:
- **Input Image**: 5×5 matrix
- **Filter**: 3×3 matrix
- **Stride**: 1
- **Padding**: 0 (valid)

**a)** Calculate the output size.

**b)** Perform convolution operation for the top-left position.

**Input**:
\[
\mathbf{X} = \begin{bmatrix}
1 & 2 & 3 & 4 & 5 \\
6 & 7 & 8 & 9 & 0 \\
1 & 2 & 3 & 4 & 5 \\
6 & 7 & 8 & 9 & 0 \\
1 & 2 & 3 & 4 & 5
\end{bmatrix}
\]

**Filter**:
\[
\mathbf{F} = \begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}
\]

---

### Solution

#### Part (a): Output Size

**Formula**:

\[
\text{Output Size} = \frac{\text{Input Size} - \text{Filter Size} + 2 \times \text{Padding}}{\text{Stride}} + 1
\]

**Calculation**:

\[
\text{Output Size} = \frac{5 - 3 + 2 \times 0}{1} + 1 = \frac{2}{1} + 1 = 3
\]

**Answer**: Output size = **3×3**

---

#### Part (b): Convolution at Top-Left Position

**Top-left 3×3 region of input**:

\[
\begin{bmatrix}
1 & 2 & 3 \\
6 & 7 & 8 \\
1 & 2 & 3
\end{bmatrix}
\]

**Convolution Operation**:

\[
\text{Output}(0, 0) = \sum_{i=0}^{2} \sum_{j=0}^{2} X(i, j) \cdot F(i, j)
\]

**Element-wise multiplication and sum**:

\[
= 1 \cdot 1 + 2 \cdot 0 + 3 \cdot (-1) + 6 \cdot 1 + 7 \cdot 0 + 8 \cdot (-1) + 1 \cdot 1 + 2 \cdot 0 + 3 \cdot (-1)
\]

\[
= 1 + 0 - 3 + 6 + 0 - 8 + 1 + 0 - 3
\]

\[
= (1 + 6 + 1) + (0 + 0 + 0) + (-3 - 8 - 3)
\]

\[
= 8 + 0 - 14 = -6
\]

**Answer**: Output at position (0, 0) = **-6**

**Interpretation**: This filter detects vertical edges (difference between left and right columns).

---

## Summary

This paper covered:
1. ✅ Perceptron Learning Algorithm with step-by-step iterations
2. ✅ Forward and Backward Propagation in multi-layer networks
3. ✅ Activation Functions (Sigmoid, ReLU) and their derivatives
4. ✅ Backpropagation through multiple layers
5. ✅ CNN Convolution Operation

**Key Takeaways**:
- Always show step-by-step calculations for perceptron updates
- Understand forward and backward propagation formulas
- Know activation function derivatives by heart
- Practice convolution operations manually
- Understand gradient flow in deep networks

---

**Good luck with your exam!** 🎯

