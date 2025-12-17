# Lecture 6: Advanced Topics - Gradients and Optimization

## Overview

This lecture covers gradients, Jacobians, Hessians, and optimization concepts - essential for understanding machine learning algorithms like gradient descent.

## 1. Partial Derivatives

### Definition

For a function $f(x_1, x_2, \ldots, x_n)$, the **partial derivative** with respect to $x_i$ is:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

### Rules

**Product Rule:**
$$\frac{\partial}{\partial x}(fg) = f \frac{\partial g}{\partial x} + g \frac{\partial f}{\partial x}$$

**Quotient Rule:**
$$\frac{\partial}{\partial x}\left(\frac{f}{g}\right) = \frac{g \frac{\partial f}{\partial x} - f \frac{\partial g}{\partial x}}{g^2}$$

**Chain Rule:**
$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u} \frac{\partial u}{\partial x}$$

## 2. Gradient

### Definition

The **gradient** of a scalar function $f: \mathbb{R}^n \to \mathbb{R}$ is:

$$\nabla f = \begin{pmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{pmatrix}$$

### Properties

- Points in direction of steepest ascent
- Perpendicular to level curves/surfaces
- $\nabla f = 0$ at critical points

### Example

For $f(x, y) = x^2 + 2xy + y^2$:

$$\nabla f = \begin{pmatrix}
2x + 2y \\
2x + 2y
\end{pmatrix} = 2(x + y)\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

## 3. Jacobian Matrix

### Definition

For a vector-valued function $F: \mathbb{R}^n \to \mathbb{R}^m$:

$$F(x) = \begin{pmatrix}
f_1(x_1, \ldots, x_n) \\
f_2(x_1, \ldots, x_n) \\
\vdots \\
f_m(x_1, \ldots, x_n)
\end{pmatrix}$$

The **Jacobian matrix** is:

$$J_F = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}$$

### Special Case

For scalar function $f: \mathbb{R}^n \to \mathbb{R}$:
$$J_f = (\nabla f)^T$$

## 4. Hessian Matrix

### Definition

The **Hessian matrix** of $f: \mathbb{R}^n \to \mathbb{R}$ is:

$$H_f = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

### Properties

- **Symmetric:** $H_{ij} = H_{ji}$ (if second partials are continuous)
- Used in second-order optimization
- Determines convexity: $H$ positive definite $\Rightarrow$ $f$ is convex

## 5. Optimization

### Critical Points

Points where $\nabla f = 0$ are **critical points**.

### Second Derivative Test

For function $f(x, y)$:

1. Compute $D = f_{xx} f_{yy} - (f_{xy})^2$
2. If $D > 0$ and $f_{xx} > 0$: Local minimum
3. If $D > 0$ and $f_{xx} < 0$: Local maximum
4. If $D < 0$: Saddle point
5. If $D = 0$: Test is inconclusive

### Gradient Descent

Update rule:
$$x^{(k+1)} = x^{(k)} - \alpha \nabla f(x^{(k)})$$

where $\alpha$ is the learning rate.

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Gradient:** Compute all partial derivatives
2. **Hessian:** Check symmetry ($H_{ij} = H_{ji}$)
3. **Critical Points:** Set $\nabla f = 0$ and solve
4. **Second Derivative Test:** Use determinant $D$ for two variables
5. **Chain Rule:** Identify all dependencies clearly
```

---

## 🔍 Worked Examples

### Example 1: Gradient and Hessian

For $f(x, y) = x^3 + 3xy + y^2$:

**Gradient:**
$$\nabla f = \begin{pmatrix} 3x^2 + 3y \\ 3x + 2y \end{pmatrix}$$

**Hessian:**
$$H_f = \begin{pmatrix}
6x & 3 \\
3 & 2
\end{pmatrix}$$

### Example 2: Critical Points

Find critical points of $f(x, y) = x^2 + y^2 - 2x - 4y + 5$.

**Gradient:**
$$\nabla f = \begin{pmatrix} 2x - 2 \\ 2y - 4 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

**Solution:** $x = 1$, $y = 2$

**Critical Point:** $(1, 2)$

**Hessian:**
$$H_f = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$$

Since $H$ is positive definite, $(1, 2)$ is a **local minimum**.

---

## 📚 Quick Revision Checklist

- [ ] Partial derivatives computation
- [ ] Gradient vector
- [ ] Jacobian matrix for vector functions
- [ ] Hessian matrix and its properties
- [ ] Critical points and optimization
- [ ] Second derivative test
- [ ] Gradient descent algorithm

