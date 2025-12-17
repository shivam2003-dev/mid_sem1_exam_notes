# Module 5: Calculus for Optimization

## Overview

Calculus is essential for optimization in machine learning. This module covers partial derivatives, gradients, Jacobians, Hessians, and Taylor series—all crucial for understanding gradient descent and other optimization algorithms.

## 1. Partial Derivatives

### Definition

For a function $f(x_1, x_2, \ldots, x_n)$, the **partial derivative** with respect to $x_i$ is:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

### Interpretation

- Rate of change of $f$ with respect to $x_i$ while keeping other variables constant
- Slope of the function in the $x_i$ direction

### Rules

**Product Rule:**
$$\frac{\partial}{\partial x}(fg) = f \frac{\partial g}{\partial x} + g \frac{\partial f}{\partial x}$$

**Quotient Rule:**
$$\frac{\partial}{\partial x}\left(\frac{f}{g}\right) = \frac{g \frac{\partial f}{\partial x} - f \frac{\partial g}{\partial x}}{g^2}$$

**Chain Rule:**
$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u} \frac{\partial u}{\partial x}$$

### Example

For $f(x, y) = x^2 y + \sin(xy)$:

$$\frac{\partial f}{\partial x} = 2xy + y\cos(xy)$$

$$\frac{\partial f}{\partial y} = x^2 + x\cos(xy)$$

---

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

- Points in the direction of steepest ascent
- Perpendicular to level curves/surfaces
- $\nabla f = 0$ at critical points (local minima/maxima/saddle points)

### Example

For $f(x, y) = x^2 + 2xy + y^2$:

$$\nabla f = \begin{pmatrix}
2x + 2y \\
2x + 2y
\end{pmatrix} = 2(x + y)\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

---

## 3. Directional Derivative

### Definition

The **directional derivative** of $f$ in direction $u$ (unit vector) is:

$$D_u f = \nabla f \cdot u = \|\nabla f\| \cos \theta$$

where $\theta$ is the angle between $\nabla f$ and $u$.

### Maximum Directional Derivative

Maximum occurs when $u$ is in the direction of $\nabla f$:
$$u = \frac{\nabla f}{\|\nabla f\|}$$

---

## 4. Jacobian Matrix

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

### Special Case: Gradient

For scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the Jacobian is the transpose of the gradient:
$$J_f = (\nabla f)^T$$

### Example

For $F(x, y) = \begin{pmatrix} x^2 + y \\ xy \end{pmatrix}$:

$$J_F = \begin{pmatrix}
2x & 1 \\
y & x
\end{pmatrix}$$

---

## 5. Hessian Matrix

### Definition

The **Hessian matrix** of a scalar function $f: \mathbb{R}^n \to \mathbb{R}$ is:

$$H_f = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

### Properties

- **Symmetric:** $H_{ij} = H_{ji}$ (if second partials are continuous)
- Used in second-order optimization methods
- Determines convexity: $H$ positive definite $\Rightarrow$ $f$ is convex

### Example

For $f(x, y) = x^2 + 2xy + y^2$:

$$H_f = \begin{pmatrix}
2 & 2 \\
2 & 2
\end{pmatrix}$$

---

## 6. Taylor Series

### Single Variable

For $f: \mathbb{R} \to \mathbb{R}$:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

**First-order approximation:**
$$f(x) \approx f(a) + f'(a)(x-a)$$

**Second-order approximation:**
$$f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2}f''(a)(x-a)^2$$

### Two Variables (MUST KNOW FOR EXAM)

For $f: \mathbb{R}^2 \to \mathbb{R}$:

**First-order (linear approximation):**
$$f(x, y) \approx f(a, b) + \frac{\partial f}{\partial x}(a, b)(x-a) + \frac{\partial f}{\partial y}(a, b)(y-b)$$

**Second-order (quadratic approximation):**
$$f(x, y) \approx f(a, b) + \frac{\partial f}{\partial x}(a, b)(x-a) + \frac{\partial f}{\partial y}(a, b)(y-b)$$
$$+ \frac{1}{2}\frac{\partial^2 f}{\partial x^2}(a, b)(x-a)^2 + \frac{\partial^2 f}{\partial x \partial y}(a, b)(x-a)(y-b) + \frac{1}{2}\frac{\partial^2 f}{\partial y^2}(a, b)(y-b)^2$$

### Matrix Form

$$f(x) \approx f(a) + \nabla f(a)^T (x-a) + \frac{1}{2}(x-a)^T H_f(a) (x-a)$$

where $x = \begin{pmatrix} x \\ y \end{pmatrix}$ and $a = \begin{pmatrix} a \\ b \end{pmatrix}$.

### Example: Two-Variable Taylor Series

Find second-order Taylor expansion of $f(x, y) = e^{xy}$ around $(0, 0)$.

**Compute derivatives:**

$$f(0, 0) = 1$$

$$\frac{\partial f}{\partial x} = ye^{xy} \Rightarrow \frac{\partial f}{\partial x}(0, 0) = 0$$

$$\frac{\partial f}{\partial y} = xe^{xy} \Rightarrow \frac{\partial f}{\partial y}(0, 0) = 0$$

$$\frac{\partial^2 f}{\partial x^2} = y^2 e^{xy} \Rightarrow \frac{\partial^2 f}{\partial x^2}(0, 0) = 0$$

$$\frac{\partial^2 f}{\partial y^2} = x^2 e^{xy} \Rightarrow \frac{\partial^2 f}{\partial y^2}(0, 0) = 0$$

$$\frac{\partial^2 f}{\partial x \partial y} = e^{xy} + xye^{xy} \Rightarrow \frac{\partial^2 f}{\partial x \partial y}(0, 0) = 1$$

**Taylor expansion:**
$$f(x, y) \approx 1 + 0 \cdot x + 0 \cdot y + \frac{1}{2}(0 \cdot x^2 + 2 \cdot 1 \cdot xy + 0 \cdot y^2) = 1 + xy$$

---

## 7. Chain Rule for Multivariable Functions

### General Form

For $z = f(x, y)$ where $x = g(t)$ and $y = h(t)$:

$$\frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$$

### Example

For $z = x^2 + y^2$ where $x = \cos t$, $y = \sin t$:

$$\frac{dz}{dt} = 2x(-\sin t) + 2y(\cos t) = -2\cos t \sin t + 2\sin t \cos t = 0$$

---

## 8. Applications in Machine Learning

### Gradient Descent

Update rule:
$$x^{(k+1)} = x^{(k)} - \alpha \nabla f(x^{(k)})$$

where $\alpha$ is the learning rate.

### Newton's Method

Uses Hessian for second-order optimization:
$$x^{(k+1)} = x^{(k)} - H_f(x^{(k)})^{-1} \nabla f(x^{(k)})$$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Two-Variable Taylor Series:** MUST know this! Practice expanding common functions
2. **Gradient:** Always compute all partial derivatives
3. **Hessian:** Check symmetry ($H_{ij} = H_{ji}$)
4. **Chain Rule:** Identify all dependencies clearly
5. **Common Functions:**
   - $f(x, y) = x^2 + y^2$: $\nabla f = (2x, 2y)^T$, $H = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$
   - $f(x, y) = xy$: $\nabla f = (y, x)^T$, $H = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$
```

---

## 🔍 Worked Examples

### Example 1: Gradient and Hessian

For $f(x, y) = x^3 + 3xy + y^2$, find $\nabla f$ and $H_f$.

**Gradient:**
$$\frac{\partial f}{\partial x} = 3x^2 + 3y$$
$$\frac{\partial f}{\partial y} = 3x + 2y$$

$$\nabla f = \begin{pmatrix} 3x^2 + 3y \\ 3x + 2y \end{pmatrix}$$

**Hessian:**
$$\frac{\partial^2 f}{\partial x^2} = 6x$$
$$\frac{\partial^2 f}{\partial y^2} = 2$$
$$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x} = 3$$

$$H_f = \begin{pmatrix} 6x & 3 \\ 3 & 2 \end{pmatrix}$$

### Example 2: Two-Variable Taylor Series

Expand $f(x, y) = \sin(x + y)$ around $(0, 0)$ to second order.

**Derivatives:**
- $f(0, 0) = 0$
- $\frac{\partial f}{\partial x} = \cos(x+y) \Rightarrow f_x(0,0) = 1$
- $\frac{\partial f}{\partial y} = \cos(x+y) \Rightarrow f_y(0,0) = 1$
- $\frac{\partial^2 f}{\partial x^2} = -\sin(x+y) \Rightarrow f_{xx}(0,0) = 0$
- $\frac{\partial^2 f}{\partial y^2} = -\sin(x+y) \Rightarrow f_{yy}(0,0) = 0$
- $\frac{\partial^2 f}{\partial x \partial y} = -\sin(x+y) \Rightarrow f_{xy}(0,0) = 0$

**Expansion:**
$$f(x, y) \approx 0 + 1 \cdot x + 1 \cdot y + \frac{1}{2}(0 \cdot x^2 + 2 \cdot 0 \cdot xy + 0 \cdot y^2) = x + y$$

---

## 📚 Quick Revision Checklist

- [ ] Partial derivatives computation
- [ ] Gradient vector
- [ ] Jacobian matrix for vector functions
- [ ] Hessian matrix and its properties
- [ ] Two-variable Taylor series (MUST!)
- [ ] Chain rule for multivariable functions
- [ ] Applications to gradient descent

