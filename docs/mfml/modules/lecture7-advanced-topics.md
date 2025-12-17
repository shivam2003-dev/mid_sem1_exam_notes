# Lecture 7: Advanced Topics - Taylor Series and Applications

## Overview

This lecture covers Taylor series expansions (especially two-variable), numerical methods, and comprehensive applications of all topics covered in the course.

## 1. Taylor Series - Single Variable

### Definition

For $f: \mathbb{R} \to \mathbb{R}$:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

### First-Order Approximation

$$f(x) \approx f(a) + f'(a)(x-a)$$

### Second-Order Approximation

$$f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2}f''(a)(x-a)^2$$

## 2. Two-Variable Taylor Series (MUST KNOW!)

### First-Order (Linear Approximation)

$$f(x, y) \approx f(a, b) + \frac{\partial f}{\partial x}(a, b)(x-a) + \frac{\partial f}{\partial y}(a, b)(y-b)$$

### Second-Order (Quadratic Approximation)

$$f(x, y) \approx f(a, b) + \frac{\partial f}{\partial x}(a, b)(x-a) + \frac{\partial f}{\partial y}(a, b)(y-b)$$
$$+ \frac{1}{2}\frac{\partial^2 f}{\partial x^2}(a, b)(x-a)^2 + \frac{\partial^2 f}{\partial x \partial y}(a, b)(x-a)(y-b) + \frac{1}{2}\frac{\partial^2 f}{\partial y^2}(a, b)(y-b)^2$$

### Matrix Form

$$f(x) \approx f(a) + \nabla f(a)^T (x-a) + \frac{1}{2}(x-a)^T H_f(a) (x-a)$$

where $x = \begin{pmatrix} x \\ y \end{pmatrix}$ and $a = \begin{pmatrix} a \\ b \end{pmatrix}$.

## 3. Common Taylor Expansions

### Exponential Function

$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots = \sum_{n=0}^{\infty} \frac{x^n}{n!}$$

### Sine Function

$$\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}$$

### Cosine Function

$$\cos(x) = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!}$$

### Natural Logarithm

$$\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots = \sum_{n=1}^{\infty} \frac{(-1)^{n+1} x^n}{n}$$

## 4. Applications in Machine Learning

### Gradient Descent

Uses first-order Taylor approximation:

$$f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x$$

To minimize, move in direction of negative gradient:
$$\Delta x = -\alpha \nabla f(x)$$

### Newton's Method

Uses second-order Taylor approximation:

$$f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2}\Delta x^T H_f(x) \Delta x$$

Setting derivative to zero:
$$H_f(x) \Delta x = -\nabla f(x)$$

Update: $x^{(k+1)} = x^{(k)} - H_f(x^{(k)})^{-1} \nabla f(x^{(k)})$

## 5. Numerical Methods

### Condition Number

For matrix $A$:
$$\kappa(A) = \|A\| \|A^{-1}\|$$

For 2-norm:
$$\kappa_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}}$$

### Numerical Stability

- **Well-conditioned:** $\kappa \approx 1$ (small errors in input → small errors in output)
- **Ill-conditioned:** $\kappa \gg 1$ (small errors in input → large errors in output)

## 6. Integration of Topics

### Complete Workflow

1. **Formulate Problem:** System of equations, optimization, etc.
2. **Matrix Representation:** Convert to matrix form
3. **Decomposition:** Use appropriate decomposition (LU, Cholesky, SVD, etc.)
4. **Solve:** Apply numerical methods
5. **Verify:** Check solution and stability

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Two-Variable Taylor:** MUST memorize the second-order formula
2. **Common Expansions:** Know $e^x$, $\sin(x)$, $\cos(x)$, $\ln(1+x)$
3. **Applications:** Understand gradient descent and Newton's method
4. **Condition Number:** High value indicates numerical instability
5. **Integration:** Be able to combine multiple concepts in one problem
```

---

## 🔍 Worked Examples

### Example 1: Two-Variable Taylor Series

Expand $f(x, y) = \sin(x + y)$ around $(0, 0)$ to second order.

**Derivatives:**
- $f(0, 0) = 0$
- $f_x = \cos(x+y) \Rightarrow f_x(0,0) = 1$
- $f_y = \cos(x+y) \Rightarrow f_y(0,0) = 1$
- $f_{xx} = -\sin(x+y) \Rightarrow f_{xx}(0,0) = 0$
- $f_{yy} = -\sin(x+y) \Rightarrow f_{yy}(0,0) = 0$
- $f_{xy} = -\sin(x+y) \Rightarrow f_{xy}(0,0) = 0$

**Expansion:**
$$f(x, y) \approx 0 + 1 \cdot x + 1 \cdot y + \frac{1}{2}[0 \cdot x^2 + 2 \cdot 0 \cdot xy + 0 \cdot y^2] = x + y$$

### Example 2: Condition Number

Compute condition number of $A = \begin{pmatrix} 1 & 1 \\ 1 & 1.0001 \end{pmatrix}$.

**Singular values:**
- $AA^T = \begin{pmatrix} 2 & 2.0001 \\ 2.0001 & 2.0002 \end{pmatrix}$
- Eigenvalues approximately: $\lambda_1 \approx 4.0002$, $\lambda_2 \approx 0.0001$
- Singular values: $\sigma_1 \approx 2.00005$, $\sigma_2 \approx 0.01$

**Condition number:**
$$\kappa_2(A) \approx \frac{2.00005}{0.01} = 200$$ (ill-conditioned)

---

## 📚 Quick Revision Checklist

- [ ] Single-variable Taylor series
- [ ] Two-variable Taylor series (MUST!)
- [ ] Common Taylor expansions
- [ ] Applications to gradient descent
- [ ] Applications to Newton's method
- [ ] Condition numbers and numerical stability
- [ ] Integration of all course topics

