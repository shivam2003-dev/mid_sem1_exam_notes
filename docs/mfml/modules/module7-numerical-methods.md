# Module 7: Numerical Methods

## Overview

Numerical methods are essential for practical computation in machine learning, especially for solving systems, matrix operations, and optimization problems.

## 1. Numerical Linear Algebra

### Floating Point Arithmetic

**IEEE 754 Standard:**
- Single precision: 32 bits (1 sign + 8 exponent + 23 mantissa)
- Double precision: 64 bits (1 sign + 11 exponent + 52 mantissa)

**Machine Epsilon ($\epsilon$):** Smallest number such that $1 + \epsilon > 1$

### Round-off Error

Errors introduced by finite precision arithmetic.

**Example:**
$$\frac{1}{3} = 0.333\ldots \approx 0.33333333 \text{ (finite precision)}$$

---

## 2. Condition Number

### Definition

For matrix $A$, the **condition number** is:

$$\kappa(A) = \|A\| \|A^{-1}\|\]

For 2-norm:
$$\kappa_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}}$$

where $\sigma_{\max}$ and $\sigma_{\min}$ are largest and smallest singular values.

### Interpretation

- $\kappa(A) \approx 1$: Well-conditioned (small errors in input → small errors in output)
- $\kappa(A) \gg 1$: Ill-conditioned (small errors in input → large errors in output)

### Example

$$A = \begin{pmatrix} 1 & 1 \\ 1 & 1.0001 \end{pmatrix}$$

Eigenvalues: $\lambda_1 \approx 2.0001$, $\lambda_2 \approx 0.0001$

Condition number: $\kappa(A) \approx \frac{2.0001}{0.0001} = 20001$ (ill-conditioned)

---

## 3. Numerical Stability

### Stable Algorithm

Small errors in input produce small errors in output.

### Unstable Algorithm

Small errors in input produce large errors in output.

### Example: Subtractive Cancellation

Computing $f(x) = \sqrt{x+1} - \sqrt{x}$ for large $x$:

**Direct computation (unstable):**
$$f(10^8) = \sqrt{10^8 + 1} - \sqrt{10^8} \approx 0$$ (loss of precision)

**Stable computation:**
$$f(x) = \frac{1}{\sqrt{x+1} + \sqrt{x}}$$

---

## 4. Iterative Methods

### Jacobi Method

For solving $Ax = b$:

**Algorithm:**
$$x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j \neq i} a_{ij} x_j^{(k)}\right)$$

### Gauss-Seidel Method

Similar to Jacobi but uses updated values immediately:

$$x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)}\right)$$

### Convergence

Both methods converge if $A$ is **strictly diagonally dominant**:
$$|a_{ii}| > \sum_{j \neq i} |a_{ij}| \quad \text{for all } i$$

---

## 5. Numerical Differentiation

### Forward Difference

$$f'(x) \approx \frac{f(x+h) - f(x)}{h}$$

**Error:** $O(h)$

### Central Difference

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

**Error:** $O(h^2)$ (more accurate)

### Second Derivative

$$f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$$

---

## 6. Numerical Integration

### Trapezoidal Rule

$$\int_a^b f(x) dx \approx \frac{h}{2}[f(a) + 2\sum_{i=1}^{n-1} f(x_i) + f(b)]$$

where $h = \frac{b-a}{n}$.

### Simpson's Rule

$$\int_a^b f(x) dx \approx \frac{h}{3}[f(a) + 4\sum_{i \text{ odd}} f(x_i) + 2\sum_{i \text{ even}} f(x_i) + f(b)]$$

---

## 7. Root Finding

### Newton's Method

For finding roots of $f(x) = 0$:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

**Convergence:** Quadratic (very fast if close to root)

### Bisection Method

**Algorithm:**
1. Start with interval $[a, b]$ where $f(a)$ and $f(b)$ have opposite signs
2. Compute midpoint $c = \frac{a+b}{2}$
3. If $f(c) = 0$, done
4. Otherwise, replace $[a, b]$ with $[a, c]$ or $[c, b]$ based on sign
5. Repeat until convergence

**Convergence:** Linear (slower but guaranteed)

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Condition Number:** High condition number → numerical instability
2. **Iterative Methods:** Check diagonal dominance for convergence
3. **Numerical Differentiation:** Central difference is more accurate
4. **Stability:** Avoid subtractive cancellation
5. **Floating Point:** Be aware of precision limitations
```

---

## 🔍 Worked Examples

### Example 1: Condition Number

Compute condition number of $A = \begin{pmatrix} 1 & 2 \\ 2 & 4.0001 \end{pmatrix}$.

**Singular values:**
- $AA^T = \begin{pmatrix} 5 & 10.0002 \\ 10.0002 & 20.0004 \end{pmatrix}$
- Eigenvalues: $\lambda_1 \approx 25.0004$, $\lambda_2 \approx 0.0001$
- Singular values: $\sigma_1 \approx 5.00004$, $\sigma_2 \approx 0.01$

**Condition number:**
$$\kappa_2(A) \approx \frac{5.00004}{0.01} = 500$$ (ill-conditioned)

### Example 2: Jacobi Method

Solve $Ax = b$ where:
$$A = \begin{pmatrix} 4 & 1 \\ 1 & 3 \end{pmatrix}, \quad b = \begin{pmatrix} 5 \\ 4 \end{pmatrix}$$

**Initial guess:** $x^{(0)} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

**Iteration 1:**
$$x_1^{(1)} = \frac{1}{4}(5 - 1 \cdot 0) = 1.25$$
$$x_2^{(1)} = \frac{1}{3}(4 - 1 \cdot 0) = 1.33$$

**Iteration 2:**
$$x_1^{(2)} = \frac{1}{4}(5 - 1 \cdot 1.33) = 0.9175$$
$$x_2^{(2)} = \frac{1}{3}(4 - 1 \cdot 1.25) = 0.9167$$

Continue until convergence...

---

## 📚 Quick Revision Checklist

- [ ] Floating point arithmetic and round-off errors
- [ ] Condition number and its interpretation
- [ ] Numerical stability concepts
- [ ] Iterative methods (Jacobi, Gauss-Seidel)
- [ ] Numerical differentiation formulas
- [ ] Numerical integration methods
- [ ] Root finding algorithms

