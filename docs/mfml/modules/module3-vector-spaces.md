# Module 3: Vector Spaces and Inner Products

## Overview

This module covers inner product spaces, orthogonality, and the Gram-Schmidt process—essential concepts for understanding geometric properties of vectors in machine learning.

## 1. Inner Product Spaces

### Definition

An **inner product** on a vector space $V$ is a function $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$ satisfying:

1. **Positivity:** $\langle v, v \rangle \geq 0$ and $\langle v, v \rangle = 0$ iff $v = 0$
2. **Symmetry:** $\langle u, v \rangle = \langle v, u \rangle$
3. **Linearity:** $\langle au + bw, v \rangle = a\langle u, v \rangle + b\langle w, v \rangle$

### Standard Inner Product (Dot Product)

For vectors in $\mathbb{R}^n$:
$$\langle u, v \rangle = u^T v = \sum_{i=1}^{n} u_i v_i$$

### Weighted Inner Product

Given a positive definite matrix $M$:
$$\langle u, v \rangle_M = u^T M v$$

This defines a valid inner product if $M$ is symmetric and positive definite.

### Properties

- **Cauchy-Schwarz Inequality:** $|\langle u, v \rangle| \leq \|u\| \|v\|$
- **Triangle Inequality:** $\|u + v\| \leq \|u\| + \|v\|$
- **Parallelogram Law:** $\|u + v\|^2 + \|u - v\|^2 = 2(\|u\|^2 + \|v\|^2)$

---

## 2. Norms and Distances

### Norm Definition

A **norm** on vector space $V$ is a function $\|\cdot\|: V \to \mathbb{R}$ satisfying:

1. **Positivity:** $\|v\| \geq 0$ and $\|v\| = 0$ iff $v = 0$
2. **Homogeneity:** $\|cv\| = |c| \|v\|$
3. **Triangle Inequality:** $\|u + v\| \leq \|u\| + \|v\|$

### Norm from Inner Product

Given an inner product, the **induced norm** is:
$$\|v\| = \sqrt{\langle v, v \rangle}$$

### Common Norms

**Euclidean Norm (L2):**
$$\|v\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}$$

**Manhattan Norm (L1):**
$$\|v\|_1 = \sum_{i=1}^{n} |v_i|$$

**Maximum Norm (L∞):**
$$\|v\|_\infty = \max_{i} |v_i|$$

### Distance

The **distance** between vectors $u$ and $v$ is:
$$d(u, v) = \|u - v\|$$

---

## 3. Orthogonality

### Orthogonal Vectors

Two vectors $u$ and $v$ are **orthogonal** if:
$$\langle u, v \rangle = 0$$

### Orthogonal Set

A set of vectors $\{v_1, v_2, \ldots, v_k\}$ is **orthogonal** if:
$$\langle v_i, v_j \rangle = 0 \quad \text{for all } i \neq j$$

### Orthonormal Set

An **orthonormal set** is an orthogonal set where all vectors have unit norm:
$$\langle v_i, v_j \rangle = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

### Orthogonal Complement

For subspace $W \subseteq V$, the **orthogonal complement** is:
$$W^\perp = \{v \in V : \langle v, w \rangle = 0 \text{ for all } w \in W\}$$

---

## 4. Gram-Schmidt Orthogonalization Process

### Algorithm

Given linearly independent vectors $\{v_1, v_2, \ldots, v_n\}$, construct an orthonormal set $\{u_1, u_2, \ldots, u_n\}$:

**Step 1:** Normalize the first vector
$$u_1 = \frac{v_1}{\|v_1\|}$$

**Step 2:** For $i = 2, \ldots, n$:
1. Project $v_i$ onto previous vectors:
   $$w_i = v_i - \sum_{j=1}^{i-1} \langle v_i, u_j \rangle u_j$$
2. Normalize:
   $$u_i = \frac{w_i}{\|w_i\|}$$

### Geometric Interpretation

At each step, we:
1. Remove the component of $v_i$ that lies in the span of previous vectors
2. Normalize the remaining orthogonal component

### Example

Apply Gram-Schmidt to:
$$v_1 = \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix}, \quad v_2 = \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}, \quad v_3 = \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix}$$

**Step 1:**
$$\|v_1\| = \sqrt{1^2 + 1^2 + 0^2} = \sqrt{2}$$
$$u_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix}$$

**Step 2:**
$$\langle v_2, u_1 \rangle = \frac{1}{\sqrt{2}}(1 \cdot 1 + 0 \cdot 1 + 1 \cdot 0) = \frac{1}{\sqrt{2}}$$
$$w_2 = v_2 - \langle v_2, u_1 \rangle u_1 = \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix} - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix} = \begin{pmatrix} \frac{1}{2} \\ -\frac{1}{2} \\ 1 \end{pmatrix}$$

$$\|w_2\| = \sqrt{\frac{1}{4} + \frac{1}{4} + 1} = \sqrt{\frac{3}{2}} = \frac{\sqrt{6}}{2}$$
$$u_2 = \frac{2}{\sqrt{6}} \begin{pmatrix} \frac{1}{2} \\ -\frac{1}{2} \\ 1 \end{pmatrix} = \frac{1}{\sqrt{6}} \begin{pmatrix} 1 \\ -1 \\ 2 \end{pmatrix}$$

**Step 3:**
$$\langle v_3, u_1 \rangle = \frac{1}{\sqrt{2}}, \quad \langle v_3, u_2 \rangle = \frac{1}{\sqrt{6}}$$
$$w_3 = v_3 - \langle v_3, u_1 \rangle u_1 - \langle v_3, u_2 \rangle u_2$$
$$= \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix} - \frac{1}{2} \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix} - \frac{1}{6} \begin{pmatrix} 1 \\ -1 \\ 2 \end{pmatrix} = \begin{pmatrix} -\frac{2}{3} \\ \frac{2}{3} \\ \frac{2}{3} \end{pmatrix}$$

$$\|w_3\| = \frac{2}{\sqrt{3}}$$
$$u_3 = \frac{\sqrt{3}}{2} \begin{pmatrix} -\frac{2}{3} \\ \frac{2}{3} \\ \frac{2}{3} \end{pmatrix} = \frac{1}{\sqrt{3}} \begin{pmatrix} -1 \\ 1 \\ 1 \end{pmatrix}$$

---

## 5. Projections

### Orthogonal Projection

The **orthogonal projection** of vector $v$ onto vector $u$ is:
$$\text{proj}_u(v) = \frac{\langle v, u \rangle}{\langle u, u \rangle} u = \frac{\langle v, u \rangle}{\|u\|^2} u$$

### Projection onto Subspace

For orthonormal basis $\{u_1, \ldots, u_k\}$ of subspace $W$:
$$\text{proj}_W(v) = \sum_{i=1}^{k} \langle v, u_i \rangle u_i$$

### Projection Matrix

For subspace $W$ with orthonormal basis matrix $Q$:
$$P = QQ^T$$

Then $\text{proj}_W(v) = Pv$.

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Inner Product Verification:** Always check all three properties (positivity, symmetry, linearity)
2. **Gram-Schmidt:** Work step-by-step, don't skip normalization
3. **Orthogonality Check:** Verify $\langle u_i, u_j \rangle = 0$ for $i \neq j$
4. **Norm Calculation:** Use $\|v\| = \sqrt{\langle v, v \rangle}$ for weighted inner products
5. **Common Mistake:** Forgetting to normalize in Gram-Schmidt process
```

---

## 🔍 Worked Examples

### Example 1: Verify Inner Product

Verify that $\langle x, y \rangle = x^T M y$ defines an inner product where:
$$M = \begin{pmatrix} 2 & -1 \\ -1 & 3 \end{pmatrix}$$

**Solution:**

1. **Positivity:** Need to check $M$ is positive definite
   - Leading minors: $2 > 0$, $\det(M) = 6 - 1 = 5 > 0$ ✓
   - So $\langle x, x \rangle = x^T M x > 0$ for $x \neq 0$ ✓

2. **Symmetry:** Need $M = M^T$
   - $M^T = \begin{pmatrix} 2 & -1 \\ -1 & 3 \end{pmatrix} = M$ ✓

3. **Linearity:**
   $$\langle ax + by, z \rangle = (ax + by)^T M z = a(x^T M z) + b(y^T M z) = a\langle x, z \rangle + b\langle y, z \rangle$$ ✓

### Example 2: Compute Norm with Weighted Inner Product

Given $M = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix}$ and $v = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$, compute $\|v\|$ with respect to $\langle \cdot, \cdot \rangle_M$.

**Solution:**

$$\langle v, v \rangle_M = v^T M v = \begin{pmatrix} 1 & 2 \end{pmatrix} \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$

$$= \begin{pmatrix} 1 & 2 \end{pmatrix} \begin{pmatrix} 6 \\ 5 \end{pmatrix} = 6 + 10 = 16$$

$$\|v\| = \sqrt{16} = 4$$

---

## 📚 Quick Revision Checklist

- [ ] Inner product definition and properties
- [ ] Norm from inner product
- [ ] Orthogonality and orthonormality
- [ ] Gram-Schmidt process (step-by-step)
- [ ] Orthogonal projections
- [ ] Weighted inner products with matrices

