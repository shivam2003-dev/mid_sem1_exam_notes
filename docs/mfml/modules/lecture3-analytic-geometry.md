# Lecture 3: Analytic Geometry and Inner Products

## Overview

This lecture covers analytic geometry, distance measures, and inner products - essential concepts for understanding similarity and dissimilarity in machine learning.

## 1. Motivation: Distance Between Vectors

### Similarity and Dissimilarity

**Similarity** is a numerical measure of how alike two data objects are. Higher values indicate more similarity.

**Dissimilarity** is a numerical measure of how different two data objects are. Lower values indicate more similarity.

### Why Study Distance?

- **Clustering:** Group similar data points
- **Classification:** Measure distance to class centers
- **Recommendation Systems:** Find similar users/items
- **Dimensionality Reduction:** Preserve distances

## 2. The Data Matrix and Dissimilarity Matrix

### Data Matrix

A **data matrix** is an $n \times p$ matrix where:
- $n$ rows represent data objects
- $p$ columns represent attributes/features

$$\begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{np}
\end{pmatrix}$$

### Dissimilarity Matrix

A **dissimilarity matrix** is an $n \times n$ matrix where entry $d_{ij}$ represents the dissimilarity between objects $i$ and $j$.

**Properties:**
- $d_{ii} = 0$ (distance from object to itself is zero)
- $d_{ij} = d_{ji}$ (symmetric)
- $d_{ij} \geq 0$ (non-negative)

## 3. Dissimilarity Measures

### Euclidean Distance

For vectors $x = (x_1, \ldots, x_p)$ and $y = (y_1, \ldots, y_p)$:

$$d(x, y) = \sqrt{\sum_{i=1}^{p} (x_i - y_i)^2} = \|x - y\|_2$$

### Manhattan Distance

$$d(x, y) = \sum_{i=1}^{p} |x_i - y_i| = \|x - y\|_1$$

### Example: Dissimilarity Matrix with Euclidean Distance

Suppose our data matrix of four data objects is:

$$X = \begin{pmatrix}
1 & 2 \\
3 & 4 \\
5 & 6 \\
7 & 8
\end{pmatrix}$$

Compute Euclidean distances:

- $d(x_1, x_2) = \sqrt{(1-3)^2 + (2-4)^2} = \sqrt{4 + 4} = \sqrt{8} = 2\sqrt{2}$
- $d(x_1, x_3) = \sqrt{(1-5)^2 + (2-6)^2} = \sqrt{16 + 16} = \sqrt{32} = 4\sqrt{2}$
- $d(x_1, x_4) = \sqrt{(1-7)^2 + (2-8)^2} = \sqrt{36 + 36} = \sqrt{72} = 6\sqrt{2}$
- $d(x_2, x_3) = \sqrt{(3-5)^2 + (4-6)^2} = 2\sqrt{2}$
- $d(x_2, x_4) = \sqrt{(3-7)^2 + (4-8)^2} = 4\sqrt{2}$
- $d(x_3, x_4) = \sqrt{(5-7)^2 + (6-8)^2} = 2\sqrt{2}$

Dissimilarity matrix:
$$D = \begin{pmatrix}
0 & 2\sqrt{2} & 4\sqrt{2} & 6\sqrt{2} \\
2\sqrt{2} & 0 & 2\sqrt{2} & 4\sqrt{2} \\
4\sqrt{2} & 2\sqrt{2} & 0 & 2\sqrt{2} \\
6\sqrt{2} & 4\sqrt{2} & 2\sqrt{2} & 0
\end{pmatrix}$$

## 4. Dot Product in $\mathbb{R}^n$

### Definition

For vectors $x = (x_1, \ldots, x_n)$ and $y = (y_1, \ldots, y_n)$ in $\mathbb{R}^n$:

$$x \cdot y = \sum_{i=1}^{n} x_i y_i = x^T y$$

### Properties

- **Commutative:** $x \cdot y = y \cdot x$
- **Distributive:** $x \cdot (y + z) = x \cdot y + x \cdot z$
- **Scalar multiplication:** $(\lambda x) \cdot y = \lambda(x \cdot y)$
- **Relation to norm:** $x \cdot x = \|x\|^2$

## 5. Bilinear, Symmetric, and Positive Definite Mappings

### Bilinear Mapping

A **bilinear mapping** $\Omega : V \times V \to \mathbb{R}$ is a mapping with two arguments that is linear in both arguments:

- $\Omega(\lambda x + \mu y, z) = \lambda \Omega(x, z) + \mu \Omega(y, z)$
- $\Omega(x, \lambda y + \mu z) = \lambda \Omega(x, y) + \mu \Omega(x, z)$

### Symmetric Mapping

A mapping $\Omega$ is **symmetric** if:

$$\Omega(x, y) = \Omega(y, x) \quad \forall x, y \in V$$

### Positive Definite Mapping

A mapping $\Omega$ is **positive definite** if:

$$\Omega(x, x) > 0 \quad \forall x \neq 0$$

and

$$\Omega(0, 0) = 0$$

## 6. Inner Products

### Definition

An **inner product** is a positive-definite, symmetric bilinear mapping:

$$\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$$

### Standard Inner Product (Dot Product)

In $\mathbb{R}^n$, the standard inner product is:

$$\langle x, y \rangle = x^T y = \sum_{i=1}^{n} x_i y_i$$

### Weighted Inner Product

Given a symmetric, positive-definite matrix $A$, we can define an inner product:

$$\langle x, y \rangle_A = x^T A y$$

### Example

Let $A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$.

**Check if $A$ is symmetric:**
$$A^T = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix} = A$$ ✓

**Check if $A$ is positive definite:**
- Leading minor 1: $2 > 0$ ✓
- Leading minor 2: $\det(A) = 6 - 1 = 5 > 0$ ✓

So $A$ is symmetric and positive definite.

**Inner product:**
$$\langle x, y \rangle_A = x^T A y = \begin{pmatrix} x_1 & x_2 \end{pmatrix} \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix} \begin{pmatrix} y_1 \\ y_2 \end{pmatrix}$$

$$= \begin{pmatrix} x_1 & x_2 \end{pmatrix} \begin{pmatrix} 2y_1 + y_2 \\ y_1 + 3y_2 \end{pmatrix} = 2x_1 y_1 + x_1 y_2 + x_2 y_1 + 3x_2 y_2$$

## 7. Symmetric and Positive Definite Matrices

### Symmetric Matrix

A matrix $A$ is **symmetric** if $A = A^T$.

**Properties:**
- All eigenvalues are real
- Eigenvectors corresponding to distinct eigenvalues are orthogonal
- Can be diagonalized: $A = PDP^T$ where $P$ is orthogonal

### Positive Definite Matrix

A symmetric matrix $A$ is **positive definite** if:

$$x^T A x > 0 \quad \forall x \neq 0$$

### Equivalent Conditions

A symmetric matrix $A$ is positive definite if and only if:

1. All eigenvalues are positive
2. All leading principal minors are positive
3. $A = B^T B$ for some invertible matrix $B$
4. $x^T A x > 0$ for all $x \neq 0$

### Checking Positive Definiteness

**Method 1: Leading Principal Minors**

For $A = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$:
- $D_1 = a_{11} > 0$
- $D_2 = \det(A) = a_{11}a_{22} - a_{12}a_{21} > 0$

**Method 2: Eigenvalues**

Compute eigenvalues. If all are positive, matrix is positive definite.

## 8. Norm from Inner Product

Given an inner product $\langle \cdot, \cdot \rangle$, the **induced norm** is:

$$\|x\| = \sqrt{\langle x, x \rangle}$$

### Properties

- $\|x\| \geq 0$ and $\|x\| = 0$ iff $x = 0$
- $\|\lambda x\| = |\lambda| \|x\|$
- **Cauchy-Schwarz:** $|\langle x, y \rangle| \leq \|x\| \|y\|$
- **Triangle inequality:** $\|x + y\| \leq \|x\| + \|y\|$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Dissimilarity Matrix:** Always symmetric with zeros on diagonal
2. **Euclidean Distance:** Most common measure, use $\sqrt{\sum (x_i - y_i)^2}$
3. **Inner Product Verification:** Check symmetry, bilinearity, and positive definiteness
4. **Positive Definite:** Use leading principal minors or eigenvalues
5. **Weighted Inner Product:** $\langle x, y \rangle_A = x^T A y$ where $A$ is symmetric positive definite
```

---

## 🔍 Worked Examples

### Example 1: Compute Dissimilarity Matrix

Given data points: $(1, 2)$, $(4, 6)$, $(7, 3)$

**Euclidean distances:**
- $d((1,2), (4,6)) = \sqrt{(1-4)^2 + (2-6)^2} = \sqrt{9 + 16} = 5$
- $d((1,2), (7,3)) = \sqrt{(1-7)^2 + (2-3)^2} = \sqrt{36 + 1} = \sqrt{37}$
- $d((4,6), (7,3)) = \sqrt{(4-7)^2 + (6-3)^2} = \sqrt{9 + 9} = 3\sqrt{2}$

**Dissimilarity matrix:**
$$D = \begin{pmatrix}
0 & 5 & \sqrt{37} \\
5 & 0 & 3\sqrt{2} \\
\sqrt{37} & 3\sqrt{2} & 0
\end{pmatrix}$$

### Example 2: Weighted Inner Product

Given $A = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}$ and $x = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$, $y = \begin{pmatrix} 1 \\ 3 \end{pmatrix}$.

Compute $\langle x, y \rangle_A$:

$$\langle x, y \rangle_A = x^T A y = \begin{pmatrix} 2 & 1 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix} \begin{pmatrix} 1 \\ 3 \end{pmatrix}$$

$$= \begin{pmatrix} 2 & 1 \end{pmatrix} \begin{pmatrix} 6 \\ 7 \end{pmatrix} = 12 + 7 = 19$$

---

## 📚 Quick Revision Checklist

- [ ] Similarity vs dissimilarity concepts
- [ ] Data matrix and dissimilarity matrix
- [ ] Euclidean and Manhattan distances
- [ ] Dot product in $\mathbb{R}^n$
- [ ] Bilinear, symmetric, positive definite mappings
- [ ] Inner product definition and properties
- [ ] Weighted inner products with matrices
- [ ] Symmetric and positive definite matrices
- [ ] Checking positive definiteness
- [ ] Norm from inner product

