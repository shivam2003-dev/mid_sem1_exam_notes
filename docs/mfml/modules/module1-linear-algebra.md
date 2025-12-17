# Module 1: Linear Algebra Fundamentals

## Overview

Linear algebra forms the mathematical foundation for machine learning. This module covers vectors, matrices, and fundamental operations that are essential for understanding ML algorithms.

## 1. Vectors and Vector Spaces

### Vectors

A **vector** is an ordered collection of numbers (scalars). In $\mathbb{R}^n$, a vector is written as:

$$v = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

### Vector Operations

**Addition:**
$$u + v = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

**Scalar Multiplication:**
$$c \cdot v = \begin{pmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{pmatrix}$$

**Dot Product (Inner Product):**
$$\langle u, v \rangle = u \cdot v = u^T v = \sum_{i=1}^{n} u_i v_i$$

### Vector Spaces

A **vector space** $V$ over a field $\mathbb{R}$ is a set with two operations (addition and scalar multiplication) satisfying:
- Closure under addition and scalar multiplication
- Associativity and commutativity of addition
- Existence of zero vector and additive inverses
- Distributivity properties

---

## 2. Matrices and Matrix Operations

### Matrix Definition

A **matrix** $A$ of size $m \times n$ is a rectangular array:

$$A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}$$

### Matrix Operations

**Matrix Addition:**
$$(A + B)_{ij} = a_{ij} + b_{ij}$$

**Matrix Multiplication:**
$$(AB)_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

**Matrix Transpose:**
$$(A^T)_{ij} = a_{ji}$$

**Matrix Inverse:**
For square matrix $A$, if $\det(A) \neq 0$:
$$AA^{-1} = A^{-1}A = I$$

### Special Matrices

- **Identity Matrix:** $I$ where $I_{ij} = \delta_{ij}$ (Kronecker delta)
- **Diagonal Matrix:** $D$ where $D_{ij} = 0$ for $i \neq j$
- **Symmetric Matrix:** $A = A^T$
- **Orthogonal Matrix:** $A^T A = AA^T = I$

---

## 3. Linear Transformations

A **linear transformation** $T: \mathbb{R}^n \to \mathbb{R}^m$ satisfies:

$$T(au + bv) = aT(u) + bT(v)$$

Every linear transformation can be represented by a matrix $A$ such that:

$$T(x) = Ax$$

---

## 4. Determinants

### 2×2 Matrix

$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

### 3×3 Matrix (Sarrus' Rule)

$$\det\begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{pmatrix} = a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})$$

### Properties

- $\det(AB) = \det(A)\det(B)$
- $\det(A^T) = \det(A)$
- $\det(A^{-1}) = \frac{1}{\det(A)}$
- $\det(cA) = c^n \det(A)$ for $n \times n$ matrix

---

## 5. Eigenvalues and Eigenvectors

### Definition

For matrix $A$, if there exists scalar $\lambda$ and non-zero vector $v$ such that:

$$Av = \lambda v$$

then $\lambda$ is an **eigenvalue** and $v$ is the corresponding **eigenvector**.

### Characteristic Equation

$$\det(A - \lambda I) = 0$$

This gives the characteristic polynomial, whose roots are eigenvalues.

### Finding Eigenvectors

For each eigenvalue $\lambda_i$, solve:

$$(A - \lambda_i I)v = 0$$

### Properties

- Sum of eigenvalues = trace of $A$
- Product of eigenvalues = determinant of $A$
- If $A$ is symmetric, all eigenvalues are real
- Eigenvectors corresponding to distinct eigenvalues are linearly independent

---

## 6. Matrix Rank

The **rank** of a matrix $A$ is:
- The maximum number of linearly independent rows (row rank)
- The maximum number of linearly independent columns (column rank)
- The dimension of the column space (range)

**Properties:**
- $\text{rank}(A) = \text{rank}(A^T)$
- $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$
- For $n \times n$ matrix: $\text{rank}(A) = n$ iff $A$ is invertible

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Matrix Multiplication:** Always check dimensions: $(m \times n) \times (n \times p) = (m \times p)$
2. **Determinant:** Use cofactor expansion for larger matrices
3. **Eigenvalues:** Always verify by checking $\det(A - \lambda I) = 0$
4. **Eigenvectors:** Remember they are defined up to a scalar multiple
```

---

## 🔍 Worked Examples

### Example 1: Matrix Multiplication

Compute $AB$ where:

$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$$

**Solution:**

$$AB = \begin{pmatrix}
1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\
3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8
\end{pmatrix} = \begin{pmatrix}
19 & 22 \\
43 & 50
\end{pmatrix}$$

### Example 2: Finding Eigenvalues and Eigenvectors

Find eigenvalues and eigenvectors of:

$$A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$$

**Solution:**

Characteristic equation:
$$\det(A - \lambda I) = \det\begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix} = (3-\lambda)(2-\lambda) = 0$$

Eigenvalues: $\lambda_1 = 3$, $\lambda_2 = 2$

For $\lambda_1 = 3$:
$$(A - 3I)v = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives $v_2 = 0$, so $v_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ (normalized)

For $\lambda_2 = 2$:
$$(A - 2I)v = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives $v_1 + v_2 = 0$, so $v_2 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$ (normalized: $\frac{1}{\sqrt{2}}\begin{pmatrix} -1 \\ 1 \end{pmatrix}$)

---

## 📚 Quick Revision Checklist

- [ ] Vector addition, scalar multiplication, dot product
- [ ] Matrix addition, multiplication, transpose
- [ ] Determinant computation (2×2, 3×3, general)
- [ ] Matrix inverse (using adjugate or row operations)
- [ ] Finding eigenvalues from characteristic equation
- [ ] Finding eigenvectors for each eigenvalue
- [ ] Understanding matrix rank and its properties

