# Lecture 5: Matrix Decompositions

## Overview

This lecture covers important matrix decompositions: Cholesky decomposition, diagonalization, and Singular Value Decomposition (SVD) - all crucial for machine learning applications.

## 1. Cholesky Decomposition

### Introduction

This decomposition is similar to the square-root operation that gives us a decomposition of the number. For matrices, Cholesky decomposition factors a matrix into a "square root" form.

### Definition

For a symmetric, positive-definite matrix $A$, the **Cholesky decomposition** is:

$$A = LL^T$$

where $L$ is a **lower triangular** matrix with positive diagonal entries.

### Conditions

- $A$ must be **symmetric:** $A = A^T$
- $A$ must be **positive definite:** $x^T A x > 0$ for all $x \neq 0$

### Algorithm

For $A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn}
\end{pmatrix}$

Compute $L = \begin{pmatrix}
l_{11} & 0 & \cdots & 0 \\
l_{21} & l_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
l_{n1} & l_{n2} & \cdots & l_{nn}
\end{pmatrix}$ such that $A = LL^T$

**Formulas:**
- $l_{11} = \sqrt{a_{11}}$
- For $i > 1$: $l_{i1} = \frac{a_{i1}}{l_{11}}$
- For $j = 2, \ldots, n$:
  - $l_{jj} = \sqrt{a_{jj} - \sum_{k=1}^{j-1} l_{jk}^2}$
  - For $i > j$: $l_{ij} = \frac{1}{l_{jj}}\left(a_{ij} - \sum_{k=1}^{j-1} l_{ik} l_{jk}\right)$

### Example: 2×2 Matrix

Decompose $A = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$.

**Check conditions:**
- Symmetric: $A^T = A$ ✓
- Positive definite: Leading minors are $4 > 0$ and $\det(A) = 12 - 4 = 8 > 0$ ✓

**Decomposition:**
- $l_{11} = \sqrt{4} = 2$
- $l_{21} = \frac{2}{2} = 1$
- $l_{22} = \sqrt{3 - 1^2} = \sqrt{2}$

$$L = \begin{pmatrix} 2 & 0 \\ 1 & \sqrt{2} \end{pmatrix}$$

**Verify:**
$$LL^T = \begin{pmatrix} 2 & 0 \\ 1 & \sqrt{2} \end{pmatrix} \begin{pmatrix} 2 & 1 \\ 0 & \sqrt{2} \end{pmatrix} = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix} = A$$ ✓

### Example: 3×3 Matrix

Decompose $A = \begin{pmatrix}
9 & 3 & 0 \\
3 & 5 & 2 \\
0 & 2 & 4
\end{pmatrix}$.

**Step 1:** $l_{11} = \sqrt{9} = 3$

**Step 2:** $l_{21} = \frac{3}{3} = 1$, $l_{31} = \frac{0}{3} = 0$

**Step 3:** $l_{22} = \sqrt{5 - 1^2} = \sqrt{4} = 2$

**Step 4:** $l_{32} = \frac{1}{2}(2 - 0 \cdot 1) = 1$

**Step 5:** $l_{33} = \sqrt{4 - 0^2 - 1^2} = \sqrt{3}$

$$L = \begin{pmatrix}
3 & 0 & 0 \\
1 & 2 & 0 \\
0 & 1 & \sqrt{3}
\end{pmatrix}$$

### Applications in ML

In ML, symmetric, positive-definite matrices require frequent manipulations:
- **Covariance matrices** are symmetric positive-definite
- **Kernel matrices** in kernel methods
- **Hessian matrices** in optimization (when positive definite)

Cholesky decomposition is computationally efficient (about half the operations of LU decomposition).

## 2. Diagonalization Theorem

### Theorem

An $n \times n$ matrix $A$ is **diagonalizable** if and only if $A$ has $n$ linearly independent eigenvectors.

If $A$ is diagonalizable, then:

$$A = PDP^{-1}$$

where:
- $D$ is a diagonal matrix containing eigenvalues
- $P$ is a matrix whose columns are eigenvectors

### Conditions for Diagonalization

1. $A$ has $n$ linearly independent eigenvectors
2. If $A$ is symmetric, it is always diagonalizable
3. If eigenvalues are distinct, eigenvectors are linearly independent

### Algorithm

1. Find eigenvalues $\lambda_1, \ldots, \lambda_n$
2. Find corresponding eigenvectors $v_1, \ldots, v_n$
3. Form $P = [v_1 | v_2 | \cdots | v_n]$
4. Form $D = \text{diag}(\lambda_1, \ldots, \lambda_n)$
5. Verify: $A = PDP^{-1}$

### Example

Diagonalize $A = \begin{pmatrix} 4 & 1 \\ 0 & 3 \end{pmatrix}$.

**Eigenvalues:** $\lambda_1 = 4$, $\lambda_2 = 3$ (from diagonal entries, since triangular)

**Eigenvectors:**
- For $\lambda_1 = 4$: $v_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$
- For $\lambda_2 = 3$: $v_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$

$$P = \begin{pmatrix} 1 & 1 \\ 0 & -1 \end{pmatrix}, \quad D = \begin{pmatrix} 4 & 0 \\ 0 & 3 \end{pmatrix}$$

$$P^{-1} = \begin{pmatrix} 1 & 1 \\ 0 & -1 \end{pmatrix}^{-1} = \begin{pmatrix} 1 & 1 \\ 0 & -1 \end{pmatrix}$$

**Verify:** $PDP^{-1} = A$ ✓

## 3. Singular Value Decomposition (SVD)

### Definition

For any $m \times n$ matrix $A$ (not necessarily square), the **Singular Value Decomposition** is:

$$A = U \Sigma V^T$$

where:
- $U$ is an $m \times m$ orthogonal matrix (left singular vectors)
- $\Sigma$ is an $m \times n$ diagonal matrix (singular values)
- $V$ is an $n \times n$ orthogonal matrix (right singular vectors)

### Properties

- Singular values: $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ where $r = \text{rank}(A)$
- $U$ columns are eigenvectors of $AA^T$
- $V$ columns are eigenvectors of $A^T A$
- $\sigma_i^2$ are eigenvalues of $AA^T$ (or $A^T A$)

### Computing SVD

1. Compute $AA^T$ and find its eigenvalues/eigenvectors → gives $U$ and $\sigma_i^2$
2. Compute $A^T A$ and find its eigenvalues/eigenvectors → gives $V$
3. Arrange singular values in descending order

### Example

Find SVD of $A = \begin{pmatrix} 3 & 0 \\ 0 & -2 \end{pmatrix}$.

**Step 1:** $AA^T = \begin{pmatrix} 9 & 0 \\ 0 & 4 \end{pmatrix}$

Eigenvalues: $\lambda_1 = 9$, $\lambda_2 = 4$
Eigenvectors: $u_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, $u_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$

Singular values: $\sigma_1 = 3$, $\sigma_2 = 2$

**Step 2:** $A^T A = \begin{pmatrix} 9 & 0 \\ 0 & 4 \end{pmatrix}$ (same as $AA^T$)

$V = U = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$

**Result:**
$$A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}^T$$

### Applications of SVD in ML

1. **Dimensionality Reduction:** High-dimensional data can be noisy and redundant. We truncate small singular values to reduce dimensions (used in PCA)

2. **Image Compression:** Keep only largest singular values

3. **Recommendation Systems:** Matrix factorization

4. **Pseudoinverse:** $A^+ = V \Sigma^+ U^T$ where $\Sigma^+$ has $1/\sigma_i$ for non-zero $\sigma_i$

5. **Low-rank Approximation:** $A_k = U_k \Sigma_k V_k^T$ approximates $A$ with rank $k$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Cholesky:** Always verify matrix is symmetric and positive definite first
2. **Diagonalization:** Check that eigenvectors are linearly independent
3. **SVD:** Works for any matrix (not just square)
4. **Computational Cost:** Cholesky is O(n³/3), faster than LU for SPD matrices
5. **Common Mistake:** Forgetting to check positive definiteness for Cholesky
```

---

## 🔍 Worked Examples

### Example 1: Cholesky Decomposition

Decompose $A = \begin{pmatrix} 16 & 4 \\ 4 & 5 \end{pmatrix}$.

**Check:** Symmetric ✓, Positive definite (16 > 0, det = 64 > 0) ✓

**Decomposition:**
- $l_{11} = \sqrt{16} = 4$
- $l_{21} = \frac{4}{4} = 1$
- $l_{22} = \sqrt{5 - 1^2} = 2$

$$L = \begin{pmatrix} 4 & 0 \\ 1 & 2 \end{pmatrix}$$

### Example 2: Diagonalization

Diagonalize $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$.

**Eigenvalues:**
$$\det(A - \lambda I) = \det\begin{pmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{pmatrix} = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = 0$$

$\lambda_1 = 3$, $\lambda_2 = 1$

**Eigenvectors:**
- For $\lambda_1 = 3$: $v_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$
- For $\lambda_2 = 1$: $v_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$

$$P = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad D = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}$$

---

## 📚 Quick Revision Checklist

- [ ] Cholesky decomposition algorithm
- [ ] Conditions for Cholesky (symmetric, positive definite)
- [ ] Diagonalization theorem
- [ ] When a matrix is diagonalizable
- [ ] SVD definition and properties
- [ ] Computing SVD
- [ ] Applications in ML (dimensionality reduction, compression, etc.)

