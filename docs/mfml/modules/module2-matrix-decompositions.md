# Module 2: Matrix Decompositions

## Overview

Matrix decompositions are fundamental tools in linear algebra and machine learning. They break down matrices into simpler, more manageable components that reveal important properties and enable efficient computations.

## 1. LU Decomposition

### Definition

**LU Decomposition** factors a square matrix $A$ into:
$$A = LU$$

where:
- $L$ is a **lower triangular** matrix (all entries above diagonal are zero)
- $U$ is an **upper triangular** matrix (all entries below diagonal are zero)

### Algorithm

1. Start with $A$ and perform Gaussian elimination
2. Record the multipliers in $L$
3. The resulting upper triangular matrix is $U$

### Example

Decompose $A = \begin{pmatrix} 2 & 1 & 0 \\ 4 & 3 & 1 \\ 2 & 1 & 1 \end{pmatrix}$

**Step 1:** Eliminate $a_{21}$:
- Multiplier: $l_{21} = \frac{4}{2} = 2$
- $R_2 \leftarrow R_2 - 2R_1$

**Step 2:** Eliminate $a_{31}$:
- Multiplier: $l_{31} = \frac{2}{2} = 1$
- $R_3 \leftarrow R_3 - R_1$

**Step 3:** Eliminate $a_{32}$:
- Multiplier: $l_{32} = \frac{0}{1} = 0$
- Already zero

Result:
$$L = \begin{pmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 1 & 0 & 1 \end{pmatrix}, \quad U = \begin{pmatrix} 2 & 1 & 0 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{pmatrix}$$

### Solving Linear Systems

For $Ax = b$:
1. Factor $A = LU$
2. Solve $Ly = b$ (forward substitution)
3. Solve $Ux = y$ (backward substitution)

---

## 2. QR Decomposition

### Definition

**QR Decomposition** factors a matrix $A$ into:
$$A = QR$$

where:
- $Q$ is an **orthogonal** matrix ($Q^T Q = I$)
- $R$ is an **upper triangular** matrix

### Gram-Schmidt Process

The columns of $Q$ are orthonormal vectors obtained by applying Gram-Schmidt to columns of $A$.

**Algorithm:**
1. $q_1 = \frac{a_1}{\|a_1\|}$
2. For $i = 2, \ldots, n$:
   - $v_i = a_i - \sum_{j=1}^{i-1} \langle a_i, q_j \rangle q_j$
   - $q_i = \frac{v_i}{\|v_i\|}$

### Properties

- $R = Q^T A$ (upper triangular)
- Used in least squares problems
- Numerically stable

---

## 3. Cholesky Decomposition

### Definition

**Cholesky Decomposition** factors a symmetric positive definite matrix $A$ into:
$$A = LL^T$$

where $L$ is a **lower triangular** matrix with positive diagonal entries.

### Conditions

- $A$ must be **symmetric**: $A = A^T$
- $A$ must be **positive definite**: $x^T A x > 0$ for all $x \neq 0$

### Algorithm

For $i = 1, \ldots, n$:
- $l_{ii} = \sqrt{a_{ii} - \sum_{k=1}^{i-1} l_{ik}^2}$
- For $j = i+1, \ldots, n$:
  - $l_{ji} = \frac{1}{l_{ii}}\left(a_{ji} - \sum_{k=1}^{i-1} l_{jk} l_{ik}\right)$

### Example

Decompose $A = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$

**Step 1:** $l_{11} = \sqrt{4} = 2$

**Step 2:** $l_{21} = \frac{2}{2} = 1$

**Step 3:** $l_{22} = \sqrt{3 - 1^2} = \sqrt{2}$

$$L = \begin{pmatrix} 2 & 0 \\ 1 & \sqrt{2} \end{pmatrix}$$

---

## 4. Eigen-decomposition (Diagonalization)

### Definition

For a diagonalizable matrix $A$, we can write:
$$A = PDP^{-1}$$

where:
- $D$ is a **diagonal** matrix containing eigenvalues
- $P$ is a matrix whose columns are eigenvectors

### Conditions for Diagonalization

- $A$ must have $n$ linearly independent eigenvectors
- If $A$ is symmetric, it is always diagonalizable

### Algorithm

1. Find eigenvalues $\lambda_1, \ldots, \lambda_n$
2. Find corresponding eigenvectors $v_1, \ldots, v_n$
3. Form $P = [v_1 | v_2 | \cdots | v_n]$
4. Form $D = \text{diag}(\lambda_1, \ldots, \lambda_n)$
5. Verify: $A = PDP^{-1}$

### Powers of Matrices

If $A = PDP^{-1}$, then:
$$A^k = PD^k P^{-1}$$

This is computationally efficient for large powers.

---

## 5. Singular Value Decomposition (SVD)

### Definition

**SVD** decomposes any matrix $A$ (not necessarily square) into:
$$A = U \Sigma V^T$$

where:
- $U$ is an $m \times m$ orthogonal matrix (left singular vectors)
- $\Sigma$ is an $m \times n$ diagonal matrix (singular values)
- $V$ is an $n \times n$ orthogonal matrix (right singular vectors)

### Properties

- Singular values: $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ (where $r = \text{rank}(A)$)
- $U$ columns are eigenvectors of $AA^T$
- $V$ columns are eigenvectors of $A^T A$
- $\sigma_i^2$ are eigenvalues of $AA^T$ (or $A^T A$)

### Computing SVD

1. Compute $AA^T$ and find its eigenvalues/eigenvectors → gives $U$ and $\sigma_i^2$
2. Compute $A^T A$ and find its eigenvalues/eigenvectors → gives $V$
3. Arrange singular values in descending order

### Applications

- Principal Component Analysis (PCA)
- Low-rank approximations
- Image compression
- Pseudoinverse computation

### Example

Find SVD of $A = \begin{pmatrix} 3 & 0 \\ 0 & -2 \end{pmatrix}$

**Step 1:** $AA^T = \begin{pmatrix} 9 & 0 \\ 0 & 4 \end{pmatrix}$

Eigenvalues: $\lambda_1 = 9, \lambda_2 = 4$
Eigenvectors: $u_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, u_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$

Singular values: $\sigma_1 = 3, \sigma_2 = 2$

**Step 2:** $A^T A = \begin{pmatrix} 9 & 0 \\ 0 & 4 \end{pmatrix}$ (same as $AA^T$)

$V = U = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$

**Result:**
$$A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}^T$$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **LU Decomposition:** Always check if pivoting is needed
2. **Cholesky:** Verify matrix is symmetric and positive definite first
3. **Eigen-decomposition:** Check that eigenvectors are linearly independent
4. **SVD:** Remember it works for any matrix (not just square)
5. **Computational Cost:** LU is O(n³), Cholesky is faster for SPD matrices
```

---

## 🔍 Worked Examples

### Example 1: LU Decomposition

Decompose $A = \begin{pmatrix} 2 & 1 \\ 4 & 3 \end{pmatrix}$

**Solution:**

Using Gaussian elimination:
- $R_2 \leftarrow R_2 - 2R_1$

$$L = \begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix}, \quad U = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix}$$

Verify: $LU = \begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix} \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 4 & 3 \end{pmatrix}$ ✓

### Example 2: Cholesky Decomposition

Decompose $A = \begin{pmatrix} 9 & 3 \\ 3 & 2 \end{pmatrix}$

**Check:** $A$ is symmetric ✓

**Check positive definite:** Leading minors: $9 > 0$, $\det(A) = 9 > 0$ ✓

**Decomposition:**
- $l_{11} = \sqrt{9} = 3$
- $l_{21} = \frac{3}{3} = 1$
- $l_{22} = \sqrt{2 - 1^2} = 1$

$$L = \begin{pmatrix} 3 & 0 \\ 1 & 1 \end{pmatrix}$$

---

## 📚 Quick Revision Checklist

- [ ] LU decomposition algorithm and solving systems
- [ ] QR decomposition via Gram-Schmidt
- [ ] Cholesky decomposition for SPD matrices
- [ ] Eigen-decomposition and diagonalization
- [ ] SVD computation and properties
- [ ] When to use each decomposition

