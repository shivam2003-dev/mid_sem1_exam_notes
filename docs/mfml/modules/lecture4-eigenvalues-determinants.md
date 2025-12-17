# Lecture 4: Eigenvalues, Eigenvectors, and Determinants

## Overview

This lecture covers eigenvalues, eigenvectors, and determinants - fundamental concepts for matrix decompositions and understanding linear transformations.

## 1. Matrix Decompositions

We studied vectors and how to manipulate them in preceding lectures. Now we study **mappings and transformations** of vectors, which leads to matrix decompositions.

## 2. Eigenvalues and Eigenvectors

### Definition

For an $n \times n$ matrix $A$, if there exists a scalar $\lambda$ and a non-zero vector $v$ such that:

$$Av = \lambda v$$

then:
- $\lambda$ is called an **eigenvalue** of $A$
- $v$ is called an **eigenvector** of $A$ corresponding to $\lambda$

### Geometrical Interpretation

- **Eigenvalue:** Scaling factor
- **Eigenvector:** Direction that remains unchanged under the transformation
- The transformation $A$ stretches or compresses the eigenvector by the eigenvalue factor

### Why Study Eigenvalues?

There are many applications of eigenvalues and eigenvectors:
- **Weather prediction**
- **Population estimation**
- **Principal Component Analysis (PCA)**
- **PageRank algorithm**
- **Vibration analysis**
- **Image compression**

## 3. Triangular Matrices

A **triangular matrix** is a special case of a square matrix where all elements above or below the principal diagonal are zero.

### Upper Triangular Matrix

$$U = \begin{pmatrix}
u_{11} & u_{12} & u_{13} \\
0 & u_{22} & u_{23} \\
0 & 0 & u_{33}
\end{pmatrix}$$

### Lower Triangular Matrix

$$L = \begin{pmatrix}
l_{11} & 0 & 0 \\
l_{21} & l_{22} & 0 \\
l_{31} & l_{32} & l_{33}
\end{pmatrix}$$

### Eigenvalues of Triangular Matrix

**Theorem:** The eigenvalues of a triangular matrix are its diagonal entries.

**Proof:** For an upper triangular matrix $U$, the characteristic equation is:

$$\det(U - \lambda I) = \prod_{i=1}^{n} (u_{ii} - \lambda) = 0$$

So eigenvalues are $\lambda_i = u_{ii}$ for $i = 1, \ldots, n$.

## 4. How to Find Eigenvalues

### Characteristic Equation

To find eigenvalues of matrix $A$:

1. Form $A - \lambda I$
2. Compute $\det(A - \lambda I) = 0$
3. Solve the characteristic polynomial

The equation $|A - \lambda I| = 0$ is called the **characteristic equation**.

The polynomial $|A - \lambda I|$ is called the **characteristic polynomial**.

## 5. Determinants

### Minor and Cofactor

For an $n \times n$ matrix $A$:

- **Minor** $M_{ij}$: Determinant of the $(n-1) \times (n-1)$ matrix obtained by deleting row $i$ and column $j$ from $A$
- **Cofactor** $C_{ij} = (-1)^{i+j} M_{ij}$

### Definition of Determinant

For an $n \times n$ matrix $A$, the **determinant** is:

$$\det(A) = \sum_{j=1}^{n} a_{1j} C_{1j} = \sum_{j=1}^{n} a_{1j} (-1)^{1+j} M_{1j}$$

(Expansion along the first row)

### Determinant of 2×2 Matrix

For $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:

$$\det(A) = ad - bc$$

**Example:**
$$\det\begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix} = 2 \cdot 4 - 3 \cdot 1 = 8 - 3 = 5$$

### Determinant of 3×3 Matrix

For $A = \begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{pmatrix}$:

Using expansion along first row:

$$\det(A) = a_{11}M_{11} - a_{12}M_{12} + a_{13}M_{13}$$

where:
- $M_{11} = \det\begin{pmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{pmatrix} = a_{22}a_{33} - a_{23}a_{32}$
- $M_{12} = \det\begin{pmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{pmatrix} = a_{21}a_{33} - a_{23}a_{31}$
- $M_{13} = \det\begin{pmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{pmatrix} = a_{21}a_{32} - a_{22}a_{31}$

**Example:**

$$\det\begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{pmatrix} = 1 \cdot \det\begin{pmatrix} 5 & 6 \\ 8 & 9 \end{pmatrix} - 2 \cdot \det\begin{pmatrix} 4 & 6 \\ 7 & 9 \end{pmatrix} + 3 \cdot \det\begin{pmatrix} 4 & 5 \\ 7 & 8 \end{pmatrix}$$

$$= 1(45 - 48) - 2(36 - 42) + 3(32 - 35) = -3 + 12 - 9 = 0$$

## 6. Determinant and Elementary Row Operations

### Properties

1. **Row Interchange:** If $B$ is obtained from $A$ by interchanging two rows, then $\det(B) = -\det(A)$

2. **Row Scaling:** If $B$ is obtained from $A$ by multiplying a row by $k$, then $\det(B) = k \det(A)$

3. **Row Addition:** If $B$ is obtained from $A$ by adding a multiple of one row to another, then $\det(B) = \det(A)$

### Alternating Way of Defining Determinant

Let $A$ be an $n \times n$ matrix and $U$ be the echelon form of $A$ obtained by elementary row operations.

Then:
$$\det(A) = (-1)^r \cdot \frac{\det(U)}{\prod k_i}$$

where:
- $r$ = number of row interchanges
- $k_i$ = scaling factors used

## 7. Equivalent Conditions for $\det(A) \neq 0$

**Theorem:** The following are equivalent for an $n \times n$ matrix $A$:

1. $\det(A) \neq 0$
2. $A$ is invertible
3. $A$ has full rank ($\text{rank}(A) = n$)
4. The columns (rows) of $A$ are linearly independent
5. The system $Ax = 0$ has only the trivial solution
6. The system $Ax = b$ has a unique solution for any $b$

## 8. Computing Eigenvalues and Eigenvectors

### Algorithm

1. **Find Eigenvalues:**
   - Solve $\det(A - \lambda I) = 0$
   - This gives the characteristic polynomial
   - Find roots (eigenvalues)

2. **Find Eigenvectors:**
   - For each eigenvalue $\lambda_i$, solve $(A - \lambda_i I)v = 0$
   - The non-zero solutions are eigenvectors

### Example 1

Find eigenvalues and eigenvectors of $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$.

**Step 1: Characteristic equation**

$$A - \lambda I = \begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix}$$

$$\det(A - \lambda I) = (3-\lambda)(2-\lambda) = 0$$

Eigenvalues: $\lambda_1 = 3$, $\lambda_2 = 2$

**Step 2: Eigenvectors**

For $\lambda_1 = 3$:
$$(A - 3I)v = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives $v_2 = 0$, so $v_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ (or any scalar multiple)

For $\lambda_2 = 2$:
$$(A - 2I)v = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives $v_1 + v_2 = 0$, so $v_2 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$ (or any scalar multiple)

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Eigenvalue Equation:** Always use $Av = \lambda v$ or $\det(A - \lambda I) = 0$
2. **Determinant 2×2:** Use formula $ad - bc$
3. **Determinant 3×3:** Expand along first row using cofactors
4. **Triangular Matrix:** Eigenvalues are diagonal entries
5. **Finding Eigenvectors:** Solve $(A - \lambda I)v = 0$ for each eigenvalue
6. **Common Mistake:** Forgetting that eigenvectors are defined up to scalar multiples
```

---

## 🔍 Worked Examples

### Example 1: Find Eigenvalues

Find eigenvalues of $A = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix}$.

**Characteristic equation:**
$$\det(A - \lambda I) = \det\begin{pmatrix} 4-\lambda & 2 \\ 1 & 3-\lambda \end{pmatrix} = (4-\lambda)(3-\lambda) - 2 = 0$$

$$12 - 7\lambda + \lambda^2 - 2 = \lambda^2 - 7\lambda + 10 = 0$$

$$(\lambda - 2)(\lambda - 5) = 0$$

Eigenvalues: $\lambda_1 = 2$, $\lambda_2 = 5$

### Example 2: Determinant of 3×3

Compute $\det\begin{pmatrix}
2 & 0 & 1 \\
1 & 3 & 2 \\
0 & 1 & 1
\end{pmatrix}$.

**Expansion along first row:**
$$\det = 2 \cdot \det\begin{pmatrix} 3 & 2 \\ 1 & 1 \end{pmatrix} - 0 \cdot \det\begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix} + 1 \cdot \det\begin{pmatrix} 1 & 3 \\ 0 & 1 \end{pmatrix}$$

$$= 2(3 - 2) + 0 + 1(1 - 0) = 2 + 1 = 3$$

---

## 📚 Quick Revision Checklist

- [ ] Definition of eigenvalues and eigenvectors
- [ ] Geometrical interpretation
- [ ] Triangular matrices and their eigenvalues
- [ ] Characteristic equation and polynomial
- [ ] Minor and cofactor
- [ ] Determinant of 2×2 and 3×3 matrices
- [ ] Determinant and elementary row operations
- [ ] Conditions for $\det(A) \neq 0$
- [ ] Algorithm for finding eigenvalues and eigenvectors
- [ ] Applications of eigenvalues

