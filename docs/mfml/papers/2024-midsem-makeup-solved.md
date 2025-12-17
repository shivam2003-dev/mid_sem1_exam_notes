# 2024 MidSem Makeup MFML - Solved Paper

## Question 1: Cholesky Decomposition

**Question:** Perform Cholesky decomposition of:

$$A = \begin{pmatrix}
9 & 3 & 0 \\
3 & 5 & 2 \\
0 & 2 & 4
\end{pmatrix}$$

**Solution:**

**Step 1: Verify Conditions**

**Symmetry:** $A^T = A$ ✓

**Positive Definiteness:**
- Leading minor 1: $9 > 0$ ✓
- Leading minor 2: $\det\begin{pmatrix} 9 & 3 \\ 3 & 5 \end{pmatrix} = 45 - 9 = 36 > 0$ ✓
- Leading minor 3: $\det(A) = 9(20-4) - 3(12-0) + 0 = 144 - 36 = 108 > 0$ ✓

**Step 2: Cholesky Decomposition**

We want $A = LL^T$ where $L$ is lower triangular.

**Computations:**
- $l_{11} = \sqrt{9} = 3$
- $l_{21} = \frac{3}{3} = 1$
- $l_{31} = \frac{0}{3} = 0$
- $l_{22} = \sqrt{5 - 1^2} = \sqrt{4} = 2$
- $l_{32} = \frac{1}{2}(2 - 0 \cdot 1) = 1$
- $l_{33} = \sqrt{4 - 0^2 - 1^2} = \sqrt{3}$

**Result:**
$$L = \begin{pmatrix}
3 & 0 & 0 \\
1 & 2 & 0 \\
0 & 1 & \sqrt{3}
\end{pmatrix}$$

**Verification:**
$$LL^T = \begin{pmatrix}
3 & 0 & 0 \\
1 & 2 & 0 \\
0 & 1 & \sqrt{3}
\end{pmatrix} \begin{pmatrix}
3 & 1 & 0 \\
0 & 2 & 1 \\
0 & 0 & \sqrt{3}
\end{pmatrix} = \begin{pmatrix}
9 & 3 & 0 \\
3 & 5 & 2 \\
0 & 2 & 4
\end{pmatrix} = A$$ ✓

---

## Question 2: SVD Decomposition

**Question:** Find the SVD of:

$$A = \begin{pmatrix}
2 & 0 \\
0 & 3
\end{pmatrix}$$

**Solution:**

**Step 1: Compute $AA^T$**

$$AA^T = \begin{pmatrix}
2 & 0 \\
0 & 3
\end{pmatrix} \begin{pmatrix}
2 & 0 \\
0 & 3
\end{pmatrix} = \begin{pmatrix}
4 & 0 \\
0 & 9
\end{pmatrix}$$

**Eigenvalues:** $\lambda_1 = 9$, $\lambda_2 = 4$

**Eigenvectors:**
- For $\lambda_1 = 9$: $u_1 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$
- For $\lambda_2 = 4$: $u_2 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$

**Singular Values:** $\sigma_1 = 3$, $\sigma_2 = 2$

**Step 2: Compute $A^T A$**

$$A^T A = \begin{pmatrix}
4 & 0 \\
0 & 9
\end{pmatrix}$$ (same as $AA^T$)

**Eigenvectors:**
- For $\lambda_1 = 9$: $v_1 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$
- For $\lambda_2 = 4$: $v_2 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$

**Step 3: SVD**

$$U = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}, \quad V = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

$$A = U \Sigma V^T = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}^T$$

---

## Question 3: Determinants

**Question:** Compute the determinant of:

$$A = \begin{pmatrix}
1 & 2 & 3 \\
0 & 4 & 5 \\
0 & 0 & 6
\end{pmatrix}$$

**Solution:**

Since $A$ is upper triangular, the determinant is the product of diagonal entries:

$$\det(A) = 1 \times 4 \times 6 = 24$$

**Alternative (Expansion):**

Expanding along first column:

$$\det(A) = 1 \cdot \det\begin{pmatrix} 4 & 5 \\ 0 & 6 \end{pmatrix} - 0 + 0 = 1(24 - 0) = 24$$

---

## Question 4: Linear System Solving

**Question:** Solve the system using matrix methods:

$$\begin{cases}
2x + y = 5 \\
x + 3y = 10
\end{cases}$$

**Solution:**

**Matrix Form:**
$$\begin{pmatrix}
2 & 1 \\
1 & 3
\end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 5 \\ 10 \end{pmatrix}$$

**Method 1: Matrix Inversion**

$$A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}, \quad A^{-1} = \frac{1}{5}\begin{pmatrix} 3 & -1 \\ -1 & 2 \end{pmatrix}$$

$$\begin{pmatrix} x \\ y \end{pmatrix} = A^{-1} \begin{pmatrix} 5 \\ 10 \end{pmatrix} = \frac{1}{5}\begin{pmatrix} 3 & -1 \\ -1 & 2 \end{pmatrix} \begin{pmatrix} 5 \\ 10 \end{pmatrix}$$

$$= \frac{1}{5}\begin{pmatrix} 15 - 10 \\ -5 + 20 \end{pmatrix} = \begin{pmatrix} 1 \\ 3 \end{pmatrix}$$

**Solution:** $x = 1$, $y = 3$

**Method 2: RREF**

Augmented matrix:
$$\begin{pmatrix}
2 & 1 & | & 5 \\
1 & 3 & | & 10
\end{pmatrix}$$

$R_1 \leftrightarrow R_2$:
$$\begin{pmatrix}
1 & 3 & | & 10 \\
2 & 1 & | & 5
\end{pmatrix}$$

$R_2 \leftarrow R_2 - 2R_1$:
$$\begin{pmatrix}
1 & 3 & | & 10 \\
0 & -5 & | & -15
\end{pmatrix}$$

$R_2 \leftarrow -\frac{1}{5}R_2$:
$$\begin{pmatrix}
1 & 3 & | & 10 \\
0 & 1 & | & 3
\end{pmatrix}$$

$R_1 \leftarrow R_1 - 3R_2$:
$$\begin{pmatrix}
1 & 0 & | & 1 \\
0 & 1 & | & 3
\end{pmatrix}$$

**Solution:** $x = 1$, $y = 3$

---

## Summary

All questions solved with multiple methods where applicable.

