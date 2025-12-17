# 2024 MidSem Regular MFML - Solved Paper

## Question 1: REF/RREF

**Question:** Transform the following matrix to REF and RREF:

$$A = \begin{pmatrix}
2 & 1 & 3 & 4 \\
4 & 2 & 6 & 8 \\
1 & 2 & 1 & 2
\end{pmatrix}$$

**Solution:**

### Row Echelon Form (REF)

**Augmented Matrix:**
$$\begin{pmatrix}
2 & 1 & 3 & 4 \\
4 & 2 & 6 & 8 \\
1 & 2 & 1 & 2
\end{pmatrix}$$

**Step 1:** $R_1 \leftrightarrow R_3$ (to get 1 in top-left)

$$\begin{pmatrix}
1 & 2 & 1 & 2 \\
4 & 2 & 6 & 8 \\
2 & 1 & 3 & 4
\end{pmatrix}$$

**Step 2:** $R_2 \leftarrow R_2 - 4R_1$

$$\begin{pmatrix}
1 & 2 & 1 & 2 \\
0 & -6 & 2 & 0 \\
2 & 1 & 3 & 4
\end{pmatrix}$$

**Step 3:** $R_3 \leftarrow R_3 - 2R_1$

$$\begin{pmatrix}
1 & 2 & 1 & 2 \\
0 & -6 & 2 & 0 \\
0 & -3 & 1 & 0
\end{pmatrix}$$

**Step 4:** $R_3 \leftarrow R_3 - \frac{1}{2}R_2$

$$\begin{pmatrix}
1 & 2 & 1 & 2 \\
0 & -6 & 2 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix}$$

This is in **REF**.

### Reduced Row Echelon Form (RREF)

**Step 5:** $R_2 \leftarrow -\frac{1}{6}R_2$

$$\begin{pmatrix}
1 & 2 & 1 & 2 \\
0 & 1 & -\frac{1}{3} & 0 \\
0 & 0 & 0 & 0
\end{pmatrix}$$

**Step 6:** $R_1 \leftarrow R_1 - 2R_2$

$$\begin{pmatrix}
1 & 0 & \frac{5}{3} & 2 \\
0 & 1 & -\frac{1}{3} & 0 \\
0 & 0 & 0 & 0
\end{pmatrix}$$

This is in **RREF**.

---

## Question 2: Linear Independence

**Question:** Determine if the following vectors are linearly independent:

$$v_1 = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}, \quad v_2 = \begin{pmatrix} 2 \\ 4 \\ 6 \end{pmatrix}, \quad v_3 = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}$$

**Solution:**

**Form Matrix:**
$$A = \begin{pmatrix}
1 & 2 & 1 \\
2 & 4 & 1 \\
3 & 6 & 1
\end{pmatrix}$$

**Transform to REF:**

**Step 1:** $R_2 \leftarrow R_2 - 2R_1$

$$\begin{pmatrix}
1 & 2 & 1 \\
0 & 0 & -1 \\
3 & 6 & 1
\end{pmatrix}$$

**Step 2:** $R_3 \leftarrow R_3 - 3R_1$

$$\begin{pmatrix}
1 & 2 & 1 \\
0 & 0 & -1 \\
0 & 0 & -2
\end{pmatrix}$$

**Step 3:** $R_3 \leftarrow R_3 - 2R_2$

$$\begin{pmatrix}
1 & 2 & 1 \\
0 & 0 & -1 \\
0 & 0 & 0
\end{pmatrix}$$

**Rank Analysis:**
- Number of non-zero rows = 2
- Number of vectors = 3
- Rank = 2 < 3

**Conclusion:** The vectors are **linearly dependent**.

**Verification:** Notice that $v_2 = 2v_1$, confirming linear dependence.

---

## Question 3: Inner Products and Gram-Schmidt

**Question:** Given inner product $\langle x, y \rangle = x^T M y$ where:

$$M = \begin{pmatrix}
3 & 1 \\
1 & 2
\end{pmatrix}$$

And vectors:
$$u = \begin{pmatrix} 1 \\ 2 \end{pmatrix}, \quad v = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$$

(a) Verify $M$ defines a valid inner product
(b) Compute $\langle u, v \rangle$ and $\|u\|$
(c) Apply Gram-Schmidt to $\{u, v\}$

**Solution:**

**(a) Verification:**

**Symmetry:** $M^T = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix} = M$ ✓

**Positive Definiteness:**
- Leading minor 1: $3 > 0$ ✓
- Leading minor 2: $\det(M) = 6 - 1 = 5 > 0$ ✓

Therefore, $M$ defines a valid inner product.

**(b) Inner Product and Norm:**

**Inner Product:**
$$\langle u, v \rangle = u^T M v = \begin{pmatrix} 1 & 2 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix} \begin{pmatrix} 2 \\ 1 \end{pmatrix}$$

$$= \begin{pmatrix} 1 & 2 \end{pmatrix} \begin{pmatrix} 7 \\ 4 \end{pmatrix} = 7 + 8 = 15$$

**Norm:**
$$\|u\|^2 = \langle u, u \rangle = \begin{pmatrix} 1 & 2 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$

$$= \begin{pmatrix} 1 & 2 \end{pmatrix} \begin{pmatrix} 5 \\ 5 \end{pmatrix} = 5 + 10 = 15$$

$$\|u\| = \sqrt{15}$$

**(c) Gram-Schmidt Process:**

**Step 1: Normalize $u$**
$$e_1 = \frac{u}{\|u\|} = \frac{1}{\sqrt{15}} \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$

**Step 2: Orthogonalize $v$**

$$\langle v, e_1 \rangle = \frac{1}{\sqrt{15}} \langle v, u \rangle = \frac{15}{\sqrt{15}} = \sqrt{15}$$

$$w_2 = v - \langle v, e_1 \rangle e_1 = \begin{pmatrix} 2 \\ 1 \end{pmatrix} - \sqrt{15} \cdot \frac{1}{\sqrt{15}} \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$

$$= \begin{pmatrix} 2 \\ 1 \end{pmatrix} - \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

**Normalize:**
$$\|w_2\|^2 = \langle w_2, w_2 \rangle = \begin{pmatrix} 1 & -1 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix} \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

$$= \begin{pmatrix} 1 & -1 \end{pmatrix} \begin{pmatrix} 2 \\ -1 \end{pmatrix} = 2 + 1 = 3$$

$$\|w_2\| = \sqrt{3}$$

$$e_2 = \frac{w_2}{\|w_2\|} = \frac{1}{\sqrt{3}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

**Orthonormal Set:**
$$\left\{ \frac{1}{\sqrt{15}} \begin{pmatrix} 1 \\ 2 \end{pmatrix}, \frac{1}{\sqrt{3}} \begin{pmatrix} 1 \\ -1 \end{pmatrix} \right\}$$

---

## Question 4: Eigen Decomposition

**Question:** Find eigenvalues and eigenvectors of:

$$A = \begin{pmatrix}
4 & 2 \\
1 & 3
\end{pmatrix}$$

**Solution:**

**Characteristic Equation:**
$$\det(A - \lambda I) = \det\begin{pmatrix}
4-\lambda & 2 \\
1 & 3-\lambda
\end{pmatrix} = 0$$

$$(4-\lambda)(3-\lambda) - 2 = 12 - 7\lambda + \lambda^2 - 2 = \lambda^2 - 7\lambda + 10 = 0$$

$$(\lambda - 2)(\lambda - 5) = 0$$

**Eigenvalues:** $\lambda_1 = 2$, $\lambda_2 = 5$

**Eigenvector for $\lambda_1 = 2$:**

$$(A - 2I)v = \begin{pmatrix}
2 & 2 \\
1 & 1
\end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives $v_1 + v_2 = 0$, so $v_2 = -v_1$.

Choose $v_1 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$ (normalized: $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$)

**Eigenvector for $\lambda_2 = 5$:**

$$(A - 5I)v = \begin{pmatrix}
-1 & 2 \\
1 & -2
\end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives $-v_1 + 2v_2 = 0$, so $v_1 = 2v_2$.

Choose $v_2 = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$ (normalized: $\frac{1}{\sqrt{5}}\begin{pmatrix} 2 \\ 1 \end{pmatrix}$)

**Diagonalization:**
$$A = PDP^{-1}$$

where:
$$P = \begin{pmatrix} 1 & 2 \\ -1 & 1 \end{pmatrix}, \quad D = \begin{pmatrix} 2 & 0 \\ 0 & 5 \end{pmatrix}$$

---

## Question 5: Two-Variable Taylor Series

**Question:** Expand $f(x, y) = e^{x+y}$ around $(0, 0)$ to second order.

**Solution:**

**Compute Derivatives:**

$$f(0, 0) = e^0 = 1$$

$$f_x = e^{x+y} \Rightarrow f_x(0, 0) = 1$$
$$f_y = e^{x+y} \Rightarrow f_y(0, 0) = 1$$

$$f_{xx} = e^{x+y} \Rightarrow f_{xx}(0, 0) = 1$$
$$f_{yy} = e^{x+y} \Rightarrow f_{yy}(0, 0) = 1$$
$$f_{xy} = e^{x+y} \Rightarrow f_{xy}(0, 0) = 1$$

**Second-Order Taylor Expansion:**

$$f(x, y) \approx f(0, 0) + f_x(0, 0)x + f_y(0, 0)y$$
$$+ \frac{1}{2}[f_{xx}(0, 0)x^2 + 2f_{xy}(0, 0)xy + f_{yy}(0, 0)y^2]$$

$$= 1 + x + y + \frac{1}{2}[x^2 + 2xy + y^2]$$

$$= 1 + x + y + \frac{1}{2}(x + y)^2$$

---

## Question 6: Gradients and Hessians

**Question:** For $f(x, y) = x^2 + 2xy + 3y^2$, find:

(a) Gradient
(b) Hessian matrix
(c) Critical points

**Solution:**

**(a) Gradient:**

$$f_x = 2x + 2y$$
$$f_y = 2x + 6y$$

$$\nabla f = \begin{pmatrix} 2x + 2y \\ 2x + 6y \end{pmatrix}$$

**(b) Hessian Matrix:**

$$f_{xx} = 2, \quad f_{yy} = 6, \quad f_{xy} = f_{yx} = 2$$

$$H_f = \begin{pmatrix}
2 & 2 \\
2 & 6
\end{pmatrix}$$

**(c) Critical Points:**

Set $\nabla f = 0$:

$$\begin{cases}
2x + 2y = 0 \\
2x + 6y = 0
\end{cases}$$

From first equation: $x = -y$

Substitute into second: $2(-y) + 6y = 4y = 0 \Rightarrow y = 0$

Therefore: $x = 0$

**Critical Point:** $(0, 0)$

**Check Nature:**
- $\det(H_f) = 12 - 4 = 8 > 0$
- $f_{xx} = 2 > 0$

Therefore, $(0, 0)$ is a **local minimum**.

---

## Summary

All questions solved with detailed step-by-step calculations following exam format.

