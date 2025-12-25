# Mathematics for Machine Learning - Mid-Semester Examination (December 2025) - Solved

This document contains detailed solutions to the Mathematics for Machine Learning Mid-Semester Examination conducted in December 2025.

---

## **Question 1: Linear Systems and Matrix Properties**

### **Part (a): Solve System of Equations [3 Marks]**

**Problem:** Solve the system of linear equations $AX = b$ by finding the Echelon form of the augmented matrix.

**Given System (Reconstructed):**
Let's assume:
$$A = \begin{bmatrix} 2 & 1 & -1 \\ 4 & 3 & 1 \\ -2 & 1 & 3 \end{bmatrix}, \quad b = \begin{bmatrix} 5 \\ 13 \\ 7 \end{bmatrix}$$

**Solution:**

**Step 1: Form augmented matrix $[A|b]$**

$$[A|b] = \begin{bmatrix} 2 & 1 & -1 & | & 5 \\ 4 & 3 & 1 & | & 13 \\ -2 & 1 & 3 & | & 7 \end{bmatrix}$$

**Step 2: Row reduction to Echelon form**

**Operation 1:** $R_2 \leftarrow R_2 - 2R_1$ (eliminate first element of row 2)

$$\begin{bmatrix} 2 & 1 & -1 & | & 5 \\ 0 & 1 & 3 & | & 3 \\ -2 & 1 & 3 & | & 7 \end{bmatrix}$$

Calculation: $[4, 3, 1, 13] - 2[2, 1, -1, 5] = [0, 1, 3, 3]$

**Operation 2:** $R_3 \leftarrow R_3 + R_1$ (eliminate first element of row 3)

$$\begin{bmatrix} 2 & 1 & -1 & | & 5 \\ 0 & 1 & 3 & | & 3 \\ 0 & 2 & 2 & | & 12 \end{bmatrix}$$

Calculation: $[-2, 1, 3, 7] + [2, 1, -1, 5] = [0, 2, 2, 12]$

**Operation 3:** $R_3 \leftarrow R_3 - 2R_2$ (eliminate second element of row 3)

$$\begin{bmatrix} 2 & 1 & -1 & | & 5 \\ 0 & 1 & 3 & | & 3 \\ 0 & 0 & -4 & | & 6 \end{bmatrix}$$

Calculation: $[0, 2, 2, 12] - 2[0, 1, 3, 3] = [0, 0, -4, 6]$

**Echelon Form:**
$$\begin{bmatrix} 2 & 1 & -1 & | & 5 \\ 0 & 1 & 3 & | & 3 \\ 0 & 0 & -4 & | & 6 \end{bmatrix}$$

**Step 3: Back substitution**

From row 3: $-4x_3 = 6 \Rightarrow x_3 = -\frac{3}{2}$

From row 2: $x_2 + 3x_3 = 3 \Rightarrow x_2 + 3(-\frac{3}{2}) = 3 \Rightarrow x_2 = 3 + \frac{9}{2} = \frac{15}{2}$

From row 1: $2x_1 + x_2 - x_3 = 5 \Rightarrow 2x_1 + \frac{15}{2} - (-\frac{3}{2}) = 5$
$$2x_1 + \frac{18}{2} = 5 \Rightarrow 2x_1 = -4 \Rightarrow x_1 = -2$$

**Answer:**
$$X = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} -2 \\ \frac{15}{2} \\ -\frac{3}{2} \end{bmatrix}$$

---

### **Part (b): Determinant and Rank [2 Marks]**

**Problem:** Find the echelon form of matrix $B$, then determine its determinant and rank.

**Given (Example):**
$$B = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{bmatrix}$$

**Solution:**

**Step 1: Row reduction to Echelon form**

$$B = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{bmatrix}$$

$R_2 \leftarrow R_2 - 2R_1$:
$$\begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & -1 \\ 3 & 5 & 6 \end{bmatrix}$$

$R_3 \leftarrow R_3 - 3R_1$:
$$\begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & -1 \\ 0 & -1 & -3 \end{bmatrix}$$

Swap $R_2 \leftrightarrow R_3$ (to get leading entries in descending column order):
$$\begin{bmatrix} 1 & 2 & 3 \\ 0 & -1 & -3 \\ 0 & 0 & -1 \end{bmatrix}$$

**Echelon Form:**
$$\text{Echelon}(B) = \begin{bmatrix} 1 & 2 & 3 \\ 0 & -1 & -3 \\ 0 & 0 & -1 \end{bmatrix}$$

**Step 2: Calculate Determinant**

For the determinant, we account for row operations:
- Each row swap changes sign
- We performed 1 swap, so multiply by $-1$

Determinant of echelon form = product of diagonal elements:
$$\det(\text{Echelon}) = 1 \times (-1) \times (-1) = 1$$

Accounting for the swap:
$$\det(B) = -1 \times 1 = -1$$

**Alternative (Direct calculation):**
$$\det(B) = 1(4 \cdot 6 - 5 \cdot 5) - 2(2 \cdot 6 - 5 \cdot 3) + 3(2 \cdot 5 - 4 \cdot 3)$$
$$= 1(24 - 25) - 2(12 - 15) + 3(10 - 12)$$
$$= 1(-1) - 2(-3) + 3(-2) = -1 + 6 - 6 = -1$$

**Step 3: Calculate Rank**

**Rank** = number of non-zero rows in echelon form = **3**

All three rows have leading entries (pivots), so the matrix has full rank.

**Answer:**
- Determinant of $B$: $\det(B) = -1$
- Rank of $B$: $\text{rank}(B) = 3$

---

## **Question 2: Vector Spaces and Linear Dependence**

### **Part (a): Linear Combination and Dependence [3 Marks]**

**Problem:** Write vector $v$ as a linear combination of vectors $u$ and $w$. Explain how this implies linear dependence in $\mathbb{R}^3$.

**Given (Example):**
$$u = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \quad w = \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix}, \quad v = \begin{bmatrix} 5 \\ 5 \\ 6 \end{bmatrix}$$

**Solution:**

**Step 1: Express $v$ as linear combination**

We seek scalars $\alpha$ and $\beta$ such that:
$$v = \alpha u + \beta w$$

$$\begin{bmatrix} 5 \\ 5 \\ 6 \end{bmatrix} = \alpha \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + \beta \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix}$$

This gives us the system:
$$\begin{cases}
\alpha + 2\beta = 5 \\
2\alpha + \beta = 5 \\
3\alpha = 6
\end{cases}$$

From equation 3: $\alpha = 2$

Substitute into equation 1: $2 + 2\beta = 5 \Rightarrow \beta = \frac{3}{2}$

Verify with equation 2: $2(2) + \frac{3}{2} = 4 + 1.5 = 5.5 \neq 5$

Let me recalculate with corrected values. Let's use:
$$v = \begin{bmatrix} 5 \\ 5 \\ 6 \end{bmatrix}, \quad u = \begin{bmatrix} 1 \\ 1 \\ 2 \end{bmatrix}, \quad w = \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix}$$

System:
$$\begin{cases}
\alpha + 2\beta = 5 \\
\alpha + \beta = 5 \\
2\alpha = 6
\end{cases}$$

From equation 3: $\alpha = 3$

Substitute into equation 2: $3 + \beta = 5 \Rightarrow \beta = 2$

Verify equation 1: $3 + 2(2) = 7 \neq 5$

**Let me use a consistent example:**
$$u = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \quad w = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}, \quad v = \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix}$$

$$v = \alpha u + \beta w$$
$$\begin{cases}
\alpha = 2 \\
\beta = 3 \\
\alpha + \beta = 5
\end{cases}$$

All equations are consistent: $v = 2u + 3w$

**Answer:**
$$v = 2u + 3w$$

**Step 2: Explain Linear Dependence**

**Definition of Linear Dependence:**
Vectors $\{u, w, v\}$ are linearly dependent if there exist scalars $c_1, c_2, c_3$ (not all zero) such that:
$$c_1 u + c_2 w + c_3 v = 0$$

**Proof of Dependence:**

Since $v = 2u + 3w$, we can rearrange:
$$2u + 3w - v = 0$$

This is a non-trivial linear combination (coefficients $2, 3, -1$ are not all zero) that equals zero.

Therefore, $\{u, w, v\}$ are **linearly dependent** in $\mathbb{R}^3$.

**Geometric Interpretation:**
- $v$ lies in the plane spanned by $u$ and $w$
- The three vectors don't span all of $\mathbb{R}^3$
- They span at most a 2-dimensional subspace (the plane containing all three)

**General Principle:**
If one vector can be written as a linear combination of others, the set is linearly dependent.

---

### **Part (b): Prove Subspace, Find Basis and Dimension [2 Marks]**

**Problem:** Prove that $V$ is a subspace of $\mathbb{R}^3$. Find a basis and dimension of $V$.

**Given (Example):**
$$V = \left\{ \begin{bmatrix} x \\ y \\ z \end{bmatrix} \in \mathbb{R}^3 : x + 2y - z = 0 \right\}$$

**Solution:**

**Part 1: Prove $V$ is a subspace**

A subset $V \subseteq \mathbb{R}^3$ is a subspace if:
1. **Contains zero vector:** $\vec{0} \in V$
2. **Closed under addition:** If $u, w \in V$, then $u + w \in V$
3. **Closed under scalar multiplication:** If $u \in V$ and $c \in \mathbb{R}$, then $cu \in V$

**Proof:**

**Property 1: Zero vector**

Check if $\vec{0} = [0, 0, 0]^T$ satisfies $x + 2y - z = 0$:
$$0 + 2(0) - 0 = 0 \quad \checkmark$$

So $\vec{0} \in V$.

**Property 2: Closed under addition**

Let $u = [x_1, y_1, z_1]^T \in V$ and $w = [x_2, y_2, z_2]^T \in V$.

Then:
- $x_1 + 2y_1 - z_1 = 0$
- $x_2 + 2y_2 - z_2 = 0$

Consider $u + w = [x_1 + x_2, y_1 + y_2, z_1 + z_2]^T$:

$$(x_1 + x_2) + 2(y_1 + y_2) - (z_1 + z_2)$$
$$= (x_1 + 2y_1 - z_1) + (x_2 + 2y_2 - z_2) = 0 + 0 = 0 \quad \checkmark$$

So $u + w \in V$.

**Property 3: Closed under scalar multiplication**

Let $u = [x, y, z]^T \in V$ and $c \in \mathbb{R}$.

Then $x + 2y - z = 0$.

Consider $cu = [cx, cy, cz]^T$:

$$(cx) + 2(cy) - (cz) = c(x + 2y - z) = c \cdot 0 = 0 \quad \checkmark$$

So $cu \in V$.

**Conclusion:** $V$ is a subspace of $\mathbb{R}^3$. ✓

**Part 2: Find basis of $V$**

Express the constraint $x + 2y - z = 0$ in parametric form:

$$z = x + 2y$$

So any vector in $V$ can be written as:
$$\begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} x \\ y \\ x + 2y \end{bmatrix} = x \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} + y \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix}$$

Let:
$$v_1 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix}$$

**Verify linear independence:**

$c_1 v_1 + c_2 v_2 = 0$ implies:
$$\begin{bmatrix} c_1 \\ c_2 \\ c_1 + 2c_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$

This gives $c_1 = 0$, $c_2 = 0$ (only trivial solution).

Therefore, $\{v_1, v_2\}$ are linearly independent.

**Basis of $V$:**
$$\mathcal{B} = \left\{ \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix} \right\}$$

**Part 3: Find dimension**

$$\dim(V) = |\mathcal{B}| = 2$$

**Geometric Interpretation:**
$V$ is a plane through the origin in $\mathbb{R}^3$ with normal vector $[1, 2, -1]^T$.

**Answer:**
- $V$ is a subspace (proven above)
- Basis: $\left\{ \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix} \right\}$
- Dimension: $\dim(V) = 2$

---

### **Part (c): Element Membership in Subspace [1 Mark]**

**Problem:** Is the element $p = [1, 1, 2]^T$ in $V$, where $V$ is defined in part (b)?

**Solution:**

**Method 1: Check constraint directly**

For $V = \{[x, y, z]^T : x + 2y - z = 0\}$, check if $p$ satisfies the constraint:

$$x + 2y - z = 1 + 2(1) - 2 = 1 + 2 - 2 = 1 \neq 0$$

Since $1 \neq 0$, the constraint is **not satisfied**.

**Method 2: Express using basis**

Try to write $p$ as a linear combination of basis vectors:
$$\begin{bmatrix} 1 \\ 1 \\ 2 \end{bmatrix} = \alpha \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} + \beta \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix}$$

This gives:
$$\begin{cases}
\alpha = 1 \\
\beta = 1 \\
\alpha + 2\beta = 2
\end{cases}$$

Check third equation: $1 + 2(1) = 3 \neq 2$

**Inconsistent system** → $p$ cannot be expressed using basis of $V$.

**Answer:** No, $p = [1, 1, 2]^T$ is **not** in $V$.

---

## **Question 3: Singular Value Decomposition and Diagonalization**

### **Part (a): Singular Value Decomposition [4 Marks]**

**Problem:** Find the Singular Value Decomposition (SVD) of matrix $A$.

**Given (Example):**
$$A = \begin{bmatrix} 4 & 0 \\ 3 & -5 \end{bmatrix}$$

**Solution:**

SVD decomposes $A = U \Sigma V^T$ where:
- $U$: left singular vectors (eigenvectors of $AA^T$)
- $\Sigma$: diagonal matrix of singular values
- $V$: right singular vectors (eigenvectors of $A^T A$)

**Step 1: Compute $A^T A$**

$$A^T = \begin{bmatrix} 4 & 3 \\ 0 & -5 \end{bmatrix}$$

$$A^T A = \begin{bmatrix} 4 & 3 \\ 0 & -5 \end{bmatrix} \begin{bmatrix} 4 & 0 \\ 3 & -5 \end{bmatrix} = \begin{bmatrix} 25 & -15 \\ -15 & 25 \end{bmatrix}$$

**Step 2: Find eigenvalues of $A^T A$**

$$\det(A^T A - \lambda I) = 0$$

$$\det\begin{bmatrix} 25 - \lambda & -15 \\ -15 & 25 - \lambda \end{bmatrix} = (25 - \lambda)^2 - 225 = 0$$

$$\lambda^2 - 50\lambda + 625 - 225 = 0$$
$$\lambda^2 - 50\lambda + 400 = 0$$

Using quadratic formula:
$$\lambda = \frac{50 \pm \sqrt{2500 - 1600}}{2} = \frac{50 \pm \sqrt{900}}{2} = \frac{50 \pm 30}{2}$$

$$\lambda_1 = 40, \quad \lambda_2 = 10$$

**Step 3: Find singular values**

$$\sigma_1 = \sqrt{\lambda_1} = \sqrt{40} = 2\sqrt{10} \approx 6.325$$
$$\sigma_2 = \sqrt{\lambda_2} = \sqrt{10} \approx 3.162$$

**Step 4: Find right singular vectors (eigenvectors of $A^T A$)**

**For $\lambda_1 = 40$:**

$$(A^T A - 40I)v = 0$$
$$\begin{bmatrix} -15 & -15 \\ -15 & -15 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$-15v_1 - 15v_2 = 0 \Rightarrow v_1 = -v_2$$

Normalized eigenvector: $v^{(1)} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}$

**For $\lambda_2 = 10$:**

$$(A^T A - 10I)v = 0$$
$$\begin{bmatrix} 15 & -15 \\ -15 & 15 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$15v_1 - 15v_2 = 0 \Rightarrow v_1 = v_2$$

Normalized eigenvector: $v^{(2)} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$

$$V = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix}$$

**Step 5: Compute $AA^T$**

$$AA^T = \begin{bmatrix} 4 & 0 \\ 3 & -5 \end{bmatrix} \begin{bmatrix} 4 & 3 \\ 0 & -5 \end{bmatrix} = \begin{bmatrix} 16 & 12 \\ 12 & 34 \end{bmatrix}$$

**Step 6: Find left singular vectors**

Eigenvalues should be the same: $\lambda_1 = 40$, $\lambda_2 = 10$

**For $\lambda_1 = 40$:**

$$(AA^T - 40I)u = 0$$
$$\begin{bmatrix} -24 & 12 \\ 12 & -6 \end{bmatrix} \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$-24u_1 + 12u_2 = 0 \Rightarrow u_2 = 2u_1$$

Normalized: $u^{(1)} = \frac{1}{\sqrt{5}} \begin{bmatrix} 1 \\ 2 \end{bmatrix}$

**For $\lambda_2 = 10$:**

$$(AA^T - 10I)u = 0$$
$$\begin{bmatrix} 6 & 12 \\ 12 & 24 \end{bmatrix} \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$6u_1 + 12u_2 = 0 \Rightarrow u_2 = -\frac{1}{2}u_1$$

Normalized: $u^{(2)} = \frac{1}{\sqrt{5}} \begin{bmatrix} 2 \\ -1 \end{bmatrix}$

$$U = \begin{bmatrix} \frac{1}{\sqrt{5}} & \frac{2}{\sqrt{5}} \\ \frac{2}{\sqrt{5}} & -\frac{1}{\sqrt{5}} \end{bmatrix}$$

**Step 7: Construct $\Sigma$**

$$\Sigma = \begin{bmatrix} 2\sqrt{10} & 0 \\ 0 & \sqrt{10} \end{bmatrix}$$

**Answer: SVD of $A$**

$$A = U \Sigma V^T$$

$$A = \begin{bmatrix} \frac{1}{\sqrt{5}} & \frac{2}{\sqrt{5}} \\ \frac{2}{\sqrt{5}} & -\frac{1}{\sqrt{5}} \end{bmatrix} \begin{bmatrix} 2\sqrt{10} & 0 \\ 0 & \sqrt{10} \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix}$$

---

### **Part (b): Diagonalizability [2 Marks]**

**Problem:** Is matrix $A$ diagonalizable? Justify your answer.

**Given:** Same matrix $A = \begin{bmatrix} 4 & 0 \\ 3 & -5 \end{bmatrix}$

**Solution:**

**Definition:** A matrix $A$ is diagonalizable if it can be written as $A = PDP^{-1}$ where $D$ is diagonal and $P$ is invertible.

**Sufficient condition:** An $n \times n$ matrix is diagonalizable if it has $n$ linearly independent eigenvectors.

**Step 1: Find eigenvalues of $A$**

$$\det(A - \lambda I) = 0$$

$$\det\begin{bmatrix} 4 - \lambda & 0 \\ 3 & -5 - \lambda \end{bmatrix} = (4 - \lambda)(-5 - \lambda) - 0 = 0$$

$$(4 - \lambda)(-5 - \lambda) = 0$$

$$\lambda_1 = 4, \quad \lambda_2 = -5$$

**Step 2: Find eigenvectors**

**For $\lambda_1 = 4$:**

$$(A - 4I)v = 0$$
$$\begin{bmatrix} 0 & 0 \\ 3 & -9 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$3v_1 - 9v_2 = 0 \Rightarrow v_1 = 3v_2$$

Eigenvector: $v^{(1)} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$

**For $\lambda_2 = -5$:**

$$(A - (-5)I)v = 0$$
$$\begin{bmatrix} 9 & 0 \\ 3 & 0 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$9v_1 = 0 \Rightarrow v_1 = 0$$

Eigenvector: $v^{(2)} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$

**Step 3: Check linear independence**

$$P = \begin{bmatrix} 3 & 0 \\ 1 & 1 \end{bmatrix}$$

$$\det(P) = 3(1) - 0(1) = 3 \neq 0$$

Since $\det(P) \neq 0$, the eigenvectors are linearly independent.

**Conclusion:**

**Yes, $A$ is diagonalizable** because:
1. $A$ is a $2 \times 2$ matrix
2. $A$ has 2 distinct eigenvalues ($\lambda_1 = 4$, $\lambda_2 = -5$)
3. $A$ has 2 linearly independent eigenvectors
4. Therefore, $A = PDP^{-1}$ where:

$$D = \begin{bmatrix} 4 & 0 \\ 0 & -5 \end{bmatrix}, \quad P = \begin{bmatrix} 3 & 0 \\ 1 & 1 \end{bmatrix}$$

**Alternative justification:**
- A matrix with $n$ distinct eigenvalues is always diagonalizable
- $A$ has 2 distinct eigenvalues → diagonalizable ✓

---

## **Question 4: Multivariable Calculus and Optimization**

### **Part (a): Compute Gradient [2 Marks]**

**Problem:** Compute the gradient $\nabla f(x, y)$ for $f(x, y) = x^2 + xy + 2y^2 - 4x + 3$.

**Solution:**

The gradient is the vector of partial derivatives:

$$\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}$$

**Calculate $\frac{\partial f}{\partial x}$:**

$$\frac{\partial f}{\partial x} = \frac{\partial}{\partial x}(x^2 + xy + 2y^2 - 4x + 3)$$
$$= 2x + y - 4$$

**Calculate $\frac{\partial f}{\partial y}$:**

$$\frac{\partial f}{\partial y} = \frac{\partial}{\partial y}(x^2 + xy + 2y^2 - 4x + 3)$$
$$= x + 4y$$

**Answer:**

$$\nabla f(x, y) = \begin{bmatrix} 2x + y - 4 \\ x + 4y \end{bmatrix}$$

---

### **Part (b): Find Critical Points [1 Mark]**

**Problem:** Using part (a), find the critical point(s) of $f$.

**Solution:**

Critical points occur where $\nabla f = \vec{0}$:

$$\begin{bmatrix} 2x + y - 4 \\ x + 4y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

This gives the system:
$$\begin{cases}
2x + y = 4 \\
x + 4y = 0
\end{cases}$$

From equation 2: $x = -4y$

Substitute into equation 1:
$$2(-4y) + y = 4$$
$$-8y + y = 4$$
$$-7y = 4$$
$$y = -\frac{4}{7}$$

Then:
$$x = -4\left(-\frac{4}{7}\right) = \frac{16}{7}$$

**Answer:**

Critical point: $\left(\frac{16}{7}, -\frac{4}{7}\right)$

---

### **Part (c): Classify Critical Point [2 Marks]**

**Problem:** Determine whether the critical point is a local minimum, local maximum, or saddle point.

**Solution:**

Use the **Second Derivative Test** (Hessian matrix):

$$H = \begin{bmatrix} f_{xx} & f_{xy} \\ f_{yx} & f_{yy} \end{bmatrix}$$

**Calculate second partial derivatives:**

$$f_{xx} = \frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}(2x + y - 4) = 2$$

$$f_{yy} = \frac{\partial^2 f}{\partial y^2} = \frac{\partial}{\partial y}(x + 4y) = 4$$

$$f_{xy} = \frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial y}(2x + y - 4) = 1$$

$$f_{yx} = \frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial x}(x + 4y) = 1$$

**Hessian matrix:**

$$H = \begin{bmatrix} 2 & 1 \\ 1 & 4 \end{bmatrix}$$

**Compute determinant:**

$$\det(H) = 2(4) - 1(1) = 8 - 1 = 7 > 0$$

**Check $f_{xx}$:**

$$f_{xx} = 2 > 0$$

**Second Derivative Test:**
- If $\det(H) > 0$ and $f_{xx} > 0$: **local minimum**
- If $\det(H) > 0$ and $f_{xx} < 0$: local maximum
- If $\det(H) < 0$: saddle point
- If $\det(H) = 0$: test inconclusive

**Answer:**

The critical point $\left(\frac{16}{7}, -\frac{4}{7}\right)$ is a **local minimum** because:
- $\det(H) = 7 > 0$ (positive definite)
- $f_{xx} = 2 > 0$ (upward curvature in x-direction)

**Geometric interpretation:** The function is bowl-shaped (paraboloid) opening upward at this point.

---

### **Part (d): Taylor Series Expansion [2 Marks]**

**Problem:** Let $g(x) = f(x, 1)$. Find the Taylor series expansion of $g(x)$ about $x = 1$.

**Solution:**

**Step 1: Define $g(x)$**

$$g(x) = f(x, 1) = x^2 + x(1) + 2(1)^2 - 4x + 3$$
$$= x^2 + x + 2 - 4x + 3 = x^2 - 3x + 5$$

**Step 2: Taylor series formula**

$$g(x) = g(a) + g'(a)(x - a) + \frac{g''(a)}{2!}(x - a)^2 + \frac{g'''(a)}{3!}(x - a)^3 + \cdots$$

For $a = 1$:

$$g(x) = g(1) + g'(1)(x - 1) + \frac{g''(1)}{2}(x - 1)^2 + \cdots$$

**Step 3: Calculate derivatives**

$$g(x) = x^2 - 3x + 5$$

$$g'(x) = 2x - 3$$

$$g''(x) = 2$$

$$g'''(x) = 0$$ (and all higher derivatives)

**Step 4: Evaluate at $x = 1$**

$$g(1) = 1^2 - 3(1) + 5 = 1 - 3 + 5 = 3$$

$$g'(1) = 2(1) - 3 = -1$$

$$g''(1) = 2$$

**Step 5: Write Taylor series**

$$g(x) = 3 + (-1)(x - 1) + \frac{2}{2}(x - 1)^2$$

$$g(x) = 3 - (x - 1) + (x - 1)^2$$

**Verification (expand and simplify):**
$$g(x) = 3 - x + 1 + x^2 - 2x + 1 = x^2 - 3x + 5 \quad \checkmark$$

**Answer:**

Taylor series expansion of $g(x)$ about $x = 1$:

$$g(x) = 3 - (x - 1) + (x - 1)^2$$

Or equivalently:
$$g(x) = \sum_{n=0}^{2} \frac{g^{(n)}(1)}{n!}(x - 1)^n = 3 - (x-1) + (x-1)^2$$

(Series terminates at $n=2$ since $g$ is a quadratic polynomial)

---

## **Question 5: Inner Products and Orthogonality**

### **Part (a): Inner Product Definition and Properties [3 Marks]**

**Problem:** Define an inner product on $\mathbb{R}^2$ using matrix $A$ by $\langle X, Y \rangle = X^T A Y$.
- What properties of $A$ ensure this is a valid inner product?
- Prove these properties.

**Given (Example):**
$$A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$$

**Solution:**

**Part 1: Required properties of $A$**

For $\langle X, Y \rangle = X^T A Y$ to be a valid inner product, matrix $A$ must be:

1. **Symmetric:** $A = A^T$
2. **Positive definite:** $X^T A X > 0$ for all $X \neq \vec{0}$

**Part 2: Verify properties for given $A$**

**Property 1: Symmetry**

$$A^T = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}^T = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} = A \quad \checkmark$$

**Property 2: Positive definiteness**

For any $X = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \neq \vec{0}$:

$$X^T A X = \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

$$= \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} 2x_1 \\ 3x_2 \end{bmatrix} = 2x_1^2 + 3x_2^2$$

Since $X \neq \vec{0}$, at least one of $x_1, x_2$ is non-zero.
- If $x_1 \neq 0$: $2x_1^2 > 0$
- If $x_2 \neq 0$: $3x_2^2 > 0$

Therefore, $X^T A X = 2x_1^2 + 3x_2^2 > 0$ ✓

**Part 3: Prove inner product axioms**

An inner product must satisfy (for all $X, Y, Z \in \mathbb{R}^2$ and $c \in \mathbb{R}$):

**Axiom 1: Positivity**
$$\langle X, X \rangle > 0 \text{ for } X \neq \vec{0}$$

**Proof:** $\langle X, X \rangle = X^T A X > 0$ by positive definiteness ✓

**Axiom 2: Definiteness**
$$\langle X, X \rangle = 0 \iff X = \vec{0}$$

**Proof:**
- Forward: If $X = \vec{0}$, then $\langle \vec{0}, \vec{0} \rangle = 0^T A 0 = 0$ ✓
- Reverse: If $2x_1^2 + 3x_2^2 = 0$, since both terms are non-negative, we need $x_1 = 0$ and $x_2 = 0$, so $X = \vec{0}$ ✓

**Axiom 3: Symmetry**
$$\langle X, Y \rangle = \langle Y, X \rangle$$

**Proof:**
$$\langle X, Y \rangle = X^T A Y$$

This is a scalar, so:
$$\langle X, Y \rangle = (X^T A Y)^T = Y^T A^T X = Y^T A X = \langle Y, X \rangle$$

(Used $A^T = A$ for symmetric matrices) ✓

**Axiom 4: Linearity in first argument**
$$\langle cX + Z, Y \rangle = c\langle X, Y \rangle + \langle Z, Y \rangle$$

**Proof:**
$$\langle cX + Z, Y \rangle = (cX + Z)^T A Y = (cX^T + Z^T) A Y$$
$$= cX^T A Y + Z^T A Y = c\langle X, Y \rangle + \langle Z, Y \rangle$$ ✓

**Conclusion:** Since $A$ is symmetric and positive definite, $\langle X, Y \rangle = X^T A Y$ defines a valid inner product on $\mathbb{R}^2$.

---

### **Part (b): Distance Calculation [2 Marks]**

**Problem:** Using the inner product defined in part (a), find the distance between vectors $u$ and $w$.

**Given (Example):**
$$u = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad w = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$$

**Solution:**

**Distance formula using inner product:**

$$d(u, w) = \sqrt{\langle u - w, u - w \rangle}$$

**Step 1: Calculate $u - w$**

$$u - w = \begin{bmatrix} 1 \\ 2 \end{bmatrix} - \begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} -2 \\ 1 \end{bmatrix}$$

**Step 2: Calculate inner product $\langle u - w, u - w \rangle$**

$$\langle u - w, u - w \rangle = (u - w)^T A (u - w)$$

$$= \begin{bmatrix} -2 & 1 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} -2 \\ 1 \end{bmatrix}$$

$$= \begin{bmatrix} -2 & 1 \end{bmatrix} \begin{bmatrix} -4 \\ 3 \end{bmatrix}$$

$$= (-2)(-4) + (1)(3) = 8 + 3 = 11$$

**Step 3: Calculate distance**

$$d(u, w) = \sqrt{11}$$

**Answer:**

The distance between $u$ and $w$ with respect to the inner product $\langle X, Y \rangle = X^T A Y$ is:

$$d(u, w) = \sqrt{11} \approx 3.317$$

**Note:** This is different from standard Euclidean distance:
$$d_{\text{Euclidean}} = \sqrt{(-2)^2 + 1^2} = \sqrt{5} \approx 2.236$$

The inner product with $A$ "stretches" space differently in different directions (factor of $\sqrt{2}$ in x-direction, $\sqrt{3}$ in y-direction).

---

### **Part (c): Orthogonality Verification [1 Mark]**

**Problem:** Verify whether vectors $v_1$ and $v_2$ are orthogonal with respect to the inner product defined in part (a).

**Given (Example):**
$$v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

**Solution:**

**Definition:** Vectors are orthogonal if $\langle v_1, v_2 \rangle = 0$.

**Calculate inner product:**

$$\langle v_1, v_2 \rangle = v_1^T A v_2$$

$$= \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

$$= \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 3 \end{bmatrix} = 1(0) + 0(3) = 0$$

**Conclusion:**

Since $\langle v_1, v_2 \rangle = 0$, vectors $v_1$ and $v_2$ are **orthogonal** with respect to the inner product $\langle X, Y \rangle = X^T A Y$.

**Answer:** Yes, $v_1$ and $v_2$ are orthogonal.

**Note:** These standard basis vectors are orthogonal with respect to this inner product because $A$ is diagonal. For a general symmetric positive definite matrix, standard basis vectors may not be orthogonal.

---

## **Summary**

This examination comprehensively tested:

1. **Linear Algebra Fundamentals:**
   - Solving linear systems using row reduction
   - Matrix properties: determinant, rank
   - Eigenvalues and eigenvectors

2. **Vector Spaces:**
   - Subspace verification
   - Basis and dimension
   - Linear dependence/independence

3. **Matrix Decompositions:**
   - Singular Value Decomposition (SVD)
   - Diagonalization

4. **Multivariable Calculus:**
   - Gradients and critical points
   - Hessian matrix and second derivative test
   - Taylor series expansion

5. **Inner Products:**
   - Custom inner product spaces
   - Distance and orthogonality

**Key Mathematical Concepts for Machine Learning:**
- SVD: Used in PCA, recommender systems, data compression
- Gradients: Foundation of gradient descent optimization
- Inner products: Kernel methods, similarity measures
- Eigenvalues: Understanding covariance matrices, stability analysis
- Taylor series: Understanding loss function landscapes

These mathematical foundations are essential for understanding modern machine learning algorithms!
