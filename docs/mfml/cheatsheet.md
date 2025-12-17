# MFML Cheat Sheet

Quick reference guide for all important formulas, concepts, and algorithms in Mathematical Foundations for Machine Learning.

## 📐 Key Formulas

### Matrices and Systems

**Matrix Multiplication:**
$$(AB)_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

**System in Matrix Form:**
$$Ax = b$$

**Augmented Matrix:**
$$[A|b]$$

### Determinants

**2×2 Matrix:**
$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

**3×3 Matrix (Expansion along first row):**
$$\det(A) = a_{11}M_{11} - a_{12}M_{12} + a_{13}M_{13}$$

where $M_{ij}$ is the minor.

### Eigenvalues and Eigenvectors

**Eigenvalue Equation:**
$$Av = \lambda v$$

**Characteristic Equation:**
$$\det(A - \lambda I) = 0$$

**Eigenvalues of Triangular Matrix:**
Diagonal entries are eigenvalues.

### Inner Products

**Standard Inner Product:**
$$\langle x, y \rangle = x^T y = \sum_{i=1}^{n} x_i y_i$$

**Weighted Inner Product:**
$$\langle x, y \rangle_A = x^T A y$$

where $A$ is symmetric positive-definite.

**Norm from Inner Product:**
$$\|x\| = \sqrt{\langle x, x \rangle}$$

### Distances

**Euclidean Distance:**
$$d(x, y) = \sqrt{\sum_{i=1}^{p} (x_i - y_i)^2} = \|x - y\|_2$$

**Manhattan Distance:**
$$d(x, y) = \sum_{i=1}^{p} |x_i - y_i| = \|x - y\|_1$$

### Matrix Decompositions

**Cholesky Decomposition:**
$$A = LL^T$$

(For symmetric positive-definite $A$)

**Diagonalization:**
$$A = PDP^{-1}$$

where $D$ contains eigenvalues, $P$ contains eigenvectors.

**SVD:**
$$A = U \Sigma V^T$$

where $U$ and $V$ are orthogonal, $\Sigma$ contains singular values.

### REF/RREF

**Row Echelon Form (REF):**
- All zero rows at bottom
- Leading entry of each row is to the right of row above
- All entries below leading entry are zero

**Reduced Row Echelon Form (RREF):**
- In REF
- All leading entries are 1
- All entries above and below leading 1 are zero

### Linear Independence

Vectors $\{v_1, \ldots, v_k\}$ are **linearly independent** if:

$$c_1 v_1 + \cdots + c_k v_k = 0 \Rightarrow c_1 = \cdots = c_k = 0$$

### Gram-Schmidt Process

Given linearly independent $\{v_1, \ldots, v_n\}$:

1. $u_1 = \frac{v_1}{\|v_1\|}$
2. For $i = 2, \ldots, n$:
   - $w_i = v_i - \sum_{j=1}^{i-1} \langle v_i, u_j \rangle u_j$
   - $u_i = \frac{w_i}{\|w_i\|}$

### Two-Variable Taylor Series

**First-order:**
$$f(x, y) \approx f(a, b) + f_x(a, b)(x-a) + f_y(a, b)(y-b)$$

**Second-order:**
$$f(x, y) \approx f(a, b) + f_x(a, b)(x-a) + f_y(a, b)(y-b)$$
$$+ \frac{1}{2}[f_{xx}(a, b)(x-a)^2 + 2f_{xy}(a, b)(x-a)(y-b) + f_{yy}(a, b)(y-b)^2]$$

### Gradients and Derivatives

**Gradient:**
$$\nabla f = \begin{pmatrix}
\frac{\partial f}{\partial x_1} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{pmatrix}$$

**Jacobian Matrix:**
$$J_F = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}$$

**Hessian Matrix:**
$$H_f = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{pmatrix}$$

## 🔑 Important Properties

### Matrix Properties

- $\det(AB) = \det(A)\det(B)$
- $\det(A^T) = \det(A)$
- $\det(A^{-1}) = \frac{1}{\det(A)}$
- $\det(cA) = c^n \det(A)$ for $n \times n$ matrix

### Eigenvalue Properties

- Sum of eigenvalues = trace of $A$
- Product of eigenvalues = determinant of $A$
- If $A$ is symmetric, all eigenvalues are real
- Eigenvectors for distinct eigenvalues are linearly independent

### Positive Definite Matrix

$A$ is positive definite if:
- All eigenvalues > 0, OR
- All leading principal minors > 0, OR
- $x^T A x > 0$ for all $x \neq 0$

### Vector Space Properties

**Subspace Test:** $U$ is a subspace if:
1. $U \neq \emptyset$
2. Closed under addition: $u, v \in U \Rightarrow u + v \in U$
3. Closed under scalar multiplication: $u \in U, \lambda \in \mathbb{R} \Rightarrow \lambda u \in U$

## 📊 Quick Reference Tables

### Solution Types for Linear Systems

| Condition | Solution Type |
|----------|--------------|
| $\text{rank}(A) = \text{rank}([A|b]) = n$ | Unique solution |
| $\text{rank}(A) = \text{rank}([A|b]) < n$ | Infinitely many solutions |
| $\text{rank}(A) < \text{rank}([A|b])$ | No solution |

### Decomposition Comparison

| Decomposition | Requirements | Use Case |
|--------------|--------------|----------|
| **LU** | Any square matrix | General solving |
| **Cholesky** | Symmetric positive definite | Efficient for SPD |
| **QR** | Any matrix | Least squares |
| **Eigen** | Diagonalizable | Powers, analysis |
| **SVD** | Any matrix | Dimensionality reduction |

## 🎯 Exam Strategy

1. **REF/RREF:** Show all row operations step-by-step
2. **Linear Independence:** Set up $c_1 v_1 + \cdots + c_k v_k = 0$ and solve
3. **Inner Products:** Verify all three properties (symmetry, bilinearity, positive definiteness)
4. **Gram-Schmidt:** Work step-by-step, don't skip normalization
5. **Eigenvalues:** Always verify with $\det(A - \lambda I) = 0$
6. **Cholesky:** Check symmetric and positive definite first
7. **Taylor Series:** MUST know two-variable expansion
8. **Gradients:** Compute all partial derivatives

## 📝 Common Mistakes to Avoid

- ❌ Forgetting to check conditions (symmetric, positive definite)
- ❌ Not normalizing in Gram-Schmidt
- ❌ Confusing $P(A|B)$ with $P(B|A)$
- ❌ Forgetting eigenvectors are defined up to scalar multiples
- ❌ Using wrong formula for determinant
- ❌ Not showing all steps in REF/RREF
- ❌ Forgetting $(n-1)$ in sample variance

---

**Remember:** Show all steps clearly - marks are given for methodology, not just final answers!

