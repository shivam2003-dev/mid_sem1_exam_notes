# Module 4: Systems of Linear Equations

## Overview

Solving systems of linear equations is fundamental in machine learning. This module covers Gaussian elimination, REF/RREF, and methods for solving linear systems.

## 1. Systems of Linear Equations

### General Form

A system of $m$ linear equations in $n$ unknowns:

$$\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}$$

### Matrix Form

$$Ax = b$$

where:
- $A$ is the $m \times n$ coefficient matrix
- $x$ is the $n \times 1$ vector of unknowns
- $b$ is the $m \times 1$ constant vector

### Augmented Matrix

$$[A|b] = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} & | & b_1 \\
a_{21} & a_{22} & \cdots & a_{2n} & | & b_2 \\
\vdots & \vdots & \ddots & \vdots & | & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} & | & b_m
\end{pmatrix}$$

---

## 2. Row Echelon Form (REF)

### Definition

A matrix is in **Row Echelon Form (REF)** if:

1. All zero rows are at the bottom
2. The first nonzero entry (leading entry/pivot) of each row is to the right of the leading entry of the row above it
3. All entries below a leading entry are zero

### Example

$$\begin{pmatrix}
1 & 2 & 3 & 4 \\
0 & 5 & 6 & 7 \\
0 & 0 & 0 & 8 \\
0 & 0 & 0 & 0
\end{pmatrix}$$

### Properties

- Leading entries (pivots) are in positions $(1,1), (2,2), (3,4)$
- All entries below pivots are zero
- Zero row is at the bottom

---

## 3. Reduced Row Echelon Form (RREF)

### Definition

A matrix is in **Reduced Row Echelon Form (RREF)** if:

1. It is in REF
2. All leading entries are 1
3. All entries above and below each leading 1 are zero

### Example

$$\begin{pmatrix}
1 & 0 & 0 & 2 \\
0 & 1 & 0 & 3 \\
0 & 0 & 1 & 4 \\
0 & 0 & 0 & 0
\end{pmatrix}$$

### Properties

- Each pivot is 1
- Each pivot is the only nonzero entry in its column
- Easier to read solutions directly

---

## 4. Gaussian Elimination

### Algorithm for REF

1. Start with the leftmost nonzero column (pivot column)
2. Select a pivot (nonzero entry) in that column
3. Use row operations to make all entries below the pivot zero
4. Move to the next column and repeat

### Elementary Row Operations

1. **Swap:** $R_i \leftrightarrow R_j$
2. **Scale:** $R_i \leftarrow cR_i$ (where $c \neq 0$)
3. **Replace:** $R_i \leftarrow R_i + cR_j$

### Algorithm for RREF

After obtaining REF:

1. Scale each row so the leading entry is 1
2. Use row operations to make all entries above each pivot zero

### Example: REF

Transform to REF:

$$A = \begin{pmatrix}
2 & 4 & 6 \\
1 & 2 & 3 \\
3 & 6 & 9
\end{pmatrix}$$

**Step 1:** $R_1 \leftrightarrow R_2$ (to get 1 in top-left)

$$\begin{pmatrix}
1 & 2 & 3 \\
2 & 4 & 6 \\
3 & 6 & 9
\end{pmatrix}$$

**Step 2:** $R_2 \leftarrow R_2 - 2R_1$, $R_3 \leftarrow R_3 - 3R_1$

$$\begin{pmatrix}
1 & 2 & 3 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{pmatrix}$$

This is in REF (zero rows at bottom).

### Example: RREF

Transform to RREF:

$$A = \begin{pmatrix}
2 & 1 & 3 \\
4 & 2 & 7 \\
6 & 3 & 10
\end{pmatrix}$$

**Step 1:** $R_1 \leftarrow \frac{1}{2}R_1$

$$\begin{pmatrix}
1 & \frac{1}{2} & \frac{3}{2} \\
4 & 2 & 7 \\
6 & 3 & 10
\end{pmatrix}$$

**Step 2:** $R_2 \leftarrow R_2 - 4R_1$, $R_3 \leftarrow R_3 - 6R_1$

$$\begin{pmatrix}
1 & \frac{1}{2} & \frac{3}{2} \\
0 & 0 & 1 \\
0 & 0 & 1
\end{pmatrix}$$

**Step 3:** $R_3 \leftarrow R_3 - R_2$

$$\begin{pmatrix}
1 & \frac{1}{2} & \frac{3}{2} \\
0 & 0 & 1 \\
0 & 0 & 0
\end{pmatrix}$$

**Step 4:** $R_1 \leftarrow R_1 - \frac{3}{2}R_2$ (eliminate above pivot)

$$\begin{pmatrix}
1 & \frac{1}{2} & 0 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{pmatrix}$$

This is in RREF.

---

## 5. Solving Linear Systems

### Using REF/RREF

1. Form augmented matrix $[A|b]$
2. Transform to REF (or RREF)
3. Use back substitution (REF) or read solution directly (RREF)

### Types of Solutions

**Unique Solution:**
- Number of pivots = number of variables
- System is consistent

**Infinitely Many Solutions:**
- Number of pivots < number of variables
- Free variables exist
- System is consistent

**No Solution:**
- Inconsistent system
- Row of form $[0 \ 0 \ \cdots \ 0 \ | \ c]$ where $c \neq 0$

### Example: Unique Solution

Solve:
$$\begin{cases}
x + 2y = 5 \\
3x + 4y = 11
\end{cases}$$

**Augmented matrix:**
$$\begin{pmatrix}
1 & 2 & | & 5 \\
3 & 4 & | & 11
\end{pmatrix}$$

**RREF:**
- $R_2 \leftarrow R_2 - 3R_1$: $\begin{pmatrix} 1 & 2 & | & 5 \\ 0 & -2 & | & -4 \end{pmatrix}$
- $R_2 \leftarrow -\frac{1}{2}R_2$: $\begin{pmatrix} 1 & 2 & | & 5 \\ 0 & 1 & | & 2 \end{pmatrix}$
- $R_1 \leftarrow R_1 - 2R_2$: $\begin{pmatrix} 1 & 0 & | & 1 \\ 0 & 1 & | & 2 \end{pmatrix}$

**Solution:** $x = 1, y = 2$

### Example: Infinitely Many Solutions

Solve:
$$\begin{cases}
x + y + z = 3 \\
2x + 2y + 2z = 6
\end{cases}$$

**RREF:**
$$\begin{pmatrix}
1 & 1 & 1 & | & 3 \\
0 & 0 & 0 & | & 0
\end{pmatrix}$$

**Solution:** $x = 3 - y - z$ (free variables: $y, z$)

### Example: No Solution

Solve:
$$\begin{cases}
x + y = 1 \\
x + y = 2
\end{cases}$$

**RREF:**
$$\begin{pmatrix}
1 & 1 & | & 1 \\
0 & 0 & | & 1
\end{pmatrix}$$

**Result:** Inconsistent (contradiction: $0 = 1$)

---

## 6. Linear Dependence and Independence

### Definition

Vectors $\{v_1, v_2, \ldots, v_k\}$ are **linearly dependent** if there exist scalars $c_1, c_2, \ldots, c_k$ (not all zero) such that:

$$c_1 v_1 + c_2 v_2 + \cdots + c_k v_k = 0$$

Otherwise, they are **linearly independent**.

### Testing Linear Independence

Form matrix $A = [v_1 | v_2 | \cdots | v_k]$ and check:

- **Linearly independent** if $\text{rank}(A) = k$ (number of vectors)
- **Linearly dependent** if $\text{rank}(A) < k$

Or check if the homogeneous system $Ax = 0$ has only the trivial solution.

### Example

Check if $v_1 = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$, $v_2 = \begin{pmatrix} 2 \\ 4 \end{pmatrix}$ are linearly independent.

**Form matrix:**
$$A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$$

**REF:**
- $R_2 \leftarrow R_2 - 2R_1$: $\begin{pmatrix} 1 & 2 \\ 0 & 0 \end{pmatrix}$

**Rank = 1 < 2**, so vectors are **linearly dependent**.

Indeed: $2v_1 - v_2 = 0$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **REF vs RREF:** REF is sufficient for solving, but RREF gives cleaner solutions
2. **Row Operations:** Always show each step clearly
3. **Pivot Positions:** Track pivot columns for rank determination
4. **Linear Independence:** Use REF to find rank, compare with number of vectors
5. **Common Mistakes:**
   - Forgetting to update all rows when doing row operations
   - Not checking for inconsistent systems (row of zeros with nonzero constant)
   - Confusing free variables with unique solution
```

---

## 🔍 Worked Examples

### Example 1: REF and RREF

Transform to REF and RREF:

$$A = \begin{pmatrix}
1 & 2 & 3 \\
2 & 4 & 6 \\
1 & 1 & 1
\end{pmatrix}$$

**REF:**
- $R_2 \leftarrow R_2 - 2R_1$, $R_3 \leftarrow R_3 - R_1$:
  $$\begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 0 & -1 & -2 \end{pmatrix}$$
- Swap $R_2$ and $R_3$:
  $$\begin{pmatrix} 1 & 2 & 3 \\ 0 & -1 & -2 \\ 0 & 0 & 0 \end{pmatrix}$$

**RREF:**
- $R_2 \leftarrow -R_2$:
  $$\begin{pmatrix} 1 & 2 & 3 \\ 0 & 1 & 2 \\ 0 & 0 & 0 \end{pmatrix}$$
- $R_1 \leftarrow R_1 - 2R_2$:
  $$\begin{pmatrix} 1 & 0 & -1 \\ 0 & 1 & 2 \\ 0 & 0 & 0 \end{pmatrix}$$

### Example 2: Linear Independence

Check linear independence of:
$$v_1 = \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}, \quad v_2 = \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix}, \quad v_3 = \begin{pmatrix} 1 \\ 1 \\ 2 \end{pmatrix}$$

**Form matrix:**
$$A = \begin{pmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
1 & 1 & 2
\end{pmatrix}$$

**REF:**
- $R_3 \leftarrow R_3 - R_1$:
  $$\begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix}$$
- $R_3 \leftarrow R_3 - R_2$:
  $$\begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

**Rank = 2 < 3**, so vectors are **linearly dependent**.

Indeed: $v_3 = v_1 + v_2$

---

## 📚 Quick Revision Checklist

- [ ] REF definition and properties
- [ ] RREF definition and properties
- [ ] Gaussian elimination algorithm
- [ ] Elementary row operations
- [ ] Solving systems using REF/RREF
- [ ] Identifying solution types (unique, infinite, none)
- [ ] Testing linear independence using REF

