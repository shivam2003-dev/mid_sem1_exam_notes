# Lecture 1: Matrices and Solving System of Linear Equations

## Overview

This lecture introduces matrices and systems of linear equations, which form the foundation of linear algebra and are essential for machine learning.

## 1. Why Study Vectors and Matrices?

Vectors are fundamental in machine learning and are used throughout the entire workflow:
- **Input:** Data is represented as vectors
- **Processing:** Mathematical operations on vectors
- **Output:** ML model outputs are often vectors
- **Optimization:** Vectors in optimization algorithms

## 2. What is Linear Algebra?

**Linear algebra** is the study of vectors and rules to manipulate vectors.

Vectors are not only the familiar arrows in space, but also:
- Data points in high-dimensional spaces
- Features in machine learning
- Representations in neural networks

## 3. Systems of Linear Equations

Systems of linear equations form a central part of linear algebra. Many problems can be formulated as systems of linear equations.

### Motivating Problem

Consider a company that produces products $N_1, N_2, \ldots, N_n$ using resources $R_1, R_2, \ldots, R_m$.

**Formulation:**
If we produce $x_1, x_2, \ldots, x_n$ units of products $N_1, N_2, \ldots, N_n$, we need:
- Total of $a_{1i}x_1 + a_{2i}x_2 + \cdots + a_{ni}x_n$ units of resource $R_i$

This leads to a system of linear equations.

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

## 4. Augmented Matrix

The **augmented matrix** combines the coefficient matrix and constant vector:

$$[A|b] = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} & | & b_1 \\
a_{21} & a_{22} & \cdots & a_{2n} & | & b_2 \\
\vdots & \vdots & \ddots & \vdots & | & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} & | & b_m
\end{pmatrix}$$

## 5. Solving System of Equations

### Geometrical Interpretation (2 Variables)

For a system with two variables:
- Each equation represents a line
- Solution is the intersection point(s)

### Three Possibilities

A system of linear equations has:

1. **No solution** - Lines are parallel (inconsistent)
2. **Exactly one solution** - Lines intersect at one point (consistent and independent)
3. **Infinitely many solutions** - Lines coincide (consistent and dependent)

## 6. Matrices

### Definition

A **matrix** is a rectangular array of numbers arranged in rows and columns.

$$A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}$$

### Matrix Operations

**Addition:** $(A + B)_{ij} = a_{ij} + b_{ij}$

**Scalar Multiplication:** $(cA)_{ij} = c \cdot a_{ij}$

**Matrix Multiplication:** $(AB)_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$

## 7. Representing Systems in Matrix Form

The system:
$$\begin{cases}
a_{11}x_1 + a_{12}x_2 = b_1 \\
a_{21}x_1 + a_{22}x_2 = b_2
\end{cases}$$

Can be written as:
$$\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} b_1 \\ b_2 \end{pmatrix}$$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Matrix Form:** Always convert systems to matrix form $Ax = b$
2. **Augmented Matrix:** Use $[A|b]$ for solving systems
3. **Solution Types:** Identify if system has no solution, one solution, or infinitely many
4. **Geometrical Interpretation:** Understand what solutions mean geometrically
```

---

## 📚 Quick Revision Checklist

- [ ] Understanding why vectors and matrices are important in ML
- [ ] Converting word problems to systems of linear equations
- [ ] Writing systems in matrix form $Ax = b$
- [ ] Forming augmented matrices $[A|b]$
- [ ] Understanding the three types of solutions
- [ ] Matrix operations (addition, scalar multiplication, multiplication)

