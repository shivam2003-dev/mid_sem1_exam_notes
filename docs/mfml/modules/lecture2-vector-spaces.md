# Lecture 2: Vector Spaces

## Overview

This lecture shifts focus from systems of linear equations to vector spaces, which provide the mathematical framework for understanding vectors in machine learning.

## 1. Groups

We already have some notion about the structure of a vector space - adding two vectors returns a vector, multiplying a vector by a scalar returns a vector.

### Definition

A **group** $(G, \circ)$ is a set $G$ with a binary operation $\circ$ satisfying:

1. **Closure:** $\forall a, b \in G: a \circ b \in G$
2. **Associativity:** $\forall a, b, c \in G: (a \circ b) \circ c = a \circ (b \circ c)$
3. **Identity:** $\exists e \in G: \forall a \in G, a \circ e = e \circ a = a$
4. **Inverse:** $\forall a \in G, \exists a^{-1} \in G: a \circ a^{-1} = a^{-1} \circ a = e$

### Abelian Group

If the operation is also **commutative** ($a \circ b = b \circ a$), the group is called **Abelian**.

### Examples

- $(\mathbb{Z}, +)$ where $\mathbb{Z}$ is the set of all integers is an Abelian group
- Identity: $0$
- Inverse: $-a$ for each $a$

## 2. Vector Spaces

### Definition

A **vector space** $V$ over a field $\mathbb{R}$ is a set with two operations:
- **Vector addition:** $+ : V \times V \to V$
- **Scalar multiplication:** $\cdot : \mathbb{R} \times V \to V$

Satisfying the following properties:

**Vector Addition Properties:**
1. **Closure:** $\forall u, v \in V: u + v \in V$
2. **Commutativity:** $\forall u, v \in V: u + v = v + u$
3. **Associativity:** $\forall u, v, w \in V: (u + v) + w = u + (v + w)$
4. **Zero Vector:** $\exists 0 \in V: \forall v \in V, v + 0 = v$
5. **Additive Inverse:** $\forall v \in V, \exists -v \in V: v + (-v) = 0$

**Scalar Multiplication Properties:**
6. **Closure:** $\forall \lambda \in \mathbb{R}, v \in V: \lambda v \in V$
7. **Distributivity 1:** $\forall \lambda \in \mathbb{R}, u, v \in V: \lambda(u + v) = \lambda u + \lambda v$
8. **Distributivity 2:** $\forall \lambda, \mu \in \mathbb{R}, v \in V: (\lambda + \mu)v = \lambda v + \mu v$
9. **Associativity:** $\forall \lambda, \mu \in \mathbb{R}, v \in V: (\lambda \mu)v = \lambda(\mu v)$
10. **Multiplicative Identity:** $\forall v \in V: 1 \cdot v = v$

## 3. Examples of Vector Spaces

### $\mathbb{R}^n$

Consider the $n$-tuple $\mathbb{R}^n = \{(x_1, x_2, \ldots, x_n): x_i \in \mathbb{R}, i = 1, 2, \ldots, n\}$

Define:
- **Vector addition:** $(x_1, \ldots, x_n) + (y_1, \ldots, y_n) = (x_1 + y_1, \ldots, x_n + y_n)$
- **Scalar multiplication:** $\lambda(x_1, \ldots, x_n) = (\lambda x_1, \ldots, \lambda x_n)$

This forms a vector space.

### Geometrical Interpretation in $\mathbb{R}^2$

- **Vector Addition:** Parallelogram law
- **Scalar Multiplication:** Scaling and direction change

## 4. Why Study Vector Spaces in ML?

### Applications

1. **Input:** Data is represented as vectors in vector spaces
2. **Output:** ML model outputs are vectors
3. **Optimization:** Vectors in optimization algorithms (gradient descent, etc.)
4. **Feature Spaces:** High-dimensional vector spaces for features

## 5. Matrices Form a Vector Space

The set of all $m \times n$ matrices with real entries forms a vector space with:
- **Addition:** $(A + B)_{ij} = a_{ij} + b_{ij}$
- **Scalar multiplication:** $(\lambda A)_{ij} = \lambda a_{ij}$

## 6. Vector Subspaces

### Definition

Let $V = (V, +, \cdot)$ be a vector space and let $U \subseteq V$, $U \neq \emptyset$. Then $U = (U, +, \cdot)$ is called a **vector subspace** of $V$ if $U$ is itself a vector space with the same operations.

### How to Show $U$ is a Subspace

We need to show:
1. $U \neq \emptyset$
2. **Closure under addition:** $\forall u, v \in U: u + v \in U$
3. **Closure under scalar multiplication:** $\forall \lambda \in \mathbb{R}, u \in U: \lambda u \in U$

### Examples

**Example 1:** Let $V = \mathbb{R}^2$ and $U$ be the $y$-axis. Is $U$ a subspace?

$U = \{(0, y): y \in \mathbb{R}\}$

- $U \neq \emptyset$ ✓
- Closure under addition: $(0, y_1) + (0, y_2) = (0, y_1 + y_2) \in U$ ✓
- Closure under scalar multiplication: $\lambda(0, y) = (0, \lambda y) \in U$ ✓

Yes, $U$ is a subspace.

**Example 2:** What about the subset of $\mathbb{R}^2$ that represents a square region?

No - not closed under scalar multiplication (scaling a square can go outside the square).

## 7. Linear Combination and Linear Independence

### Linear Combination

Consider a vector space $V$ and a set of vectors $\{v_1, v_2, \ldots, v_k\} \subseteq V$.

A vector $v \in V$ is a **linear combination** of $\{v_1, v_2, \ldots, v_k\}$ if:

$$v = c_1 v_1 + c_2 v_2 + \cdots + c_k v_k$$

for some scalars $c_1, c_2, \ldots, c_k \in \mathbb{R}$.

### Examples

**Example 1:** Let $S = \{[1, 0, 1], [1, 1, 0], [1, 1, 1]\}$ in $\mathbb{R}^3$.

Is $[2, 1, 1]$ a linear combination of vectors in $S$?

We need to find $c_1, c_2, c_3$ such that:
$$c_1[1, 0, 1] + c_2[1, 1, 0] + c_3[1, 1, 1] = [2, 1, 1]$$

This gives the system:
$$\begin{cases}
c_1 + c_2 + c_3 = 2 \\
c_2 + c_3 = 1 \\
c_1 + c_3 = 1
\end{cases}$$

Solving: $c_1 = 1, c_2 = 0, c_3 = 1$

Yes, $[2, 1, 1] = 1 \cdot [1, 0, 1] + 0 \cdot [1, 1, 0] + 1 \cdot [1, 1, 1]$

### Linear Independence

Vectors $\{v_1, v_2, \ldots, v_k\}$ are **linearly independent** if the only solution to:

$$c_1 v_1 + c_2 v_2 + \cdots + c_k v_k = 0$$

is $c_1 = c_2 = \cdots = c_k = 0$.

Otherwise, they are **linearly dependent**.

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Subspace Test:** Always check three conditions (non-empty, closure under addition, closure under scalar multiplication)
2. **Linear Independence:** Set up equation $c_1 v_1 + \cdots + c_k v_k = 0$ and solve
3. **Linear Combination:** Express vector as combination of given vectors
4. **Common Mistake:** Forgetting to check $U \neq \emptyset$ for subspaces
```

---

## 📚 Quick Revision Checklist

- [ ] Definition of groups and Abelian groups
- [ ] Vector space axioms (10 properties)
- [ ] Examples of vector spaces ($\mathbb{R}^n$, matrices)
- [ ] Subspace test (3 conditions)
- [ ] Linear combinations
- [ ] Linear independence/dependence
- [ ] Applications in ML

