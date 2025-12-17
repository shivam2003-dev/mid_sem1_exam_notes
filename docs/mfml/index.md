# Mathematical Foundations for Machine Learning (MFML) - Complete Revision Guide

Welcome to the Mathematical Foundations for Machine Learning revision guide. This section covers all modules with detailed explanations, proofs, and important mathematical concepts based on the actual course lectures.

## 📖 Modules Overview (Based on Lectures)

1. **[Lecture 1: Matrices and Solving System of Linear Equations](modules/lecture1-matrices-linear-systems.md)**
    - Matrices and Matrix Operations
    - Systems of Linear Equations
    - Augmented Matrix
    - Echelon Form and RREF
    - Gauss-Jordan Elimination
    - Matrices and Matrix Operations
    - Systems of Linear Equations
    - Augmented Matrix
    - Solving System of Equations
    - Geometrical Interpretation

2. **[Lecture 2: Vector Spaces](modules/lecture2-vector-spaces.md)**
    - Groups and Abelian Groups
    - Vector Spaces Definition and Properties
    - Examples of Vector Spaces
    - Vector Subspaces
    - Linear Combinations and Linear Independence
    - Groups and Abelian Groups
    - Vector Spaces Definition and Properties
    - Examples of Vector Spaces (Rn, Matrices)
    - Vector Subspaces
    - Applications in ML

3. **[Lecture 3: Analytic Geometry and Inner Products](modules/lecture3-analytic-geometry.md)**
    - Distance between Vectors
    - Similarity and Dissimilarity Measures
    - Dot Product in Rn
    - Inner Products
    - Symmetric and Positive Definite Matrices
    - Distance between Vectors
    - Similarity and Dissimilarity Measures
    - Dot Product in Rn
    - Inner Products
    - Symmetric and Positive Definite Matrices

4. **[Lecture 4: Eigenvalues, Eigenvectors, and Determinants](modules/lecture4-eigenvalues-determinants.md)**
    - Eigenvalues and Eigenvectors
    - Triangular Matrices
    - Determinants (2×2, 3×3, general)
    - Characteristic Equation
    - Finding Eigenvalues and Eigenvectors
    - Eigenvalues and Eigenvectors
    - Geometrical Interpretation
    - Triangular Matrices
    - Determinants (2x2, 3x3, general)
    - Minor and Cofactor

5. **[Lecture 5: Matrix Decompositions](modules/lecture5-matrix-decompositions.md)**
    - Cholesky Decomposition
    - Diagonalization Theorem
    - Singular Value Decomposition (SVD)
    - Applications in ML
    - Cholesky Decomposition
    - Diagonalization Theorem
    - Singular Value Decomposition (SVD)
    - Applications in ML

6. **[Lecture 6: Advanced Topics](modules/lecture6-advanced-topics.md)**
    - (Content based on Lecture 6 PDF)

7. **[Lecture 7: Advanced Topics](modules/lecture7-advanced-topics.md)**
    - (Content based on Lecture 7 PDF)

## 📋 Cheat Sheet

- [MFML Cheat Sheet - Quick Reference](cheatsheet.md)

## 📚 Solved Previous Year Papers

- [2024 MidSem Regular Paper - Complete Solutions](papers/2024-midsem-regular-solved.md)
- [2024 MidSem Makeup Paper - Detailed Solutions](papers/2024-midsem-makeup-solved.md)
- [2023 MidSem Regular Paper - Step-by-Step Solutions](papers/2023-midsem-regular-solved.md)
- [2022 MidSem Paper - Comprehensive Solutions](papers/2022-midsem-solved.md)

## 📝 Assignments

- [Assignment 1 - Linear Systems & Inner Products](assignments/assignment1-solved.md)

## 🎯 MidSem Important Topics

### Expected Question Types

1. **REF/RREF (Easy to Score)**
   - Transform matrices to Row Echelon Form
   - Transform matrices to Reduced Row Echelon Form
   - Use Gaussian elimination step-by-step

2. **Linear Dependence/Independence**
   - Test if vectors are linearly independent
   - Use REF to find rank
   - Express dependent vectors as linear combinations

3. **Inner/Dot Products or Gram-Schmidt Orthogonalization**
   - Compute inner products with weighted matrices
   - Compute norms from inner products
   - Apply Gram-Schmidt process step-by-step
   - Verify orthonormality

4. **Eigen Decomposition/Diagonalization/SVD Decomposition**
   - Find eigenvalues and eigenvectors
   - Diagonalize matrices
   - Compute SVD (lengthy but easy to score)
   - Use decomposition to solve systems

5. **Taylor's Series (Easy to Score)**
   - **MUST KNOW: Two-variable Taylor series**
   - Expand functions around given points
   - First-order and second-order approximations

6. **Partial Derivatives or Applications**
   - Compute partial derivatives
   - Find gradients
   - Compute Jacobian matrices
   - Compute Hessian matrices
   - Applications in optimization

### Must Solve

- **MFML Practice Problems.pdf** - Essential practice material
- All past papers topic-wise

### Key Formulas

- Matrix multiplication and inversion
- Determinant computation
- Eigenvalue equation: $Av = \lambda v$
- Inner product: $\langle x, y \rangle = x^T M y$
- Norm: $\|x\| = \sqrt{\langle x, x \rangle}$
- Gradient: $\nabla f(x) = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)^T$
- Two-variable Taylor: $f(x,y) \approx f(a,b) + f_x(a,b)(x-a) + f_y(a,b)(y-b) + \frac{1}{2}[f_{xx}(a,b)(x-a)^2 + 2f_{xy}(a,b)(x-a)(y-b) + f_{yy}(a,b)(y-b)^2]$

### Study Resources

- Companion docs + Topic explanations: MFML Additional Resources
- Understanding the Gradient
- Finding the Gradient of a Vector Function
- Matrix Calculus Overview
- Detailed Explanation of Linear Independence
- Example Calculation of LU Decomposition
- SVD Problem with Full Explanation
- Eigenvectors and Eigenvalues (Essence of Linear Algebra)
- Powers in Matrix Product
- Past papers topic wise: MFML Past Papers
- Useful cheatsheet: [Linear Algebra Cheat Sheet](https://jiha-kim.github.io/crash-courses/linear-algebra/999-cheat-sheet/)
- Linear algebra Formula sheet: Linear Algebra Formula Sheet.pdf

---

**Course Information:**
- Course No.: AIMLC ZC416 / DSECLZC416
- Course Title: Mathematical Foundations for Machine Language / Mathematical Foundations for Data Science
