# 2025 Mid-Semester Exam Preparation Guide

This guide provides curated YouTube resources and key concepts to revise before your December 2025 mid-semester exams for DNN, MFML, and ML courses.

---

## 🎯 Quick Revision Strategy

1. **Review solved papers first** to understand question patterns
2. **Watch concept videos** for topics you find challenging
3. **Practice calculations** for mathematical topics
4. **Focus on implementation** for coding questions

---

## 🧠 Deep Neural Networks (DNN)

### 📺 YouTube Playlists & Videos

#### Complete Course Playlists
- **[StatQuest Neural Networks Playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)** ⭐
  - Clear explanations with visual animations
  - Covers perceptrons, backpropagation, CNNs

- **[3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)** ⭐⭐⭐
  - Best visual intuition for neural networks
  - Deep dive into backpropagation

- **[DeepLearning.AI - Neural Networks Basics](https://www.youtube.com/c/Deeplearningai)**
  - Andrew Ng's comprehensive lectures

#### Topic-Specific Videos

**Perceptron & Linear Models:**
- [Perceptron Algorithm Explained](https://www.youtube.com/watch?v=4Gac5I64LM4) - StatQuest
- [Linear Regression from Scratch](https://www.youtube.com/watch?v=nk2CQITm_eo) - StatQuest
- [Gradient Descent Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) - StatQuest

**Logistic Regression & Softmax:**
- [Logistic Regression Clearly Explained](https://www.youtube.com/watch?v=yIYKR4sgzI8) - StatQuest
- [Softmax Regression Explained](https://www.youtube.com/watch?v=LLux1SW--oM) - Serrano.Academy
- [Cross-Entropy Loss Explained](https://www.youtube.com/watch?v=6ArSys5qHAU)

**Deep Feedforward Neural Networks:**
- [Neural Networks from Scratch](https://www.youtube.com/watch?v=aircAruvnKk) - 3Blue1Brown ⭐⭐⭐
- [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) - 3Blue1Brown
- [Activation Functions Explained](https://www.youtube.com/watch?v=m0pIlLfpXWE)

**Evaluation Metrics:**
- [Confusion Matrix Explained](https://www.youtube.com/watch?v=Kdsp6soqA7o) - StatQuest
- [Precision, Recall, F1-Score](https://www.youtube.com/watch?v=qWfzIYCvBqo) - StatQuest
- [ROC and AUC Clearly Explained](https://www.youtube.com/watch?v=4jRBRDbJemM) - StatQuest

### 🎯 Key Concepts to Revise

#### 1. Perceptron (Q1)
- ✅ Perceptron learning rule: $w_{new} = w_{old} + \eta \cdot (y - \hat{y}) \cdot x$
- ✅ Step activation function
- ✅ Linear separability limitations
- ✅ Weight magnitude interpretation
- ✅ Feature engineering and overfitting

**Practice:** Weight updates, decision boundaries, feature analysis

#### 2. Linear Regression (Q2)
- ✅ MSE loss: $J = \frac{1}{2m}\sum(h(x) - y)^2$
- ✅ Batch gradient descent algorithm
- ✅ Gradient calculation: $\nabla J = \frac{1}{m}X^T(Xw - y)$
- ✅ Learning rate effects
- ✅ Imbalanced data diagnostics

**Practice:** Gradient descent iterations, loss calculations, accuracy vs other metrics

#### 3. Logistic Regression (Q3)
- ✅ Sigmoid function: $\sigma(z) = \frac{1}{1+e^{-z}}$
- ✅ Binary cross-entropy loss
- ✅ Probability interpretation
- ✅ Threshold selection
- ✅ Mini-batch gradient descent

**Practice:** Probability calculations, gradient updates, threshold tuning

#### 4. Softmax Regression (Q4)
- ✅ Softmax formula: $P(y=k) = \frac{e^{z_k}}{\sum_j e^{z_j}}$
- ✅ Categorical cross-entropy
- ✅ One-hot encoding
- ✅ Confusion matrix analysis
- ✅ Multi-class evaluation metrics

**Practice:** Multi-class classification, confusion matrix interpretation

#### 5. Deep Feedforward Neural Networks (Q5)
- ✅ Forward propagation
- ✅ Activation functions (ReLU, Sigmoid)
- ✅ Backpropagation algorithm
- ✅ Architecture design principles
- ✅ Regularization techniques (dropout, L2)

**Practice:** Manual forward/backward pass, architecture justification

### 💻 Python Implementation Tips
- Master NumPy operations: `np.dot()`, `np.sum()`, broadcasting
- Understand vectorization for efficiency
- Practice writing gradient calculations
- Know how to compute predictions in batch

---

## 📐 Mathematics for Machine Learning (MFML)

### 📺 YouTube Playlists & Videos

#### Complete Course Playlists
- **[MIT 18.06 Linear Algebra - Gilbert Strang](https://www.youtube.com/playlist?list=PL49CF3715CB9EF31D)** ⭐⭐⭐
  - Gold standard for linear algebra
  - Essential for ML foundations

- **[3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)** ⭐⭐⭐
  - Best visual intuition
  - Perfect for conceptual understanding

- **[Khan Academy - Linear Algebra](https://www.youtube.com/playlist?list=PLFD0EB975BA0CC1E0)**
  - Step-by-step problem solving

#### Topic-Specific Videos

**Linear Systems & Matrices:**
- [Gaussian Elimination Explained](https://www.youtube.com/watch?v=eYSASx8_nyg) - Khan Academy
- [Row Echelon Form Step-by-Step](https://www.youtube.com/watch?v=5RRLL9OWEgU)
- [Matrix Operations Visualized](https://www.youtube.com/watch?v=fNk_zzaMoSs) - 3Blue1Brown

**Vector Spaces:**
- [Vector Spaces and Subspaces](https://www.youtube.com/watch?v=Qm_OS-8COwU) - MIT OCW
- [Linear Independence](https://www.youtube.com/watch?v=CrV1xCWdY-g) - Khan Academy
- [Basis and Dimension](https://www.youtube.com/watch?v=eeMJg4uI7o0) - Khan Academy

**Eigenvalues & Eigenvectors:**
- [Eigenvectors and Eigenvalues Visualized](https://www.youtube.com/watch?v=PFDu9oVAE-g) - 3Blue1Brown ⭐⭐⭐
- [Finding Eigenvalues Example](https://www.youtube.com/watch?v=IdsV0RaC9jM) - Khan Academy
- [Diagonalization Explained](https://www.youtube.com/watch?v=U0xlKuFqCuI)

**Singular Value Decomposition:**
- [SVD Visualized](https://www.youtube.com/watch?v=mBcLRGuAFUk) - Steve Brunton ⭐⭐
- [SVD Step-by-Step Calculation](https://www.youtube.com/watch?v=cOUTpqlX-Xs)
- [Applications of SVD](https://www.youtube.com/watch?v=rYz83XPxiZo)

**Multivariable Calculus:**
- [Gradient and Directional Derivatives](https://www.youtube.com/watch?v=TEB2z7ZlRAw) - Khan Academy
- [Second Derivative Test](https://www.youtube.com/watch?v=x1NXdTnZDDc) - Khan Academy
- [Taylor Series Visualization](https://www.youtube.com/watch?v=3d6DsjIBzJ4) - 3Blue1Brown

**Inner Products:**
- [Inner Product Spaces](https://www.youtube.com/watch?v=LyGKycYT2v0)
- [Orthogonality and Projections](https://www.youtube.com/watch?v=uNsCkP9mgRk) - MIT OCW

### 🎯 Key Concepts to Revise

#### 1. Linear Systems (Q1)
- ✅ Augmented matrix $[A|b]$
- ✅ Row operations (elementary matrices)
- ✅ Echelon and reduced echelon forms
- ✅ Back substitution
- ✅ Determinant calculation methods
- ✅ Rank interpretation

**Practice:** Solve 3×3 systems, find determinants, compute ranks

#### 2. Vector Spaces (Q2)
- ✅ Linear combination: $v = \alpha u + \beta w$
- ✅ Linear dependence/independence
- ✅ Subspace verification (3 properties)
- ✅ Basis and dimension
- ✅ Span of vectors

**Practice:** Express vectors as combinations, prove subspace properties

#### 3. SVD & Diagonalization (Q3)
- ✅ SVD decomposition: $A = U\Sigma V^T$
- ✅ Compute $A^TA$ and $AA^T$
- ✅ Find eigenvalues and eigenvectors
- ✅ Singular values: $\sigma = \sqrt{\lambda}$
- ✅ Diagonalizability conditions

**Practice:** Full SVD calculation, diagonalize 2×2 matrices

#### 4. Multivariable Calculus (Q4)
- ✅ Gradient: $\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}]^T$
- ✅ Critical points: solve $\nabla f = 0$
- ✅ Hessian matrix for classification
- ✅ Second derivative test
- ✅ Taylor series expansion

**Practice:** Find and classify critical points, compute Hessians

#### 5. Inner Products (Q5)
- ✅ Inner product definition: $\langle x, y \rangle = x^TAy$
- ✅ Symmetry and positive definiteness
- ✅ Distance formula: $d(u,v) = \sqrt{\langle u-v, u-v \rangle}$
- ✅ Orthogonality: $\langle u, v \rangle = 0$
- ✅ Properties verification

**Practice:** Custom inner products, compute distances, check orthogonality

### 🧮 Calculation Checklist
- [ ] Solve 3×3 linear systems (15 min)
- [ ] Compute 3×3 determinants (5 min)
- [ ] Find eigenvalues/eigenvectors (20 min)
- [ ] Calculate SVD for 2×2 matrix (30 min)
- [ ] Find and classify critical points (15 min)

---

## 🤖 Machine Learning (ML)

### 📺 YouTube Playlists & Videos

#### Complete Course Playlists
- **[StatQuest Machine Learning Playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)** ⭐⭐⭐
  - Best explanations for ML concepts
  - Covers all core topics

- **[Andrew Ng's Machine Learning Course](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)** ⭐⭐⭐
  - Comprehensive and authoritative
  - Stanford CS229

- **[Google Developers - Machine Learning Crash Course](https://www.youtube.com/playlist?list=PLRKtJ4IpxJpDxl0NTvNYQWKCYzHNuy2xG)**
  - Practical ML concepts

#### Topic-Specific Videos

**Logistic Regression:**
- [Logistic Regression in Depth](https://www.youtube.com/watch?v=yIYKR4sgzI8) - StatQuest ⭐
- [Gradient Descent for Logistic Regression](https://www.youtube.com/watch?v=z_xiwjEdAC4)
- [Feature Normalization](https://www.youtube.com/watch?v=mnKm3YP56PY)

**Bias-Variance Tradeoff:**
- [Bias and Variance Explained](https://www.youtube.com/watch?v=EuBBz3bI-aA) - StatQuest ⭐⭐
- [Understanding Overfitting](https://www.youtube.com/watch?v=pFWJsOBwQF0)
- [Regularization Techniques](https://www.youtube.com/watch?v=Q81RR3yKn30) - StatQuest

**Decision Trees:**
- [Decision Trees Clearly Explained](https://www.youtube.com/watch?v=_L39rN6gz7Y) - StatQuest ⭐⭐
- [Entropy and Information Gain](https://www.youtube.com/watch?v=nodQ2s0CUbI)
- [ID3 Algorithm Tutorial](https://www.youtube.com/watch?v=UdTKxGQvYdc)

**Model Evaluation:**
- [Confusion Matrix Deep Dive](https://www.youtube.com/watch?v=Kdsp6soqA7o) - StatQuest
- [Sensitivity and Specificity](https://www.youtube.com/watch?v=vP06aMoz4v8) - StatQuest
- [F1 Score Explained](https://www.youtube.com/watch?v=jJ7ff7Gcq34)

**Linear Regression:**
- [Linear Regression Details](https://www.youtube.com/watch?v=nk2CQITm_eo) - StatQuest
- [Interpreting Coefficients](https://www.youtube.com/watch?v=Qa0xVEqQBpQ)
- [Residual Analysis](https://www.youtube.com/watch?v=sGIKPgwUXzE)

### 🎯 Key Concepts to Revise

#### 1. Logistic Regression (Q1)
- ✅ Missing value imputation strategies
- ✅ Z-score normalization: $z = \frac{x-\mu}{\sigma}$
- ✅ Batch gradient descent steps
- ✅ Cross-entropy loss calculation
- ✅ Probability interpretation

**Practice:** Complete gradient descent iteration, normalize features

#### 2. Bias-Variance (Q2)
- ✅ Training vs validation error patterns
- ✅ Overfitting symptoms and causes
- ✅ Regularization techniques (L1, L2)
- ✅ Model complexity vs performance
- ✅ Cross-validation strategies

**Practice:** Diagnose learning curves, suggest remedies

#### 3. Decision Trees (Q3)
- ✅ Entropy formula: $H(S) = -\sum p_i \log_2(p_i)$
- ✅ Information gain calculation
- ✅ Attribute selection criteria
- ✅ Tree construction algorithm
- ✅ Majority voting for classification

**Practice:** Calculate entropy, build decision tree manually

#### 4. Model Evaluation (Q4)
- ✅ Confusion matrix construction
- ✅ Precision: $\frac{TP}{TP+FP}$
- ✅ Recall: $\frac{TP}{TP+FN}$
- ✅ F1-score: $\frac{2PR}{P+R}$
- ✅ Threshold selection impact

**Practice:** Build confusion matrices, compute all metrics

#### 5. Regression Interpretation (Q5)
- ✅ Coefficient interpretation
- ✅ Unit scaling effects
- ✅ Gradient analysis
- ✅ Feature importance comparison
- ✅ Standardized coefficients

**Practice:** Interpret coefficients, compare feature influence

### 📊 Key Formulas Quick Reference

**Normalization:**
- Z-score: $z = \frac{x - \mu}{\sigma}$
- Min-max: $x_{norm} = \frac{x - \min}{\max - \min}$

**Entropy & Information Gain:**
- Entropy: $H(S) = -\sum_{i=1}^{k} p_i \log_2(p_i)$
- Information Gain: $IG = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$

**Evaluation Metrics:**
- Accuracy: $\frac{TP + TN}{Total}$
- Precision: $\frac{TP}{TP + FP}$
- Recall: $\frac{TP}{TP + FN}$
- F1: $\frac{2 \times Precision \times Recall}{Precision + Recall}$

---

## 📅 7-Day Study Plan

### Day 1-2: Deep Neural Networks
- Watch 3Blue1Brown neural network series
- Review all 5 questions from DNN paper
- Practice gradient descent calculations
- Code perceptron and logistic regression

### Day 3-4: Mathematics for ML
- Watch 3Blue1Brown linear algebra series
- Practice solving linear systems
- Calculate SVD for sample matrices
- Work through calculus optimization problems

### Day 5-6: Machine Learning
- Watch StatQuest ML playlist (key videos)
- Build decision trees by hand
- Practice confusion matrix calculations
- Review bias-variance concepts

### Day 7: Revision & Mock Test
- Solve all three papers timed
- Review weak areas
- Memorize key formulas
- Practice Python implementations

---

## 🎓 Exam Day Tips

### Before the Exam
- [ ] Review formula sheet one last time
- [ ] Bring calculator (if allowed)
- [ ] Get good sleep (7-8 hours)
- [ ] Eat proper breakfast

### During the Exam
1. **Read all questions first** (5 min)
2. **Solve easy questions first** to build confidence
3. **Show all calculations** for partial credit
4. **Verify dimensions** in matrix operations
5. **Check answers** if time permits

### Time Management
- **DNN:** 15-20 min per question
- **MFML:** 12-18 min per question (more calculation)
- **ML:** 12-15 min per question
- Reserve 15 min for review

---

## 📚 Additional Resources

### Online Tools
- **[Matrix Calculator](https://matrixcalc.org/)** - Verify matrix operations
- **[Symbolab](https://www.symbolab.com/)** - Step-by-step solutions
- **[Desmos](https://www.desmos.com/calculator)** - Visualize functions
- **[Wolfram Alpha](https://www.wolframalpha.com/)** - Compute anything

### Practice Platforms
- **[Kaggle Learn](https://www.kaggle.com/learn)** - Hands-on ML tutorials
- **[Google Colab](https://colab.research.google.com/)** - Free Python notebooks
- **[Brilliant.org](https://brilliant.org/)** - Interactive problem solving

### Cheat Sheets
- [Stanford CS229 Cheat Sheet](https://stanford.edu/~shervine/teaching/cs-229/)
- [ML Cheat Sheet](https://ml-cheatsheet.readthedocs.io/)
- [Linear Algebra Review](https://www.cs.cmu.edu/~zkolter/course/linalg/linalg_notes.pdf)

---

## ⚡ Quick Formula Reference

### DNN Quick Formulas
```
Perceptron: w_new = w_old + η(y - ŷ)x
MSE Loss: J = (1/2m)Σ(ŷ - y)²
Sigmoid: σ(z) = 1/(1 + e^(-z))
Softmax: P(k) = e^(zk) / Σe^(zj)
ReLU: max(0, z)
```

### MFML Quick Formulas
```
Determinant (2×2): ad - bc
Eigenvalues: det(A - λI) = 0
Gradient: ∇f = [∂f/∂x, ∂f/∂y]ᵀ
Hessian: H = [[fxx, fxy], [fyx, fyy]]
Distance: d = √(<u-v, u-v>)
```

### ML Quick Formulas
```
Z-score: z = (x - μ) / σ
Entropy: H = -Σ p log₂(p)
Precision: TP / (TP + FP)
Recall: TP / (TP + FN)
F1: 2PR / (P + R)
```

---

## 🎯 Final Checklist

### Theoretical Understanding
- [ ] Can explain perceptron learning rule
- [ ] Understand bias-variance tradeoff
- [ ] Know when to use different activation functions
- [ ] Understand SVD applications in ML
- [ ] Can interpret confusion matrices

### Practical Skills
- [ ] Can perform gradient descent manually
- [ ] Can solve 3×3 linear systems
- [ ] Can build decision tree from data
- [ ] Can calculate all evaluation metrics
- [ ] Can write NumPy implementations

### Exam Readiness
- [ ] Attempted all 2025 practice papers
- [ ] Timed practice runs completed
- [ ] Weak topics identified and reviewed
- [ ] Formula sheet memorized
- [ ] Feeling confident! 💪

---

## 💡 Pro Tips

1. **Focus on understanding, not memorizing** - Concepts help you adapt to new questions
2. **Practice hand calculations** - Most exam questions require showing work
3. **Watch at 1.5x speed** - YouTube videos can be consumed faster
4. **Use active recall** - Test yourself instead of passive re-reading
5. **Teach someone else** - Best way to identify gaps in understanding
6. **Take breaks** - Pomodoro technique (25 min study, 5 min break)
7. **Sleep well** - Memory consolidation happens during sleep

---

## 🚀 Good Luck!

Remember: **You've got this!** 🎓

The fact that you're using this preparation guide shows you're taking your exam seriously. Stay focused, practice consistently, and trust in your preparation.

**"Success is where preparation and opportunity meet."**

---

*Last Updated: December 2025*
*Based on: DNN, MFML, and ML December 2025 Mid-Semester Papers*
