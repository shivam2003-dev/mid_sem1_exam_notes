# 2025 Practice Set - Complete Step-by-Step Solutions

## Question 1: Gradient Descent Convergence

### Problem Statement

Given the cost function $J(\theta) = (\theta - 3)^2$:

**a)** What is the optimal value of $\theta$?

**b)** Starting from $\theta = 0$, perform 3 iterations of gradient descent with $\alpha = 0.5$.

**c)** Will gradient descent converge? Why?

---

### Solution

#### Part (a): Optimal Value

**Cost Function**: $J(\theta) = (\theta - 3)^2$

**To find minimum, set derivative to zero**:
$$
\frac{dJ}{d\theta} = 2(\theta - 3) = 0
$$
$$
\theta - 3 = 0
$$
$$
\theta = 3
$$

**Answer**: Optimal $\theta = 3$ (minimum cost = 0)

---

#### Part (b): Gradient Descent Iterations

**Gradient**: $\frac{dJ}{d\theta} = 2(\theta - 3)$

**Update Rule**: $\theta := \theta - \alpha \frac{dJ}{d\theta}$

**Given**: $\theta_0 = 0$, $\alpha = 0.5$

**Iteration 1**:
$$
\theta_1 := 0 - 0.5 \times 2(0 - 3) = 0 - 0.5 \times (-6) = 0 + 3 = 3
$$

**Iteration 2**:
$$
\theta_2 := 3 - 0.5 \times 2(3 - 3) = 3 - 0.5 \times 0 = 3
$$

**Iteration 3**:
$$
\theta_3 := 3 - 0.5 \times 2(3 - 3) = 3
$$

**Answer**: 
- After iteration 1: $\theta = 3$
- After iteration 2: $\theta = 3$ (converged)
- After iteration 3: $\theta = 3$ (converged)

---

#### Part (c): Convergence Analysis

**Yes, gradient descent will converge** because:

1. **Convex Function**: $J(\theta) = (\theta - 3)^2$ is a convex function (parabola)
2. **Single Global Minimum**: No local minima
3. **Appropriate Learning Rate**: $\alpha = 0.5$ is suitable for this quadratic function
4. **Gradient Approaches Zero**: As $\theta \to 3$, gradient $\to 0$

**Mathematical Proof**:
- At $\theta = 3$: Gradient = $2(3-3) = 0$ (stationary point)
- Second derivative: $\frac{d^2J}{d\theta^2} = 2 > 0$ (minimum confirmed)

**Answer**: Yes, converges to $\theta = 3$ in 1 iteration with this learning rate.

---

## Question 2: Logistic Regression Decision Boundary

### Problem Statement

Given a logistic regression model with parameters $\theta = [2, -1, 1]^T$:

**a)** Write the decision boundary equation.

**b)** Classify the following points:
- $(1, 1)$
- $(2, 0)$
- $(0, 3)$

**c)** Plot the decision boundary (sketch).

---

### Solution

#### Part (a): Decision Boundary Equation

**Given**: $\theta = [\theta_0, \theta_1, \theta_2]^T = [2, -1, 1]^T$

**Hypothesis**: $h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2)$

**Decision Boundary**: Where $h_\theta(x) = 0.5$, which occurs when:
$$
\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0
$$

**Substituting values**:
$$
2 + (-1)x_1 + (1)x_2 = 0
$$
$$
2 - x_1 + x_2 = 0
$$
$$
x_2 = x_1 - 2
$$

**Answer**: Decision boundary: $x_2 = x_1 - 2$ (or $x_1 - x_2 = 2$)

---

#### Part (b): Classification

**Classification Rule**: 
- If $\theta^T x \geq 0$, predict $y = 1$
- If $\theta^T x < 0$, predict $y = 0$

**For point $(1, 1)$**:
$$
z = 2 + (-1)(1) + (1)(1) = 2 - 1 + 1 = 2 \geq 0
$$
**Prediction**: $y = 1$

**For point $(2, 0)$**:
$$
z = 2 + (-1)(2) + (1)(0) = 2 - 2 + 0 = 0
$$
**Prediction**: $y = 1$ (since $z = 0 \geq 0$)

**For point $(0, 3)$**:
$$
z = 2 + (-1)(0) + (1)(3) = 2 + 0 + 3 = 5 \geq 0
$$
**Prediction**: $y = 1$

**Answer**:
- $(1, 1)$ → **Class 1**
- $(2, 0)$ → **Class 1** (on boundary)
- $(0, 3)$ → **Class 1**

---

#### Part (c): Decision Boundary Plot

**Equation**: $x_2 = x_1 - 2$

**Key Points**:
- When $x_1 = 0$: $x_2 = -2$ → Point $(0, -2)$
- When $x_2 = 0$: $0 = x_1 - 2$ → $x_1 = 2$ → Point $(2, 0)$
- When $x_1 = 4$: $x_2 = 2$ → Point $(4, 2)$

**Sketch**:
```
x₂
  |
 3|                    ● (0,3)
  |                   /
 2|                  /  ● (4,2)
  |                 /
 1|    ● (1,1)     /
  |              /
 0|_____________● (2,0)___________ x₁
  |            /
-1|           /
-2|          ● (0,-2)
  |
```

**Region Above Line** ($x_2 > x_1 - 2$): Class 1
**Region Below Line** ($x_2 < x_1 - 2$): Class 0

---

## Question 3: Confusion Matrix and Metrics

### Problem Statement

A classifier is evaluated on a test set of 200 examples. The results are:

- Correctly classified as Positive: 60
- Incorrectly classified as Positive: 20
- Correctly classified as Negative: 100
- Incorrectly classified as Negative: 20

**a)** Construct the confusion matrix.

**b)** Calculate all evaluation metrics.

**c)** Interpret the results.

---

### Solution

#### Part (a): Confusion Matrix

**Given Information**:
- TP = 60 (Correctly classified as Positive)
- FP = 20 (Incorrectly classified as Positive)
- TN = 100 (Correctly classified as Negative)
- FN = 20 (Incorrectly classified as Negative)

**Confusion Matrix**:

```
                Predicted
              Positive  Negative
Actual Positive   60      20
       Negative   20     100
```

**Verification**: Total = 60 + 20 + 20 + 100 = 200 ✓

---

#### Part (b): Evaluation Metrics

**1. Accuracy**:
$$
\text{Accuracy} = \frac{TP + TN}{Total} = \frac{60 + 100}{200} = \frac{160}{200} = 0.80 = 80\%
$$

**2. Precision**:
$$
\text{Precision} = \frac{TP}{TP + FP} = \frac{60}{60 + 20} = \frac{60}{80} = 0.75 = 75\%
$$

**3. Recall (Sensitivity)**:
$$
\text{Recall} = \frac{TP}{TP + FN} = \frac{60}{60 + 20} = \frac{60}{80} = 0.75 = 75\%
$$

**4. Specificity**:
$$
\text{Specificity} = \frac{TN}{TN + FP} = \frac{100}{100 + 20} = \frac{100}{120} = 0.833 = 83.3\%
$$

**5. F1-Score**:
$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.75 \times 0.75}{0.75 + 0.75} = 2 \times \frac{0.5625}{1.5} = 0.75 = 75\%
$$

**6. Error Rate**:
$$
\text{Error Rate} = \frac{FP + FN}{Total} = \frac{20 + 20}{200} = \frac{40}{200} = 0.20 = 20\%
$$

**7. False Positive Rate (FPR)**:
$$
\text{FPR} = \frac{FP}{FP + TN} = \frac{20}{20 + 100} = \frac{20}{120} = 0.167 = 16.7\%
$$

**8. False Negative Rate (FNR)**:
$$
\text{FNR} = \frac{FN}{FN + TP} = \frac{20}{20 + 60} = \frac{20}{80} = 0.25 = 25\%
$$

**Answer**: All metrics calculated above

---

#### Part (c): Interpretation

**Overall Performance**:
- **Accuracy = 80%**: Model correctly classifies 4 out of 5 examples
- **Balanced Performance**: Precision = Recall = 75% (balanced classifier)

**Class-Specific Performance**:
- **Positive Class**: 
  - Precision = 75%: When predicting positive, 75% are correct
  - Recall = 75%: Catches 75% of actual positives
  - 20 false negatives (missed 25% of positives)

- **Negative Class**:
  - Specificity = 83.3%: Correctly identifies 83.3% of negatives
  - 20 false positives (incorrectly flagged 16.7% of negatives)

**Error Analysis**:
- Equal false positives and false negatives (20 each)
- Balanced error distribution
- No strong bias toward either class

**Conclusion**: The classifier shows **balanced performance** with equal precision and recall. It's suitable when both false positives and false negatives have similar costs.

---

## Question 4: K-Means Clustering

### Problem Statement

Given 5 data points:
- A(0, 0)
- B(1, 1)
- C(2, 2)
- D(5, 5)
- E(6, 6)

Perform K-Means with $K = 2$:
- Initial centroids: $C_1 = (0, 0)$, $C_2 = (5, 5)$
- Perform iterations until convergence.

---

### Solution

#### Iteration 1

**Step 1: Assign Points to Nearest Centroid**

**Distance Calculations**:

**Point A(0, 0)**:
- To $C_1(0, 0)$: $\sqrt{(0-0)^2 + (0-0)^2} = 0$
- To $C_2(5, 5)$: $\sqrt{(0-5)^2 + (0-5)^2} = \sqrt{50} = 7.07$
- **Assign to Cluster 1**

**Point B(1, 1)**:
- To $C_1(0, 0)$: $\sqrt{(1-0)^2 + (1-0)^2} = \sqrt{2} = 1.41$
- To $C_2(5, 5)$: $\sqrt{(1-5)^2 + (1-5)^2} = \sqrt{32} = 5.66$
- **Assign to Cluster 1**

**Point C(2, 2)**:
- To $C_1(0, 0)$: $\sqrt{(2-0)^2 + (2-0)^2} = \sqrt{8} = 2.83$
- To $C_2(5, 5)$: $\sqrt{(2-5)^2 + (2-5)^2} = \sqrt{18} = 4.24$
- **Assign to Cluster 1**

**Point D(5, 5)**:
- To $C_1(0, 0)$: $\sqrt{(5-0)^2 + (5-0)^2} = \sqrt{50} = 7.07$
- To $C_2(5, 5)$: $0$
- **Assign to Cluster 2**

**Point E(6, 6)**:
- To $C_1(0, 0)$: $\sqrt{(6-0)^2 + (6-0)^2} = \sqrt{72} = 8.49$
- To $C_2(5, 5)$: $\sqrt{(6-5)^2 + (6-5)^2} = \sqrt{2} = 1.41$
- **Assign to Cluster 2**

**Clusters**:
- **Cluster 1**: A, B, C
- **Cluster 2**: D, E

**Step 2: Update Centroids**

**New $C_1$** (mean of A, B, C):
$$
C_1 = \left(\frac{0+1+2}{3}, \frac{0+1+2}{3}\right) = (1, 1)
$$

**New $C_2$** (mean of D, E):
$$
C_2 = \left(\frac{5+6}{2}, \frac{5+6}{2}\right) = (5.5, 5.5)
$$

**After Iteration 1**: $C_1 = (1, 1)$, $C_2 = (5.5, 5.5)$

---

#### Iteration 2

**Step 1: Reassign Points**

**Point A(0, 0)**:
- To $C_1(1, 1)$: $\sqrt{2} = 1.41$
- To $C_2(5.5, 5.5)$: $\sqrt{60.5} = 7.78$
- **Assign to Cluster 1**

**Point B(1, 1)**:
- To $C_1(1, 1)$: $0$
- To $C_2(5.5, 5.5)$: $\sqrt{40.5} = 6.36$
- **Assign to Cluster 1**

**Point C(2, 2)**:
- To $C_1(1, 1)$: $\sqrt{2} = 1.41$
- To $C_2(5.5, 5.5)$: $\sqrt{24.5} = 4.95$
- **Assign to Cluster 1**

**Point D(5, 5)**:
- To $C_1(1, 1)$: $\sqrt{32} = 5.66$
- To $C_2(5.5, 5.5)$: $\sqrt{0.5} = 0.71$
- **Assign to Cluster 2**

**Point E(6, 6)**:
- To $C_1(1, 1)$: $\sqrt{50} = 7.07$
- To $C_2(5.5, 5.5)$: $\sqrt{0.5} = 0.71$
- **Assign to Cluster 2**

**Clusters** (same as before):
- **Cluster 1**: A, B, C
- **Cluster 2**: D, E

**Step 2: Update Centroids**

**New $C_1$**: $(1, 1)$ (same as before)
**New $C_2$**: $(5.5, 5.5)$ (same as before)

**Convergence**: Centroids unchanged → **Algorithm converged!**

**Final Result**:
- **Cluster 1**: A(0,0), B(1,1), C(2,2) with centroid $(1, 1)$
- **Cluster 2**: D(5,5), E(6,6) with centroid $(5.5, 5.5)$

---

## Question 5: Decision Tree - Information Gain

### Problem Statement

Given dataset:

| Weather | Temperature | Play? |
|---------|-------------|-------|
| Sunny   | Hot         | No    |
| Sunny   | Mild        | Yes   |
| Rainy   | Cool        | Yes   |
| Rainy   | Mild        | Yes   |
| Overcast| Hot         | Yes   |
| Overcast| Cool        | No    |

**a)** Calculate entropy of the dataset.

**b)** Calculate information gain for "Weather".

**c)** Calculate information gain for "Temperature".

**d)** Build the decision tree.

---

### Solution

#### Part (a): Dataset Entropy

**Total examples**: $m = 6$

**Class distribution**:
- Yes: 4 examples
- No: 2 examples

**Entropy**:
$$
H(S) = -\left[\frac{4}{6} \log_2\left(\frac{4}{6}\right) + \frac{2}{6} \log_2\left(\frac{2}{6}\right)\right]
$$
$$
H(S) = -\left[0.667 \times (-0.585) + 0.333 \times (-1.585)\right]
$$
$$
H(S) = -[-0.390 - 0.528] = 0.918
$$

**Answer**: $H(S) = 0.918$

---

#### Part (b): Information Gain for Weather

**Weather has 3 values**: Sunny, Rainy, Overcast

**Split by Weather**:

1. **Sunny**:
   - Examples: [Sunny/Hot/No, Sunny/Mild/Yes]
   - Yes: 1, No: 1
   - $H(S_{\text{Sunny}}) = -\left[\frac{1}{2} \log_2\left(\frac{1}{2}\right) + \frac{1}{2} \log_2\left(\frac{1}{2}\right)\right] = 1$

2. **Rainy**:
   - Examples: [Rainy/Cool/Yes, Rainy/Mild/Yes]
   - Yes: 2, No: 0
   - $H(S_{\text{Rainy}}) = 0$ (pure - all Yes)

3. **Overcast**:
   - Examples: [Overcast/Hot/Yes, Overcast/Cool/No]
   - Yes: 1, No: 1
   - $H(S_{\text{Overcast}}) = 1$

**Weighted Average Entropy**:
$$
H(S|\text{Weather}) = \frac{2}{6} \times 1 + \frac{2}{6} \times 0 + \frac{2}{6} \times 1 = \frac{4}{6} = 0.667
$$

**Information Gain**:
$$
\text{IG}(S, \text{Weather}) = 0.918 - 0.667 = 0.251
$$

**Answer**: IG(Weather) = **0.251**

---

#### Part (c): Information Gain for Temperature

**Temperature has 3 values**: Hot, Mild, Cool

**Split by Temperature**:

1. **Hot**:
   - Examples: [Sunny/Hot/No, Overcast/Hot/Yes]
   - Yes: 1, No: 1
   - $H(S_{\text{Hot}}) = 1$

2. **Mild**:
   - Examples: [Sunny/Mild/Yes, Rainy/Mild/Yes]
   - Yes: 2, No: 0
   - $H(S_{\text{Mild}}) = 0$ (pure)

3. **Cool**:
   - Examples: [Rainy/Cool/Yes, Overcast/Cool/No]
   - Yes: 1, No: 1
   - $H(S_{\text{Cool}}) = 1$

**Weighted Average Entropy**:
$$
H(S|\text{Temperature}) = \frac{2}{6} \times 1 + \frac{2}{6} \times 0 + \frac{2}{6} \times 1 = 0.667
$$

**Information Gain**:
$$
\text{IG}(S, \text{Temperature}) = 0.918 - 0.667 = 0.251
$$

**Answer**: IG(Temperature) = **0.251**

**Tie!** Both have same information gain.

---

#### Part (d): Decision Tree Construction

**Since IG(Weather) = IG(Temperature) = 0.251**, we can choose either. Let's choose **Weather** as root.

**Step 1: Root Split on Weather**

```
                    Weather
                   /   |   \
              Sunny Rainy Overcast
              [1Y,1N] [2Y,0N] [1Y,1N]
```

**Step 2: Handle Pure Nodes**

- **Rainy**: Pure (all Yes) → Leaf: **Yes**

**Step 3: Split Impure Nodes**

**For Sunny branch** (needs further splitting):
- Split on Temperature:
  - Hot → No
  - Mild → Yes

**For Overcast branch** (needs further splitting):
- Split on Temperature:
  - Hot → Yes
  - Cool → No

**Final Decision Tree**:

```
                    Weather
                   /   |   \
              Sunny Rainy Overcast
              /      |        \
        Temperature  Yes   Temperature
         /     \              /     \
      Hot    Mild         Hot     Cool
       |      |            |       |
      No     Yes          Yes     No
```

**Answer**: Decision tree constructed as shown above

---

## Summary

This practice set covered:
1. ✅ Gradient Descent Convergence Analysis
2. ✅ Logistic Regression Decision Boundaries
3. ✅ Comprehensive Evaluation Metrics
4. ✅ K-Means Clustering
5. ✅ Decision Tree Construction with Information Gain

**Key Takeaways**:
- Understand gradient descent convergence conditions
- Know how to find and plot decision boundaries
- Master all evaluation metrics
- Practice K-Means iterations
- Build decision trees step-by-step

---

**Practice makes perfect! Keep solving problems!** 🎯

