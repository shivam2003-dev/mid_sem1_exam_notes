# 2024 Mid Semester Regular Paper - Complete Solutions

## Question 1: Linear Regression and Gradient Descent

### Problem Statement

Given the following training data:

| x | y |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 5 |
| 4 | 4 |

**a)** Find the hypothesis function $h_\theta(x) = \theta_0 + \theta_1 x$ using gradient descent with:
- Initial values: $\theta_0 = 0$, $\theta_1 = 0$
- Learning rate: $\alpha = 0.1$
- Perform 2 iterations

**b)** Calculate the cost function $J(\theta)$ after 2 iterations.

---

### Solution

#### Part (a): Gradient Descent

**Given**:
- Training examples: $m = 4$
- Features: $x = [1, 2, 3, 4]^T$
- Targets: $y = [2, 4, 5, 4]^T$
- Initial: $\theta_0 = 0$, $\theta_1 = 0$
- Learning rate: $\alpha = 0.1$

**Hypothesis**: $h_\theta(x) = \theta_0 + \theta_1 x$

**Gradient Descent Update Rules**:
$$
\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})
$$
$$
\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)}
$$

---

#### Iteration 0 (Initial)

$\theta_0 = 0$, $\theta_1 = 0$

**Predictions**:
- $h_\theta(1) = 0 + 0 \times 1 = 0$
- $h_\theta(2) = 0 + 0 \times 2 = 0$
- $h_\theta(3) = 0 + 0 \times 3 = 0$
- $h_\theta(4) = 0 + 0 \times 4 = 0$

**Errors**:
- $(0 - 2) = -2$
- $(0 - 4) = -4$
- $(0 - 5) = -5$
- $(0 - 4) = -4$

---

#### Iteration 1

**Update $\theta_0$**:
$$
\theta_0 := 0 - 0.1 \times \frac{1}{4} \times [(-2) + (-4) + (-5) + (-4)]
$$
$$
\theta_0 := 0 - 0.1 \times \frac{1}{4} \times (-15)
$$
$$
\theta_0 := 0 - 0.1 \times (-3.75)
$$
$$
\theta_0 := 0 + 0.375 = 0.375
$$

**Update $\theta_1$**:
$$
\theta_1 := 0 - 0.1 \times \frac{1}{4} \times [(-2) \times 1 + (-4) \times 2 + (-5) \times 3 + (-4) \times 4]
$$
$$
\theta_1 := 0 - 0.1 \times \frac{1}{4} \times [-2 - 8 - 15 - 16]
$$
$$
\theta_1 := 0 - 0.1 \times \frac{1}{4} \times (-41)
$$
$$
\theta_1 := 0 - 0.1 \times (-10.25)
$$
$$
\theta_1 := 0 + 1.025 = 1.025
$$

**After Iteration 1**: $\theta_0 = 0.375$, $\theta_1 = 1.025$

**New Predictions**:
- $h_\theta(1) = 0.375 + 1.025 \times 1 = 1.4$
- $h_\theta(2) = 0.375 + 1.025 \times 2 = 2.425$
- $h_\theta(3) = 0.375 + 1.025 \times 3 = 3.45$
- $h_\theta(4) = 0.375 + 1.025 \times 4 = 4.475$

**New Errors**:
- $(1.4 - 2) = -0.6$
- $(2.425 - 4) = -1.575$
- $(3.45 - 5) = -1.55$
- $(4.475 - 4) = 0.475$

---

#### Iteration 2

**Update $\theta_0$**:
$$
\theta_0 := 0.375 - 0.1 \times \frac{1}{4} \times [(-0.6) + (-1.575) + (-1.55) + (0.475)]
$$
$$
\theta_0 := 0.375 - 0.1 \times \frac{1}{4} \times (-3.25)
$$
$$
\theta_0 := 0.375 - 0.1 \times (-0.8125)
$$
$$
\theta_0 := 0.375 + 0.08125 = 0.45625
$$

**Update $\theta_1$**:
$$
\theta_1 := 1.025 - 0.1 \times \frac{1}{4} \times [(-0.6) \times 1 + (-1.575) \times 2 + (-1.55) \times 3 + (0.475) \times 4]
$$
$$
\theta_1 := 1.025 - 0.1 \times \frac{1}{4} \times [-0.6 - 3.15 - 4.65 + 1.9]
$$
$$
\theta_1 := 1.025 - 0.1 \times \frac{1}{4} \times (-6.5)
$$
$$
\theta_1 := 1.025 - 0.1 \times (-1.625)
$$
$$
\theta_1 := 1.025 + 0.1625 = 1.1875
$$

**After Iteration 2**: $\theta_0 = 0.45625$, $\theta_1 = 1.1875$

**Final Hypothesis**: $h_\theta(x) = 0.45625 + 1.1875x$

---

#### Part (b): Cost Function

**Cost Function**:
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
$$

**With $\theta_0 = 0.45625$, $\theta_1 = 1.1875$**:

**Predictions**:
- $h_\theta(1) = 0.45625 + 1.1875 \times 1 = 1.64375$
- $h_\theta(2) = 0.45625 + 1.1875 \times 2 = 2.83125$
- $h_\theta(3) = 0.45625 + 1.1875 \times 3 = 4.01875$
- $h_\theta(4) = 0.45625 + 1.1875 \times 4 = 5.20625$

**Squared Errors**:
- $(1.64375 - 2)^2 = (-0.35625)^2 = 0.1269$
- $(2.83125 - 4)^2 = (-1.16875)^2 = 1.3660$
- $(4.01875 - 5)^2 = (-0.98125)^2 = 0.9629$
- $(5.20625 - 4)^2 = (1.20625)^2 = 1.4550$

**Cost**:
$$
J(\theta) = \frac{1}{2 \times 4} \times (0.1269 + 1.3660 + 0.9629 + 1.4550)
$$
$$
J(\theta) = \frac{1}{8} \times 3.9108 = 0.48885
$$

**Answer**: $J(\theta) = 0.48885$

---

## Question 2: Logistic Regression

### Problem Statement

Given training data for binary classification:

| x₁ | x₂ | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 0 |
| 1  | 0  | 0 |
| 1  | 1  | 1 |

**a)** Calculate the hypothesis $h_\theta(x)$ for point $(1, 0)$ with parameters $\theta = [0, 1, 1]^T$ (where $\theta_0 = 0$, $\theta_1 = 1$, $\theta_2 = 1$).

**b)** What is the predicted class for this point?

**c)** Calculate the cost for this single training example.

---

### Solution

#### Part (a): Hypothesis Calculation

**Given**:
- Features: $x = [1, 1, 0]^T$ (with bias term $x_0 = 1$)
- Parameters: $\theta = [0, 1, 1]^T$

**Linear Combination**:
$$
z = \theta^T x = 0 \times 1 + 1 \times 1 + 1 \times 0 = 0 + 1 + 0 = 1
$$

**Sigmoid Function**:
$$
h_\theta(x) = g(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-1}} = \frac{1}{1 + 0.3679} = \frac{1}{1.3679} = 0.731
$$

**Answer**: $h_\theta(x) = 0.731$

---

#### Part (b): Predicted Class

**Decision Rule**:
- If $h_\theta(x) \geq 0.5$, predict $y = 1$
- If $h_\theta(x) < 0.5$, predict $y = 0$

Since $h_\theta(x) = 0.731 \geq 0.5$:

**Answer**: Predicted class = **1**

---

#### Part (c): Cost Calculation

**Actual label**: $y = 0$ (from table, point $(1, 0)$ has $y = 0$)

**Logistic Regression Cost Function** (for single example):
$$Cost(h_\theta(x), y) = \begin{cases}
-\log(h_\theta(x)) & \text{if } y = 1 \\
-\log(1 - h_\theta(x)) & \text{if } y = 0
\end{cases}$$

Since $y = 0$:
$$
Cost = -\log(1 - h_\theta(x)) = -\log(1 - 0.731) = -\log(0.269) = -(-1.313) = 1.313
$$

**Answer**: Cost = **1.313**

---

## Question 3: Evaluation Metrics

### Problem Statement

Given the following confusion matrix for a binary classification problem:

```
                Predicted
              Positive  Negative
Actual Positive   85      15
       Negative   20      80
```

Calculate:
**a)** Accuracy
**b)** Precision
**c)** Recall
**d)** F1-Score

---

### Solution

**From Confusion Matrix**:
- TP = 85 (True Positives)
- TN = 80 (True Negatives)
- FP = 20 (False Positives)
- FN = 15 (False Negatives)
- Total = 85 + 80 + 20 + 15 = 200

---

#### Part (a): Accuracy

**Formula**:
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Calculation**:
$$
\text{Accuracy} = \frac{85 + 80}{200} = \frac{165}{200} = 0.825 = 82.5\%
$$

**Answer**: Accuracy = **0.825** or **82.5%**

---

#### Part (b): Precision

**Formula**:
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Calculation**:
$$
\text{Precision} = \frac{85}{85 + 20} = \frac{85}{105} = 0.8095 = 80.95\%
$$

**Answer**: Precision = **0.8095** or **80.95%**

---

#### Part (c): Recall

**Formula**:
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**Calculation**:
$$
\text{Recall} = \frac{85}{85 + 15} = \frac{85}{100} = 0.85 = 85\%
$$

**Answer**: Recall = **0.85** or **85%**

---

#### Part (d): F1-Score

**Formula**:
$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Using calculated values**:
$$
\text{F1-Score} = 2 \times \frac{0.8095 \times 0.85}{0.8095 + 0.85}
$$
$$
\text{F1-Score} = 2 \times \frac{0.6881}{1.6595}
$$
$$
\text{F1-Score} = 2 \times 0.4148 = 0.8296
$$

**Answer**: F1-Score = **0.8296**

---

## Question 4: Decision Trees

### Problem Statement

Given the following dataset:

| Outlook | Temperature | Play? |
|---------|-------------|-------|
| Sunny   | Hot         | No    |
| Sunny   | Hot         | No    |
| Overcast| Hot         | Yes   |
| Rainy   | Mild        | Yes   |
| Rainy   | Cool        | Yes   |
| Rainy   | Cool        | No    |
| Overcast| Cool       | Yes   |
| Sunny   | Mild        | No    |

**a)** Calculate the entropy of the dataset.

**b)** Calculate the information gain for splitting on "Outlook".

**c)** Which feature should be chosen for the root node? Justify.

---

### Solution

#### Part (a): Entropy of Dataset

**Total examples**: $m = 8$

**Class distribution**:
- Yes: 4 examples
- No: 4 examples

**Entropy Formula**:
$$
H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
$$

**Calculation**:
$$
H(S) = -\left[\frac{4}{8} \log_2\left(\frac{4}{8}\right) + \frac{4}{8} \log_2\left(\frac{4}{8}\right)\right]
$$
$$
H(S) = -\left[0.5 \times \log_2(0.5) + 0.5 \times \log_2(0.5)\right]
$$
$$
H(S) = -\left[0.5 \times (-1) + 0.5 \times (-1)\right]
$$
$$
H(S) = -[-0.5 - 0.5] = -[-1] = 1
$$

**Answer**: $H(S) = 1$ (maximum entropy - completely impure)

---

#### Part (b): Information Gain for Outlook

**Information Gain Formula**:
$$
\text{IG}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
$$

**Outlook has 3 values**: Sunny, Overcast, Rainy

**Split by Outlook**:

1. **Sunny** ($S_{\text{Sunny}}$):
   - Examples: [Sunny/Hot/No, Sunny/Hot/No, Sunny/Mild/No]
   - Yes: 0, No: 3
   - $H(S_{\text{Sunny}}) = -\left[0 \times \log_2(0) + 1 \times \log_2(1)\right] = 0$ (pure - all No)

2. **Overcast** ($S_{\text{Overcast}}$):
   - Examples: [Overcast/Hot/Yes, Overcast/Cool/Yes]
   - Yes: 2, No: 0
   - $H(S_{\text{Overcast}}) = -\left[1 \times \log_2(1) + 0 \times \log_2(0)\right] = 0$ (pure - all Yes)

3. **Rainy** ($S_{\text{Rainy}}$):
   - Examples: [Rainy/Mild/Yes, Rainy/Cool/Yes, Rainy/Cool/No]
   - Yes: 2, No: 1
   - $H(S_{\text{Rainy}}) = -\left[\frac{2}{3} \log_2\left(\frac{2}{3}\right) + \frac{1}{3} \log_2\left(\frac{1}{3}\right)\right]$
   - $H(S_{\text{Rainy}}) = -\left[0.667 \times (-0.585) + 0.333 \times (-1.585)\right]$
   - $H(S_{\text{Rainy}}) = -[-0.390 - 0.528] = 0.918$

**Weighted Average Entropy**:
$$
H(S|\text{Outlook}) = \frac{3}{8} \times 0 + \frac{2}{8} \times 0 + \frac{3}{8} \times 0.918
$$
$$
H(S|\text{Outlook}) = 0 + 0 + 0.344 = 0.344
$$

**Information Gain**:
$$
\text{IG}(S, \text{Outlook}) = H(S) - H(S|\text{Outlook}) = 1 - 0.344 = 0.656
$$

**Answer**: Information Gain = **0.656**

---

#### Part (c): Root Node Selection

**Calculate Information Gain for Temperature**:

**Temperature has 3 values**: Hot, Mild, Cool

1. **Hot**: [Sunny/Hot/No, Sunny/Hot/No, Overcast/Hot/Yes]
   - Yes: 1, No: 2
   - $H(S_{\text{Hot}}) = -\left[\frac{1}{3} \log_2\left(\frac{1}{3}\right) + \frac{2}{3} \log_2\left(\frac{2}{3}\right)\right] = 0.918$

2. **Mild**: [Rainy/Mild/Yes, Sunny/Mild/No]
   - Yes: 1, No: 1
   - $H(S_{\text{Mild}}) = -\left[\frac{1}{2} \log_2\left(\frac{1}{2}\right) + \frac{1}{2} \log_2\left(\frac{1}{2}\right)\right] = 1$

3. **Cool**: [Rainy/Cool/Yes, Rainy/Cool/No, Overcast/Cool/Yes]
   - Yes: 2, No: 1
   - $H(S_{\text{Cool}}) = 0.918$ (same as Hot)

**Weighted Average**:
$$
H(S|\text{Temperature}) = \frac{3}{8} \times 0.918 + \frac{2}{8} \times 1 + \frac{3}{8} \times 0.918 = 0.938
$$

**Information Gain**:
$$
\text{IG}(S, \text{Temperature}) = 1 - 0.938 = 0.062
$$

**Comparison**:
- IG(Outlook) = 0.656
- IG(Temperature) = 0.062

**Answer**: **Outlook** should be chosen as the root node because it has the **highest information gain (0.656)**, meaning it provides the best split and reduces entropy the most.

---

## Question 5: K-Means Clustering

### Problem Statement

Given 4 data points in 2D:
- A(1, 1)
- B(1, 0)
- C(0, 2)
- D(2, 1)

Perform K-Means clustering with $K = 2$:
- Initial centroids: $C_1 = (1, 0)$, $C_2 = (2, 1)$
- Perform 2 iterations

---

### Solution

#### Iteration 1

**Step 1: Assign Points to Nearest Centroid**

**Distance from A(1, 1)**:
- To $C_1(1, 0)$: $\sqrt{(1-1)^2 + (1-0)^2} = \sqrt{0 + 1} = 1$
- To $C_2(2, 1)$: $\sqrt{(1-2)^2 + (1-1)^2} = \sqrt{1 + 0} = 1$
- **Tie**: Assign to $C_1$ (arbitrary choice)

**Distance from B(1, 0)**:
- To $C_1(1, 0)$: $\sqrt{(1-1)^2 + (0-0)^2} = 0$
- To $C_2(2, 1)$: $\sqrt{(1-2)^2 + (0-1)^2} = \sqrt{1 + 1} = \sqrt{2} = 1.414$
- **Assign to $C_1$**

**Distance from C(0, 2)**:
- To $C_1(1, 0)$: $\sqrt{(0-1)^2 + (2-0)^2} = \sqrt{1 + 4} = \sqrt{5} = 2.236$
- To $C_2(2, 1)$: $\sqrt{(0-2)^2 + (2-1)^2} = \sqrt{4 + 1} = \sqrt{5} = 2.236$
- **Tie**: Assign to $C_1$

**Distance from D(2, 1)**:
- To $C_1(1, 0)$: $\sqrt{(2-1)^2 + (1-0)^2} = \sqrt{1 + 1} = \sqrt{2} = 1.414$
- To $C_2(2, 1)$: $\sqrt{(2-2)^2 + (1-1)^2} = 0$
- **Assign to $C_2$**

**Clusters**:
- **Cluster 1**: A, B, C → Centroid: $C_1 = (1, 0)$
- **Cluster 2**: D → Centroid: $C_2 = (2, 1)$

**Step 2: Update Centroids**

**New $C_1$** (mean of A, B, C):
$$
C_1 = \left(\frac{1+1+0}{3}, \frac{1+0+2}{3}\right) = \left(\frac{2}{3}, 1\right) = (0.667, 1)
$$

**New $C_2$** (mean of D):
$$
C_2 = (2, 1)
$$ (unchanged)

**After Iteration 1**: $C_1 = (0.667, 1)$, $C_2 = (2, 1)$

---

#### Iteration 2

**Step 1: Reassign Points**

**Distance from A(1, 1)**:
- To $C_1(0.667, 1)$: $\sqrt{(1-0.667)^2 + (1-1)^2} = 0.333$
- To $C_2(2, 1)$: $\sqrt{(1-2)^2 + (1-1)^2} = 1$
- **Assign to $C_1$**

**Distance from B(1, 0)**:
- To $C_1(0.667, 1)$: $\sqrt{(1-0.667)^2 + (0-1)^2} = \sqrt{0.111 + 1} = 1.054$
- To $C_2(2, 1)$: $\sqrt{(1-2)^2 + (0-1)^2} = \sqrt{1 + 1} = 1.414$
- **Assign to $C_1$**

**Distance from C(0, 2)**:
- To $C_1(0.667, 1)$: $\sqrt{(0-0.667)^2 + (2-1)^2} = \sqrt{0.445 + 1} = 1.205$
- To $C_2(2, 1)$: $\sqrt{(0-2)^2 + (2-1)^2} = \sqrt{4 + 1} = 2.236$
- **Assign to $C_1$**

**Distance from D(2, 1)**:
- To $C_1(0.667, 1)$: $\sqrt{(2-0.667)^2 + (1-1)^2} = 1.333$
- To $C_2(2, 1)$: $0$
- **Assign to $C_2$**

**Clusters** (same as before):
- **Cluster 1**: A, B, C
- **Cluster 2**: D

**Step 2: Update Centroids**

**New $C_1$**: $(0.667, 1)$ (same as before)
**New $C_2$**: $(2, 1)$ (same as before)

**Convergence**: Centroids unchanged → **Algorithm converged!**

**Final Clusters**:
- **Cluster 1**: A(1, 1), B(1, 0), C(0, 2) with centroid $(0.667, 1)$
- **Cluster 2**: D(2, 1) with centroid $(2, 1)$

---

## Summary

This paper covered:
1. ✅ Linear Regression with Gradient Descent
2. ✅ Logistic Regression and Sigmoid Function
3. ✅ Evaluation Metrics (Accuracy, Precision, Recall, F1-Score)
4. ✅ Decision Trees (Entropy, Information Gain)
5. ✅ K-Means Clustering

**Key Takeaways**:
- Always show step-by-step calculations
- Understand formulas and their applications
- Practice gradient descent iterations
- Know evaluation metrics by heart
- Understand decision tree construction

---

**Good luck with your exam!** 🎯

