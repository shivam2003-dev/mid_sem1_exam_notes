# Module 5: Decision Trees

## Overview

Decision Trees are versatile algorithms used for both classification and regression. They create a tree-like model of decisions based on feature values.

---

## Introduction to Decision Trees

### What is a Decision Tree?

A **Decision Tree** is a flowchart-like structure where:
- **Internal nodes**: Represent features/attributes
- **Branches**: Represent decision rules (feature values)
- **Leaf nodes**: Represent class labels (classification) or values (regression)

### Example

```
                    Outlook
                   /   |   \
              Sunny Overcast Rainy
              /        |        \
          Humidity    Yes    Wind
          /     \              /   \
      High    Normal      Strong  Weak
       /        |           /       \
      No       Yes         No       Yes
```

**Interpretation**: 
- If Outlook = Sunny and Humidity = High → No (don't play)
- If Outlook = Overcast → Yes (play)
- If Outlook = Rainy and Wind = Weak → Yes (play)

### Advantages

✅ Easy to understand and interpret (visual)
✅ Requires little data preparation
✅ Handles both numerical and categorical data
✅ Can model non-linear relationships
✅ Feature importance is clear

### Disadvantages

❌ Prone to overfitting
❌ Unstable (small data changes → different tree)
❌ Biased toward features with more levels
❌ Can create biased trees if classes are imbalanced

---

## Decision Tree Construction

### Algorithm Overview

**Top-Down Approach (Recursive Partitioning)**:

1. **Start**: All training examples at root
2. **Select Best Feature**: Choose feature that best splits data
3. **Split**: Partition data based on feature values
4. **Recurse**: Repeat for each subset until stopping criterion met
5. **Leaf Node**: Assign class label (majority class) or value (mean)

### Key Questions

1. **Which feature to split on?**
   - Use impurity measures (Entropy, Gini, Information Gain)

2. **When to stop splitting?**
   - All examples in node have same class
   - No more features to split on
   - Maximum depth reached
   - Minimum samples per node
   - Impurity reduction too small

3. **What value to assign to leaf?**
   - **Classification**: Majority class
   - **Regression**: Mean (or median) of target values

---

## Impurity Measures

### Entropy

**Entropy** measures uncertainty/randomness in data.

**Formula**:

$$
H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
$$

Where:
- $S$ = set of examples
- $c$ = number of classes
- $p_i$ = proportion of class $i$ in $S$

**Properties**:
- **Entropy = 0**: Pure node (all same class)
- **Entropy = 1**: Maximum impurity (equal distribution for binary)
- **Maximum Entropy**: $\log_2(c)$ for $c$ classes

**Example** (Binary Classification):
- **Pure node**: [10 Yes, 0 No] → $H = -1 \cdot \log_2(1) - 0 \cdot \log_2(0) = -1 \cdot 0 - 0 = 0$
- **Impure node**: [5 Yes, 5 No] → $H = -0.5 \cdot \log_2(0.5) - 0.5 \cdot \log_2(0.5) = -0.5 \cdot (-1) - 0.5 \cdot (-1) = 1$
- **Mixed node**: [7 Yes, 3 No] → $H = -0.7 \cdot \log_2(0.7) - 0.3 \cdot \log_2(0.3) \approx 0.88$

!!! tip "Remember"
    When $p_i = 0$, we define $p_i \log_2(p_i) = 0$ (by convention) to avoid $\log(0)$ which is undefined.

!!! recommendation "Exam Tip"
    For binary classification, memorize: Maximum entropy = 1 when classes are perfectly balanced (50-50 split).

### Gini Impurity (Gini Index)

**Gini Impurity** measures the probability of misclassifying a randomly chosen element.

**Formula**:

$$
\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2
$$

Where:
- $S$ = set of examples
- $c$ = number of classes
- $p_i$ = proportion of class $i$ in $S$

**Properties**:
- **Gini = 0**: Pure node (all same class)
- **Gini = 0.5**: Maximum impurity for binary classification
- **Maximum Gini**: $1 - \frac{1}{c}$ for $c$ classes

**Example** (Binary Classification):
- **Pure node**: [10 Yes, 0 No] → $\text{Gini} = 1 - (1^2 + 0^2) = 1 - 1 = 0$
- **Impure node**: [5 Yes, 5 No] → $\text{Gini} = 1 - (0.5^2 + 0.5^2) = 1 - 0.5 = 0.5$
- **Mixed node**: [7 Yes, 3 No] → $\text{Gini} = 1 - (0.7^2 + 0.3^2) = 1 - (0.49 + 0.09) = 0.42$

!!! note "Key Point"
    Gini Impurity is computationally faster than Entropy because it doesn't require logarithms. Use Gini when performance is critical.

!!! warning "Common Mistake"
    Don't confuse Gini Impurity with Gini Coefficient (used in economics). They are different concepts!

### Comparison: Entropy vs Gini

| Aspect | Entropy | Gini |
|--------|---------|------|
| **Range** | [0, $\log_2(c)$] | [0, $1-\frac{1}{c}$] |
| **Calculation** | More complex (log) | Simpler (squares) |
| **Sensitivity** | More sensitive to changes | Less sensitive |
| **Performance** | Slightly slower | Faster |
| **Common Use** | ID3, C4.5 | CART |

**Note**: Both work well; choice is often based on convention or performance.

---

## Information Gain

### Definition

**Information Gain** measures reduction in entropy after splitting on a feature.

**Formula**:

$$
\text{IG}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
$$

Where:
- $S$ = set of examples
- $A$ = feature/attribute
- $S_v$ = subset where feature $A$ has value $v$
- $H(S)$ = entropy of $S$
- $H(S_v)$ = entropy of subset $S_v$

**Interpretation**:
- **High IG**: Feature provides good split (reduces uncertainty)
- **IG = 0**: Feature doesn't help (no reduction in entropy)

### Information Gain Ratio

**Problem**: Information Gain favors features with many values.

**Solution**: Normalize by **Split Information**

**Split Information**:

$$
\text{SplitInfo}(S, A) = -\sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)
$$

**Information Gain Ratio**:

$$
\text{IGR}(S, A) = \frac{\text{IG}(S, A)}{\text{SplitInfo}(S, A)}
$$

**Use**: C4.5 algorithm uses Information Gain Ratio

---

## Decision Tree Algorithms

### ID3 (Iterative Dichotomiser 3)

**Algorithm**:
1. Calculate entropy of dataset
2. For each feature, calculate information gain
3. Choose feature with highest information gain
4. Split dataset on chosen feature
5. Recurse for each subset

**Characteristics**:
- Uses Entropy and Information Gain
- Handles categorical features only
- No pruning
- No handling of missing values

### C4.5

**Improvements over ID3**:
- Uses **Information Gain Ratio** (handles many-valued features)
- Handles **continuous features** (creates thresholds)
- Handles **missing values**
- **Pruning** to reduce overfitting

### CART (Classification and Regression Trees)

**Characteristics**:
- Uses **Gini Impurity** (classification) or **MSE** (regression)
- Handles both classification and regression
- Binary splits only (each node has 2 children)
- Uses **Cost Complexity Pruning**

---

## Decision Tree Construction Example

### Example Dataset

| Outlook | Temperature | Humidity | Wind | Play? |
|---------|-------------|----------|------|-------|
| Sunny   | Hot         | High     | Weak | No    |
| Sunny   | Hot         | High     | Strong| No   |
| Overcast| Hot        | High     | Weak | Yes   |
| Rainy   | Mild        | High     | Weak | Yes   |
| Rainy   | Cool        | Normal   | Weak | Yes   |
| Rainy   | Cool        | Normal   | Strong| No   |
| Overcast| Cool       | Normal   | Strong| Yes   |
| Sunny   | Mild        | High     | Weak | No    |
| Sunny   | Cool        | Normal   | Weak | Yes   |
| Rainy   | Mild        | Normal   | Weak | Yes   |
| Sunny   | Mild        | Normal   | Strong| Yes  |
| Overcast| Mild       | High     | Strong| Yes  |
| Overcast| Hot        | Normal   | Weak | Yes   |
| Rainy   | Mild        | High     | Strong| No   |

### Step 1: Calculate Root Entropy

Total: 14 examples
- Yes: 9
- No: 5

$$
H(S) = -\frac{9}{14}\log_2\left(\frac{9}{14}\right) - \frac{5}{14}\log_2\left(\frac{5}{14}\right) \approx 0.940
$$

### Step 2: Calculate Information Gain for Each Feature

**For Outlook**:
- Sunny: [2 Yes, 3 No] → $H = 0.971$
- Overcast: [4 Yes, 0 No] → $H = 0$
- Rainy: [3 Yes, 2 No] → $H = 0.971$

$$
\text{IG}(S, \text{Outlook}) = 0.940 - \left(\frac{5}{14} \times 0.971 + \frac{4}{14} \times 0 + \frac{5}{14} \times 0.971\right) = 0.246
$$

**For Humidity**:
- High: [3 Yes, 4 No] → $H = 0.985$
- Normal: [6 Yes, 1 No] → $H = 0.592$

$$
\text{IG}(S, \text{Humidity}) = 0.940 - \left(\frac{7}{14} \times 0.985 + \frac{7}{14} \times 0.592\right) = 0.152
$$

**For Wind**:
- Weak: [6 Yes, 2 No] → $H = 0.811$
- Strong: [3 Yes, 3 No] → $H = 1.0$

$$
\text{IG}(S, \text{Wind}) = 0.940 - \left(\frac{8}{14} \times 0.811 + \frac{6}{14} \times 1.0\right) = 0.048
$$

**For Temperature**:
- Hot: [2 Yes, 2 No] → $H = 1.0$
- Mild: [4 Yes, 2 No] → $H = 0.918$
- Cool: [3 Yes, 1 No] → $H = 0.811$

$$
\text{IG}(S, \text{Temperature}) = 0.940 - \left(\frac{4}{14} \times 1.0 + \frac{6}{14} \times 0.918 + \frac{4}{14} \times 0.811\right) = 0.029
$$

**Result**: Outlook has highest Information Gain (0.246) → **Split on Outlook**

### Step 3: Build Tree Recursively

```
                    Outlook
                   /   |   \
              Sunny Overcast Rainy
              [2Y,3N] [4Y,0N] [3Y,2N]
              /        |        \
          (Yes)    Humidity    Wind
                    /     \      /   \
                High  Normal Strong Weak
               [0Y,2N] [2Y,1N] [0Y,2N] [3Y,0N]
                 /       |       /       |
               (No)    (Yes)   (No)    (Yes)
```

**Final Tree**:
- If Outlook = Overcast → **Yes**
- If Outlook = Sunny and Humidity = High → **No**
- If Outlook = Sunny and Humidity = Normal → **Yes**
- If Outlook = Rainy and Wind = Strong → **No**
- If Outlook = Rainy and Wind = Weak → **Yes**

---

## Pruning

### Why Prune?

**Overfitting**: Tree too complex, memorizes training data, poor generalization

**Solution**: Remove branches that don't improve generalization

### Types of Pruning

#### 1. Pre-pruning (Early Stopping)

Stop splitting before perfect classification:

**Criteria**:
- Maximum depth
- Minimum samples per node
- Minimum information gain
- Maximum number of leaf nodes

**Advantages**: Faster, simpler
**Disadvantages**: May stop too early (underfitting)

#### 2. Post-pruning

Build full tree, then remove branches:

**Methods**:
- **Reduced Error Pruning**: Remove branch if validation error doesn't increase
- **Cost Complexity Pruning**: Balance tree complexity vs accuracy

**Advantages**: Better results, uses all data
**Disadvantages**: More expensive

### Cost Complexity Pruning

**Objective**: Minimize
$$
\text{Cost} = \text{Error} + \alpha \times \text{Complexity}
$$

Where:
- $\alpha$ = complexity parameter
- Larger $\alpha$ → Simpler tree

**Algorithm**:
1. Build full tree
2. For each $\alpha$, find subtree that minimizes cost
3. Choose $\alpha$ using cross-validation

---

## Regression Trees

### Difference from Classification

**Classification Tree**:
- Predicts class labels
- Uses Entropy/Gini for splitting
- Leaf = majority class

**Regression Tree**:
- Predicts continuous values
- Uses **MSE (Mean Squared Error)** for splitting
- Leaf = mean (or median) of target values

### Splitting Criterion for Regression

**MSE (Mean Squared Error)**:
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2
$$

**Information Gain (Regression)**:
$$
\text{IG} = \text{MSE}_{\text{parent}} - \left(\frac{n_{\text{left}}}{n} \text{MSE}_{\text{left}} + \frac{n_{\text{right}}}{n} \text{MSE}_{\text{right}}\right)
$$

Choose split that maximizes information gain (minimizes weighted MSE).

---

## Key Formulas Summary

### Entropy
$$
H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
$$

### Gini Impurity
$$
\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2
$$

### Information Gain
$$
\text{IG}(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)
$$

### Information Gain Ratio
$$
\text{IGR}(S, A) = \frac{\text{IG}(S, A)}{\text{SplitInfo}(S, A)}
$$

### Regression MSE
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2
$$

---

## Important Points to Remember

✅ **Decision Trees**: Tree structure, easy to interpret

✅ **Entropy**: Measures uncertainty, range [0, $\log_2(c)$]

✅ **Gini**: Measures impurity, range [0, $1-\frac{1}{c}$]

✅ **Information Gain**: Reduction in entropy after split

✅ **ID3**: Uses Entropy, categorical features only

✅ **C4.5**: Uses Information Gain Ratio, handles continuous features

✅ **CART**: Uses Gini, binary splits, handles regression

✅ **Pruning**: Prevents overfitting, improves generalization

---

**Previous**: [Module 4 - Unsupervised Learning](module4-unsupervised-learning.md) | **Back to**: [ML Overview](index.md)

