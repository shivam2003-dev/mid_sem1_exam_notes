# Module 3: Classification & Evaluation Metrics

## Overview

This module covers classification algorithms and how to evaluate their performance using various metrics.

---

## Classification Algorithms

### 1. Logistic Regression (Review)

- Binary classification using sigmoid function
- Outputs probability of class membership
- Decision boundary: $\theta^T x = 0$

### 2. K-Nearest Neighbors (KNN)

**Algorithm**:
1. Choose parameter $K$ (number of neighbors)
2. For new data point:
   - Find $K$ nearest training examples
   - Classify based on majority vote of neighbors

**Distance Metrics**:
- Euclidean: $d = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$
- Manhattan: $d = \sum_{i=1}^{n} |x_i - y_i|$

**Choosing K**:
- Small K: More sensitive to noise, complex boundaries
- Large K: Smoother boundaries, may underfit
- Rule of thumb: $K = \sqrt{n}$ where $n$ is number of samples

### 3. Support Vector Machines (SVM)

**Goal**: Find optimal hyperplane that separates classes with maximum margin

**Key Concepts**:
- **Support Vectors**: Data points closest to decision boundary
- **Margin**: Distance between decision boundary and nearest points
- **Kernel Trick**: Transform data to higher dimensions for non-linear separation

---

## Confusion Matrix

A confusion matrix is a table used to evaluate classification performance.

### Binary Classification Confusion Matrix

```
                    Predicted
                 Positive  Negative
Actual Positive    TP       FN
       Negative    FP       TN
```

**Terminology**:
- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Incorrectly predicted positive (Type I error)
- **FN (False Negative)**: Incorrectly predicted negative (Type II error)

### Multi-class Confusion Matrix

For $n$ classes, it's an $n \times n$ matrix where:
- Rows = Actual classes
- Columns = Predicted classes
- Diagonal elements = Correct predictions
- Off-diagonal elements = Misclassifications

---

## Evaluation Metrics

### 1. Accuracy

**Definition**: Proportion of correct predictions

**Formula**:
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

**Range**: [0, 1] or [0%, 100%]

**When to Use**:
- ✅ Balanced classes
- ✅ Equal cost for all errors
- ❌ Not good for imbalanced datasets

**Limitation**: Can be misleading with imbalanced data
- Example: 95% accuracy with 95% negative class → Always predicting negative gives 95% accuracy!

### 2. Precision

**Definition**: Of all positive predictions, how many were actually positive?

**Formula**:
$$
\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{True Positives}}{\text{All Predicted Positives}}
$$

**Interpretation**: 
- High precision = Low false positive rate
- "When we predict positive, how often are we right?"

**Use Cases**:
- Spam detection (minimize false positives - don't want to mark important emails as spam)
- Medical diagnosis (minimize false alarms)

### 3. Recall (Sensitivity)

**Definition**: Of all actual positives, how many did we correctly identify?

**Formula**:
$$
\text{Recall} = \frac{TP}{TP + FN} = \frac{\text{True Positives}}{\text{All Actual Positives}}
$$

**Also called**:
- Sensitivity
- True Positive Rate (TPR)

**Interpretation**:
- High recall = Low false negative rate
- "Of all actual positives, how many did we catch?"

**Use Cases**:
- Disease detection (don't want to miss actual cases)
- Fraud detection (don't want to miss fraudulent transactions)

### 4. Specificity

**Definition**: Of all actual negatives, how many did we correctly identify?

**Formula**:
$$
\text{Specificity} = \frac{TN}{TN + FP} = \frac{\text{True Negatives}}{\text{All Actual Negatives}}
$$

**Also called**: True Negative Rate (TNR)

**Interpretation**: Ability to correctly identify negative cases

### 5. F1-Score

**Definition**: Harmonic mean of Precision and Recall

**Formula**:
$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
$$

**Range**: [0, 1]

**Why Harmonic Mean?**
- Penalizes extreme values
- Better than arithmetic mean when one metric is very low

**When to Use**:
- ✅ Need balance between precision and recall
- ✅ Imbalanced datasets
- ✅ Single metric to optimize

### 6. Fβ-Score

**Generalized F-Score**:

$$
\text{F}_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{(\beta^2 \times \text{Precision}) + \text{Recall}}
$$

**Common Values**:
- $\beta = 1$: F1-Score (equal weight)
- $\beta = 2$: F2-Score (more weight to recall)
- $\beta = 0.5$: F0.5-Score (more weight to precision)

### 7. Error Rate

**Formula**:
$$
\text{Error Rate} = \frac{FP + FN}{TP + TN + FP + FN} = 1 - \text{Accuracy}
$$

---

## ROC Curve and AUC

### ROC Curve (Receiver Operating Characteristic)

**Definition**: Plot of True Positive Rate (TPR) vs False Positive Rate (FPR) at different classification thresholds.

**Axes**:
- **X-axis**: False Positive Rate (FPR) = $\frac{FP}{FP + TN}$
- **Y-axis**: True Positive Rate (TPR) = Recall = $\frac{TP}{TP + FN}$

**How it works**:
1. Vary classification threshold from 0 to 1
2. For each threshold, calculate TPR and FPR
3. Plot points and connect to form curve

**Interpretation**:
- **Top-left corner** (0, 1): Perfect classifier
  - TPR = 1, FPR = 0
- **Diagonal line**: Random classifier (no better than guessing)
- **Above diagonal**: Better than random
- **Below diagonal**: Worse than random

**Key Points**:
- **(0, 0)**: Threshold = 1, predict all negative
- **(1, 1)**: Threshold = 0, predict all positive

### AUC (Area Under the Curve)

**Definition**: Area under the ROC curve

**Range**: [0, 1]

**Interpretation**:
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random classifier (diagonal line)
- **AUC > 0.5**: Better than random
- **AUC < 0.5**: Worse than random (flip predictions!)

**Meaning**: 
- Probability that classifier ranks a random positive example higher than a random negative example
- Higher AUC = Better classifier at distinguishing classes

**Advantages**:
- ✅ Threshold-independent
- ✅ Works well with imbalanced data
- ✅ Single number summary

**When to Use**:
- Binary classification
- Need threshold-independent metric
- Comparing different models

---

## Precision-Recall Curve

**Definition**: Plot of Precision vs Recall at different thresholds

**When to Use Instead of ROC**:
- ✅ Highly imbalanced datasets
- ✅ More informative when positive class is rare
- ✅ Focus on positive class performance

**AUC-PR**: Area under Precision-Recall curve
- Higher is better
- More sensitive to class imbalance than ROC-AUC

---

## Multi-class Classification Metrics

### Macro-Averaging

Calculate metric for each class, then average:

$$
\text{Macro-Precision} = \frac{1}{C} \sum_{i=1}^{C} \text{Precision}_i
$$

$$
\text{Macro-Recall} = \frac{1}{C} \sum_{i=1}^{C} \text{Recall}_i
$$

**Treats all classes equally**

### Micro-Averaging

Aggregate all TP, FP, FN across classes, then calculate:

$$
\text{Micro-Precision} = \frac{\sum_{i=1}^{C} TP_i}{\sum_{i=1}^{C} (TP_i + FP_i)}
$$

**Gives equal weight to each sample (not each class)**

### Weighted-Averaging

Weight by number of samples in each class:

$$
\text{Weighted-Precision} = \sum_{i=1}^{C} w_i \times \text{Precision}_i
$$

where $w_i = \frac{n_i}{N}$ (proportion of class $i$)

---

## Choosing the Right Metric

### When to Use Each Metric

| Metric | Best For | Example Use Case |
|--------|----------|------------------|
| **Accuracy** | Balanced classes, equal error costs | General classification |
| **Precision** | Minimize false positives | Spam detection, medical screening |
| **Recall** | Minimize false negatives | Disease diagnosis, fraud detection |
| **F1-Score** | Balance precision and recall | General binary classification |
| **AUC-ROC** | Threshold-independent, imbalanced data | Model comparison, binary classification |
| **AUC-PR** | Highly imbalanced data | Rare event detection |

### Decision Framework

1. **Is the dataset balanced?**
   - Balanced → Accuracy, F1-Score
   - Imbalanced → Precision, Recall, F1-Score, AUC

2. **What's the cost of errors?**
   - False positives expensive → Optimize Precision
   - False negatives expensive → Optimize Recall
   - Both important → Optimize F1-Score

3. **Do you need threshold-independent metric?**
   - Yes → AUC-ROC or AUC-PR
   - No → Precision, Recall, F1-Score

---

## Key Formulas Summary

### Binary Classification Metrics

- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **Specificity**: $\frac{TN}{TN + FP}$
- **F1-Score**: $\frac{2TP}{2TP + FP + FN}$
- **FPR**: $\frac{FP}{FP + TN}$
- **FNR**: $\frac{FN}{FN + TP}$

### ROC Curve
- **TPR (Y-axis)**: $\frac{TP}{TP + FN}$ = Recall
- **FPR (X-axis)**: $\frac{FP}{FP + TN}$

---

## Worked Examples (Exam-Style)

### Worked Example 1: Metrics from Confusion Matrix

Given:
- $TP=50,\ FP=10,\ FN=5,\ TN=35$

1) **Accuracy**:

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}=\frac{50+35}{50+35+10+5}=\frac{85}{100}=0.85
$$

2) **Precision**:

$$
\text{Precision}=\frac{TP}{TP+FP}=\frac{50}{50+10}=\frac{50}{60}=0.833
$$

3) **Recall**:

$$
\text{Recall}=\frac{TP}{TP+FN}=\frac{50}{50+5}=\frac{50}{55}=0.909
$$

4) **F1**:

$$
F1=2\cdot\frac{PR}{P+R}
=2\cdot\frac{0.833\cdot0.909}{0.833+0.909}
\approx 0.869
$$

```{admonition} How to score full marks
:class: note
Write the **formula** first, then substitute numbers, then final value (with 3 decimals).

```

### Worked Example 2: ROC Point at a Threshold

If a classifier at threshold $t$ gives $TP=40,\ FN=10,\ FP=20,\ TN=30$:

$$
TPR=\frac{TP}{TP+FN}=\frac{40}{50}=0.8
$$

$$
FPR=\frac{FP}{FP+TN}=\frac{20}{50}=0.4
$$

So ROC has a point **(0.4, 0.8)**.

---

## Common Pitfalls

- **Using Accuracy on imbalanced data**: can look “high” even for a useless model.
- **Mixing up Precision and Recall**:
  - Precision cares about **FP**
  - Recall cares about **FN**
- **Forgetting class meaning**: always state which class is “positive”.

---

## Quick Revision Checklist (2 minutes)

- Can you draw confusion matrix and label **TP/FP/FN/TN**?
- Do you remember all metric formulas?
- Can you explain when to use **F1 vs ROC-AUC vs PR-AUC**?
- Can you compute one ROC point from counts?

---

## Important Points to Remember

✅ **Confusion Matrix**: Foundation for all classification metrics

✅ **Precision**: "Of predictions, how many correct?" → Minimize false positives

✅ **Recall**: "Of actual positives, how many found?" → Minimize false negatives

✅ **F1-Score**: Harmonic mean, balances precision and recall

✅ **ROC-AUC**: Threshold-independent, good for imbalanced data

✅ **Choose metric based on**: Class balance, error costs, use case

---

**Previous**: [Module 2 - Supervised Learning](module2-supervised-learning.md) | **Next**: [Module 4 - Unsupervised Learning](module4-unsupervised-learning.md)

