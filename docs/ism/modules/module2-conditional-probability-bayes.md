# Module 2: Conditional Probability & Bayes Theorem

## Overview

This module covers conditional probability, its relationship with independence, Bayes' theorem, and its application to Naïve Bayes classification.

## 2.1 Conditional Probability

### Definition

The **conditional probability** of event $A$ given event $B$ is:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

provided $P(B) > 0$.

### Interpretation

- Probability of $A$ occurring given that $B$ has occurred
- "Restricted" sample space to outcomes in $B$
- Updates our belief about $A$ after observing $B$

### Multiplication Rule

From the definition:

$$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

### Extended Multiplication Rule

For events $A_1, A_2, \ldots, A_n$:

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1 \cap A_2) \cdots P(A_n|A_1 \cap \cdots \cap A_{n-1})$$

### Example

In a class of 30 students: 18 are girls, 12 are boys. 10 girls and 6 boys wear glasses.

Find $P(\text{Girl}|\text{Wears Glasses})$:

**Given:**
- $P(\text{Girl}) = \frac{18}{30} = \frac{3}{5}$
- $P(\text{Wears Glasses}|\text{Girl}) = \frac{10}{18} = \frac{5}{9}$
- $P(\text{Wears Glasses}) = \frac{16}{30} = \frac{8}{15}$

**Solution:**
$$P(\text{Girl}|\text{Wears Glasses}) = \frac{P(\text{Girl} \cap \text{Wears Glasses})}{P(\text{Wears Glasses})}$$

$$= \frac{P(\text{Wears Glasses}|\text{Girl}) \cdot P(\text{Girl})}{P(\text{Wears Glasses})} = \frac{\frac{5}{9} \times \frac{3}{5}}{\frac{8}{15}} = \frac{\frac{1}{3}}{\frac{8}{15}} = \frac{5}{8}$$

---

## 2.2 Conditional Probability of Independent Events

### Key Result

If events $A$ and $B$ are **independent**, then:

$$P(A|B) = P(A)$$

and

$$P(B|A) = P(B)$$

### Proof

If $A$ and $B$ are independent:
$$P(A \cap B) = P(A) \cdot P(B)$$

Then:
$$P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{P(A) \cdot P(B)}{P(B)} = P(A)$$

### Interpretation

- Knowing $B$ occurred doesn't change the probability of $A$
- Information about $B$ provides no information about $A$
- Events are "unrelated" in probabilistic sense

### Example

Tossing a fair coin twice:
- $A$ = first toss is heads
- $B$ = second toss is heads

Since $A$ and $B$ are independent:
$$P(A|B) = P(A) = \frac{1}{2}$$

The outcome of the second toss doesn't affect the probability of the first toss being heads.

---

## 2.3 Bayes' Theorem

### Statement

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

### Components

- **Prior:** $P(A)$ - initial probability of $A$
- **Likelihood:** $P(B|A)$ - probability of $B$ given $A$
- **Evidence:** $P(B)$ - probability of $B$
- **Posterior:** $P(A|B)$ - updated probability of $A$ after observing $B$

### Law of Total Probability

If $B_1, B_2, \ldots, B_n$ form a partition of the sample space (mutually exclusive and exhaustive):

$$P(A) = \sum_{i=1}^{n} P(A|B_i) \cdot P(B_i)$$

### Extended Bayes' Theorem

For a partition $\{B_1, B_2, \ldots, B_n\}$:

$$P(B_i|A) = \frac{P(A|B_i) \cdot P(B_i)}{\sum_{j=1}^{n} P(A|B_j) \cdot P(B_j)}$$

### Example: Medical Test

A disease affects 1% of population. Test is 95% accurate:
- 95% true positive rate: $P(\text{Test +}|\text{Disease}) = 0.95$
- 95% true negative rate: $P(\text{Test -}|\text{No Disease}) = 0.95$

Find $P(\text{Disease}|\text{Test +})$:

**Given:**
- $P(\text{Disease}) = 0.01$
- $P(\text{Test +}|\text{Disease}) = 0.95$
- $P(\text{Test +}|\text{No Disease}) = 0.05$

**Solution:**

Using Law of Total Probability:
$$P(\text{Test +}) = P(\text{Test +}|\text{Disease})P(\text{Disease}) + P(\text{Test +}|\text{No Disease})P(\text{No Disease})$$
$$= 0.95(0.01) + 0.05(0.99) = 0.0095 + 0.0495 = 0.059$$

Using Bayes' Theorem:
$$P(\text{Disease}|\text{Test +}) = \frac{P(\text{Test +}|\text{Disease}) \cdot P(\text{Disease})}{P(\text{Test +})}$$
$$= \frac{0.95 \times 0.01}{0.059} = \frac{0.0095}{0.059} = 0.161$$

**Interpretation:** Despite a positive test, only 16.1% chance of having the disease! This is because the disease is rare (1%).

---

## 2.4 Introduction to Naïve Bayes Concept

### Naïve Bayes Classifier

A probabilistic classifier based on Bayes' theorem with a "naïve" assumption of independence.

### Classification Problem

Given features $X = (X_1, X_2, \ldots, X_n)$, predict class $C_k$.

### Bayes' Rule for Classification

$$P(C_k|X) = \frac{P(X|C_k) \cdot P(C_k)}{P(X)}$$

### Naïve Assumption

Assume features are **conditionally independent** given the class:

$$P(X|C_k) = P(X_1, X_2, \ldots, X_n|C_k) = \prod_{i=1}^{n} P(X_i|C_k)$$

### Naïve Bayes Classifier

$$P(C_k|X) = \frac{P(C_k) \prod_{i=1}^{n} P(X_i|C_k)}{P(X)}$$

Since $P(X)$ is constant for all classes, we can ignore it for classification:

$$\hat{C} = \arg\max_{k} P(C_k) \prod_{i=1}^{n} P(X_i|C_k)$$

### Steps

1. **Training:**
   - Estimate $P(C_k)$ for each class
   - Estimate $P(X_i|C_k)$ for each feature and class

2. **Prediction:**
   - For new instance $X$, compute $P(C_k|X)$ for each class
   - Predict class with highest probability

### Example: Spam Classification

**Training Data:**
- 100 emails: 60 spam, 40 not spam
- Word "money" appears in 40 spam, 5 not spam
- Word "meeting" appears in 10 spam, 20 not spam

**New Email:** Contains both "money" and "meeting"

**Probabilities:**
- $P(\text{Spam}) = \frac{60}{100} = 0.6$
- $P(\text{Not Spam}) = \frac{40}{100} = 0.4$
- $P(\text{"money"}|\text{Spam}) = \frac{40}{60} = \frac{2}{3}$
- $P(\text{"money"}|\text{Not Spam}) = \frac{5}{40} = \frac{1}{8}$
- $P(\text{"meeting"}|\text{Spam}) = \frac{10}{60} = \frac{1}{6}$
- $P(\text{"meeting"}|\text{Not Spam}) = \frac{20}{40} = \frac{1}{2}$

**Classification:**

$$P(\text{Spam}|\text{"money", "meeting"}) \propto 0.6 \times \frac{2}{3} \times \frac{1}{6} = 0.6 \times \frac{2}{18} = \frac{1.2}{18} = 0.067$$

$$P(\text{Not Spam}|\text{"money", "meeting"}) \propto 0.4 \times \frac{1}{8} \times \frac{1}{2} = 0.4 \times \frac{1}{16} = 0.025$$

Since $0.067 > 0.025$, classify as **Spam**.

### Advantages

- Simple and fast
- Works well with small datasets
- Handles multiple classes
- Can handle missing data

### Limitations

- Naïve independence assumption (often violated)
- Requires smoothing for unseen features
- Sensitive to irrelevant features

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Conditional Probability:** Always use definition $P(A|B) = P(A \cap B)/P(B)$
2. **Independence:** Check $P(A \cap B) = P(A) \cdot P(B)$ or $P(A|B) = P(A)$
3. **Bayes' Theorem:** Identify prior, likelihood, evidence, and posterior clearly
4. **Law of Total Probability:** Use when $P(B)$ is not directly given
5. **Naïve Bayes:** Remember the independence assumption
6. **Common Mistake:** Confusing $P(A|B)$ with $P(B|A)$
```

---

## 🔍 Worked Examples

### Example 1: Conditional Probability

Two cards drawn without replacement from deck of 52.

Find $P(\text{Second is Ace}|\text{First is Ace})$:

**Solution:**
After first ace drawn, 51 cards remain with 3 aces:

$$P(\text{Second is Ace}|\text{First is Ace}) = \frac{3}{51} = \frac{1}{17}$$

### Example 2: Bayes' Theorem

Three boxes contain:
- Box 1: 2 red, 3 blue
- Box 2: 1 red, 4 blue
- Box 3: 3 red, 2 blue

A box is selected at random and a ball drawn is red. Find probability it came from Box 1.

**Solution:**

Let $B_i$ = event box $i$ is selected
Let $R$ = event red ball is drawn

**Given:**
- $P(B_1) = P(B_2) = P(B_3) = \frac{1}{3}$
- $P(R|B_1) = \frac{2}{5}$, $P(R|B_2) = \frac{1}{5}$, $P(R|B_3) = \frac{3}{5}$

**Using Law of Total Probability:**
$$P(R) = P(R|B_1)P(B_1) + P(R|B_2)P(B_2) + P(R|B_3)P(B_3)$$
$$= \frac{2}{5} \cdot \frac{1}{3} + \frac{1}{5} \cdot \frac{1}{3} + \frac{3}{5} \cdot \frac{1}{3} = \frac{6}{15} = \frac{2}{5}$$

**Using Bayes' Theorem:**
$$P(B_1|R) = \frac{P(R|B_1) \cdot P(B_1)}{P(R)} = \frac{\frac{2}{5} \times \frac{1}{3}}{\frac{2}{5}} = \frac{1}{3}$$

---

## 📚 Quick Revision Checklist

- [ ] Conditional probability definition
- [ ] Multiplication rule
- [ ] Conditional probability for independent events
- [ ] Bayes' theorem statement and components
- [ ] Law of total probability
- [ ] Extended Bayes' theorem
- [ ] Naïve Bayes classifier concept
- [ ] Independence assumption in Naïve Bayes
- [ ] Steps for Naïve Bayes classification
- [ ] Advantages and limitations

