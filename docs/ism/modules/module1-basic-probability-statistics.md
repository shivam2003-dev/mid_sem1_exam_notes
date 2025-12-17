# Module 1: Basic Probability & Statistics

## Overview

This module introduces fundamental concepts in probability and statistics, including measures of central tendency, variability, and basic probability theory.

## 1.1 Measures of Central Tendency

### Mean (Arithmetic Average)

**Population Mean:**
$$\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$$

**Sample Mean:**
$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i = \frac{x_1 + x_2 + \cdots + x_n}{n}$$

**Properties:**
- Uses all data values
- Sensitive to outliers
- Sum of deviations from mean is zero: $\sum(x_i - \bar{x}) = 0$

**Example:**
Data: 5, 7, 8, 9, 10, 12, 15

$$\bar{x} = \frac{5 + 7 + 8 + 9 + 10 + 12 + 15}{7} = \frac{66}{7} = 9.43$$

### Median

The middle value when data is arranged in ascending order.

**For odd number of observations ($n$ odd):**
$$\text{Median} = x_{\frac{n+1}{2}}$$

**For even number of observations ($n$ even):**
$$\text{Median} = \frac{x_{\frac{n}{2}} + x_{\frac{n}{2}+1}}{2}$$

**Properties:**
- Not affected by outliers
- Divides data into two equal halves
- 50% of observations are below median, 50% above

**Example:**
Data: 5, 7, 8, 9, 10, 12, 15

Since $n = 7$ (odd), median = $x_4 = 9$

### Mode

The value that occurs most frequently in the data.

**Properties:**
- Can have multiple modes (bimodal, multimodal)
- Useful for categorical data
- Not always unique
- May not exist (if all values are unique)

**Example:**
Data: 2, 3, 3, 4, 5, 5, 5, 6

Mode = 5 (appears 3 times)

### Relationship Between Measures

- **Symmetric distribution:** Mean ≈ Median ≈ Mode
- **Right-skewed (positive skew):** Mean > Median > Mode
- **Left-skewed (negative skew):** Mean < Median < Mode

---

## 1.2 Measures of Variability

### Range

$$\text{Range} = \text{Maximum} - \text{Minimum}$$

**Limitations:**
- Only uses two values
- Ignores the rest of the data
- Highly sensitive to outliers

**Example:**
Data: 5, 7, 8, 9, 10, 12, 15

Range = 15 - 5 = 10

### Variance

**Population Variance:**
$$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2$$

**Sample Variance:**
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**Note:** Sample variance uses $(n-1)$ (Bessel's correction) for unbiased estimation.

**Alternative Formula:**
$$s^2 = \frac{1}{n-1}\left[\sum_{i=1}^{n} x_i^2 - n\bar{x}^2\right]$$

### Standard Deviation

**Population:**
$$\sigma = \sqrt{\sigma^2}$$

**Sample:**
$$s = \sqrt{s^2}$$

**Properties:**
- Same units as the data
- Measures spread around the mean
- Larger values indicate more variability
- Always non-negative

**Example:**
Data: 2, 4, 6, 8, 10

Mean: $\bar{x} = 6$

Variance:
$$s^2 = \frac{1}{4}[(2-6)^2 + (4-6)^2 + (6-6)^2 + (8-6)^2 + (10-6)^2]$$
$$= \frac{1}{4}[16 + 4 + 0 + 4 + 16] = \frac{40}{4} = 10$$

Standard Deviation: $s = \sqrt{10} = 3.16$

### Coefficient of Variation (CV)

$$CV = \frac{s}{\bar{x}} \times 100\%$$

**Use:** Compare variability across different units/scales

**Interpretation:**
- Lower CV = less variability relative to mean
- Higher CV = more variability relative to mean

### Interquartile Range (IQR)

$$IQR = Q_3 - Q_1$$

where:
- $Q_1$ = First quartile (25th percentile)
- $Q_3$ = Third quartile (75th percentile)

**Properties:**
- Not affected by outliers
- Measures spread of middle 50% of data
- Used in box plots

**Outlier Detection:**
- Lower fence: $Q_1 - 1.5 \times IQR$
- Upper fence: $Q_3 + 1.5 \times IQR$
- Values outside fences are considered outliers

---

## 1.3 Basic Probability Concepts

### 1.3.1 Axioms of Probability

**Axiom 1 (Non-negativity):**
For any event $E$:
$$P(E) \geq 0$$

**Axiom 2 (Normalization):**
For the sample space $\Omega$:
$$P(\Omega) = 1$$

**Axiom 3 (Additivity):**
For mutually exclusive events $E_1, E_2, \ldots$:
$$P(E_1 \cup E_2 \cup \cdots) = P(E_1) + P(E_2) + \cdots$$

### Consequences

From these axioms, we can derive:

**Complement Rule:**
$$P(E^c) = 1 - P(E)$$

**Union Rule:**
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**For mutually exclusive events:**
If $A \cap B = \emptyset$, then:
$$P(A \cup B) = P(A) + P(B)$$

### 1.3.2 Definition of Probability

### Classical Definition

If all outcomes in sample space $\Omega$ are equally likely:

$$P(E) = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}} = \frac{|E|}{|\Omega|}$$

**Example:**
Rolling a fair die:
- Sample space: $\Omega = \{1, 2, 3, 4, 5, 6\}$
- $P(\text{even number}) = P(\{2, 4, 6\}) = \frac{3}{6} = \frac{1}{2}$

### Relative Frequency Definition

$$P(E) = \lim_{n \to \infty} \frac{\text{Number of times E occurs}}{n}$$

where $n$ is the number of trials.

### Subjective Definition

Probability as a measure of belief or confidence in an event occurring.

### 1.3.3 Mutually Exclusive and Independent Events

### Mutually Exclusive Events

Events $A$ and $B$ are **mutually exclusive** (disjoint) if:

$$A \cap B = \emptyset$$

**Properties:**
- Cannot occur simultaneously
- $P(A \cap B) = 0$
- $P(A \cup B) = P(A) + P(B)$

**Example:**
Rolling a die:
- $A$ = getting an even number = $\{2, 4, 6\}$
- $B$ = getting an odd number = $\{1, 3, 5\}$
- $A \cap B = \emptyset$ → Mutually exclusive

### Independent Events

Events $A$ and $B$ are **independent** if:

$$P(A \cap B) = P(A) \cdot P(B)$$

**Equivalently:**
- $P(A|B) = P(A)$
- $P(B|A) = P(B)$

**Properties:**
- Can occur simultaneously
- Occurrence of one doesn't affect probability of the other
- If $A$ and $B$ are independent, then:
  - $A$ and $B^c$ are independent
  - $A^c$ and $B$ are independent
  - $A^c$ and $B^c$ are independent

**Example:**
Tossing a coin twice:
- $A$ = first toss is heads
- $B$ = second toss is heads
- $P(A) = \frac{1}{2}$, $P(B) = \frac{1}{2}$
- $P(A \cap B) = P(\text{both heads}) = \frac{1}{4} = P(A) \cdot P(B)$

Therefore, $A$ and $B$ are independent.

### Difference

| Property | Mutually Exclusive | Independent |
|----------|-------------------|-------------|
| **Can occur together?** | No | Yes |
| **Condition** | $A \cap B = \emptyset$ | $P(A \cap B) = P(A)P(B)$ |
| **Probability** | $P(A \cap B) = 0$ | $P(A \cap B) = P(A)P(B)$ |
| **Example** | Heads and tails on same toss | Heads on toss 1 and heads on toss 2 |

**Important:** Mutually exclusive events are **NOT** independent (unless one has probability 0)!

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Mean vs Median:** Use median when data has outliers
2. **Sample Variance:** Always use $(n-1)$ in denominator
3. **Mutually Exclusive vs Independent:** Understand the difference clearly
4. **Axioms:** Know all three axioms and their consequences
5. **IQR:** Better than range for skewed data
6. **Common Mistake:** Confusing mutually exclusive with independent
```

---

## 🔍 Worked Examples

### Example 1: Measures of Central Tendency and Variability

Given data: 10, 12, 14, 16, 18, 20, 22

**Mean:**
$$\bar{x} = \frac{10 + 12 + 14 + 16 + 18 + 20 + 22}{7} = \frac{112}{7} = 16$$

**Median:**
Since $n = 7$ (odd), median = $x_4 = 16$

**Mode:**
No mode (all values unique)

**Variance:**
$$s^2 = \frac{1}{6}[(10-16)^2 + (12-16)^2 + (14-16)^2 + (16-16)^2 + (18-16)^2 + (20-16)^2 + (22-16)^2]$$
$$= \frac{1}{6}[36 + 16 + 4 + 0 + 4 + 16 + 36] = \frac{112}{6} = 18.67$$

**Standard Deviation:**
$$s = \sqrt{18.67} = 4.32$$

**Range:**
Range = 22 - 10 = 12

### Example 2: Mutually Exclusive vs Independent

**Scenario 1:** Draw one card from deck
- $A$ = card is a heart
- $B$ = card is a spade

$A \cap B = \emptyset$ → **Mutually exclusive**

$P(A \cap B) = 0 \neq P(A) \cdot P(B) = \frac{1}{4} \cdot \frac{1}{4} = \frac{1}{16}$ → **Not independent**

**Scenario 2:** Toss coin twice
- $A$ = first toss is heads
- $B$ = second toss is heads

$A \cap B \neq \emptyset$ → **Not mutually exclusive**

$P(A \cap B) = \frac{1}{4} = P(A) \cdot P(B) = \frac{1}{2} \cdot \frac{1}{2}$ → **Independent**

---

## 📚 Quick Revision Checklist

- [ ] Mean, median, mode calculation and interpretation
- [ ] Variance and standard deviation formulas
- [ ] Coefficient of variation
- [ ] Interquartile range and outlier detection
- [ ] Three axioms of probability
- [ ] Classical, relative frequency, and subjective definitions
- [ ] Mutually exclusive events (definition and properties)
- [ ] Independent events (definition and properties)
- [ ] Difference between mutually exclusive and independent
- [ ] When to use mean vs median

