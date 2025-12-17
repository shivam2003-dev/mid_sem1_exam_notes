# Module 2: Probability Fundamentals

## Overview

Probability is the foundation of statistical inference. This module covers basic probability concepts, conditional probability, and Bayes' theorem.

## 1. Basic Probability Concepts

### Sample Space

The **sample space** ($\Omega$ or $S$) is the set of all possible outcomes of an experiment.

**Example:** Rolling a die
$$\Omega = \{1, 2, 3, 4, 5, 6\}$$

### Event

An **event** ($E$) is a subset of the sample space.

**Example:** Event "even number" = $\{2, 4, 6\}$

### Probability Axioms

For any event $E$:

1. **Non-negativity:** $P(E) \geq 0$
2. **Normalization:** $P(\Omega) = 1$
3. **Additivity:** For mutually exclusive events $E_1, E_2, \ldots$:
   $$P(E_1 \cup E_2 \cup \cdots) = P(E_1) + P(E_2) + \cdots$$

### Basic Rules

**Complement Rule:**
$$P(E^c) = 1 - P(E)$$

**Union Rule:**
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Mutually Exclusive Events:**
If $A \cap B = \emptyset$, then:
$$P(A \cup B) = P(A) + P(B)$$

---

## 2. Conditional Probability

### Definition

The **conditional probability** of event $A$ given event $B$ is:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

provided $P(B) > 0$.

### Interpretation

- Probability of $A$ occurring given that $B$ has occurred
- "Restricted" sample space to outcomes in $B$

### Multiplication Rule

$$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

### Example

In a class of 30 students: 18 are girls, 12 are boys. 10 girls and 6 boys wear glasses.

Find $P(\text{Girl}|\text{Wears Glasses})$:

**Solution:**
- $P(\text{Wears Glasses}) = \frac{16}{30} = \frac{8}{15}$
- $P(\text{Girl} \cap \text{Wears Glasses}) = \frac{10}{30} = \frac{1}{3}$

$$P(\text{Girl}|\text{Wears Glasses}) = \frac{1/3}{8/15} = \frac{5}{8}$$

---

## 3. Independence

### Definition

Events $A$ and $B$ are **independent** if:

$$P(A \cap B) = P(A) \cdot P(B)$$

Equivalently:
$$P(A|B) = P(A) \quad \text{or} \quad P(B|A) = P(B)$$

### Properties

- Independence is symmetric
- If $A$ and $B$ are independent, then:
  - $A$ and $B^c$ are independent
  - $A^c$ and $B$ are independent
  - $A^c$ and $B^c$ are independent

### Example

Tossing a fair coin twice:
- $A$ = First toss is heads
- $B$ = Second toss is heads

$P(A) = \frac{1}{2}$, $P(B) = \frac{1}{2}$, $P(A \cap B) = \frac{1}{4}$

Since $P(A \cap B) = P(A) \cdot P(B)$, events are independent.

---

## 4. Bayes' Theorem

### Statement

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

### Law of Total Probability

If $B_1, B_2, \ldots, B_n$ partition the sample space:

$$P(A) = \sum_{i=1}^{n} P(A|B_i) \cdot P(B_i)$$

### Extended Bayes' Theorem

$$P(A_i|B) = \frac{P(B|A_i) \cdot P(A_i)}{\sum_{j=1}^{n} P(B|A_j) \cdot P(A_j)}$$

### Example: Medical Test

A disease affects 1% of population. Test is 95% accurate (95% true positive, 95% true negative).

Find $P(\text{Disease}|\text{Test Positive})$:

**Given:**
- $P(\text{Disease}) = 0.01$
- $P(\text{Test +}|\text{Disease}) = 0.95$
- $P(\text{Test +}|\text{No Disease}) = 0.05$

**Solution:**

$$P(\text{Disease}|\text{Test +}) = \frac{P(\text{Test +}|\text{Disease}) \cdot P(\text{Disease})}{P(\text{Test +})}$$

$$P(\text{Test +}) = 0.95 \times 0.01 + 0.05 \times 0.99 = 0.059$$

$$P(\text{Disease}|\text{Test +}) = \frac{0.95 \times 0.01}{0.059} = \frac{0.0095}{0.059} \approx 0.161$$

Only 16.1% chance of having disease despite positive test!

---

## 5. Permutations and Combinations

### Permutations

Number of ways to arrange $r$ objects from $n$:

$$P(n, r) = \frac{n!}{(n-r)!}$$

**Example:** Ways to arrange 3 books from 5:
$$P(5, 3) = \frac{5!}{2!} = 60$$

### Combinations

Number of ways to choose $r$ objects from $n$:

$$C(n, r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}$$

**Example:** Ways to choose 3 books from 5:
$$C(5, 3) = \frac{5!}{3!2!} = 10$$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Conditional Probability:** Always use definition $P(A|B) = P(A \cap B)/P(B)$
2. **Independence:** Check $P(A \cap B) = P(A) \cdot P(B)$
3. **Bayes' Theorem:** Identify prior $P(A)$, likelihood $P(B|A)$, and posterior $P(A|B)$
4. **Common Mistake:** Confusing $P(A|B)$ with $P(B|A)$
5. **Practice:** Use https://seeing-theory.brown.edu/ for conditional probability intuition
```

---

## 🔍 Worked Examples

### Example 1: Conditional Probability

Two cards drawn without replacement from deck of 52.

Find $P(\text{Second is Ace}|\text{First is Ace})$:

**Solution:**
After first ace drawn, 51 cards remain with 3 aces:

$$P(\text{Second is Ace}|\text{First is Ace}) = \frac{3}{51} = \frac{1}{17}$$

### Example 2: Independence

$P(A) = 0.4$, $P(B) = 0.5$, $P(A \cap B) = 0.2$

Are $A$ and $B$ independent?

**Check:** $P(A) \cdot P(B) = 0.4 \times 0.5 = 0.2 = P(A \cap B)$

Yes, they are independent.

---

## 📚 Quick Revision Checklist

- [ ] Sample space and events
- [ ] Probability axioms
- [ ] Conditional probability definition
- [ ] Independence definition and properties
- [ ] Bayes' theorem
- [ ] Law of total probability
- [ ] Permutations vs combinations

