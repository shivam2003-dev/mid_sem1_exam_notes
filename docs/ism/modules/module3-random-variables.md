# Module 3: Random Variables and Distributions

## Overview

Random variables are fundamental in statistics. This module covers discrete and continuous random variables, probability distributions, and their properties.

## 1. Random Variables

### Definition

A **random variable** $X$ is a function that assigns a real number to each outcome in the sample space.

- **Discrete Random Variable:** Takes countable values
- **Continuous Random Variable:** Takes uncountable values (any value in an interval)

## 2. Discrete Random Variables

### Probability Mass Function (PMF)

For a discrete random variable $X$, the **PMF** is:

$$p_X(x) = P(X = x)$$

**Properties:**
- $p_X(x) \geq 0$ for all $x$
- $\sum_x p_X(x) = 1$

### Cumulative Distribution Function (CDF)

$$F_X(x) = P(X \leq x) = \sum_{k \leq x} p_X(k)$$

**Properties:**
- $F_X(x)$ is non-decreasing
- $\lim_{x \to -\infty} F_X(x) = 0$
- $\lim_{x \to \infty} F_X(x) = 1$

## 3. Continuous Random Variables

### Probability Density Function (PDF)

For a continuous random variable $X$, the **PDF** is $f_X(x)$ such that:

$$P(a \leq X \leq b) = \int_a^b f_X(x) dx$$

**Properties:**
- $f_X(x) \geq 0$ for all $x$
- $\int_{-\infty}^{\infty} f_X(x) dx = 1$

### Cumulative Distribution Function (CDF)

$$F_X(x) = P(X \leq x) = \int_{-\infty}^{x} f_X(t) dt$$

**Relationship:**
$$f_X(x) = \frac{d}{dx} F_X(x)$$

## 4. Expectation and Variance

### Expectation (Mean)

**Discrete:**
$$\mathbb{E}[X] = \sum_x x \cdot p_X(x)$$

**Continuous:**
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) dx$$

### Properties of Expectation

- $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$
- $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$
- $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ (if $X$ and $Y$ are independent)

### Variance

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Standard Deviation:**
$$\sigma_X = \sqrt{\text{Var}(X)}$$

### Properties of Variance

- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ (if $X$ and $Y$ are independent)

## 5. Common Discrete Distributions

### Bernoulli Distribution

$X \sim \text{Bernoulli}(p)$

**PMF:**
$$p_X(x) = \begin{cases}
p & \text{if } x = 1 \\
1-p & \text{if } x = 0
\end{cases}$$

**Mean:** $\mathbb{E}[X] = p$

**Variance:** $\text{Var}(X) = p(1-p)$

### Binomial Distribution

$X \sim \text{Binomial}(n, p)$

**PMF:**
$$p_X(k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

**Mean:** $\mathbb{E}[X] = np$

**Variance:** $\text{Var}(X) = np(1-p)$

### Poisson Distribution

$X \sim \text{Poisson}(\lambda)$

**PMF:**
$$p_X(k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

**Mean:** $\mathbb{E}[X] = \lambda$

**Variance:** $\text{Var}(X) = \lambda$

## 6. Common Continuous Distributions

### Uniform Distribution

$X \sim \text{Uniform}(a, b)$

**PDF:**
$$f_X(x) = \begin{cases}
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}$$

**Mean:** $\mathbb{E}[X] = \frac{a+b}{2}$

**Variance:** $\text{Var}(X) = \frac{(b-a)^2}{12}$

### Normal (Gaussian) Distribution

$X \sim \mathcal{N}(\mu, \sigma^2)$

**PDF:**
$$f_X(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**Mean:** $\mathbb{E}[X] = \mu$

**Variance:** $\text{Var}(X) = \sigma^2$

**Standard Normal:** $Z \sim \mathcal{N}(0, 1)$

**Z-score:** $z = \frac{x - \mu}{\sigma}$

### Exponential Distribution

$X \sim \text{Exp}(\lambda)$

**PDF:**
$$f_X(x) = \begin{cases}
\lambda e^{-\lambda x} & \text{if } x \geq 0 \\
0 & \text{otherwise}
\end{cases}$$

**Mean:** $\mathbb{E}[X] = \frac{1}{\lambda}$

**Variance:** $\text{Var}(X) = \frac{1}{\lambda^2}$

## 7. Moment Generating Functions

### Definition

The **moment generating function (MGF)** of $X$ is:

$$M_X(t) = \mathbb{E}[e^{tX}]$$

**Discrete:**
$$M_X(t) = \sum_x e^{tx} p_X(x)$$

**Continuous:**
$$M_X(t) = \int_{-\infty}^{\infty} e^{tx} f_X(x) dx$$

### Properties

- $M_X(0) = 1$
- $\mathbb{E}[X^n] = M_X^{(n)}(0)$ (n-th derivative at 0)
- If $X$ and $Y$ are independent: $M_{X+Y}(t) = M_X(t) M_Y(t)$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **PMF vs PDF:** PMF for discrete, PDF for continuous
2. **Expectation:** Use appropriate formula (sum for discrete, integral for continuous)
3. **Variance:** Use $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$
4. **Common Distributions:** Know mean and variance for each
5. **Normal Distribution:** Use z-scores and standard normal table
```

---

## 🔍 Worked Examples

### Example 1: Binomial Distribution

A coin is tossed 5 times. Find probability of getting exactly 3 heads.

$X \sim \text{Binomial}(5, 0.5)$

$$P(X = 3) = \binom{5}{3} (0.5)^3 (0.5)^2 = 10 \times 0.125 \times 0.25 = 0.3125$$

### Example 2: Normal Distribution

$X \sim \mathcal{N}(100, 25)$. Find $P(95 < X < 110)$.

**Standardize:**
$$P(95 < X < 110) = P\left(\frac{95-100}{5} < Z < \frac{110-100}{5}\right) = P(-1 < Z < 2)$$

Using standard normal table:
$$= \Phi(2) - \Phi(-1) = 0.9772 - 0.1587 = 0.8185$$

---

## 📚 Quick Revision Checklist

- [ ] Discrete vs continuous random variables
- [ ] PMF and PDF definitions
- [ ] CDF for both types
- [ ] Expectation and variance formulas
- [ ] Common discrete distributions (Bernoulli, Binomial, Poisson)
- [ ] Common continuous distributions (Uniform, Normal, Exponential)
- [ ] Properties of each distribution
- [ ] Moment generating functions

