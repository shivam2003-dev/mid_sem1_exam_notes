# Module 3: Probability Distributions

## Overview

This module covers random variables (discrete and continuous), their properties, transformations, and important probability distributions used in statistics and machine learning.

## 3.1 Random Variables

### Definition

A **random variable** $X$ is a function that assigns a real number to each outcome in the sample space.

- **Discrete Random Variable:** Takes countable values (finite or countably infinite)
- **Continuous Random Variable:** Takes uncountable values (any value in an interval)

### 3.1.1 Discrete Random Variable – Single Variable

### Probability Mass Function (PMF)

For discrete $X$, the **PMF** is:

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
- $P(a < X \leq b) = F_X(b) - F_X(a)$

### Expectation (Mean)

$$\mathbb{E}[X] = \sum_x x \cdot p_X(x)$$

### Variance

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

where $\mathbb{E}[X^2] = \sum_x x^2 \cdot p_X(x)$

### Example

$X$ has PMF:
$$p_X(0) = 0.2, \quad p_X(1) = 0.5, \quad p_X(2) = 0.3$$

**Mean:**
$$\mathbb{E}[X] = 0(0.2) + 1(0.5) + 2(0.3) = 0 + 0.5 + 0.6 = 1.1$$

**Variance:**
$$\mathbb{E}[X^2] = 0^2(0.2) + 1^2(0.5) + 2^2(0.3) = 0 + 0.5 + 1.2 = 1.7$$
$$\text{Var}(X) = 1.7 - (1.1)^2 = 1.7 - 1.21 = 0.49$$

### 3.1.2 Continuous Random Variable – Single Variable

### Probability Density Function (PDF)

For continuous $X$, the **PDF** is $f_X(x)$ such that:

$$P(a \leq X \leq b) = \int_a^b f_X(x) dx$$

**Properties:**
- $f_X(x) \geq 0$ for all $x$
- $\int_{-\infty}^{\infty} f_X(x) dx = 1$

### Cumulative Distribution Function (CDF)

$$F_X(x) = P(X \leq x) = \int_{-\infty}^{x} f_X(t) dt$$

**Relationship:**
$$f_X(x) = \frac{d}{dx} F_X(x)$$

### Expectation (Mean)

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) dx$$

### Variance

$$\text{Var}(X) = \int_{-\infty}^{\infty} (x - \mathbb{E}[X])^2 f_X(x) dx = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

where $\mathbb{E}[X^2] = \int_{-\infty}^{\infty} x^2 f_X(x) dx$

### Example

$X$ has PDF: $f_X(x) = \begin{cases} 2x & \text{if } 0 \leq x \leq 1 \\ 0 & \text{otherwise} \end{cases}$

**Mean:**
$$\mathbb{E}[X] = \int_0^1 x \cdot 2x dx = \int_0^1 2x^2 dx = \frac{2}{3}x^3\Big|_0^1 = \frac{2}{3}$$

**Variance:**
$$\mathbb{E}[X^2] = \int_0^1 x^2 \cdot 2x dx = \int_0^1 2x^3 dx = \frac{1}{2}x^4\Big|_0^1 = \frac{1}{2}$$
$$\text{Var}(X) = \frac{1}{2} - \left(\frac{2}{3}\right)^2 = \frac{1}{2} - \frac{4}{9} = \frac{9-8}{18} = \frac{1}{18}$$

### 3.1.3 Mean, Variance, Co-Variance of Random Variables

### Mean (Expectation)

**Properties:**
- $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$
- $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$
- $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ (if $X$ and $Y$ are independent)

### Variance

**Properties:**
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ (if $X$ and $Y$ are independent)

### Covariance

For two random variables $X$ and $Y$:

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

**Properties:**
- $\text{Cov}(X, X) = \text{Var}(X)$
- $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
- $\text{Cov}(aX + b, cY + d) = ac \text{Cov}(X, Y)$
- If $X$ and $Y$ are independent: $\text{Cov}(X, Y) = 0$

### Correlation Coefficient

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

**Properties:**
- $-1 \leq \rho_{XY} \leq 1$
- $\rho_{XY} = 1$: Perfect positive linear relationship
- $\rho_{XY} = -1$: Perfect negative linear relationship
- $\rho_{XY} = 0$: Uncorrelated (but not necessarily independent)

### 3.1.4 Transformation of Random Variables

### Discrete Case

If $Y = g(X)$ where $X$ is discrete:

$$p_Y(y) = \sum_{x: g(x) = y} p_X(x)$$

### Continuous Case

If $Y = g(X)$ where $X$ is continuous and $g$ is monotonic:

**Method 1: CDF Method**
1. Find $F_Y(y) = P(Y \leq y) = P(g(X) \leq y)$
2. Differentiate to get $f_Y(y) = \frac{d}{dy} F_Y(y)$

**Method 2: Change of Variables**

If $Y = g(X)$ and $g$ is strictly monotonic:

$$f_Y(y) = f_X(g^{-1}(y)) \left|\frac{d}{dy} g^{-1}(y)\right|$$

### Example: Linear Transformation

If $Y = aX + b$ where $X$ has PDF $f_X(x)$:

$$f_Y(y) = \frac{1}{|a|} f_X\left(\frac{y-b}{a}\right)$$

**Mean and Variance:**
- $\mathbb{E}[Y] = a\mathbb{E}[X] + b$
- $\text{Var}(Y) = a^2 \text{Var}(X)$

---

## 3.2 Probability Distributions

### 3.2.1 Bernoulli Distribution

**Notation:** $X \sim \text{Bernoulli}(p)$

**PMF:**
$$p_X(x) = \begin{cases}
p & \text{if } x = 1 \\
1-p & \text{if } x = 0
\end{cases} = p^x(1-p)^{1-x}$$

**Mean:** $\mathbb{E}[X] = p$

**Variance:** $\text{Var}(X) = p(1-p)$

**Use:** Single trial with two outcomes (success/failure)

### 3.2.2 Binomial Distribution

**Notation:** $X \sim \text{Binomial}(n, p)$

**PMF:**
$$p_X(k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$

**Mean:** $\mathbb{E}[X] = np$

**Variance:** $\text{Var}(X) = np(1-p)$

**Use:** $n$ independent Bernoulli trials

**Example:**
Toss coin 10 times. Probability of exactly 3 heads:

$$P(X = 3) = \binom{10}{3} (0.5)^3 (0.5)^7 = 120 \times 0.125 \times 0.0078125 = 0.117$$

### 3.2.3 Poisson Distribution

**Notation:** $X \sim \text{Poisson}(\lambda)$

**PMF:**
$$p_X(k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

**Mean:** $\mathbb{E}[X] = \lambda$

**Variance:** $\text{Var}(X) = \lambda$

**Use:** 
- Number of events in fixed interval
- Rare events
- Approximation to Binomial when $n$ is large and $p$ is small

**Poisson Approximation to Binomial:**
If $X \sim \text{Binomial}(n, p)$ with $n$ large and $p$ small, then $X \approx \text{Poisson}(np)$

**Example:**
Average 2 calls per hour. Probability of exactly 3 calls in an hour:

$$P(X = 3) = \frac{2^3 e^{-2}}{3!} = \frac{8 e^{-2}}{6} = \frac{4}{3e^2} \approx 0.180$$

### 3.2.4 Normal (Gaussian) Distribution

**Notation:** $X \sim \mathcal{N}(\mu, \sigma^2)$

**PDF:**
$$f_X(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**Mean:** $\mathbb{E}[X] = \mu$

**Variance:** $\text{Var}(X) = \sigma^2$

**Properties:**
- Symmetric about mean
- Bell-shaped curve
- 68-95-99.7 rule:
  - 68% within 1 standard deviation
  - 95% within 2 standard deviations
  - 99.7% within 3 standard deviations

### Standard Normal Distribution

$Z \sim \mathcal{N}(0, 1)$ with PDF:

$$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}$$

**Z-score:**
$$z = \frac{x - \mu}{\sigma}$$

**Standardization:**
If $X \sim \mathcal{N}(\mu, \sigma^2)$, then $Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$

**Example:**
$X \sim \mathcal{N}(100, 25)$. Find $P(95 < X < 110)$.

**Standardize:**
$$P(95 < X < 110) = P\left(\frac{95-100}{5} < Z < \frac{110-100}{5}\right) = P(-1 < Z < 2)$$

$$= \Phi(2) - \Phi(-1) = 0.9772 - 0.1587 = 0.8185$$

### 3.2.5 Introduction to t-Distribution, F-Distribution, Chi-Square Distribution

### t-Distribution

**Notation:** $T \sim t(\nu)$ where $\nu$ is degrees of freedom

**Definition:**
If $Z \sim \mathcal{N}(0, 1)$ and $V \sim \chi^2(\nu)$ are independent:

$$T = \frac{Z}{\sqrt{V/\nu}} \sim t(\nu)$$

**Properties:**
- Symmetric about 0
- Heavier tails than normal
- As $\nu \to \infty$, $t(\nu) \to \mathcal{N}(0, 1)$
- Used when population variance is unknown

**Application:**
For sample from normal distribution with unknown variance:

$$\frac{\bar{X} - \mu}{s/\sqrt{n}} \sim t(n-1)$$

### Chi-Square Distribution

**Notation:** $X \sim \chi^2(\nu)$ where $\nu$ is degrees of freedom

**Definition:**
If $Z_1, \ldots, Z_\nu$ are i.i.d. $\mathcal{N}(0, 1)$:

$$X = \sum_{i=1}^{\nu} Z_i^2 \sim \chi^2(\nu)$$

**Properties:**
- Non-negative
- Skewed to the right
- Mean = $\nu$
- Variance = $2\nu$

**Application:**
For sample variance from normal distribution:

$$\frac{(n-1)s^2}{\sigma^2} \sim \chi^2(n-1)$$

### F-Distribution

**Notation:** $F \sim F(\nu_1, \nu_2)$ where $\nu_1, \nu_2$ are degrees of freedom

**Definition:**
If $U \sim \chi^2(\nu_1)$ and $V \sim \chi^2(\nu_2)$ are independent:

$$F = \frac{U/\nu_1}{V/\nu_2} \sim F(\nu_1, \nu_2)$$

**Properties:**
- Non-negative
- Skewed to the right
- Used in ANOVA and regression

**Application:**
For comparing two variances:

$$F = \frac{s_1^2/\sigma_1^2}{s_2^2/\sigma_2^2} \sim F(n_1-1, n_2-1)$$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **PMF vs PDF:** PMF for discrete, PDF for continuous
2. **Expectation:** Use sum for discrete, integral for continuous
3. **Variance:** Use $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$
4. **Covariance:** $\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$
5. **Normal Distribution:** Always standardize to use standard normal table
6. **t-Distribution:** Use when variance is unknown and sample size is small
7. **Common Distributions:** Know mean and variance for each
```

---

## 🔍 Worked Examples

### Example 1: Binomial Distribution

A test has 10 multiple-choice questions, each with 4 options. Student guesses randomly. Find probability of getting exactly 6 correct.

**Solution:**
$X \sim \text{Binomial}(10, 0.25)$

$$P(X = 6) = \binom{10}{6} (0.25)^6 (0.75)^4 = 210 \times 0.000244 \times 0.3164 = 0.0162$$

### Example 2: Normal Distribution

Scores are normally distributed with mean 75 and standard deviation 10. Find the 90th percentile.

**Solution:**
We need $x$ such that $P(X \leq x) = 0.90$

$$P\left(Z \leq \frac{x-75}{10}\right) = 0.90$$

From standard normal table: $z_{0.90} = 1.28$

$$\frac{x-75}{10} = 1.28 \Rightarrow x = 75 + 12.8 = 87.8$$

### Example 3: Covariance

Given joint PMF:
$$P(X=0, Y=0) = 0.2, \quad P(X=0, Y=1) = 0.3$$
$$P(X=1, Y=0) = 0.3, \quad P(X=1, Y=1) = 0.2$$

Find $\text{Cov}(X, Y)$.

**Marginal PMFs:**
- $P(X=0) = 0.5$, $P(X=1) = 0.5$
- $P(Y=0) = 0.5$, $P(Y=1) = 0.5$

**Expectations:**
- $\mathbb{E}[X] = 0.5$, $\mathbb{E}[Y] = 0.5$
- $\mathbb{E}[XY] = 0 \cdot 0 \cdot 0.2 + 0 \cdot 1 \cdot 0.3 + 1 \cdot 0 \cdot 0.3 + 1 \cdot 1 \cdot 0.2 = 0.2$

**Covariance:**
$$\text{Cov}(X, Y) = 0.2 - 0.5 \times 0.5 = 0.2 - 0.25 = -0.05$$

---

## 📚 Quick Revision Checklist

- [ ] Discrete vs continuous random variables
- [ ] PMF and PDF definitions
- [ ] CDF for both types
- [ ] Expectation and variance formulas
- [ ] Covariance and correlation
- [ ] Transformation of random variables
- [ ] Bernoulli, Binomial, Poisson distributions
- [ ] Normal distribution and standardization
- [ ] t-distribution, chi-square, F-distribution
- [ ] Properties and applications of each distribution

