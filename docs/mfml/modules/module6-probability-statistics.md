# Module 6: Probability and Statistics

## Overview

Probability and statistics form the foundation for understanding uncertainty in machine learning, Bayesian methods, and statistical inference.

## 1. Probability Fundamentals

### Basic Definitions

**Sample Space ($\Omega$):** Set of all possible outcomes

**Event ($E$):** Subset of sample space

**Probability Function $P$:** Maps events to $[0, 1]$ satisfying:
- $P(\Omega) = 1$
- $P(\emptyset) = 0$
- For disjoint events: $P(E_1 \cup E_2) = P(E_1) + P(E_2)$

### Conditional Probability

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

### Bayes' Theorem

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

### Independence

Events $A$ and $B$ are independent if:
$$P(A \cap B) = P(A)P(B)$$

---

## 2. Random Variables

### Discrete Random Variable

Takes countable values. **Probability Mass Function (PMF):**
$$p_X(x) = P(X = x)$$

**Properties:**
- $\sum_x p_X(x) = 1$
- $p_X(x) \geq 0$ for all $x$

### Continuous Random Variable

Takes uncountable values. **Probability Density Function (PDF):**
$$f_X(x) = \frac{d}{dx} F_X(x)$$

where $F_X(x) = P(X \leq x)$ is the **Cumulative Distribution Function (CDF)**.

**Properties:**
- $\int_{-\infty}^{\infty} f_X(x) dx = 1$
- $f_X(x) \geq 0$ for all $x$
- $P(a \leq X \leq b) = \int_a^b f_X(x) dx$

---

## 3. Expectation and Variance

### Expectation (Mean)

**Discrete:**
$$\mathbb{E}[X] = \sum_x x \cdot p_X(x)$$

**Continuous:**
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) dx$$

### Properties

- $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$
- $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$
- $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ (if $X$ and $Y$ are independent)

### Variance

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Standard Deviation:**
$$\sigma_X = \sqrt{\text{Var}(X)}$$

### Properties

- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ (if $X$ and $Y$ are independent)

---

## 4. Common Distributions

### Discrete Distributions

**Bernoulli:**
- $X \sim \text{Bernoulli}(p)$
- $P(X = 1) = p$, $P(X = 0) = 1-p$
- $\mathbb{E}[X] = p$, $\text{Var}(X) = p(1-p)$

**Binomial:**
- $X \sim \text{Binomial}(n, p)$
- $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$
- $\mathbb{E}[X] = np$, $\text{Var}(X) = np(1-p)$

**Poisson:**
- $X \sim \text{Poisson}(\lambda)$
- $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$
- $\mathbb{E}[X] = \lambda$, $\text{Var}(X) = \lambda$

### Continuous Distributions

**Uniform:**
- $X \sim \text{Uniform}(a, b)$
- $f_X(x) = \frac{1}{b-a}$ for $x \in [a, b]$
- $\mathbb{E}[X] = \frac{a+b}{2}$, $\text{Var}(X) = \frac{(b-a)^2}{12}$

**Normal (Gaussian):**
- $X \sim \mathcal{N}(\mu, \sigma^2)$
- $f_X(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- $\mathbb{E}[X] = \mu$, $\text{Var}(X) = \sigma^2$

**Exponential:**
- $X \sim \text{Exp}(\lambda)$
- $f_X(x) = \lambda e^{-\lambda x}$ for $x \geq 0$
- $\mathbb{E}[X] = \frac{1}{\lambda}$, $\text{Var}(X) = \frac{1}{\lambda^2}$

---

## 5. Covariance and Correlation

### Covariance

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

### Properties

- $\text{Cov}(X, X) = \text{Var}(X)$
- $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
- $\text{Cov}(aX + b, cY + d) = ac \text{Cov}(X, Y)$
- If $X$ and $Y$ are independent: $\text{Cov}(X, Y) = 0$

### Correlation

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

**Properties:**
- $-1 \leq \rho_{XY} \leq 1$
- $\rho_{XY} = 1$: perfect positive correlation
- $\rho_{XY} = -1$: perfect negative correlation
- $\rho_{XY} = 0$: uncorrelated (but not necessarily independent)

---

## 6. Maximum Likelihood Estimation (MLE)

### Likelihood Function

For observations $x_1, x_2, \ldots, x_n$ with parameter $\theta$:

**Discrete:**
$$L(\theta) = \prod_{i=1}^{n} p(x_i | \theta)$$

**Continuous:**
$$L(\theta) = \prod_{i=1}^{n} f(x_i | \theta)$$

### Log-Likelihood

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log p(x_i | \theta)$$

### MLE Estimator

$$\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta) = \arg\max_{\theta} \ell(\theta)$$

**Method:** Set $\frac{\partial \ell}{\partial \theta} = 0$ and solve.

### Example: MLE for Normal Distribution

Given $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$, find MLE for $\mu$ and $\sigma^2$.

**Log-likelihood:**
$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**Differentiate with respect to $\mu$:**
$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0$$

$$\Rightarrow \hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

**Differentiate with respect to $\sigma^2$:**
$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{n}(x_i - \mu)^2 = 0$$

$$\Rightarrow \hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

---

## 7. Bayesian Inference

### Prior and Posterior

**Prior:** $P(\theta)$ - belief about $\theta$ before seeing data

**Likelihood:** $P(D|\theta)$ - probability of data given $\theta$

**Posterior:** $P(\theta|D)$ - updated belief after seeing data

### Bayes' Rule

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)} = \frac{P(D|\theta)P(\theta)}{\int P(D|\theta)P(\theta) d\theta}$$

### Conjugate Priors

Prior and posterior belong to the same family:
- **Beta** prior for **Bernoulli/Binomial** likelihood
- **Normal** prior for **Normal** likelihood (with known variance)
- **Gamma** prior for **Poisson/Exponential** likelihood

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Conditional Probability:** Always use the definition $P(A|B) = P(A \cap B)/P(B)$
2. **MLE:** Take log-likelihood, differentiate, set to zero
3. **Covariance:** Remember $\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$
4. **Common Distributions:** Know mean and variance for each
5. **Bayes' Theorem:** Identify prior, likelihood, and posterior clearly
```

---

## 🔍 Worked Examples

### Example 1: Covariance

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
$$\text{Cov}(X, Y) = 0.2 - 0.5 \cdot 0.5 = -0.05$$

### Example 2: MLE

Find MLE for parameter $\lambda$ of exponential distribution given $x_1, \ldots, x_n$.

**Likelihood:**
$$L(\lambda) = \prod_{i=1}^{n} \lambda e^{-\lambda x_i} = \lambda^n e^{-\lambda \sum x_i}$$

**Log-likelihood:**
$$\ell(\lambda) = n\log\lambda - \lambda\sum_{i=1}^{n} x_i$$

**Differentiate:**
$$\frac{\partial \ell}{\partial \lambda} = \frac{n}{\lambda} - \sum_{i=1}^{n} x_i = 0$$

$$\Rightarrow \hat{\lambda}_{MLE} = \frac{n}{\sum_{i=1}^{n} x_i} = \frac{1}{\bar{x}}$$

---

## 📚 Quick Revision Checklist

- [ ] Basic probability rules and conditional probability
- [ ] Bayes' theorem
- [ ] Expectation and variance properties
- [ ] Common distributions (PMF/PDF, mean, variance)
- [ ] Covariance and correlation
- [ ] Maximum likelihood estimation
- [ ] Bayesian inference basics

