# Module 4: Sampling and Sampling Distributions

## Overview

This module covers sampling methods, the Central Limit Theorem, and sampling distributions - essential for making inferences about populations from samples.

## 1. Sampling Methods

### Simple Random Sampling

Each member of the population has an equal chance of being selected.

### Stratified Sampling

Population divided into strata (groups), then random sampling within each stratum.

### Systematic Sampling

Select every $k$-th element from a list.

### Cluster Sampling

Population divided into clusters, then entire clusters are randomly selected.

## 2. Sampling Distribution of the Mean

### Sample Mean

For a sample $X_1, X_2, \ldots, X_n$:

$$\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$$

### Properties

If $X_1, \ldots, X_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$:

- **Mean:** $\mathbb{E}[\bar{X}] = \mu$
- **Variance:** $\text{Var}(\bar{X}) = \frac{\sigma^2}{n}$
- **Standard Error:** $\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$

## 3. Central Limit Theorem (CLT)

### Statement

If $X_1, X_2, \ldots, X_n$ are i.i.d. random variables with mean $\mu$ and variance $\sigma^2$, then for large $n$:

$$\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

Or equivalently:

$$\bar{X} \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$

### Interpretation

- Sample mean is approximately normally distributed for large $n$
- Works regardless of the original distribution (if variance exists)
- Typically $n \geq 30$ is considered "large enough"

### Applications

- Confidence intervals for population mean
- Hypothesis testing
- Quality control

## 4. Sampling Distribution of Proportion

### Sample Proportion

For $n$ trials with $X$ successes:

$$\hat{p} = \frac{X}{n}$$

### Properties

If $X \sim \text{Binomial}(n, p)$:

- **Mean:** $\mathbb{E}[\hat{p}] = p$
- **Variance:** $\text{Var}(\hat{p}) = \frac{p(1-p)}{n}$
- **Standard Error:** $\sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}$

### Normal Approximation

For large $n$ (typically $np \geq 5$ and $n(1-p) \geq 5$):

$$\hat{p} \sim \mathcal{N}\left(p, \frac{p(1-p)}{n}\right)$$

## 5. t-Distribution

### Definition

If $Z \sim \mathcal{N}(0, 1)$ and $V \sim \chi^2(\nu)$ are independent, then:

$$T = \frac{Z}{\sqrt{V/\nu}} \sim t(\nu)$$

where $\nu$ is degrees of freedom.

### Properties

- Symmetric about 0
- Heavier tails than normal distribution
- As $\nu \to \infty$, $t(\nu) \to \mathcal{N}(0, 1)$
- Used when population variance is unknown

### Application

For sample from normal distribution with unknown variance:

$$\frac{\bar{X} - \mu}{s/\sqrt{n}} \sim t(n-1)$$

where $s$ is sample standard deviation.

## 6. Chi-Square Distribution

### Definition

If $Z_1, \ldots, Z_k$ are i.i.d. $\mathcal{N}(0, 1)$, then:

$$X = \sum_{i=1}^{k} Z_i^2 \sim \chi^2(k)$$

where $k$ is degrees of freedom.

### Properties

- Non-negative
- Skewed to the right
- Mean = $k$, Variance = $2k$

### Application

For sample variance from normal distribution:

$$\frac{(n-1)s^2}{\sigma^2} \sim \chi^2(n-1)$$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **CLT:** Remember it applies to sample mean, not individual observations
2. **Standard Error:** $\sigma/\sqrt{n}$ for mean, $\sqrt{p(1-p)/n}$ for proportion
3. **t-distribution:** Use when variance is unknown and sample size is small
4. **Normal Approximation:** Check conditions ($np \geq 5$ for proportion)
5. **Sample Size:** Larger $n$ gives smaller standard error
```

---

## 🔍 Worked Examples

### Example 1: Sampling Distribution of Mean

Population: $\mu = 50$, $\sigma = 10$. Sample size $n = 25$.

Find $P(\bar{X} > 52)$.

**By CLT:**
$$\bar{X} \sim \mathcal{N}\left(50, \frac{100}{25}\right) = \mathcal{N}(50, 4)$$

$$P(\bar{X} > 52) = P\left(Z > \frac{52-50}{2}\right) = P(Z > 1) = 1 - \Phi(1) = 0.1587$$

### Example 2: Sampling Distribution of Proportion

$p = 0.3$, $n = 100$. Find $P(\hat{p} > 0.35)$.

**Check conditions:** $np = 30 \geq 5$, $n(1-p) = 70 \geq 5$ ✓

$$\hat{p} \sim \mathcal{N}\left(0.3, \frac{0.3 \times 0.7}{100}\right) = \mathcal{N}(0.3, 0.0021)$$

$$P(\hat{p} > 0.35) = P\left(Z > \frac{0.35-0.3}{\sqrt{0.0021}}\right) = P(Z > 1.09) = 0.1379$$

---

## 📚 Quick Revision Checklist

- [ ] Types of sampling methods
- [ ] Sampling distribution of mean
- [ ] Central Limit Theorem statement and conditions
- [ ] Sampling distribution of proportion
- [ ] t-distribution and when to use it
- [ ] Chi-square distribution
- [ ] Standard error formulas
- [ ] Normal approximation conditions

