# Module 4: Hypothesis Testing

## Overview

This module covers sampling methods, sampling distributions, estimation, hypothesis testing (including ANOVA), and maximum likelihood estimation.

## 4.1 Sampling – Random Sampling and Stratified Sampling

### Random Sampling

**Simple Random Sampling:**
- Each member of population has equal chance of being selected
- Every possible sample of size $n$ has equal probability
- No bias in selection

**Methods:**
- **With Replacement:** Each selected unit is returned to population
- **Without Replacement:** Selected units are not returned

### Stratified Sampling

**Definition:**
Population divided into **strata** (homogeneous groups), then random sampling within each stratum.

**Steps:**
1. Divide population into strata based on relevant characteristics
2. Select random sample from each stratum
3. Combine samples from all strata

**Advantages:**
- Ensures representation from all groups
- Can reduce sampling error
- Allows different sampling rates per stratum

**Proportional Allocation:**
Sample size from stratum $i$:
$$n_i = n \times \frac{N_i}{N}$$

where $N_i$ = size of stratum $i$, $N$ = total population size.

**Example:**
Population: 1000 students (600 male, 400 female)
Sample size: 100

**Proportional allocation:**
- Males: $n_1 = 100 \times \frac{600}{1000} = 60$
- Females: $n_2 = 100 \times \frac{400}{1000} = 40$

---

## 4.2 Sampling Distribution – Central Limit Theorem

### Sampling Distribution

The **sampling distribution** of a statistic is the probability distribution of that statistic over all possible samples.

### Sampling Distribution of the Mean

For sample $X_1, X_2, \ldots, X_n$:

$$\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$$

**Properties:**
If $X_1, \ldots, X_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$:

- **Mean:** $\mathbb{E}[\bar{X}] = \mu$
- **Variance:** $\text{Var}(\bar{X}) = \frac{\sigma^2}{n}$
- **Standard Error:** $\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$

### Central Limit Theorem (CLT)

**Statement:**

If $X_1, X_2, \ldots, X_n$ are i.i.d. random variables with mean $\mu$ and variance $\sigma^2$, then for large $n$:

$$\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

Or equivalently:

$$\bar{X} \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$

**Key Points:**
- Works regardless of original distribution (if variance exists)
- Typically $n \geq 30$ is considered "large enough"
- Sample mean is approximately normally distributed

**Applications:**
- Confidence intervals
- Hypothesis testing
- Quality control

**Example:**
Population: $\mu = 100$, $\sigma = 15$, $n = 36$

By CLT:
$$\bar{X} \sim \mathcal{N}\left(100, \frac{225}{36}\right) = \mathcal{N}(100, 6.25)$$

So $\bar{X} \sim \mathcal{N}(100, 2.5^2)$

---

## 4.3 Estimation – Interval Estimation, Confidence Level

### Point Estimation

A **point estimator** provides a single value estimate of a parameter.

**Examples:**
- $\bar{X}$ estimates $\mu$
- $s^2$ estimates $\sigma^2$
- $\hat{p}$ estimates $p$

### Interval Estimation

An **interval estimator** provides a range of values that likely contains the parameter.

### Confidence Interval

A **100(1-α)% confidence interval** for parameter $\theta$ is an interval $[L, U]$ such that:

$$P(L \leq \theta \leq U) = 1 - \alpha$$

**Interpretation:**
If we repeated the procedure many times, 100(1-α)% of intervals would contain the true parameter.

### Confidence Interval for Mean

**Known Variance:**
$$\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

**Unknown Variance (Large Sample):**
$$\bar{x} \pm z_{\alpha/2} \frac{s}{\sqrt{n}}$$

**Unknown Variance (Small Sample):**
$$\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}$$

### Confidence Interval for Proportion

**Large Sample:**
$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

**Conditions:** $n\hat{p} \geq 5$ and $n(1-\hat{p}) \geq 5$

### Common Confidence Levels

- **90%:** $\alpha = 0.10$, $z_{0.05} = 1.645$
- **95%:** $\alpha = 0.05$, $z_{0.025} = 1.96$ (most common)
- **99%:** $\alpha = 0.01$, $z_{0.005} = 2.576$

---

## 4.4 Testing of Hypothesis

### Hypothesis Testing Framework

- **Null Hypothesis ($H_0$):** Statement about parameter (usually equality)
- **Alternative Hypothesis ($H_1$):** Statement we want to prove
- **Significance Level ($\alpha$):** Probability of Type I error

### Type I and Type II Errors

- **Type I Error:** Reject $H_0$ when it's true ($P = \alpha$)
- **Type II Error:** Fail to reject $H_0$ when it's false ($P = \beta$)
- **Power:** $1 - \beta$ = Probability of rejecting $H_0$ when it's false

### 4.4.1 Mean Based

### Z-test for Mean (Known Variance)

**Test Statistic:**
$$z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$$

**Decision:** Reject $H_0$ if $|z| > z_{\alpha/2}$ (two-tailed) or $z > z_{\alpha}$ (one-tailed)

### t-test for Mean (Unknown Variance)

**Test Statistic:**
$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$$

**Distribution:** $t(n-1)$

**Decision:** Reject $H_0$ if $|t| > t_{\alpha/2, n-1}$

**Example:**
Test $H_0: \mu = 100$ vs $H_1: \mu \neq 100$ at $\alpha = 0.05$

Sample: $n = 25$, $\bar{x} = 105$, $s = 15$

**Test Statistic:**
$$t = \frac{105 - 100}{15/\sqrt{25}} = \frac{5}{3} = 1.67$$

**Critical Value:** $t_{0.025, 24} = 2.064$

**Decision:** Since $|t| = 1.67 < 2.064$, fail to reject $H_0$

### 4.4.2 Proportions Related

### Z-test for Proportion

**Test Statistic:**
$$z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}$$

**Conditions:** $np_0 \geq 5$ and $n(1-p_0) \geq 5$

**Decision:** Reject $H_0$ if $|z| > z_{\alpha/2}$

**Example:**
Test $H_0: p = 0.5$ vs $H_1: p \neq 0.5$ at $\alpha = 0.05$

Sample: $n = 400$, $\hat{p} = 0.55$

**Test Statistic:**
$$z = \frac{0.55 - 0.5}{\sqrt{\frac{0.5 \times 0.5}{400}}} = \frac{0.05}{0.025} = 2.0$$

**Critical Value:** $z_{0.025} = 1.96$

**Decision:** Since $|z| = 2.0 > 1.96$, reject $H_0$

### 4.4.3 ANOVA – Single and Dual Factor

### Analysis of Variance (ANOVA)

Tests whether means of multiple groups are equal.

### Single-Factor ANOVA

**Hypotheses:**
- $H_0: \mu_1 = \mu_2 = \cdots = \mu_k$
- $H_1:$ At least one mean is different

**Assumptions:**
- Normality
- Equal variances (homoscedasticity)
- Independence

### ANOVA Table

| Source | SS | df | MS | F |
|--------|----|----|----|---|
| Between | SSB | $k-1$ | MSB = SSB/(k-1) | MSB/MSE |
| Within | SSE | $N-k$ | MSE = SSE/(N-k) | |
| Total | SST | $N-1$ | | |

**Formulas:**
- **SST (Total Sum of Squares):** $\sum_{i,j}(x_{ij} - \bar{x})^2$
- **SSB (Between Groups):** $\sum_{i} n_i(\bar{x}_i - \bar{x})^2$
- **SSE (Within Groups):** $\sum_{i,j}(x_{ij} - \bar{x}_i)^2$

**Test Statistic:**
$$F = \frac{MSB}{MSE} \sim F(k-1, N-k)$$

**Decision:** Reject $H_0$ if $F > F_{\alpha, k-1, N-k}$

### Two-Factor ANOVA

Tests effects of two factors and their interaction.

**Model:**
$$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}$$

where:
- $\mu$ = overall mean
- $\alpha_i$ = effect of factor A level $i$
- $\beta_j$ = effect of factor B level $j$
- $(\alpha\beta)_{ij}$ = interaction effect
- $\epsilon_{ijk}$ = error

**Hypotheses:**
1. $H_0:$ No main effect of factor A
2. $H_0:$ No main effect of factor B
3. $H_0:$ No interaction effect

---

## 4.5 Maximum Likelihood

### Likelihood Function

For parameter $\theta$ and data $x_1, \ldots, x_n$:

**Discrete:**
$$L(\theta) = \prod_{i=1}^{n} p(x_i; \theta)$$

**Continuous:**
$$L(\theta) = \prod_{i=1}^{n} f(x_i; \theta)$$

### Maximum Likelihood Estimator (MLE)

The **MLE** $\hat{\theta}$ maximizes the likelihood function:

$$\hat{\theta} = \arg\max_{\theta} L(\theta)$$

### Log-Likelihood

Often easier to maximize log-likelihood:

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log p(x_i; \theta)$$

### Method

1. Write likelihood function $L(\theta)$
2. Take logarithm: $\ell(\theta) = \log L(\theta)$
3. Differentiate: $\frac{d\ell}{d\theta} = 0$
4. Solve for $\hat{\theta}$

### Example: MLE for Normal Distribution

Given $X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$:

**Likelihood:**
$$L(\mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x_i-\mu)^2}{2\sigma^2}}$$

**Log-likelihood:**
$$\ell(\mu, \sigma^2) = -n\log(\sigma\sqrt{2\pi}) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2$$

**MLE for $\mu$:**
$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i-\mu) = 0$$

$$\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

**MLE for $\sigma^2$:**
$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Sampling:** Know difference between simple random and stratified
2. **CLT:** Remember conditions and interpretation
3. **Confidence Intervals:** Use correct formula (z vs t, mean vs proportion)
4. **Hypothesis Testing:** Always state hypotheses, test statistic, decision, conclusion
5. **ANOVA:** Check assumptions before applying
6. **MLE:** Use log-likelihood for easier computation
7. **Common Mistake:** Confusing Type I and Type II errors
```

---

## 🔍 Worked Examples

### Example 1: Stratified Sampling

Population: 2000 students
- Freshmen: 800
- Sophomores: 600
- Juniors: 400
- Seniors: 200

Sample size: 200

**Proportional allocation:**
- Freshmen: $200 \times \frac{800}{2000} = 80$
- Sophomores: $200 \times \frac{600}{2000} = 60$
- Juniors: $200 \times \frac{400}{2000} = 40$
- Seniors: $200 \times \frac{200}{2000} = 20$

### Example 2: Confidence Interval

Sample: $n = 36$, $\bar{x} = 50$, $s = 12$. Find 95% CI.

**95% CI:**
$$\bar{x} \pm z_{0.025} \frac{s}{\sqrt{n}} = 50 \pm 1.96 \times \frac{12}{6} = 50 \pm 3.92$$

**Interval:** $[46.08, 53.92]$

### Example 3: Single-Factor ANOVA

Three groups with means $\bar{x}_1 = 10$, $\bar{x}_2 = 12$, $\bar{x}_3 = 14$, overall mean $\bar{x} = 12$, $n_i = 5$ each.

**SSB:**
$$SSB = 5[(10-12)^2 + (12-12)^2 + (14-12)^2] = 5[4 + 0 + 4] = 40$$

**df:** $k-1 = 2$, $N-k = 12$

**MSB:** $40/2 = 20$

If MSE = 5, then $F = 20/5 = 4$

Compare with $F_{0.05, 2, 12} = 3.89$

Since $F = 4 > 3.89$, reject $H_0$ (means are significantly different).

---

## 📚 Quick Revision Checklist

- [ ] Random sampling vs stratified sampling
- [ ] Central Limit Theorem statement and conditions
- [ ] Confidence intervals for mean and proportion
- [ ] Hypothesis testing for means (z-test, t-test)
- [ ] Hypothesis testing for proportions
- [ ] Single-factor ANOVA
- [ ] Two-factor ANOVA
- [ ] Maximum likelihood estimation
- [ ] Log-likelihood method

