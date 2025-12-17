# Module 5: Estimation and Confidence Intervals

## Overview

This module covers point estimation, properties of estimators, and confidence intervals - methods for estimating population parameters from samples.

## 1. Point Estimation

### Definition

A **point estimator** is a statistic used to estimate a population parameter.

**Examples:**
- $\bar{X}$ estimates $\mu$ (population mean)
- $s^2$ estimates $\sigma^2$ (population variance)
- $\hat{p}$ estimates $p$ (population proportion)

### Notation

- **Parameter:** $\theta$ (unknown population value)
- **Estimator:** $\hat{\theta}$ (statistic used to estimate)
- **Estimate:** Specific value of $\hat{\theta}$ from a sample

## 2. Properties of Estimators

### Unbiasedness

An estimator $\hat{\theta}$ is **unbiased** if:

$$\mathbb{E}[\hat{\theta}] = \theta$$

**Examples:**
- $\bar{X}$ is unbiased for $\mu$
- $s^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ is unbiased for $\sigma^2$

### Efficiency

Among unbiased estimators, the one with smaller variance is more **efficient**.

### Consistency

An estimator $\hat{\theta}$ is **consistent** if:

$$\hat{\theta} \xrightarrow{p} \theta$$

(Converges in probability to the true parameter)

### Sufficiency

An estimator is **sufficient** if it contains all information about the parameter in the sample.

## 3. Confidence Intervals for Mean

### Known Variance

For $X \sim \mathcal{N}(\mu, \sigma^2)$ with known $\sigma$:

**100(1-α)% Confidence Interval:**

$$\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

where $z_{\alpha/2}$ is the critical value from standard normal.

**Common Values:**
- 90% CI: $z_{0.05} = 1.645$
- 95% CI: $z_{0.025} = 1.96$
- 99% CI: $z_{0.005} = 2.576$

### Unknown Variance (Small Sample)

For normal population with unknown $\sigma$ and small $n$:

**100(1-α)% Confidence Interval:**

$$\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}$$

where $t_{\alpha/2, n-1}$ is from t-distribution with $n-1$ degrees of freedom.

### Unknown Variance (Large Sample)

For large $n$ (typically $n \geq 30$), use normal approximation:

$$\bar{x} \pm z_{\alpha/2} \frac{s}{\sqrt{n}}$$

## 4. Confidence Intervals for Proportion

### Large Sample

For large $n$ (with $np \geq 5$ and $n(1-p) \geq 5$):

**100(1-α)% Confidence Interval:**

$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

### Interpretation

"We are 95% confident that the true proportion $p$ lies in the interval $[\hat{p} - E, \hat{p} + E]$"

where $E = z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$ is the margin of error.

## 5. Sample Size Determination

### For Mean

To estimate $\mu$ with margin of error $E$ and confidence level $(1-\alpha)$:

$$n = \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2$$

If $\sigma$ is unknown, use a pilot study or estimate.

### For Proportion

To estimate $p$ with margin of error $E$ and confidence level $(1-\alpha)$:

$$n = \frac{z_{\alpha/2}^2 \cdot p(1-p)}{E^2}$$

If $p$ is unknown, use $p = 0.5$ for maximum sample size.

## 6. Interpretation of Confidence Intervals

### Common Misconception

❌ **Wrong:** "There is a 95% probability that $\mu$ is in the interval"

✅ **Correct:** "If we repeated this procedure many times, 95% of the intervals would contain $\mu$"

### Key Points

- The parameter is fixed (not random)
- The interval is random (varies with each sample)
- Confidence level refers to the procedure, not a specific interval

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Known vs Unknown Variance:** Use $z$ for known, $t$ for unknown (small sample)
2. **Sample Size:** Use $z$ for large samples even if variance unknown
3. **Proportion CI:** Check conditions ($np \geq 5$, $n(1-p) \geq 5$)
4. **Interpretation:** Confidence level is about the procedure, not the parameter
5. **Sample Size Formula:** Remember to round up
```

---

## 🔍 Worked Examples

### Example 1: CI for Mean (Known Variance)

Sample: $n = 36$, $\bar{x} = 50$, $\sigma = 12$. Find 95% CI for $\mu$.

**95% CI:**
$$\bar{x} \pm z_{0.025} \frac{\sigma}{\sqrt{n}} = 50 \pm 1.96 \times \frac{12}{6} = 50 \pm 3.92$$

**Interval:** $[46.08, 53.92]$

### Example 2: CI for Mean (Unknown Variance)

Sample: $n = 16$, $\bar{x} = 50$, $s = 12$. Find 95% CI for $\mu$.

**Use t-distribution:** $t_{0.025, 15} = 2.131$

**95% CI:**
$$\bar{x} \pm t_{0.025, 15} \frac{s}{\sqrt{n}} = 50 \pm 2.131 \times \frac{12}{4} = 50 \pm 6.393$$

**Interval:** $[43.607, 56.393]$

### Example 3: CI for Proportion

In a sample of 400, 120 favor a proposal. Find 95% CI for true proportion.

$\hat{p} = \frac{120}{400} = 0.3$

**Check:** $np = 120 \geq 5$, $n(1-p) = 280 \geq 5$ ✓

**95% CI:**
$$\hat{p} \pm z_{0.025} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = 0.3 \pm 1.96 \sqrt{\frac{0.3 \times 0.7}{400}}$$

$$= 0.3 \pm 1.96 \times 0.0229 = 0.3 \pm 0.045$$

**Interval:** $[0.255, 0.345]$

### Example 4: Sample Size

Want to estimate mean with margin of error 2, 95% confidence, $\sigma = 10$.

$$n = \left(\frac{1.96 \times 10}{2}\right)^2 = (9.8)^2 = 96.04$$

**Round up:** $n = 97$

---

## 📚 Quick Revision Checklist

- [ ] Point estimation concepts
- [ ] Properties of estimators (unbiased, efficient, consistent)
- [ ] CI for mean (known variance)
- [ ] CI for mean (unknown variance - t-distribution)
- [ ] CI for proportion
- [ ] Sample size determination
- [ ] Correct interpretation of confidence intervals
- [ ] When to use z vs t

