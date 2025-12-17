# Module 6: Hypothesis Testing

## Overview

Hypothesis testing is a statistical method for making decisions about population parameters based on sample data. This module covers the framework, types of tests, and interpretation of results.

## 1. Hypothesis Testing Framework

### Null and Alternative Hypotheses

- **Null Hypothesis ($H_0$):** Statement about population parameter (usually equality)
- **Alternative Hypothesis ($H_1$ or $H_a$):** Statement we want to prove (usually inequality)

### Types of Tests

**One-tailed (One-sided):**
- $H_0: \mu = \mu_0$ vs $H_1: \mu > \mu_0$ (right-tailed)
- $H_0: \mu = \mu_0$ vs $H_1: \mu < \mu_0$ (left-tailed)

**Two-tailed (Two-sided):**
- $H_0: \mu = \mu_0$ vs $H_1: \mu \neq \mu_0$

## 2. Type I and Type II Errors

### Type I Error (α)

Rejecting $H_0$ when it is true.

**Probability:** $P(\text{Type I Error}) = \alpha$ (significance level)

### Type II Error (β)

Failing to reject $H_0$ when it is false.

**Probability:** $P(\text{Type II Error}) = \beta$

### Power

**Power** = $1 - \beta$ = Probability of rejecting $H_0$ when it is false

### Relationship

- Decreasing $\alpha$ increases $\beta$ (and decreases power)
- Increasing sample size decreases both $\alpha$ and $\beta$

## 3. Test Statistics

### Z-test for Mean (Known Variance)

**Test Statistic:**
$$z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$$

**Distribution:** Standard normal $\mathcal{N}(0, 1)$

### t-test for Mean (Unknown Variance)

**Test Statistic:**
$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$$

**Distribution:** t-distribution with $n-1$ degrees of freedom

### Z-test for Proportion

**Test Statistic:**
$$z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}$$

**Distribution:** Standard normal (for large $n$)

## 4. P-values

### Definition

The **p-value** is the probability of observing a test statistic as extreme as (or more extreme than) the observed value, assuming $H_0$ is true.

### Decision Rule

- **Reject $H_0$** if $p\text{-value} < \alpha$
- **Fail to reject $H_0$** if $p\text{-value} \geq \alpha$

### Interpretation

- **Small p-value (< 0.05):** Strong evidence against $H_0$
- **Large p-value (> 0.05):** Weak evidence against $H_0$

## 5. Tests for Mean

### Z-test (Known Variance)

**Steps:**
1. State hypotheses: $H_0: \mu = \mu_0$ vs $H_1: \mu \neq \mu_0$ (or one-sided)
2. Choose significance level: $\alpha = 0.05$
3. Compute test statistic: $z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$
4. Find critical value: $z_{\alpha/2}$ (two-tailed) or $z_{\alpha}$ (one-tailed)
5. Decision: Reject if $|z| > z_{\alpha/2}$ (two-tailed) or $z > z_{\alpha}$ (right-tailed)
6. Conclusion: State in context

### t-test (Unknown Variance)

**Steps:**
1. State hypotheses
2. Choose $\alpha$
3. Compute: $t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$
4. Find critical value: $t_{\alpha/2, n-1}$
5. Decision: Reject if $|t| > t_{\alpha/2, n-1}$
6. Conclusion

## 6. Tests for Proportion

### Z-test for Proportion

**Steps:**
1. State hypotheses: $H_0: p = p_0$ vs $H_1: p \neq p_0$
2. Check conditions: $np_0 \geq 5$ and $n(1-p_0) \geq 5$
3. Compute: $z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}$
4. Find critical value: $z_{\alpha/2}$
5. Decision: Reject if $|z| > z_{\alpha/2}$
6. Conclusion

## 7. Significance Levels

### Common Levels

- **α = 0.10:** 10% significance level (90% confidence)
- **α = 0.05:** 5% significance level (95% confidence) - **Most common**
- **α = 0.01:** 1% significance level (99% confidence)

### Choosing α

- Lower α: More stringent, harder to reject $H_0$
- Higher α: Less stringent, easier to reject $H_0$
- Typically use α = 0.05 unless specified

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Hypotheses:** Always state clearly, $H_0$ has equality
2. **Test Statistic:** Use correct formula (z vs t, mean vs proportion)
3. **Critical Values:** Know when to use $z_{\alpha/2}$ vs $z_{\alpha}$
4. **P-values:** Compare to α for decision
5. **Conclusion:** State in context, not just "reject" or "fail to reject"
6. **Type I/II Errors:** Understand the difference
```

---

## 🔍 Worked Examples

### Example 1: Z-test for Mean

Test $H_0: \mu = 100$ vs $H_1: \mu \neq 100$ at α = 0.05.

Sample: $n = 36$, $\bar{x} = 105$, $\sigma = 12$.

**Test Statistic:**
$$z = \frac{105 - 100}{12/\sqrt{36}} = \frac{5}{2} = 2.5$$

**Critical Value:** $z_{0.025} = 1.96$

**Decision:** Since $|z| = 2.5 > 1.96$, reject $H_0$

**Conclusion:** There is sufficient evidence at 5% level to conclude $\mu \neq 100$.

### Example 2: t-test for Mean

Test $H_0: \mu = 50$ vs $H_1: \mu > 50$ at α = 0.05.

Sample: $n = 16$, $\bar{x} = 55$, $s = 10$.

**Test Statistic:**
$$t = \frac{55 - 50}{10/\sqrt{16}} = \frac{5}{2.5} = 2.0$$

**Critical Value:** $t_{0.05, 15} = 1.753$

**Decision:** Since $t = 2.0 > 1.753$, reject $H_0$

**Conclusion:** There is sufficient evidence at 5% level to conclude $\mu > 50$.

### Example 3: Z-test for Proportion

Test $H_0: p = 0.5$ vs $H_1: p \neq 0.5$ at α = 0.05.

Sample: $n = 400$, $\hat{p} = 0.55$.

**Check:** $np_0 = 200 \geq 5$, $n(1-p_0) = 200 \geq 5$ ✓

**Test Statistic:**
$$z = \frac{0.55 - 0.5}{\sqrt{\frac{0.5 \times 0.5}{400}}} = \frac{0.05}{0.025} = 2.0$$

**Critical Value:** $z_{0.025} = 1.96$

**Decision:** Since $|z| = 2.0 > 1.96$, reject $H_0$

**Conclusion:** There is sufficient evidence at 5% level to conclude $p \neq 0.5$.

---

## 📚 Quick Revision Checklist

- [ ] Hypothesis testing framework
- [ ] Null and alternative hypotheses
- [ ] Type I and Type II errors
- [ ] Test statistics (z-test, t-test)
- [ ] P-values and interpretation
- [ ] Tests for mean (known and unknown variance)
- [ ] Tests for proportion
- [ ] Significance levels
- [ ] Decision rules and conclusions

