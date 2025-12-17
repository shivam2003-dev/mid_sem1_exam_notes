# 2024 EndSem Regular ISM - Solved Paper

## Question 1: Comprehensive Statistics

**Question:** A sample of 50 observations has the following summary statistics:
- Mean = 45
- Standard deviation = 8
- Minimum = 30
- Maximum = 65
- Q₁ = 40
- Q₃ = 50

(a) Calculate the coefficient of variation
(b) Calculate the interquartile range
(c) Identify any potential outliers using the IQR method
(d) What can you say about the shape of the distribution?

**Solution:**

**(a) Coefficient of Variation:**
$$CV = \frac{s}{\bar{x}} \times 100\% = \frac{8}{45} \times 100\% = 17.78\%$$

**(b) Interquartile Range:**
$$IQR = Q_3 - Q_1 = 50 - 40 = 10$$

**(c) Outlier Detection:**

**Lower Fence:** $Q_1 - 1.5 \times IQR = 40 - 1.5(10) = 25$

**Upper Fence:** $Q_3 + 1.5 \times IQR = 50 + 1.5(10) = 65$

Since minimum = 30 > 25 and maximum = 65 = 65, there are **no outliers**.

**(d) Shape of Distribution:**

Since $Q_2$ (median) would be around 45 (close to mean), and the data is symmetric around the median, the distribution appears to be **approximately symmetric**.

---

## Question 2: Bayes' Theorem

**Question:** A factory has two machines. Machine A produces 60% of items and Machine B produces 40%. The defect rate is 5% for Machine A and 8% for Machine B.

(a) Find the probability that a randomly selected item is defective
(b) If an item is found to be defective, what is the probability it came from Machine B?

**Solution:**

Let $A$ = event item from Machine A
Let $B$ = event item from Machine B
Let $D$ = event item is defective

**Given:**
- $P(A) = 0.6$, $P(B) = 0.4$
- $P(D|A) = 0.05$, $P(D|B) = 0.08$

**(a) Probability of Defective Item:**

$$P(D) = P(D|A)P(A) + P(D|B)P(B) = 0.05(0.6) + 0.08(0.4) = 0.03 + 0.032 = 0.062$$

**(b) Probability from Machine B given Defective:**

$$P(B|D) = \frac{P(D|B)P(B)}{P(D)} = \frac{0.08(0.4)}{0.062} = \frac{0.032}{0.062} = 0.516$$

---

## Question 3: Sampling Distribution

**Question:** A population has mean 100 and standard deviation 15. A sample of size 36 is taken.

(a) What is the sampling distribution of the sample mean?
(b) Find the probability that the sample mean is between 95 and 105
(c) Find the probability that the sample mean exceeds 103

**Solution:**

**Given:**
- $\mu = 100$, $\sigma = 15$, $n = 36$

**(a) Sampling Distribution:**

By Central Limit Theorem (since $n = 36 \geq 30$):

$$\bar{X} \sim \mathcal{N}\left(100, \frac{225}{36}\right) = \mathcal{N}(100, 6.25)$$

So $\bar{X} \sim \mathcal{N}(100, 2.5^2)$

**(b) $P(95 < \bar{X} < 105)$:**

$$P(95 < \bar{X} < 105) = P\left(\frac{95-100}{2.5} < Z < \frac{105-100}{2.5}\right) = P(-2 < Z < 2)$$

$$= \Phi(2) - \Phi(-2) = 0.9772 - 0.0228 = 0.9544$$

**(c) $P(\bar{X} > 103)$:**

$$P(\bar{X} > 103) = P\left(Z > \frac{103-100}{2.5}\right) = P(Z > 1.2) = 1 - \Phi(1.2) = 1 - 0.8849 = 0.1151$$

---

## Question 4: Hypothesis Testing (Two-Sample)

**Question:** Test whether there is a significant difference between two population means at 5% level.

Sample 1: $n_1 = 25$, $\bar{x}_1 = 50$, $s_1 = 8$
Sample 2: $n_2 = 30$, $\bar{x}_2 = 45$, $s_2 = 10$

**Solution:**

**Step 1: Hypotheses**
- $H_0: \mu_1 = \mu_2$
- $H_1: \mu_1 \neq \mu_2$

**Step 2: Significance Level**
$\alpha = 0.05$

**Step 3: Test Statistic**

Assuming equal variances:

$$s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2} = \frac{24(64) + 29(100)}{53} = \frac{1536 + 2900}{53} = 83.62$$

$$s_p = 9.14$$

$$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} = \frac{50 - 45}{9.14\sqrt{\frac{1}{25} + \frac{1}{30}}}$$

$$= \frac{5}{9.14\sqrt{0.04 + 0.0333}} = \frac{5}{9.14(0.27)} = \frac{5}{2.47} = 2.02$$

**Step 4: Critical Value**
$df = n_1 + n_2 - 2 = 53$
$t_{0.025, 53} \approx 2.006$

**Step 5: Decision**
Since $|t| = 2.02 > 2.006$, we **reject** $H_0$.

**Step 6: Conclusion**
There is sufficient evidence at 5% level to conclude that the two population means are significantly different.

---

## Question 5: Multiple Regression Concepts

**Question:** Explain the following in the context of regression:
(a) Residuals
(b) Multicollinearity
(c) Adjusted R²
(d) Dummy variables

**Solution:**

**(a) Residuals:**
Residuals are the differences between observed values and predicted values:
$$e_i = y_i - \hat{y}_i$$

They are used to:
- Check model assumptions
- Identify outliers
- Assess model fit

**(b) Multicollinearity:**
Multicollinearity occurs when predictor variables are highly correlated with each other. This can cause:
- Unstable coefficient estimates
- Large standard errors
- Difficulty interpreting individual coefficients

**(c) Adjusted R²:**
Adjusted R² accounts for the number of predictors in the model:

$$R_{adj}^2 = 1 - \frac{SSE/(n-k-1)}{SST/(n-1)}$$

where $k$ is the number of predictors. It penalizes for adding unnecessary variables.

**(d) Dummy Variables:**
Dummy variables are binary (0/1) variables used to represent categorical predictors in regression. For a categorical variable with $k$ categories, we need $k-1$ dummy variables.

---

## Summary

Comprehensive solutions covering descriptive statistics, probability, sampling, hypothesis testing, and regression concepts.

