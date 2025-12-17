# 2024 MidSem Makeup ISM - Solved Paper

## Question 1: Probability and Conditional Probability

**Question:** In a survey, 60% of people like coffee, 40% like tea, and 30% like both.

(a) Find the probability that a person likes coffee or tea
(b) Find the probability that a person likes tea given they like coffee
(c) Are the events "likes coffee" and "likes tea" independent?

**Solution:**

Let $C$ = event that person likes coffee
Let $T$ = event that person likes tea

**Given:**
- $P(C) = 0.6$
- $P(T) = 0.4$
- $P(C \cap T) = 0.3$

**(a) Probability of coffee or tea:**
$$P(C \cup T) = P(C) + P(T) - P(C \cap T) = 0.6 + 0.4 - 0.3 = 0.7$$

**(b) Probability of tea given coffee:**
$$P(T|C) = \frac{P(C \cap T)}{P(C)} = \frac{0.3}{0.6} = 0.5$$

**(c) Independence Check:**
$$P(C) \times P(T) = 0.6 \times 0.4 = 0.24$$

$$P(C \cap T) = 0.3 \neq 0.24 = P(C) \times P(T)$$

Therefore, the events are **not independent**.

---

## Question 2: Normal Distribution

**Question:** Scores on an exam are normally distributed with mean 75 and standard deviation 10.

(a) Find the probability that a randomly selected score is above 85
(b) Find the probability that a score is between 70 and 80
(c) What score corresponds to the 90th percentile?

**Solution:**

$X \sim \mathcal{N}(75, 100)$ where $\mu = 75$, $\sigma = 10$

**(a) $P(X > 85)$:**

$$P(X > 85) = P\left(Z > \frac{85-75}{10}\right) = P(Z > 1) = 1 - \Phi(1) = 1 - 0.8413 = 0.1587$$

**(b) $P(70 < X < 80)$:**

$$P(70 < X < 80) = P\left(\frac{70-75}{10} < Z < \frac{80-75}{10}\right) = P(-0.5 < Z < 0.5)$$

$$= \Phi(0.5) - \Phi(-0.5) = 0.6915 - 0.3085 = 0.3830$$

**(c) 90th Percentile:**

We need $x$ such that $P(X \leq x) = 0.90$

$$P\left(Z \leq \frac{x-75}{10}\right) = 0.90$$

From standard normal table: $z_{0.90} = 1.28$

$$\frac{x-75}{10} = 1.28$$

$$x = 75 + 12.8 = 87.8$$

---

## Question 3: Hypothesis Testing for Proportion

**Question:** A company claims that at least 80% of its products are defect-free. In a sample of 200 products, 150 are defect-free. Test the claim at 5% significance level.

**Solution:**

**Step 1: State Hypotheses**
- $H_0: p \geq 0.80$ (or $p = 0.80$)
- $H_1: p < 0.80$ (left-tailed test)

**Step 2: Significance Level**
$\alpha = 0.05$

**Step 3: Sample Proportion**
$$\hat{p} = \frac{150}{200} = 0.75$$

**Step 4: Check Conditions**
$np_0 = 200(0.80) = 160 \geq 5$ ✓
$n(1-p_0) = 200(0.20) = 40 \geq 5$ ✓

**Step 5: Test Statistic**
$$z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}} = \frac{0.75 - 0.80}{\sqrt{\frac{0.80(0.20)}{200}}}$$

$$= \frac{-0.05}{\sqrt{0.0008}} = \frac{-0.05}{0.0283} = -1.77$$

**Step 6: Critical Value**
For left-tailed test: $z_{\alpha} = z_{0.05} = -1.645$

**Step 7: Decision**
Since $z = -1.77 < -1.645$, we **reject** $H_0$.

**Step 8: Conclusion**
There is sufficient evidence at 5% level to reject the claim that at least 80% of products are defect-free.

---

## Question 4: Confidence Interval for Proportion

**Question:** In a sample of 500 voters, 275 favor a proposal. Construct a 99% confidence interval for the true proportion of voters favoring the proposal.

**Solution:**

**Given:**
- $n = 500$
- $X = 275$
- $\hat{p} = \frac{275}{500} = 0.55$
- Confidence level = 99% ($\alpha = 0.01$)

**Check Conditions:**
$n\hat{p} = 275 \geq 5$ ✓
$n(1-\hat{p}) = 225 \geq 5$ ✓

**99% Confidence Interval:**
$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = 0.55 \pm 2.576 \sqrt{\frac{0.55 \times 0.45}{500}}$$

$$= 0.55 \pm 2.576 \times 0.0222 = 0.55 \pm 0.0572$$

**Interval:** $[0.4928, 0.6072]$

**Interpretation:** We are 99% confident that the true proportion of voters favoring the proposal lies between 49.28% and 60.72%.

---

## Question 5: Regression Analysis

**Question:** The following data shows hours studied (X) and exam scores (Y):

| X | Y |
|---|---|
| 2 | 60 |
| 4 | 70 |
| 6 | 80 |
| 8 | 85 |
| 10 | 90 |

(a) Find the regression equation
(b) Predict the score for 7 hours of study
(c) Calculate and interpret $R^2$

**Solution:**

**Given:**
- $\bar{x} = 6$, $\bar{y} = 77$

**Calculations:**
- $\sum x_i = 30$, $\sum y_i = 385$
- $\sum x_i^2 = 220$, $\sum y_i^2 = 30225$, $\sum x_i y_i = 2440$

**Sums of Squares:**
- $S_{xx} = 220 - 5(36) = 40$
- $S_{yy} = 30225 - 5(5929) = 580$
- $S_{xy} = 2440 - 5(462) = 130$

**(a) Regression Equation:**

**Slope:**
$$\hat{\beta_1} = \frac{130}{40} = 3.25$$

**Intercept:**
$$\hat{\beta_0} = 77 - 3.25(6) = 77 - 19.5 = 57.5$$

**Regression Line:**
$$\hat{y} = 57.5 + 3.25x$$

**(b) Prediction for $x = 7$:**
$$\hat{y} = 57.5 + 3.25(7) = 57.5 + 22.75 = 80.25$$

**(c) Coefficient of Determination:**

**Correlation:**
$$r = \frac{130}{\sqrt{40 \times 580}} = \frac{130}{\sqrt{23200}} = \frac{130}{152.3} = 0.853$$

**R²:**
$$R^2 = (0.853)^2 = 0.728$$

**Interpretation:** 72.8% of the variance in exam scores is explained by the linear relationship with hours studied.

---

## Summary

All questions solved with detailed step-by-step solutions.

