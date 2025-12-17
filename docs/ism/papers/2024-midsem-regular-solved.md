# 2024 MidSem Regular ISM - Solved Paper

## Question 1: Descriptive Statistics

**Question:** Given the following data: 12, 15, 18, 20, 22, 25, 28, 30, 32, 35

Calculate:
(a) Mean
(b) Median
(c) Mode
(d) Standard deviation
(e) Coefficient of variation

**Solution:**

**(a) Mean:**
$$\bar{x} = \frac{12 + 15 + 18 + 20 + 22 + 25 + 28 + 30 + 32 + 35}{10} = \frac{237}{10} = 23.7$$

**(b) Median:**
Since $n = 10$ (even):
$$\text{Median} = \frac{x_5 + x_6}{2} = \frac{22 + 25}{2} = 23.5$$

**(c) Mode:**
No mode (all values are unique)

**(d) Standard Deviation:**
First, compute variance:
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

$$= \frac{1}{9}[(12-23.7)^2 + (15-23.7)^2 + \cdots + (35-23.7)^2]$$

$$= \frac{1}{9}[136.89 + 75.69 + 32.49 + 13.69 + 2.89 + 1.69 + 18.49 + 39.69 + 68.89 + 127.69]$$

$$= \frac{1}{9}(519.1) = 57.68$$

$$s = \sqrt{57.68} = 7.59$$

**(e) Coefficient of Variation:**
$$CV = \frac{s}{\bar{x}} \times 100\% = \frac{7.59}{23.7} \times 100\% = 32.0\%$$

---

## Question 2: Probability

**Question:** In a class of 30 students, 18 are girls and 12 are boys. 10 girls and 6 boys wear glasses. If a student is selected at random:

(a) Find the probability that the student wears glasses
(b) Find the probability that the student is a girl given that they wear glasses
(c) Are the events "being a girl" and "wearing glasses" independent?

**Solution:**

Let $G$ = event that student is a girl
Let $Gl$ = event that student wears glasses

**Given:**
- $P(G) = \frac{18}{30} = \frac{3}{5}$
- $P(Gl|G) = \frac{10}{18} = \frac{5}{9}$
- $P(Gl|G^c) = \frac{6}{12} = \frac{1}{2}$

**(a) Probability of wearing glasses:**
$$P(Gl) = P(Gl|G)P(G) + P(Gl|G^c)P(G^c) = \frac{5}{9} \times \frac{3}{5} + \frac{1}{2} \times \frac{2}{5}$$

$$= \frac{1}{3} + \frac{1}{5} = \frac{5+3}{15} = \frac{8}{15}$$

**(b) Probability of being a girl given wearing glasses:**
$$P(G|Gl) = \frac{P(Gl|G)P(G)}{P(Gl)} = \frac{\frac{5}{9} \times \frac{3}{5}}{\frac{8}{15}} = \frac{\frac{1}{3}}{\frac{8}{15}} = \frac{15}{24} = \frac{5}{8}$$

**(c) Independence Check:**
$$P(G \cap Gl) = P(Gl|G)P(G) = \frac{5}{9} \times \frac{3}{5} = \frac{1}{3}$$

$$P(G) \times P(Gl) = \frac{3}{5} \times \frac{8}{15} = \frac{24}{75} = \frac{8}{25}$$

Since $P(G \cap Gl) = \frac{1}{3} = \frac{25}{75} \neq \frac{24}{75} = P(G) \times P(Gl)$, the events are **not independent**.

---

## Question 3: Confidence Interval

**Question:** A sample of 36 observations has mean 50 and standard deviation 12. Construct a 95% confidence interval for the population mean.

**Solution:**

**Given:**
- $n = 36$
- $\bar{x} = 50$
- $s = 12$
- Confidence level = 95% ($\alpha = 0.05$)

Since $n = 36 \geq 30$ (large sample), we use normal distribution.

**95% Confidence Interval:**
$$\bar{x} \pm z_{\alpha/2} \frac{s}{\sqrt{n}} = 50 \pm 1.96 \times \frac{12}{\sqrt{36}}$$

$$= 50 \pm 1.96 \times 2 = 50 \pm 3.92$$

**Interval:** $[46.08, 53.92]$

**Interpretation:** We are 95% confident that the true population mean lies between 46.08 and 53.92.

---

## Question 4: Hypothesis Testing

**Question:** Test the hypothesis $H_0: \mu = 100$ vs $H_1: \mu \neq 100$ at 5% significance level.

Sample: $n = 25$, $\bar{x} = 105$, $s = 15$.

**Solution:**

**Step 1: State Hypotheses**
- $H_0: \mu = 100$
- $H_1: \mu \neq 100$ (two-tailed test)

**Step 2: Significance Level**
$\alpha = 0.05$

**Step 3: Test Statistic**

Since variance is unknown and $n = 25 < 30$, use t-test:

$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} = \frac{105 - 100}{15/\sqrt{25}} = \frac{5}{3} = 1.67$$

**Step 4: Critical Value**

For two-tailed test with $\alpha = 0.05$ and $df = 24$:
$$t_{\alpha/2, n-1} = t_{0.025, 24} = 2.064$$

**Step 5: Decision**

Since $|t| = 1.67 < 2.064$, we **fail to reject** $H_0$.

**Step 6: Conclusion**

There is insufficient evidence at 5% significance level to conclude that $\mu \neq 100$.

---

## Question 5: Regression Analysis

**Question:** Given the following data:

| X | Y |
|---|---|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| 5 | 11 |

(a) Find the regression line
(b) Calculate the correlation coefficient
(c) Find $R^2$ and interpret

**Solution:**

**Given Data:**
- $\bar{x} = 3$, $\bar{y} = 7$

**Calculations:**
- $\sum x_i = 15$, $\sum y_i = 35$
- $\sum x_i^2 = 55$, $\sum y_i^2 = 285$, $\sum x_i y_i = 115$

**Sums of Squares:**
- $S_{xx} = \sum x_i^2 - n\bar{x}^2 = 55 - 5(9) = 10$
- $S_{yy} = \sum y_i^2 - n\bar{y}^2 = 285 - 5(49) = 40$
- $S_{xy} = \sum x_i y_i - n\bar{x}\bar{y} = 115 - 5(21) = 10$

**(a) Regression Line:**

**Slope:**
$$\hat{\beta_1} = \frac{S_{xy}}{S_{xx}} = \frac{10}{10} = 1$$

**Intercept:**
$$\hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x} = 7 - 1(3) = 4$$

**Regression Line:**
$$\hat{y} = 4 + x$$

**(b) Correlation Coefficient:**
$$r = \frac{S_{xy}}{\sqrt{S_{xx} S_{yy}}} = \frac{10}{\sqrt{10 \times 40}} = \frac{10}{20} = 0.5$$

**(c) Coefficient of Determination:**
$$R^2 = r^2 = (0.5)^2 = 0.25$$

**Interpretation:** 25% of the variance in $Y$ is explained by the linear relationship with $X$.

---

## Summary

All questions solved with step-by-step calculations and clear explanations.

