# Module 7: Regression and Correlation

## Overview

This module covers simple linear regression and correlation - methods for modeling relationships between variables and making predictions.

## 1. Simple Linear Regression

### Model

$$Y = \beta_0 + \beta_1 X + \epsilon$$

where:
- $Y$ = dependent variable (response)
- $X$ = independent variable (predictor)
- $\beta_0$ = y-intercept
- $\beta_1$ = slope
- $\epsilon$ = error term

### Assumptions

1. **Linearity:** Relationship between $X$ and $Y$ is linear
2. **Independence:** Observations are independent
3. **Homoscedasticity:** Constant variance of errors
4. **Normality:** Errors are normally distributed

## 2. Least Squares Method

### Objective

Minimize the sum of squared errors:

$$SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - \hat{\beta_0} - \hat{\beta_1} x_i)^2$$

### Estimates

**Slope:**
$$\hat{\beta_1} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}$$

**Intercept:**
$$\hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x}$$

where:
- $S_{xy} = \sum(x_i - \bar{x})(y_i - \bar{y}) = \sum x_i y_i - n\bar{x}\bar{y}$
- $S_{xx} = \sum(x_i - \bar{x})^2 = \sum x_i^2 - n\bar{x}^2$
- $S_{yy} = \sum(y_i - \bar{y})^2 = \sum y_i^2 - n\bar{y}^2$

## 3. Correlation Coefficient

### Pearson Correlation

$$r = \frac{S_{xy}}{\sqrt{S_{xx} S_{yy}}} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$

### Properties

- $-1 \leq r \leq 1$
- $r = 1$: Perfect positive linear relationship
- $r = -1$: Perfect negative linear relationship
- $r = 0$: No linear relationship
- $r^2$ = Coefficient of determination ($R^2$)

### Interpretation

- **|r| > 0.7:** Strong correlation
- **0.3 < |r| < 0.7:** Moderate correlation
- **|r| < 0.3:** Weak correlation

## 4. Coefficient of Determination (R²)

### Definition

$$R^2 = r^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$$

where:
- **SST (Total Sum of Squares):** $SST = \sum(y_i - \bar{y})^2 = S_{yy}$
- **SSR (Regression Sum of Squares):** $SSR = \sum(\hat{y}_i - \bar{y})^2$
- **SSE (Error Sum of Squares):** $SSE = \sum(y_i - \hat{y}_i)^2$

### Interpretation

$R^2$ represents the proportion of variance in $Y$ explained by $X$.

- $R^2 = 1$: Perfect fit (all points on line)
- $R^2 = 0$: No linear relationship
- Higher $R^2$ = Better model fit

## 5. Regression Analysis

### Fitted Line

$$\hat{y} = \hat{\beta_0} + \hat{\beta_1}x$$

### Prediction

For a new value $x_0$:

$$\hat{y}_0 = \hat{\beta_0} + \hat{\beta_1}x_0$$

### Residuals

$$e_i = y_i - \hat{y}_i$$

**Properties:**
- $\sum e_i = 0$
- $\sum x_i e_i = 0$

## 6. Residual Analysis

### Purpose

Check if assumptions are satisfied.

### Patterns to Check

- **Random scatter:** Assumptions likely satisfied
- **Curved pattern:** Non-linear relationship
- **Funnel shape:** Heteroscedasticity (non-constant variance)
- **Trend:** Missing variable or wrong model

## 7. Inference in Regression

### Standard Errors

**For slope:**
$$SE(\hat{\beta_1}) = \frac{s}{\sqrt{S_{xx}}}$$

where $s = \sqrt{\frac{SSE}{n-2}}$ is the standard error of estimate.

**For intercept:**
$$SE(\hat{\beta_0}) = s\sqrt{\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}}$$

### Confidence Intervals

**For slope:**
$$\hat{\beta_1} \pm t_{\alpha/2, n-2} \cdot SE(\hat{\beta_1})$$

**For intercept:**
$$\hat{\beta_0} \pm t_{\alpha/2, n-2} \cdot SE(\hat{\beta_0})$$

### Hypothesis Tests

**Test for slope:**
- $H_0: \beta_1 = 0$ vs $H_1: \beta_1 \neq 0$
- Test statistic: $t = \frac{\hat{\beta_1}}{SE(\hat{\beta_1})}$
- Reject if $|t| > t_{\alpha/2, n-2}$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Least Squares:** Know formulas for $\hat{\beta_0}$ and $\hat{\beta_1}$
2. **Correlation:** Use formula $r = S_{xy}/\sqrt{S_{xx}S_{yy}}$
3. **R²:** Always between 0 and 1, represents explained variance
4. **Assumptions:** Check residuals for violations
5. **Prediction:** Use fitted equation $\hat{y} = \hat{\beta_0} + \hat{\beta_1}x$
6. **Interpretation:** State in context (e.g., "for each unit increase in X, Y increases by $\hat{\beta_1}$")
```

---

## 🔍 Worked Examples

### Example 1: Simple Linear Regression

Given data:
| X | Y |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 5 |
| 4 | 7 |
| 5 | 8 |

Find regression line.

**Calculations:**
- $\bar{x} = 3$, $\bar{y} = 5.2$
- $S_{xx} = 10$, $S_{yy} = 24.8$, $S_{xy} = 14$

**Slope:**
$$\hat{\beta_1} = \frac{14}{10} = 1.4$$

**Intercept:**
$$\hat{\beta_0} = 5.2 - 1.4 \times 3 = 1.0$$

**Regression Line:**
$$\hat{y} = 1.0 + 1.4x$$

### Example 2: Correlation

From Example 1:

$$r = \frac{14}{\sqrt{10 \times 24.8}} = \frac{14}{\sqrt{248}} = \frac{14}{15.75} = 0.889$$

**Interpretation:** Strong positive linear relationship.

**R²:**
$$R^2 = (0.889)^2 = 0.79$$

**Interpretation:** 79% of variance in $Y$ is explained by $X$.

---

## 📚 Quick Revision Checklist

- [ ] Simple linear regression model
- [ ] Least squares method and formulas
- [ ] Correlation coefficient calculation and interpretation
- [ ] Coefficient of determination (R²)
- [ ] Regression assumptions
- [ ] Residual analysis
- [ ] Inference in regression (CI, hypothesis tests)
- [ ] Making predictions

