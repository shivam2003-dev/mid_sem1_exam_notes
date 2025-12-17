# Module 5: Prediction & Forecasting

## Overview

This module covers correlation, regression analysis, and time series analysis methods for prediction and forecasting.

## 5.1 Correlation

### Pearson Correlation Coefficient

For variables $X$ and $Y$:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}} = \frac{S_{xy}}{\sqrt{S_{xx} S_{yy}}}$$

where:
- $S_{xy} = \sum(x_i - \bar{x})(y_i - \bar{y}) = \sum x_i y_i - n\bar{x}\bar{y}$
- $S_{xx} = \sum(x_i - \bar{x})^2 = \sum x_i^2 - n\bar{x}^2$
- $S_{yy} = \sum(y_i - \bar{y})^2 = \sum y_i^2 - n\bar{y}^2$

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

### Correlation vs Causation

**Important:** Correlation does not imply causation!

---

## 5.2 Regression

### Simple Linear Regression

**Model:**
$$Y = \beta_0 + \beta_1 X + \epsilon$$

where:
- $Y$ = dependent variable (response)
- $X$ = independent variable (predictor)
- $\beta_0$ = y-intercept
- $\beta_1$ = slope
- $\epsilon$ = error term

### Least Squares Estimates

**Slope:**
$$\hat{\beta_1} = \frac{S_{xy}}{S_{xx}} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$$

**Intercept:**
$$\hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x}$$

### Fitted Line

$$\hat{y} = \hat{\beta_0} + \hat{\beta_1}x$$

### Coefficient of Determination (R²)

$$R^2 = r^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$$

where:
- **SST (Total Sum of Squares):** $\sum(y_i - \bar{y})^2$
- **SSR (Regression Sum of Squares):** $\sum(\hat{y}_i - \bar{y})^2$
- **SSE (Error Sum of Squares):** $\sum(y_i - \hat{y}_i)^2$

**Interpretation:** Proportion of variance in $Y$ explained by $X$.

### Multiple Linear Regression

**Model:**
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon$$

**Matrix Form:**
$$Y = X\beta + \epsilon$$

**Least Squares Estimate:**
$$\hat{\beta} = (X^T X)^{-1} X^T Y$$

---

## 5.3 Time Series Analysis

### 5.3.1 Introduction, Components of Time Series Data

### Definition

A **time series** is a sequence of observations measured at successive time points.

### Components

**1. Trend (T):**
- Long-term direction (increasing, decreasing, stable)
- Can be linear or non-linear

**2. Seasonality (S):**
- Regular patterns that repeat over fixed periods
- Examples: daily, weekly, monthly, yearly patterns

**3. Cyclical (C):**
- Irregular cycles (not fixed period)
- Business cycles, economic cycles

**4. Irregular/Random (I):**
- Random fluctuations
- Unexplained variation

### Decomposition Models

**Additive Model:**
$$Y_t = T_t + S_t + C_t + I_t$$

**Multiplicative Model:**
$$Y_t = T_t \times S_t \times C_t \times I_t$$

### 5.3.2 MA Model – Basic and Weighted MA Model

### Moving Average (MA)

**Basic Moving Average:**

For window size $k$:

$$MA_t = \frac{Y_t + Y_{t-1} + \cdots + Y_{t-k+1}}{k}$$

**Example (3-period MA):**
$$MA_3 = \frac{Y_3 + Y_2 + Y_1}{3}$$

### Weighted Moving Average

$$WMA_t = \sum_{i=0}^{k-1} w_i Y_{t-i}$$

where $\sum w_i = 1$ and typically $w_0 > w_1 > \cdots > w_{k-1}$.

**Example (3-period WMA with weights 0.5, 0.3, 0.2):**
$$WMA_3 = 0.5Y_3 + 0.3Y_2 + 0.2Y_1$$

### Exponential Moving Average (EMA)

$$EMA_t = \alpha Y_t + (1-\alpha) EMA_{t-1}$$

where $\alpha$ is smoothing constant ($0 < \alpha \leq 1$).

---

## 5.3.3 Time Series Models

### 5.3.3.1 AR Model

### Autoregressive (AR) Model

**AR(p) Model:**
$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + \epsilon_t$$

where:
- $c$ = constant
- $\phi_i$ = autoregressive coefficients
- $\epsilon_t$ = white noise error

**AR(1) Model:**
$$Y_t = c + \phi_1 Y_{t-1} + \epsilon_t$$

**Properties:**
- Stationary if $|\phi_1| < 1$
- Mean: $\mu = \frac{c}{1-\phi_1}$
- Variance: $\sigma^2_Y = \frac{\sigma^2_\epsilon}{1-\phi_1^2}$

### 5.3.3.2 ARIMA Model

### ARIMA(p, d, q) Model

**Components:**
- **AR(p):** Autoregressive of order $p$
- **I(d):** Integrated (differencing) of order $d$
- **MA(q):** Moving average of order $q$

**Differencing:**
- **First difference:** $\Delta Y_t = Y_t - Y_{t-1}$
- **Second difference:** $\Delta^2 Y_t = \Delta(\Delta Y_t)$

**ARIMA(p, d, q) Form:**
$$\phi(B)(1-B)^d Y_t = \theta(B) \epsilon_t$$

where:
- $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ (AR polynomial)
- $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ (MA polynomial)
- $B$ = backshift operator ($BY_t = Y_{t-1}$)

**Common Models:**
- **ARIMA(1,1,1):** $\Delta Y_t = c + \phi_1 \Delta Y_{t-1} + \epsilon_t + \theta_1 \epsilon_{t-1}$
- **Random Walk:** ARIMA(0,1,0): $Y_t = Y_{t-1} + \epsilon_t$

### 5.3.3.3 SARIMA, SARIMAX, VAR, VARMAX

### SARIMA (Seasonal ARIMA)

**SARIMA(p, d, q)(P, D, Q)_s:**

- $(p, d, q)$: Non-seasonal components
- $(P, D, Q)$: Seasonal components
- $s$: Seasonal period

**Model:**
$$\phi(B)\Phi(B^s)(1-B)^d(1-B^s)^D Y_t = \theta(B)\Theta(B^s) \epsilon_t$$

**Example:** SARIMA(1,1,1)(1,1,1)_12 for monthly data with yearly seasonality.

### SARIMAX

SARIMA with **eXogenous** variables:

$$Y_t = \beta_1 X_{1t} + \beta_2 X_{2t} + \cdots + \text{SARIMA component}$$

### VAR (Vector Autoregression)

For multivariate time series:

**VAR(p) Model:**
$$\mathbf{Y}_t = \mathbf{c} + \Phi_1 \mathbf{Y}_{t-1} + \Phi_2 \mathbf{Y}_{t-2} + \cdots + \Phi_p \mathbf{Y}_{t-p} + \boldsymbol{\epsilon}_t$$

where $\mathbf{Y}_t$ is a vector of variables.

**VAR(1) Example:**
$$\begin{pmatrix} Y_{1t} \\ Y_{2t} \end{pmatrix} = \begin{pmatrix} c_1 \\ c_2 \end{pmatrix} + \begin{pmatrix} \phi_{11} & \phi_{12} \\ \phi_{21} & \phi_{22} \end{pmatrix} \begin{pmatrix} Y_{1,t-1} \\ Y_{2,t-1} \end{pmatrix} + \begin{pmatrix} \epsilon_{1t} \\ \epsilon_{2t} \end{pmatrix}$$

### VARMAX

VAR with **eXogenous** variables:

$$\mathbf{Y}_t = \mathbf{c} + \sum_{i=1}^{p} \Phi_i \mathbf{Y}_{t-i} + \sum_{j=0}^{q} \Theta_j \mathbf{X}_{t-j} + \boldsymbol{\epsilon}_t$$

### 5.3.3.4 Simple Exponential Smoothing Model

### Simple Exponential Smoothing

**Forecast:**
$$\hat{Y}_{t+1} = \alpha Y_t + (1-\alpha)\hat{Y}_t$$

where $\alpha$ is smoothing parameter ($0 < \alpha \leq 1$).

**Recursive Form:**
$$\hat{Y}_{t+1} = \alpha \sum_{i=0}^{\infty} (1-\alpha)^i Y_{t-i}$$

**Properties:**
- Gives more weight to recent observations
- $\alpha$ close to 1: More responsive to recent changes
- $\alpha$ close to 0: More smoothing, less responsive

**Initialization:**
- $\hat{Y}_1 = Y_1$ (or average of first few observations)

**Example:**
Given: $Y_1 = 10$, $Y_2 = 12$, $Y_3 = 11$, $\alpha = 0.3$

- $\hat{Y}_1 = 10$
- $\hat{Y}_2 = 0.3(10) + 0.7(10) = 10$
- $\hat{Y}_3 = 0.3(12) + 0.7(10) = 10.6$
- $\hat{Y}_4 = 0.3(11) + 0.7(10.6) = 10.72$

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **Correlation:** Use formula $r = S_{xy}/\sqrt{S_{xx}S_{yy}}$
2. **Regression:** Know formulas for $\hat{\beta_0}$ and $\hat{\beta_1}$
3. **R²:** Always between 0 and 1, represents explained variance
4. **Time Series Components:** Trend, Seasonality, Cyclical, Irregular
5. **ARIMA:** Understand (p, d, q) notation
6. **Exponential Smoothing:** Higher $\alpha$ = more responsive
7. **Common Mistake:** Confusing correlation with causation
```

---

## 🔍 Worked Examples

### Example 1: Correlation and Regression

Given data:
| X | Y |
|---|---|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| 5 | 11 |

**Correlation:**
- $\bar{x} = 3$, $\bar{y} = 7$
- $S_{xx} = 10$, $S_{yy} = 40$, $S_{xy} = 20$
- $r = \frac{20}{\sqrt{10 \times 40}} = \frac{20}{20} = 1$ (perfect positive)

**Regression:**
- $\hat{\beta_1} = \frac{20}{10} = 2$
- $\hat{\beta_0} = 7 - 2(3) = 1$
- $\hat{y} = 1 + 2x$

**R²:**
$R^2 = r^2 = 1$ (perfect fit)

### Example 2: Moving Average

Data: 10, 12, 14, 16, 18

**3-period MA:**
- $MA_3 = \frac{10+12+14}{3} = 12$
- $MA_4 = \frac{12+14+16}{3} = 14$
- $MA_5 = \frac{14+16+18}{3} = 16$

### Example 3: AR(1) Model

$Y_t = 2 + 0.5Y_{t-1} + \epsilon_t$, $\epsilon_t \sim \mathcal{N}(0, 1)$

**Mean:**
$$\mu = \frac{2}{1-0.5} = 4$$

**Variance:**
$$\sigma^2_Y = \frac{1}{1-0.5^2} = \frac{1}{0.75} = \frac{4}{3}$$

---

## 📚 Quick Revision Checklist

- [ ] Correlation coefficient calculation and interpretation
- [ ] Simple linear regression formulas
- [ ] Multiple linear regression
- [ ] Coefficient of determination (R²)
- [ ] Time series components (trend, seasonality, cyclical, irregular)
- [ ] Moving average models (basic and weighted)
- [ ] AR model
- [ ] ARIMA model notation and differencing
- [ ] SARIMA, SARIMAX, VAR, VARMAX
- [ ] Simple exponential smoothing

