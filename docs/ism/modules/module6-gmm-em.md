# Module 6: Gaussian Mixture Model & Expectation Maximization

## Overview

This module covers Gaussian Mixture Models (GMM) and the Expectation-Maximization (EM) algorithm for parameter estimation in mixture models.

## 6.1 Gaussian Mixture Model (GMM)

### Definition

A **Gaussian Mixture Model** is a probabilistic model that assumes data is generated from a mixture of $K$ Gaussian distributions.

### Model

For $K$ components:

$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$

where:
- $\pi_k$ = mixing coefficient (weight) for component $k$ with $\sum_{k=1}^{K} \pi_k = 1$ and $\pi_k \geq 0$
- $\mathcal{N}(x|\mu_k, \Sigma_k)$ = $k$-th Gaussian component with mean $\mu_k$ and covariance $\Sigma_k$

### Parameters

For $K$ components:
- **Mixing coefficients:** $\boldsymbol{\pi} = \{\pi_1, \ldots, \pi_K\}$
- **Means:** $\boldsymbol{\mu} = \{\mu_1, \ldots, \mu_K\}$
- **Covariances:** $\boldsymbol{\Sigma} = \{\Sigma_1, \ldots, \Sigma_K\}$

**Total parameters:** $K-1 + Kd + Kd(d+1)/2$ (for $d$-dimensional data)

### Special Cases

**1. Spherical GMM:**
$$\Sigma_k = \sigma_k^2 I$$

**2. Diagonal GMM:**
$$\Sigma_k = \text{diag}(\sigma_{k1}^2, \ldots, \sigma_{kd}^2)$$

**3. Shared Covariance:**
$$\Sigma_k = \Sigma \quad \forall k$$

### Latent Variables

Introduce **latent variable** $z \in \{1, \ldots, K\}$ indicating which component generated the observation:

$$p(z_k = 1) = \pi_k$$

$$p(x|z_k = 1) = \mathcal{N}(x|\mu_k, \Sigma_k)$$

### Responsibilities

**Posterior probability** that component $k$ generated observation $x$:

$$\gamma_{nk} = p(z_k = 1|x_n) = \frac{\pi_k \mathcal{N}(x_n|\mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n|\mu_j, \Sigma_j)}$$

**Properties:**
- $\sum_{k=1}^{K} \gamma_{nk} = 1$
- $0 \leq \gamma_{nk} \leq 1$

---

## 6.2 Expectation-Maximization (EM) Algorithm

### Overview

EM is an iterative algorithm for finding maximum likelihood estimates in models with latent variables.

### General Framework

**Given:** Data $X = \{x_1, \ldots, x_N\}$, latent variables $Z = \{z_1, \ldots, z_N\}$, parameters $\theta$

**Goal:** Maximize log-likelihood:
$$\ell(\theta) = \log p(X|\theta) = \log \sum_Z p(X, Z|\theta)$$

### Two Steps

**1. Expectation (E-step):**
Compute expected value of log-likelihood:

$$Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z|X,\theta^{(t)}}[\log p(X, Z|\theta)]$$

**2. Maximization (M-step):**
Maximize $Q$ with respect to $\theta$:

$$\theta^{(t+1)} = \arg\max_{\theta} Q(\theta, \theta^{(t)})$$

### Algorithm

1. **Initialize:** $\theta^{(0)}$
2. **Repeat until convergence:**
   - **E-step:** Compute $Q(\theta, \theta^{(t)})$
   - **M-step:** $\theta^{(t+1)} = \arg\max_{\theta} Q(\theta, \theta^{(t)})$
3. **Return:** $\hat{\theta}$

---

## 6.3 EM for Gaussian Mixture Models

### Log-Likelihood

For GMM with data $X = \{x_1, \ldots, x_N\}$:

$$\ell(\theta) = \sum_{n=1}^{N} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(x_n|\mu_k, \Sigma_k)$$

### E-Step

Compute responsibilities:

$$\gamma_{nk}^{(t)} = \frac{\pi_k^{(t)} \mathcal{N}(x_n|\mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^{K} \pi_j^{(t)} \mathcal{N}(x_n|\mu_j^{(t)}, \Sigma_j^{(t)})}$$

### M-Step

**Update mixing coefficients:**
$$\pi_k^{(t+1)} = \frac{1}{N}\sum_{n=1}^{N} \gamma_{nk}^{(t)} = \frac{N_k}{N}$$

where $N_k = \sum_{n=1}^{N} \gamma_{nk}^{(t)}$ is effective number of points in cluster $k$.

**Update means:**
$$\mu_k^{(t+1)} = \frac{1}{N_k}\sum_{n=1}^{N} \gamma_{nk}^{(t)} x_n$$

**Update covariances:**
$$\Sigma_k^{(t+1)} = \frac{1}{N_k}\sum_{n=1}^{N} \gamma_{nk}^{(t)} (x_n - \mu_k^{(t+1)})(x_n - \mu_k^{(t+1)})^T$$

### Complete Algorithm

1. **Initialize:** $\pi_k^{(0)}, \mu_k^{(0)}, \Sigma_k^{(0)}$ for $k = 1, \ldots, K$
2. **Repeat:**
   - **E-step:** Compute $\gamma_{nk}$ for all $n, k$
   - **M-step:** Update $\pi_k, \mu_k, \Sigma_k$ for all $k$
3. **Check convergence:** $|\ell(\theta^{(t+1)}) - \ell(\theta^{(t)})| < \epsilon$

### Initialization

**Common methods:**
- Random initialization
- K-means clustering
- Random subset of data points

### Convergence

- EM converges to a local maximum (not necessarily global)
- Multiple random initializations recommended
- Monitor log-likelihood to detect convergence

---

## 6.4 Applications

### Clustering

GMM can be used for **soft clustering**:
- Each point belongs to all clusters with probabilities $\gamma_{nk}$
- Hard assignment: assign to cluster with highest $\gamma_{nk}$

### Density Estimation

GMM provides a flexible density model:
$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$

### Advantages

- Flexible (can model complex distributions)
- Probabilistic (provides uncertainty estimates)
- Soft clustering (points can belong to multiple clusters)

### Limitations

- Requires number of components $K$
- Sensitive to initialization
- Can converge to local optima
- Computationally expensive for large $K$ or high dimensions

---

## 6.5 Model Selection

### Choosing Number of Components $K$

**Methods:**
1. **Cross-validation**
2. **Information Criteria:**
   - **AIC (Akaike Information Criterion):** $AIC = -2\ell + 2p$
   - **BIC (Bayesian Information Criterion):** $BIC = -2\ell + p\log N$
   where $p$ = number of parameters, $N$ = sample size
3. **Elbow method** (plot log-likelihood vs $K$)

### Regularization

To prevent overfitting:
- Add regularization to covariance matrices
- Use diagonal or spherical covariances
- Limit number of components

---

## 📝 Exam Tips

```{admonition} Important for Exam
:class: tip

1. **GMM Formula:** $p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$
2. **Responsibilities:** $\gamma_{nk} = p(z_k=1|x_n)$
3. **EM Algorithm:** E-step (compute responsibilities), M-step (update parameters)
4. **M-step Updates:** Know formulas for $\pi_k, \mu_k, \Sigma_k$
5. **Initialization:** Important for convergence
6. **Convergence:** EM finds local maximum, not global
7. **Model Selection:** Use AIC/BIC to choose $K$
```

---

## 🔍 Worked Examples

### Example 1: Responsibilities

GMM with $K=2$:
- $\pi_1 = 0.6$, $\pi_2 = 0.4$
- $\mu_1 = 0$, $\mu_2 = 5$
- $\sigma_1 = 1$, $\sigma_2 = 1$

Observation: $x = 2$

**Compute responsibilities:**

$$\gamma_1 = \frac{0.6 \times \mathcal{N}(2|0, 1)}{0.6 \times \mathcal{N}(2|0, 1) + 0.4 \times \mathcal{N}(2|5, 1)}$$

$$\mathcal{N}(2|0, 1) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}(2-0)^2} = \frac{1}{\sqrt{2\pi}} e^{-2} \approx 0.054$$

$$\mathcal{N}(2|5, 1) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}(2-5)^2} = \frac{1}{\sqrt{2\pi}} e^{-4.5} \approx 0.004$$

$$\gamma_1 = \frac{0.6 \times 0.054}{0.6 \times 0.054 + 0.4 \times 0.004} = \frac{0.0324}{0.0324 + 0.0016} = 0.953$$

$$\gamma_2 = 1 - 0.953 = 0.047$$

### Example 2: M-Step Update

Given responsibilities for $N=100$ points:
- $N_1 = \sum \gamma_{n1} = 60$
- $N_2 = \sum \gamma_{n2} = 40$

**Update mixing coefficients:**
$$\pi_1 = \frac{60}{100} = 0.6, \quad \pi_2 = \frac{40}{100} = 0.4$$

**Update means:**
$$\mu_1 = \frac{1}{60}\sum_{n=1}^{100} \gamma_{n1} x_n, \quad \mu_2 = \frac{1}{40}\sum_{n=1}^{100} \gamma_{n2} x_n$$

---

## 📚 Quick Revision Checklist

- [ ] GMM model definition and parameters
- [ ] Latent variables and responsibilities
- [ ] EM algorithm framework (E-step, M-step)
- [ ] EM for GMM (E-step: responsibilities, M-step: parameter updates)
- [ ] Initialization strategies
- [ ] Convergence properties
- [ ] Applications (clustering, density estimation)
- [ ] Model selection (choosing $K$)
- [ ] Advantages and limitations

