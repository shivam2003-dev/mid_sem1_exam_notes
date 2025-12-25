# Machine Learning - Mid-Semester Examination (December 2025) - Solved

This document contains detailed solutions to the Regular Mid-Semester Examination for Machine Learning conducted in December 2025.

---

## **Q1: IPL Playoff Prediction (Logistic Regression)**

An IPL franchise wants to predict playoff qualification based on league stage performance (8 matches).

### **Training Dataset:**

| Team | Mean Batsmen Played | Total Runs Scored | Total Wickets Taken | Qualified (Yes=1, No=0) |
|------|---------------------|-------------------|---------------------|-------------------------|
| **A** | 7 | 820 | 40 | 1 |
| **B** | 5 | 840 | 44 | 0 |
| **C** | **? (Missing)** | 860 | 48 | 1 |
| **D** | 8 | 880 | 52 | 1 |
| **E** | 4 | 900 | 56 | 0 |

---

### **Q1(a): Handling Missing Values [1 Mark]**

**Problem:** Suggest appropriate methods to handle the missing value in "Mean Number of Batsmen Played" and justify your choice.

**Solution:**

Logistic regression cannot directly use missing values. Suitable imputation methods:

**Method 1: Mean/Median Imputation (Recommended)**
- Calculate mean or median of available values: {7, 5, 8, 4}
- **Mean:** $(7 + 5 + 8 + 4) / 4 = 24 / 4 = 6.0$
- **Median:** Sort {4, 5, 7, 8} → Median = $(5 + 7) / 2 = 6.0$

**Justification:**
- ✓ Simple and stable for small datasets
- ✓ Doesn't introduce bias if data is MCAR (Missing Completely At Random)
- ✓ Preserves distribution properties
- ✓ Computationally cheap

**Method 2: KNN Imputation**
- Use k-nearest neighbors based on other features (Runs, Wickets)
- Find teams with similar Runs and Wickets, use their mean batsmen
- More sophisticated but may overfit with only 5 samples

**Method 3: Regression Imputation**
- Predict missing value using linear regression on other features
- Can capture relationships but risks overfitting with limited data

**Method 4: Mode (if categorical)**
- Not applicable here since it's a continuous numeric feature

**Best Choice for this problem:**
**Mean imputation with value 6.0** because:
- Dataset is very small (only 5 samples)
- Simple methods are more robust
- Both mean and median converge to 6.0

**Answer:**
Use **mean/median imputation**. Impute Team C's "Mean Batsmen Played" = **6.0**

**Caveat:** With such a small dataset, any imputation introduces uncertainty. If possible, collect actual data instead.

---

### **Q1(b): Z-Score Normalization [3 Marks]**

**Problem:** Normalize only "Total Runs" and "Total Wickets" using z-score normalization. Compute mean and standard deviation.

**Solution:**

**Z-score formula:**
$$z = \frac{x - \mu}{\sigma}$$

where $\mu$ = mean, $\sigma$ = standard deviation

**Step 1: Normalize Total Runs**

Raw values: {820, 840, 860, 880, 900}

**Calculate mean:**
$$\mu_{\text{runs}} = \frac{820 + 840 + 860 + 880 + 900}{5} = \frac{4300}{5} = 860$$

**Calculate standard deviation (population):**
$$\sigma = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \mu)^2}{n}}$$

$$\sigma_{\text{runs}} = \sqrt{\frac{(820-860)^2 + (840-860)^2 + (860-860)^2 + (880-860)^2 + (900-860)^2}{5}}$$

$$= \sqrt{\frac{(-40)^2 + (-20)^2 + 0^2 + 20^2 + 40^2}{5}}$$

$$= \sqrt{\frac{1600 + 400 + 0 + 400 + 1600}{5}} = \sqrt{\frac{4000}{5}} = \sqrt{800} = 20\sqrt{2} \approx 28.2843$$

**Normalized values:**
- Team A: $(820 - 860) / 28.2843 = -40 / 28.2843 = -1.4142$
- Team B: $(840 - 860) / 28.2843 = -20 / 28.2843 = -0.7071$
- Team C: $(860 - 860) / 28.2843 = 0 / 28.2843 = 0.0000$
- Team D: $(880 - 860) / 28.2843 = 20 / 28.2843 = 0.7071$
- Team E: $(900 - 860) / 28.2843 = 40 / 28.2843 = 1.4142$

**Step 2: Normalize Total Wickets**

Raw values: {40, 44, 48, 52, 56}

**Calculate mean:**
$$\mu_{\text{wickets}} = \frac{40 + 44 + 48 + 52 + 56}{5} = \frac{240}{5} = 48$$

**Calculate standard deviation:**
$$\sigma_{\text{wickets}} = \sqrt{\frac{(40-48)^2 + (44-48)^2 + (48-48)^2 + (52-48)^2 + (56-48)^2}{5}}$$

$$= \sqrt{\frac{64 + 16 + 0 + 16 + 64}{5}} = \sqrt{\frac{160}{5}} = \sqrt{32} = 4\sqrt{2} \approx 5.6569$$

**Normalized values:**
- Team A: $(40 - 48) / 5.6569 = -8 / 5.6569 = -1.4142$
- Team B: $(44 - 48) / 5.6569 = -4 / 5.6569 = -0.7071$
- Team C: $(48 - 48) / 5.6569 = 0 / 5.6569 = 0.0000$
- Team D: $(52 - 48) / 5.6569 = 4 / 5.6569 = 0.7071$
- Team E: $(56 - 48) / 5.6569 = 8 / 5.6569 = 1.4142$

**Step 3: Updated Dataset**

| Team | Mean Batsmen | Runs (z-score) | Wickets (z-score) | Qualified |
|------|-------------|----------------|-------------------|-----------|
| A | 7 | -1.4142 | -1.4142 | 1 |
| B | 5 | -0.7071 | -0.7071 | 0 |
| C | 6 | 0.0000 | 0.0000 | 1 |
| D | 8 | 0.7071 | 0.7071 | 1 |
| E | 4 | 1.4142 | 1.4142 | 0 |

**Answer:**
- **Mean for Total Runs:** $\mu = 860$
- **Std Dev for Total Runs:** $\sigma = 28.28$
- **Mean for Total Wickets:** $\mu = 48$
- **Std Dev for Total Wickets:** $\sigma = 5.66$
- Updated dataset provided above

---

### **Q1(c): Batch Gradient Descent [4 Marks]**

**Problem:** Perform one iteration of Batch Gradient Descent given:
- Initial weights: $\theta = [0.1, 0.1, 0.1, 0.1]$
- Learning rate: $\alpha = 0.01$

**Solution:**

**Logistic Regression Formulation:**

Feature vector: $x = [1, x_1, x_2, x_3]$ where:
- $x_0 = 1$ (bias term)
- $x_1$ = Mean Batsmen Played
- $x_2$ = Runs (z-normalized)
- $x_3$ = Wickets (z-normalized)

**Hypothesis:** $h_\theta(x) = \sigma(\theta^T x)$ where $\sigma(z) = \frac{1}{1 + e^{-z}}$

**Step 1: Calculate predictions for each team**

**Team A:**
$$z_A = 0.1(1) + 0.1(7) + 0.1(-1.4142) + 0.1(-1.4142)$$
$$= 0.1 + 0.7 - 0.14142 - 0.14142 = 0.51716$$

$$h_A = \sigma(0.51716) = \frac{1}{1 + e^{-0.51716}} = \frac{1}{1.59534} = 0.6265$$

**Team B:**
$$z_B = 0.1(1) + 0.1(5) + 0.1(-0.7071) + 0.1(-0.7071)$$
$$= 0.1 + 0.5 - 0.07071 - 0.07071 = 0.45858$$

$$h_B = \sigma(0.45858) = \frac{1}{1 + e^{-0.45858}} = 0.6127$$

**Team C:**
$$z_C = 0.1(1) + 0.1(6) + 0.1(0) + 0.1(0)$$
$$= 0.1 + 0.6 = 0.7$$

$$h_C = \sigma(0.7) = \frac{1}{1 + e^{-0.7}} = 0.6682$$

**Team D:**
$$z_D = 0.1(1) + 0.1(8) + 0.1(0.7071) + 0.1(0.7071)$$
$$= 0.1 + 0.8 + 0.07071 + 0.07071 = 1.04142$$

$$h_D = \sigma(1.04142) = \frac{1}{1 + e^{-1.04142}} = 0.7391$$

**Team E:**
$$z_E = 0.1(1) + 0.1(4) + 0.1(1.4142) + 0.1(1.4142)$$
$$= 0.1 + 0.4 + 0.14142 + 0.14142 = 0.78284$$

$$h_E = \sigma(0.78284) = \frac{1}{1 + e^{-0.78284}} = 0.6863$$

**Step 2: Calculate errors**

$$\text{error}_i = h_i - y_i$$

- Team A: $0.6265 - 1 = -0.3735$
- Team B: $0.6127 - 0 = 0.6127$
- Team C: $0.6682 - 1 = -0.3318$
- Team D: $0.7391 - 1 = -0.2609$
- Team E: $0.6863 - 0 = 0.6863$

**Step 3: Calculate gradients**

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_i - y_i) x_{ij}$$

where $m = 5$ (number of examples)

**For $\theta_0$ (bias, $x_0 = 1$ for all):**
$$\frac{\partial J}{\partial \theta_0} = \frac{1}{5}(-0.3735 + 0.6127 - 0.3318 - 0.2609 + 0.6863)$$
$$= \frac{1}{5}(0.3328) = 0.06656$$

**For $\theta_1$ (Mean Batsmen):**
$$\frac{\partial J}{\partial \theta_1} = \frac{1}{5}[(-0.3735)(7) + (0.6127)(5) + (-0.3318)(6) + (-0.2609)(8) + (0.6863)(4)]$$
$$= \frac{1}{5}[-2.6145 + 3.0635 - 1.9908 - 2.0872 + 2.7452]$$
$$= \frac{1}{5}(-0.8838) = -0.17676$$

**For $\theta_2$ (Normalized Runs):**
$$\frac{\partial J}{\partial \theta_2} = \frac{1}{5}[(-0.3735)(-1.4142) + (0.6127)(-0.7071) + 0 + (-0.2609)(0.7071) + (0.6863)(1.4142)]$$
$$= \frac{1}{5}[0.5282 - 0.4333 + 0 - 0.1845 + 0.9706]$$
$$= \frac{1}{5}(0.8810) = 0.17620$$

**For $\theta_3$ (Normalized Wickets):**
$$\frac{\partial J}{\partial \theta_3} = \frac{1}{5}[(-0.3735)(-1.4142) + (0.6127)(-0.7071) + 0 + (-0.2609)(0.7071) + (0.6863)(1.4142)]$$
$$= \frac{1}{5}(0.8810) = 0.17620$$

**Step 4: Update weights**

$$\theta_j^{\text{new}} = \theta_j^{\text{old}} - \alpha \cdot \frac{\partial J}{\partial \theta_j}$$

$$\theta_0^{\text{new}} = 0.1 - 0.01(0.06656) = 0.1 - 0.0006656 = 0.09933$$

$$\theta_1^{\text{new}} = 0.1 - 0.01(-0.17676) = 0.1 + 0.0017676 = 0.10177$$

$$\theta_2^{\text{new}} = 0.1 - 0.01(0.17620) = 0.1 - 0.0017620 = 0.09824$$

$$\theta_3^{\text{new}} = 0.1 - 0.01(0.17620) = 0.1 - 0.0017620 = 0.09824$$

**Answer:**

Updated weights after one iteration:
$$\theta = [0.09933, 0.10177, 0.09824, 0.09824]$$

**Interpretation:**
- $\theta_1$ increased → model learned that more batsmen correlates with qualification
- $\theta_2, \theta_3$ decreased slightly → adjusting based on z-scored features

---

### **Q1(d): Cross-Entropy Loss [1 Mark]**

**Problem:** Calculate Cross-Entropy Loss after the first iteration.

**Solution:**

**Binary Cross-Entropy Loss formula:**
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(h_i) + (1 - y_i) \log(1 - h_i)]$$

Using new weights $\theta = [0.09933, 0.10177, 0.09824, 0.09824]$:

**Recalculate predictions with updated weights:**

**Team A:**
$$z_A = 0.09933 + 0.10177(7) + 0.09824(-1.4142) + 0.09824(-1.4142)$$
$$= 0.09933 + 0.71239 - 0.13888 - 0.13888 = 0.53396$$
$$h_A = \sigma(0.53396) = 0.6304$$

**Team B:**
$$z_B = 0.09933 + 0.10177(5) - 0.09824(0.7071) - 0.09824(0.7071)$$
$$= 0.09933 + 0.50885 - 0.06947 - 0.06947 = 0.46924$$
$$h_B = \sigma(0.46924) = 0.6152$$

**Team C:**
$$z_C = 0.09933 + 0.10177(6) + 0 + 0 = 0.70995$$
$$h_C = \sigma(0.70995) = 0.6704$$

**Team D:**
$$z_D = 0.09933 + 0.10177(8) + 0.09824(0.7071) + 0.09824(0.7071)$$
$$= 0.09933 + 0.81416 + 0.06947 + 0.06947 = 1.05243$$
$$h_D = \sigma(1.05243) = 0.7412$$

**Team E:**
$$z_E = 0.09933 + 0.10177(4) + 0.09824(1.4142) + 0.09824(1.4142)$$
$$= 0.09933 + 0.40708 + 0.13888 + 0.13888 = 0.78417$$
$$h_E = \sigma(0.78417) = 0.6866$$

**Calculate loss:**

$$J = -\frac{1}{5}[1 \cdot \log(0.6304) + 0 \cdot \log(1-0.6304) + \ldots]$$

$$= -\frac{1}{5}[\log(0.6304) + \log(1-0.6152) + \log(0.6704) + \log(0.7412) + \log(1-0.6866)]$$

$$= -\frac{1}{5}[\log(0.6304) + \log(0.3848) + \log(0.6704) + \log(0.7412) + \log(0.3134)]$$

$$= -\frac{1}{5}[-0.4605 - 0.9545 - 0.3998 - 0.2995 - 1.1616]$$

$$= -\frac{1}{5}(-3.2759) = 0.6552$$

**Answer:** Cross-Entropy Loss = **0.655**

**Note:** Loss should decrease with training. Initial loss (before any training) would be higher.

---

### **Q1(e): Predicted Probability for Team D [1 Mark]**

**Problem:** Calculate and interpret predicted probability of playoff qualification for Team D after first iteration.

**Solution:**

From Q1(d), we calculated:
$$h_D = 0.7412$$

**Answer:**

The predicted probability that Team D qualifies for playoffs is **0.741 or 74.1%**.

**Interpretation:**
- Model assigns high confidence (>70%) that Team D will qualify
- This matches the true label (Team D did qualify, $y_D = 1$)
- Since probability > 0.5, model correctly predicts "Qualified" for Team D
- The model learned that Team D's strong performance (880 runs, 52 wickets, 8 batsmen) indicates playoff-worthy performance

**Decision rule:**
- If $h_D \geq 0.5$: Predict "Qualified" (1)
- If $h_D < 0.5$: Predict "Not Qualified" (0)

Since 0.741 > 0.5, prediction = **Qualified** ✓

---

## **Q2: Bias-Variance & Overfitting**

**Scenario:** A student claims: *"I doubled my dataset size, but validation RMSE stayed high while training RMSE kept dropping. Therefore, increasing dataset size doesn't help."*

### **Q2(a): Bias and Variance Classification [1 Mark]**

**Problem:** Comment whether the model has (low/high) Bias and (low/high) Variance.

**Solution:**

**Analysis of symptoms:**
- Training RMSE keeps dropping → model fits training data well
- Validation RMSE stays high → model doesn't generalize to new data
- Gap between training and validation error is large

**Diagnosis:**

**Bias:** **LOW**
- Training RMSE is low → model has sufficient capacity to fit training data
- Model is not underfitting
- Can capture complex patterns in training set

**Variance:** **HIGH**
- Large gap between training and validation error
- Model memorizes training data but fails on validation
- Classic sign of overfitting
- Model is too sensitive to training data fluctuations

**Answer:**
- **Bias: LOW**
- **Variance: HIGH**

This is the **high variance (overfitting) regime** of the bias-variance tradeoff.

---

### **Q2(b): Likely Reason [1 Mark]**

**Problem:** Provide one likely reason for this observation.

**Solution:**

**Reason:** The model is **too complex** relative to the signal-to-noise ratio in the data.

**Detailed explanation:**

Even though the dataset size doubled, the model's excessive complexity allows it to fit noise and spurious patterns. Possible specific causes:

1. **Model has too many parameters** relative to useful signal
   - E.g., polynomial degree too high, too many features
   - More data doesn't help if model continues to memorize

2. **New data has similar noise/outliers**
   - Doubling noisy data just provides more noise to fit
   - Model learns dataset-specific artifacts instead of true patterns

3. **Features don't have strong predictive power**
   - Even with more samples, weak features → model fits noise
   - Need better features, not just more data

4. **Insufficient regularization**
   - Model not penalized for complexity
   - Allowed to fit arbitrarily complex patterns

**Best single answer:**

"The model is **overly complex** (has high capacity) and is fitting noise in the training data. Doubling the dataset size doesn't address the fundamental issue of model complexity exceeding the true signal complexity."

**Answer:** The model has **too many parameters/high complexity** relative to the amount of useful signal in the features, causing it to fit noise even with more data.

---

### **Q2(c): Corrective Action [1 Mark]**

**Problem:** Suggest one corrective action for this model.

**Solution:**

**Recommended action:** **Add regularization (L2 or L1)**

**How it helps:**
- Penalizes large weights → prevents overfitting to training noise
- Formula for L2 (Ridge): $J(\theta) = \text{Loss} + \lambda \sum_{i} \theta_i^2$
- Forces model to use simpler patterns
- Reduces effective model complexity without changing architecture

**Alternative corrective actions (any one is acceptable):**

**1. Simplify the model**
- Reduce number of features (feature selection)
- Lower polynomial degree
- Use fewer layers/neurons in neural network
- **Effect:** Directly reduces model capacity

**2. Early stopping**
- Stop training when validation error starts increasing
- Prevents model from continuing to overfit
- **Effect:** Finds sweet spot in training process

**3. Increase dropout rate** (for neural networks)
- Randomly disable neurons during training
- Prevents co-adaptation of features
- **Effect:** Ensemble-like regularization

**4. Data augmentation**
- Create synthetic variations of training data
- Increases effective dataset size with meaningful variations
- **Effect:** Provides more diverse training signal

**5. Cross-validation with model selection**
- Try multiple model complexities
- Select based on validation performance
- **Effect:** Finds optimal complexity empirically

**Answer:** **Add L2/L1 regularization** to penalize model complexity and reduce overfitting.

(Any one of the above solutions is acceptable for 1 mark)

---

## **Q3: Decision Tree (Entropy & Information Gain)**

### **Training Dataset:**

| Age Group | Income Level | Occupation | Purchased |
|-----------|--------------|------------|-----------|
| Young | Low | Student | No |
| Middle-aged | High | Professional | Yes |
| Young | Medium | Student | Yes |
| Old | Low | Retired | No |
| Young | High | Professional | Yes |
| Middle-aged | Low | Professional | No |
| Old | Medium | Retired | Yes |
| Young | Medium | Professional | Yes |

---

### **Q3(i): Entropy of Target Class [2 Marks]**

**Problem:** Calculate entropy of "Purchased" for the entire dataset.

**Solution:**

**Entropy formula:**
$$H(S) = -\sum_{i=1}^{k} p_i \log_2(p_i)$$

where $p_i$ is the proportion of class $i$.

**Count classes:**
- **Yes:** 5 instances (rows 2, 3, 5, 7, 8)
- **No:** 3 instances (rows 1, 4, 6)
- **Total:** 8 instances

**Calculate proportions:**
$$p_{\text{Yes}} = \frac{5}{8}, \quad p_{\text{No}} = \frac{3}{8}$$

**Calculate entropy:**
$$H(S) = -\frac{5}{8} \log_2\left(\frac{5}{8}\right) - \frac{3}{8} \log_2\left(\frac{3}{8}\right)$$

$$\log_2\left(\frac{5}{8}\right) = \log_2(5) - \log_2(8) = 2.3219 - 3 = -0.6781$$

$$\log_2\left(\frac{3}{8}\right) = \log_2(3) - \log_2(8) = 1.5850 - 3 = -1.4150$$

$$H(S) = -\frac{5}{8}(-0.6781) - \frac{3}{8}(-1.4150)$$

$$= \frac{5 \times 0.6781}{8} + \frac{3 \times 1.4150}{8}$$

$$= \frac{3.3905}{8} + \frac{4.2450}{8} = \frac{7.6355}{8} = 0.9544$$

**Answer:** Entropy = **0.9544 bits**

**Interpretation:**
- Maximum entropy for 50-50 split would be 1.0 bit
- 0.9544 is close to maximum → dataset is fairly mixed
- High entropy → good opportunity for information gain through splits

---

### **Q3(ii): Entropy of "Young" Branch [3 Marks]**

**Problem:**
1. Calculate entropy for "Young" branch if Age Group is root node
2. Is this branch pure?
3. If not, which attribute to choose next?

**Solution:**

**Part 1: Calculate entropy of Young subset**

**Young subset (Age Group = Young):**
| Row | Income | Occupation | Purchased |
|-----|--------|------------|-----------|
| 1 | Low | Student | No |
| 3 | Medium | Student | Yes |
| 5 | High | Professional | Yes |
| 8 | Medium | Professional | Yes |

**Count:** 4 instances
- Yes: 3
- No: 1

**Calculate entropy:**
$$H(\text{Young}) = -\frac{3}{4} \log_2\left(\frac{3}{4}\right) - \frac{1}{4} \log_2\left(\frac{1}{4}\right)$$

$$\log_2\left(\frac{3}{4}\right) = \log_2(3) - 2 = 1.5850 - 2 = -0.4150$$

$$\log_2\left(\frac{1}{4}\right) = -2$$

$$H(\text{Young}) = -\frac{3}{4}(-0.4150) - \frac{1}{4}(-2)$$

$$= \frac{3 \times 0.4150}{4} + \frac{2}{4} = \frac{1.2450}{4} + 0.5 = 0.3113 + 0.5 = 0.8113$$

**Answer:** Entropy = **0.8113 bits**

**Part 2: Is branch pure?**

**No, the branch is NOT pure** because:
- Pure branch would have entropy = 0 (all same class)
- This branch has entropy = 0.8113 (still mixed)
- Contains both "Yes" (3) and "No" (1)

**Part 3: Next attribute to split**

**Option 1: Split by Income Level**

Young subset split by Income:
- **Low:** {No} → 1 instance, all No → Entropy = 0 (pure!)
- **Medium:** {Yes, Yes} → 2 instances, all Yes → Entropy = 0 (pure!)
- **High:** {Yes} → 1 instance, all Yes → Entropy = 0 (pure!)

**Weighted entropy after Income split:**
$$H(\text{Young}|\text{Income}) = \frac{1}{4}(0) + \frac{2}{4}(0) + \frac{1}{4}(0) = 0$$

**Information Gain:**
$$\text{IG}(\text{Income}) = 0.8113 - 0 = 0.8113$$

**Option 2: Split by Occupation**

Young subset split by Occupation:
- **Student:** {No, Yes} → 2 instances
  - Entropy = $-\frac{1}{2}\log_2(\frac{1}{2}) - \frac{1}{2}\log_2(\frac{1}{2}) = 1.0$
- **Professional:** {Yes, Yes} → 2 instances, all Yes
  - Entropy = 0

**Weighted entropy:**
$$H(\text{Young}|\text{Occupation}) = \frac{2}{4}(1.0) + \frac{2}{4}(0) = 0.5$$

**Information Gain:**
$$\text{IG}(\text{Occupation}) = 0.8113 - 0.5 = 0.3113$$

**Comparison:**
- Income IG = 0.8113 (achieves perfect purity!)
- Occupation IG = 0.3113

**Answer:** Choose **Income Level** as the next attribute because:
- ✓ Achieves perfect classification (all leaves pure)
- ✓ Maximum information gain (0.8113 > 0.3113)
- ✓ No further splits needed after this

**Final tree structure:**
```
Age Group = Young
  ├─ Income = Low → No
  ├─ Income = Medium → Yes
  └─ Income = High → Yes
```

---

### **Q3(iii): Compare Occupation vs Income Level [2 Marks]**

**Problem:** Without full IG calculations, argue which attribute better separates "Yes" and "No" classes.

**Solution:**

**Qualitative Analysis:**

**Income Level:**
- **Low income:** All instances → No (100% No)
  - Rows 1, 4, 6
- **Medium income:** Mixed but mostly Yes
  - Rows 3, 7, 8
- **High income:** All instances → Yes (100% Yes)
  - Rows 2, 5

**Pattern:** Clear monotonic relationship
- Low → No
- Medium → Mixed
- High → Yes

**Strong separability!**

**Occupation:**
- **Student:** Mixed (1 No, 1 Yes)
  - Rows 1, 3
- **Professional:** Mixed (2 Yes, 1 No)
  - Rows 2, 5, 6, 8
- **Retired:** Mixed (1 Yes, 1 No)
  - Rows 4, 7

**Pattern:** No clear separation by occupation alone

**Argument:**

**Income Level is better** because:

1. **Clear decision boundary:**
   - Low income → predominantly No purchases
   - High income → predominantly Yes purchases
   - Occupation shows no such pattern

2. **Economic intuition:**
   - Income directly correlates with purchasing power
   - Higher income → more likely to purchase (makes business sense)
   - Occupation is a proxy for income but noisier

3. **Purity of subsets:**
   - Income splits create purer subsets
   - Low: all No, High: all Yes
   - Occupation: all categories are mixed

4. **Information content:**
   - Income carries most information about target
   - Occupation has overlapping class distributions

**Visual comparison:**
```
Income:     Low [N,N,N]    Medium [Y,N,Y]    High [Y,Y]
Occupation: Stu [N,Y]      Prof [Y,Y,N,Y]    Ret [N,Y]
```

Income has clearer class separation!

**Answer:** **Income Level** separates classes better because it shows a clear pattern (Low→No, High→Yes), while Occupation has mixed results across all categories with no clear discriminative power.

---

### **Q3(iv): Classify New Customer [1 Mark]**

**Problem:** New customer:
- Age: Young
- Income: Low
- Occupation: Professional

Classify using **majority voting in "Young" subset only**.

**Solution:**

**Young subset composition:**
| Income | Occupation | Purchased |
|--------|------------|-----------|
| Low | Student | No |
| Medium | Student | Yes |
| High | Professional | Yes |
| Medium | Professional | Yes |

**Majority vote:**
- Yes: 3 instances (75%)
- No: 1 instance (25%)

**Answer:** Classify as **Purchased = Yes** (majority class in Young subset).

**Note:** This is a naive approach. A better approach would follow the decision tree:
- Since Income = Low, and our earlier analysis showed Low income → No, the correct prediction should be **No**.

But the question specifically asks for **majority voting in Young subset**, so answer is **Yes**.

---

## **Q4: Model Evaluation (Posterior Probabilities)**

### **Test Data:**

| Instance | True Class | M1 $P(+)$ | M2 $P(+)$ |
|----------|------------|-----------|-----------|
| 1 | + | 0.73 | 0.61 |
| 2 | + | 0.69 | 0.03 |
| 3 | - | 0.44 | 0.68 |
| 4 | - | 0.55 | 0.31 |
| 5 | + | 0.67 | 0.45 |
| 6 | + | 0.47 | 0.09 |
| 7 | - | 0.08 | 0.38 |
| 8 | - | 0.15 | 0.05 |
| 9 | + | 0.45 | 0.01 |
| 10 | - | 0.35 | 0.04 |

**Threshold:** $t = 0.5$ (predict + if $P(+) > 0.5$)

---

### **Q4(a): Confusion Matrices [2 Marks]**

**Solution:**

**Model M1:**

Apply threshold $t = 0.5$:

| Instance | True | $P(+)$ | Predicted | Outcome |
|----------|------|--------|-----------|---------|
| 1 | + | 0.73 | + | TP |
| 2 | + | 0.69 | + | TP |
| 3 | - | 0.44 | - | TN |
| 4 | - | 0.55 | + | FP |
| 5 | + | 0.67 | + | TP |
| 6 | + | 0.47 | - | FN |
| 7 | - | 0.08 | - | TN |
| 8 | - | 0.15 | - | TN |
| 9 | + | 0.45 | - | FN |
| 10 | - | 0.35 | - | TN |

**Counts:**
- TP = 3 (instances 1, 2, 5)
- FP = 1 (instance 4)
- TN = 4 (instances 3, 7, 8, 10)
- FN = 2 (instances 6, 9)

**Confusion Matrix for M1:**
```
                 Predicted
              Positive  Negative
Actual  Pos      3         2       (5 total positive)
        Neg      1         4       (5 total negative)
```

**Model M2:**

| Instance | True | $P(+)$ | Predicted | Outcome |
|----------|------|--------|-----------|---------|
| 1 | + | 0.61 | + | TP |
| 2 | + | 0.03 | - | FN |
| 3 | - | 0.68 | + | FP |
| 4 | - | 0.31 | - | TN |
| 5 | + | 0.45 | - | FN |
| 6 | + | 0.09 | - | FN |
| 7 | - | 0.38 | - | TN |
| 8 | - | 0.05 | - | TN |
| 9 | + | 0.01 | - | FN |
| 10 | - | 0.04 | - | TN |

**Counts:**
- TP = 1 (instance 1)
- FP = 1 (instance 3)
- TN = 4 (instances 4, 7, 8, 10)
- FN = 4 (instances 2, 5, 6, 9)

**Confusion Matrix for M2:**
```
                 Predicted
              Positive  Negative
Actual  Pos      1         4       (5 total positive)
        Neg      1         4       (5 total negative)
```

**Answer:**

**M1 Confusion Matrix:**
| | Pred + | Pred - |
|---|---:|---:|
| Actual + | 3 | 2 |
| Actual - | 1 | 4 |

**M2 Confusion Matrix:**
| | Pred + | Pred - |
|---|---:|---:|
| Actual + | 1 | 4 |
| Actual - | 1 | 4 |

---

### **Q4(b): Precision, Recall, and F-measure [3 Marks]**

**Solution:**

**Formulas:**
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$\text{F-measure} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**For Model M1:**

**Precision:**
$$P_{M1} = \frac{3}{3 + 1} = \frac{3}{4} = 0.75$$

**Interpretation:** Of all instances M1 predicted as positive, 75% were actually positive.

**Recall (Sensitivity):**
$$R_{M1} = \frac{3}{3 + 2} = \frac{3}{5} = 0.60$$

**Interpretation:** Of all actual positive instances, M1 correctly identified 60%.

**F-measure (F1-score):**
$$F_{M1} = \frac{2 \times 0.75 \times 0.60}{0.75 + 0.60} = \frac{0.90}{1.35} = 0.6667$$

**For Model M2:**

**Precision:**
$$P_{M2} = \frac{1}{1 + 1} = \frac{1}{2} = 0.50$$

**Interpretation:** Of all instances M2 predicted as positive, only 50% were actually positive.

**Recall:**
$$R_{M2} = \frac{1}{1 + 4} = \frac{1}{5} = 0.20$$

**Interpretation:** M2 only caught 20% of actual positive instances.

**F-measure:**
$$F_{M2} = \frac{2 \times 0.50 \times 0.20}{0.50 + 0.20} = \frac{0.20}{0.70} = 0.2857$$

**Answer:**

**Model M1:**
- Precision: **0.75** (75%)
- Recall: **0.60** (60%)
- F-measure: **0.667**

**Model M2:**
- Precision: **0.50** (50%)
- Recall: **0.20** (20%)
- F-measure: **0.286**

**Comparison:**
- M1 outperforms M2 on all metrics
- M1 has better balance between precision and recall
- M2 is too conservative (predicts negative too often → low recall)
- **Deploy Model M1** for better overall performance

**Additional metrics:**
- **M1 Accuracy:** $(3+4)/10 = 0.70$ (70%)
- **M2 Accuracy:** $(1+4)/10 = 0.50$ (50%)

---

## **Q5: Regression Model Interpretation**

**Given model:**
$$\text{House Price} = 50 + 200 \times (\text{Area}) + 5 \times (\text{Age})$$

---

### **Q5(a): Interpret Coefficients [1 Mark]**

**Solution:**

**Intercept (50):**
- Base price when Area = 0 and Age = 0
- Not interpretable in practice (houses have area and age)
- Sets baseline for price calculation

**Area Coefficient (200):**
- **Interpretation:** For each unit increase in Area (holding Age constant), the predicted house price increases by **200 units**
- If Area is in m², each additional square meter adds $200 (assuming price in $)
- **Positive coefficient:** Larger houses cost more (expected)

**Age Coefficient (5):**
- **Interpretation:** For each year increase in Age (holding Area constant), the predicted house price increases by **5 units**
- Surprising positive sign! Usually older houses cost less
- Possible explanations:
  - Age might correlate with location quality (older neighborhoods)
  - Vintage/heritage value
  - Coefficient is small (5 vs 200), so minimal impact

**Answer:**
- **Area (200):** Each unit increase in area increases price by 200, holding age constant
- **Age (5):** Each year increase in age increases price by 5, holding area constant

---

### **Q5(b): Unit Change Effect [1 Mark]**

**Problem:** What happens if Area units change from m² to ft²?

**Solution:**

**Conversion:** 1 m² ≈ 10.7639 ft²

**Current model (Area in m²):**
$$\text{Price} = 50 + 200 \times (\text{Area}_{m²}) + 5 \times (\text{Age})$$

**After unit change (Area in ft²):**

If a house has area $A$ m² = $A \times 10.7639$ ft², the coefficient must change to maintain same predictions:

$$200 \times A = \beta_{\text{new}} \times (A \times 10.7639)$$

$$\beta_{\text{new}} = \frac{200}{10.7639} \approx 18.58$$

**New model (Area in ft²):**
$$\text{Price} = 50 + 18.58 \times (\text{Area}_{ft²}) + 5 \times (\text{Age})$$

**Answer:**

The Area coefficient **changes from 200 to approximately 18.58**.

**Key points:**
- Coefficient magnitude **inversely proportional** to unit size
- Predictions remain unchanged (same house → same price)
- Only the **numeric value** of coefficient changes, not the relationship
- Always specify units when reporting coefficients!

---

### **Q5(c): Gradient Descent Inference [1 Mark]**

**Problem:** If $\frac{\partial J}{\partial \theta_1}$ is large while $\frac{\partial J}{\partial \theta_0}$ is near zero, what can you infer?

**Solution:**

**Interpretation of gradients:**

**Large $\frac{\partial J}{\partial \theta_1}$:**
- Loss function is very sensitive to $\theta_1$ changes
- Current $\theta_1$ value is far from optimal
- Need significant update to $\theta_1$
- Steep slope in $\theta_1$ direction

**Near-zero $\frac{\partial J}{\partial \theta_0}$:**
- Loss function barely changes with $\theta_0$
- Current $\theta_0$ is close to optimal
- Already at/near a local optimum for $\theta_0$
- Flat region in $\theta_0$ direction

**Geometric visualization:**
```
   Loss
    ↑
    |     /  ← steep in θ₁ direction
    |    /
    |___/___→ θ₁
    |______→ θ₀ (nearly flat)
```

**Implications:**
1. **Next iteration:** $\theta_1$ will update significantly, $\theta_0$ minimally
2. **Convergence:** $\theta_0$ has likely converged, $\theta_1$ still adjusting
3. **Learning rate:** Might want different learning rates per parameter (adaptive methods like Adam)

**Answer:**

Parameter $\theta_1$ is **far from its optimal value** and needs significant adjustment, while $\theta_0$ is **close to optimal** and requires little/no change. This suggests $\theta_1$ is the main source of current prediction error.

---

### **Q5(d): Feature Influence Comparison [1 Mark]**

**Problem:** Which feature has stronger influence on house price: Area or Age?

**Solution:**

**Comparison of coefficients:**
- Area: 200
- Age: 5

**At first glance:** Area coefficient is 40× larger (200 vs 5)

**But careful!** Must consider:
1. **Coefficient magnitude**
2. **Typical feature ranges**

**Example scenarios:**

**Scenario 1: Typical variation**
- Area varies: 50 m² to 200 m² (range = 150)
- Age varies: 0 to 50 years (range = 50)

**Price impact:**
- Area: $200 \times 150 = 30,000$ (variation due to area)
- Age: $5 \times 50 = 250$ (variation due to age)

Area impact is **120× larger**!

**Scenario 2: Standard deviations**
If:
- $\sigma_{\text{Area}} = 30$ m²
- $\sigma_{\text{Age}} = 10$ years

**Standardized coefficients:**
- Area: $200 \times 30 = 6,000$
- Age: $5 \times 10 = 50$

**Conclusion:**

**Area has MUCH stronger influence** on house price.

**Reasoning:**
- Even if Age varied wildly (0-100 years), impact = $5 \times 100 = 500$
- Moderate Area variation (100 m²) = $200 \times 100 = 20,000$
- Area coefficient is 40× larger AND typical area ranges are comparable to age ranges

**Answer:**

**Area** has stronger influence on house price because its coefficient (200) is 40 times larger than Age's coefficient (5), meaning each unit change in Area has 40× the impact on price compared to each year of Age.

**Formal answer using standardized coefficients:**

To fairly compare, compute $\beta_j \times \sigma_{x_j}$ (coefficient × feature std dev). Even with conservative estimates, Area's influence dominates.

---

## **Summary**

This examination comprehensively tested:

1. **Logistic Regression:**
   - Missing value imputation
   - Feature normalization (z-score)
   - Batch gradient descent mechanics
   - Cross-entropy loss computation
   - Probability interpretation

2. **Bias-Variance Tradeoff:**
   - Diagnosing overfitting from learning curves
   - Understanding model complexity issues
   - Regularization strategies

3. **Decision Trees:**
   - Entropy calculation
   - Information gain
   - Attribute selection
   - Tree construction

4. **Model Evaluation:**
   - Confusion matrices
   - Precision, recall, F1-score
   - Threshold-based classification

5. **Linear Regression:**
   - Coefficient interpretation
   - Unit scaling effects
   - Gradient analysis
   - Feature importance

**Key Takeaways:**
- Always handle missing data appropriately before modeling
- Normalize features for gradient-based algorithms
- High training accuracy ≠ good model (watch for overfitting)
- Choose evaluation metrics appropriate for the problem
- Interpret coefficients in context of feature scales
- Decision trees naturally handle feature selection via information gain

**Practical ML Skills Demonstrated:**
✓ Data preprocessing
✓ Model training algorithms
✓ Diagnostic analysis
✓ Performance evaluation
✓ Model interpretation
✓ Problem-specific reasoning
