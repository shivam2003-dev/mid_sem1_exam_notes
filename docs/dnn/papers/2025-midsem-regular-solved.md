# DNN Mid-Semester Examination (December 2025) - Solved

This document contains detailed solutions to the Regular Midsem Question Paper for December 2025, covering fundamental concepts in Deep Neural Networks including Perceptrons, Linear and Logistic Regression, Softmax, and Deep Feedforward Neural Networks.

---

## **Q.1: Perceptron & Spam Classification**

### **Q.1(a) Computation and Weight Update [6 Marks]**

**Given:**
- Current weights: $w_0 = 0.5$ (bias), $w_1 = -0.3$, $w_2 = 0.4$
- Training example: $x_1 = 1$, $x_2 = 2$, true label $y = 1$
- Learning rate: $\eta = 0.1$
- Step activation: outputs 1 if $z \geq 0$, else 0

**Solution:**

**Step 1: Calculate weighted sum $z$**

$$z = w_0 \cdot x_0 + w_1 \cdot x_1 + w_2 \cdot x_2$$

where $x_0 = 1$ (bias input)

$$z = 0.5(1) + (-0.3)(1) + 0.4(2) = 0.5 - 0.3 + 0.8 = 1.0$$

**Step 2: Predicted output using step activation**

$$\hat{y} = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

Since $z = 1.0 \geq 0$:
$$\hat{y} = 1$$

**Step 3: Check for misclassification and update weights**

- True label: $y = 1$
- Predicted label: $\hat{y} = 1$
- Error: $e = y - \hat{y} = 1 - 1 = 0$

**Since the example is correctly classified, no weight update is needed.**

However, if it were misclassified, the perceptron learning rule would be:

$$w_i^{new} = w_i^{old} + \eta \cdot e \cdot x_i$$

For each weight:
- $w_0^{new} = 0.5 + 0.1(0)(1) = 0.5$ (no change)
- $w_1^{new} = -0.3 + 0.1(0)(1) = -0.3$ (no change)
- $w_2^{new} = 0.4 + 0.1(0)(2) = 0.4$ (no change)

**Answer:** $z = 1.0$, $\hat{y} = 1$, correctly classified, weights remain unchanged.

---

### **Q.1(b) Feature Analysis and Limitations [2 + 4 Marks]**

**Given:**
- Spam classifier weights: $w = [0.2, 0.8, 0.9, -0.5]^T$
- Features: Bias, suspicious_words, links, length

**Part 1: Which feature most strongly indicates spam? [2 Marks]**

**Solution:**

The magnitude of weights indicates feature importance. Examining absolute values:
- Bias: $|0.2| = 0.2$
- Suspicious words: $|0.8| = 0.8$
- Links: $|0.9| = 0.9$
- Length: $|-0.5| = 0.5$

**Answer:** The **"links"** feature (weight = 0.9) most strongly indicates spam because it has the largest positive weight magnitude. This means each additional link increases the spam score most significantly.

**Part 2: Linear separability and limitations [4 Marks]**

**Solution:**

The model achieves 75% accuracy and plateaus. This reveals:

**Linear Separability Issues:**
- The remaining 25% of data points likely cannot be correctly classified by any linear boundary
- These points are on the "wrong side" of any straight line we could draw
- The data is **not linearly separable** - there exists no hyperplane that perfectly separates spam from non-spam

**Perceptron Limitations:**
1. **Cannot learn non-linear patterns:** If spam depends on complex feature interactions (e.g., "many links AND short length" vs "few links AND long length"), a single perceptron cannot capture this
2. **Convergence guarantee only for linearly separable data:** The perceptron convergence theorem guarantees finding a solution only if one exists
3. **No probabilistic output:** Cannot express uncertainty in borderline cases
4. **Single decision boundary:** Cannot handle multi-modal distributions or XOR-like patterns

**Why 75% is the ceiling:**
- Training longer won't help - the model has converged
- Adding more similar training data won't change the fundamental limitation
- Need either: (a) better features, (b) non-linear model (neural network), or (c) kernel methods

---

### **Q.1(c) Generalization and Feature Engineering [3 Marks]**

**Given:**
- 500 reviews total
- **Approach A:** 5 selected features (sentiment words, punctuation, length, rating)
- **Approach B:** 50 features (counts of 50 most frequent words)
- Both achieve 78% training accuracy

**Solution:**

**Evaluation:**

**Approach A is likely to generalize better** for the following reasons:

**1. Lower Overfitting Risk (Sample-to-Feature Ratio):**
- Approach A: 500 samples / 5 features = 100 samples per feature
- Approach B: 500 samples / 50 features = 10 samples per feature
- **Rule of thumb:** Need 10-30 samples per feature; Approach B is at the lower bound
- With limited data, Approach B risks memorizing training-specific word patterns

**2. Feature Quality:**
- Approach A uses **domain-informed features** (sentiment words, punctuation) that capture generalizable sentiment signals
- Approach B uses **raw word frequencies** which may include:
  - Corpus-specific artifacts
  - Topic-dependent words (e.g., "movie" appears often but isn't sentiment-related)
  - Noise from rare words

**3. Interpretability:**
- Approach A: Can explain why a review is positive ("has 8 positive sentiment words, 3 exclamation marks")
- Approach B: Hard to interpret which of 50 word counts drove the decision
- Better interpretability aids debugging and feature refinement

**4. Computational Efficiency:**
- Approach A: Faster training and inference
- Approach B: 10x more computations

**Counter-consideration:**
- If Approach B's 50 words were carefully selected (e.g., using TF-IDF, chi-square feature selection), and if we had 2000+ samples, it might capture more nuance

**Answer:** Approach A will likely generalize better due to lower overfitting risk (better sample-to-feature ratio), more meaningful features, and better interpretability.

---

### **Q.1(d) Python Implementation [5 Marks]**

**Complete the perceptron training code:**

```python
import numpy as np

def perceptron_train(X, y, learning_rate=0.1, epochs=10):
    """X: (N x d) with bias, y: (N,) labels {0,1}"""
    N, d = X.shape
    weights = np.zeros(d)
    
    for epoch in range(epochs):
        for i in range(N):
            # Blank 1: Calculate weighted sum z
            z = np.dot(X[i], weights)
            
            y_pred = 1 if z >= 0 else 0
            
            if y_pred != y[i]:
                # Blank 2: Calculate error
                error = y[i] - y_pred
                
                # Blank 3: Update weights expression
                weights = weights + learning_rate * error * X[i]
    
    return weights

# Test
X = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
y = np.array([0,0,0,1])  # AND gate

w = perceptron_train(X, y)

# Blank 4: Generate predictions for test data
preds = np.array([1 if np.dot(X[i], w) >= 0 else 0 for i in range(len(X))])

# Blank 5: Calculate accuracy
acc = np.mean(preds == y)
```

**Solutions:**

- **Blank 1:** `np.dot(X[i], weights)`
  - Computes weighted sum: $z = \sum_{j} w_j x_{ij}$

- **Blank 2:** `y[i] - y_pred`
  - Error = actual - predicted
  - Will be +1 if false negative, -1 if false positive, 0 if correct

- **Blank 3:** `weights + learning_rate * error * X[i]`
  - Perceptron update rule: $w^{new} = w^{old} + \eta \cdot e \cdot x$
  - Moves weights in direction of correct classification

- **Blank 4:** `np.array([1 if np.dot(X[i], w) >= 0 else 0 for i in range(len(X))])`
  - Apply learned weights to all test examples
  - Alternatively: `(np.dot(X, w) >= 0).astype(int)`

- **Blank 5:** `np.mean(preds == y)`
  - Accuracy = fraction of correct predictions
  - `preds == y` creates boolean array, `np.mean` converts to fraction

---

## **Q.2: Linear Regression & Gradient Descent**

### **Q.2(a) Batch Gradient Descent [6 Marks]**

**Given:**
- Predict house price (in $1000s) from area (in 100s sq ft)
- Training data:

| Bias | Area ($x_1$) | Price ($y$) |
|------|--------------|-------------|
| 1    | 10           | 150         |
| 1    | 20           | 250         |

- Current weights: $w_0 = 50$, $w_1 = 8$
- Learning rate: $\eta = 0.01$

**Solution:**

**Step (i): Calculate predictions**

For each training example:

$$\hat{y} = w_0 + w_1 \cdot x_1$$

- Example 1: $\hat{y}_1 = 50 + 8(10) = 50 + 80 = 130$
- Example 2: $\hat{y}_2 = 50 + 8(20) = 50 + 160 = 210$

**Step (ii): Calculate MSE loss**

$$J = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

where $m = 2$ (number of examples)

- Error 1: $e_1 = 130 - 150 = -20$
- Error 2: $e_2 = 210 - 250 = -40$

$$J = \frac{1}{2(2)} [(−20)^2 + (−40)^2] = \frac{1}{4} [400 + 1600] = \frac{2000}{4} = 500$$

**Step (iii): Calculate gradients**

$$\frac{\partial J}{\partial w_0} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$$

$$\frac{\partial J}{\partial w_1} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot x_{1i}$$

For $w_0$:
$$\frac{\partial J}{\partial w_0} = \frac{1}{2} [(-20) + (-40)] = \frac{-60}{2} = -30$$

For $w_1$:
$$\frac{\partial J}{\partial w_1} = \frac{1}{2} [(-20)(10) + (-40)(20)]$$
$$= \frac{1}{2} [-200 - 800] = \frac{-1000}{2} = -500$$

**Step (iv): Update weights**

$$w_j^{new} = w_j^{old} - \eta \cdot \frac{\partial J}{\partial w_j}$$

$$w_0^{new} = 50 - 0.01(-30) = 50 + 0.3 = 50.3$$

$$w_1^{new} = 8 - 0.01(-500) = 8 + 5 = 13$$

**Answer:**
- Predictions: $\hat{y}_1 = 130$, $\hat{y}_2 = 210$
- MSE Loss: $J = 500$
- Gradients: $\frac{\partial J}{\partial w_0} = -30$, $\frac{\partial J}{\partial w_1} = -500$
- Updated weights: $w_0 = 50.3$, $w_1 = 13$

---

### **Q.2(b) Python Implementation [5 Marks]**

**Complete the linear regression training function:**

```python
import numpy as np

def train_linear(X, y, lr=0.01, epochs=100):
    N, d = X.shape
    weights = np.random.randn(d) * 0.01
    
    for epoch in range(epochs):
        # Blank 1: Prediction vector y_pred
        y_pred = np.dot(X, weights)
        
        # Blank 2: MSE Loss calculation
        loss = (1/(2*N)) * np.sum((y_pred - y)**2)
        
        error = y_pred - y
        
        # Blank 3: Gradient calculation
        gradient = (1/N) * np.dot(X.T, error)
        
        # Blank 4: Weight update rule
        weights = weights - lr * gradient
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights

# Test data
X = np.array([[1,1],[1,2],[1,3]])
y = np.array([2,4,5])

# Blank 5: Call the training function
w = train_linear(X, y, lr=0.01, epochs=100)
```

**Solutions:**

- **Blank 1:** `np.dot(X, weights)`
  - Matrix multiplication: predictions for all examples at once
  - Shape: (N,) where N is number of examples

- **Blank 2:** `(1/(2*N)) * np.sum((y_pred - y)**2)`
  - Mean Squared Error formula
  - Factor of 1/2 simplifies gradient derivation
  - Alternative without 1/2: `(1/N) * np.sum((y_pred - y)**2)`

- **Blank 3:** `(1/N) * np.dot(X.T, error)`
  - Vectorized gradient: $\nabla J = \frac{1}{m} X^T (X w - y)$
  - `X.T` has shape (d, N), `error` has shape (N,), result is (d,)
  - Computes gradient for all weights simultaneously

- **Blank 4:** `weights - lr * gradient`
  - Gradient descent update: move in direction opposite to gradient
  - Minus sign because gradient points uphill, we want to go downhill

- **Blank 5:** `train_linear(X, y, lr=0.01, epochs=100)`
  - Call function with appropriate learning rate and epochs
  - Can also use: `train_linear(X, y)` relying on defaults

---

### **Q.2(c) Diagnostics and Imbalanced Data [3 + 3 Marks]**

**Given:**
- Disease detection model on 1000 patients
- 50 Diseased, 950 Healthy
- Results: TP=40, FN=10, FP=95, TN=855

**Part 1: Accuracy analysis [3 Marks]**

**Solution:**

**Calculate accuracy:**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{40 + 855}{40 + 855 + 95 + 10} = \frac{895}{1000} = 0.895 = 89.5\%$$

**Why 89% accuracy is misleading:**

1. **Class imbalance:** Only 5% of patients are diseased (50/1000)
2. **Baseline comparison:** A "naive classifier" that always predicts "Healthy" would achieve:
   $$\text{Naive Accuracy} = \frac{950}{1000} = 95\%$$
   
3. **The naive classifier is better!** Our model at 89% is actually worse than predicting the majority class
4. **The problem:** Accuracy doesn't account for class imbalance and treats all errors equally

**Better metrics for this scenario:**
- **Recall (Sensitivity):** $\frac{TP}{TP + FN} = \frac{40}{50} = 0.80$ (80% of sick patients detected)
- **Precision:** $\frac{TP}{TP + FP} = \frac{40}{135} = 0.296$ (29.6% of positive predictions are correct)
- **F1-Score:** $\frac{2 \times 0.80 \times 0.296}{0.80 + 0.296} = 0.432$

**Part 2: Recall vs Precision for life-threatening diseases [3 Marks]**

**Solution:**

**Why recall is more critical:**

For life-threatening diseases, **False Negatives (FN) are catastrophic:**
- FN = telling a sick patient they're healthy → they don't get treatment → potential death
- FP = telling a healthy patient they might be sick → additional tests → inconvenience but not life-threatening

**Clinical priority hierarchy:**
1. **High Recall (Sensitivity):** Catch all/most disease cases, even at cost of false alarms
2. Moderate Precision: Accept some false positives as they can be filtered with follow-up tests
3. Balance: Use two-stage screening (sensitive test → specific confirmation)

**Effect of lowering threshold to 0.3:**

**Current threshold (likely 0.5):**
- FN = 10 (missed 10 diseased patients)
- FP = 95 (95 false alarms)

**Lowering to 0.3 makes model more "trigger-happy":**

**Changes:**
- **FN decreases:** More diseased patients classified as diseased (catches borderline cases)
  - Might reduce FN from 10 → 5 or lower
- **FP increases:** More healthy patients misclassified as diseased
  - Might increase FP from 95 → 150 or higher

**Trade-off visualization:**
```
High Threshold (0.7) → Low FP, High FN → Miss diseases (bad for deadly diseases)
Medium Threshold (0.5) → Balanced
Low Threshold (0.3) → High FP, Low FN → More false alarms but catch more diseases (better for screening)
```

**Optimal strategy:**
- Use low threshold (0.3) for screening → High recall
- Follow up positive predictions with more specific tests → Reduce false positives
- **Better to inconvenience 200 healthy people than miss 1 diseased patient**

---

### **Q.2(d) Business Impact Analysis [3 Marks]**

**Given:**
- **Model A (3 features):** 95% Accuracy, 60% fraud caught
- **Model B (20 features):** 96% Accuracy, 75% fraud caught
- **Costs:** Missed fraud = $100, Investigation = $10

**Solution:**

**Comprehensive evaluation:**

**1. Cost Analysis (Assuming 1000 transactions, 100 fraudulent):**

**Model A:**
- Fraud caught: 60% × 100 = 60 cases
- Fraud missed: 40 cases → Cost: 40 × $100 = $4,000
- False positives (estimated from 95% accuracy): ~10 cases → Investigation cost: 10 × $10 = $100
- **Total cost: $4,100**

**Model B:**
- Fraud caught: 75% × 100 = 75 cases
- Fraud missed: 25 cases → Cost: 25 × $100 = $2,500
- False positives (estimated from 96% accuracy): ~15 cases → Investigation cost: 15 × $10 = $150
- **Total cost: $2,650**

**Savings with Model B: $4,100 - $2,650 = $1,450 per 1000 transactions**

**2. Accuracy vs Business Value:**
- Model B's 1% higher accuracy (96% vs 95%) is marginal
- But 15% higher fraud detection rate (75% vs 60%) is substantial
- **Accuracy can be misleading** if classes are imbalanced

**3. Complexity Trade-offs:**

**Model A advantages:**
- Simpler: 3 features → easier to maintain, faster inference
- More interpretable: Can explain to regulators/customers
- Less prone to overfitting
- Faster to retrain

**Model B advantages:**
- Better fraud detection → saves $1,450 per 1000 transactions
- 20 features might capture complex fraud patterns
- Worth complexity if fraud patterns are sophisticated

**4. Deployment Recommendation:**

**Deploy Model B if:**
- Fraud volume is high (ROI justifies complexity)
- Have infrastructure for 20-feature models
- Fraud patterns are complex (need rich features)
- Can maintain and monitor model

**Deploy Model A if:**
- Need interpretability (regulatory requirements)
- Limited computational resources
- Fraud patterns are simple
- Prefer robustness over marginal gains

**Best practice:**
- **Start with Model B** given clear cost advantage ($1,450 savings)
- Monitor for overfitting on new data
- Keep Model A as fallback
- Consider ensemble: use both models (vote or stack)

**Final Answer:** Deploy **Model B** due to significantly better fraud detection rate (75% vs 60%), which translates to $1,450 savings per 1000 transactions, far outweighing the complexity cost. However, implement monitoring for overfitting and maintain Model A as a simpler fallback.

---

## **Q.3: Logistic Regression**

### **Q.3(a) Logistic Propagation & Update [6 Marks]**

**Given:**
- Loan approval model with weights: $w_0 = 0$ (bias), $w_1 = 0.6$ (credit score), $w_2 = 0.8$ (income)
- Training example: Credit score = 0.7, Income = 0.5, True label = 1
- Learning rate: $\eta = 0.1$

**Solution:**

**Step (i): Calculate weighted sum $z$**

$$z = w_0 + w_1 \cdot x_1 + w_2 \cdot x_2$$
$$z = 0 + 0.6(0.7) + 0.8(0.5)$$
$$z = 0.42 + 0.4 = 0.82$$

**Step (ii): Calculate sigmoid probability**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
$$\sigma(0.82) = \frac{1}{1 + e^{-0.82}} = \frac{1}{1 + 0.4404} = \frac{1}{1.4404} = 0.6944$$

**Interpretation:** Model predicts 69.44% probability of loan approval

**Step (iii): Calculate gradient**

For logistic regression, gradient for one example:

$$\frac{\partial J}{\partial w_j} = (\sigma(z) - y) \cdot x_j$$

For each weight:

$$\frac{\partial J}{\partial w_0} = (0.6944 - 1) \cdot 1 = -0.3056$$

$$\frac{\partial J}{\partial w_1} = (0.6944 - 1) \cdot 0.7 = -0.3056 \times 0.7 = -0.2139$$

$$\frac{\partial J}{\partial w_2} = (0.6944 - 1) \cdot 0.5 = -0.3056 \times 0.5 = -0.1528$$

**Step (iv): Update weights**

$$w_j^{new} = w_j^{old} - \eta \cdot \frac{\partial J}{\partial w_j}$$

$$w_0^{new} = 0 - 0.1(-0.3056) = 0 + 0.03056 = 0.0306$$

$$w_1^{new} = 0.6 - 0.1(-0.2139) = 0.6 + 0.02139 = 0.6214$$

$$w_2^{new} = 0.8 - 0.1(-0.1528) = 0.8 + 0.01528 = 0.8153$$

**Answer:**
- Weighted sum: $z = 0.82$
- Sigmoid probability: $\sigma(z) = 0.6944$
- Gradients: $\nabla w = [-0.3056, -0.2139, -0.1528]$
- Updated weights: $w = [0.0306, 0.6214, 0.8153]$

---

### **Q.3(b) Mini-batch Implementation [5 Marks]**

**Complete the logistic regression code:**

```python
import numpy as np

def sigmoid(z):
    # Blank 1: Sigmoid function return
    return 1 / (1 + np.exp(-z))

def train_logistic(X, y, batch_size=32, lr=0.01, epochs=100):
    N, d = X.shape
    weights = np.zeros(d)
    
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        
        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            X_batch, y_batch = X[idx], y[idx]
            z = np.dot(X_batch, weights)
            
            # Blank 2: Prediction using sigmoid(z)
            y_pred = sigmoid(z)
            
            # Blank 3: Gradient calculation
            gradient = (1/len(X_batch)) * np.dot(X_batch.T, (y_pred - y_batch))
            
            # Blank 4: Weight update
            weights = weights - lr * gradient
    
    return weights

def predict(X, weights):
    # Blank 5: Prediction logic for new data
    return (sigmoid(np.dot(X, weights)) >= 0.5).astype(int)
```

**Solutions:**

- **Blank 1:** `1 / (1 + np.exp(-z))`
  - Standard sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$
  - Numerically stable for most z values
  - For very negative z, can use: `np.exp(z) / (1 + np.exp(z))` to avoid overflow

- **Blank 2:** `sigmoid(z)`
  - Apply sigmoid to linear combination to get probabilities
  - Output range: (0, 1) representing probability of class 1

- **Blank 3:** `(1/len(X_batch)) * np.dot(X_batch.T, (y_pred - y_batch))`
  - Vectorized gradient for binary cross-entropy loss
  - Formula: $\nabla J = \frac{1}{m} X^T (\sigma(Xw) - y)$
  - Shape: (d,) where d is number of features

- **Blank 4:** `weights - lr * gradient`
  - Standard gradient descent update
  - Moves weights to reduce loss

- **Blank 5:** `(sigmoid(np.dot(X, weights)) >= 0.5).astype(int)`
  - Compute probabilities, threshold at 0.5, convert to {0,1}
  - Alternative: `np.round(sigmoid(np.dot(X, weights))).astype(int)`

---

## **Q.4: Softmax & Confusion Matrix**

### **Q.4(a) Softmax Sentiment Classifier [6 Marks]**

**Given:**
- 3-class sentiment: Negative (0), Neutral (1), Positive (2)
- Logits: $z = [1.2, 2.0, 0.5]$
- True label: 1 (Neutral)

**Solution:**

**Step 1: Calculate softmax probabilities**

$$P(y = k | x) = \frac{e^{z_k}}{\sum_{j=0}^{2} e^{z_j}}$$

First, calculate exponentials:
- $e^{z_0} = e^{1.2} = 3.3201$
- $e^{z_1} = e^{2.0} = 7.3891$
- $e^{z_2} = e^{0.5} = 1.6487$

Sum of exponentials:
$$\text{sum} = 3.3201 + 7.3891 + 1.6487 = 12.3579$$

Softmax probabilities:

$$P(y = 0) = \frac{3.3201}{12.3579} = 0.2686 \text{ (26.86%)}$$

$$P(y = 1) = \frac{7.3891}{12.3579} = 0.5979 \text{ (59.79%)}$$

$$P(y = 2) = \frac{1.6487}{12.3579} = 0.1335 \text{ (13.35%)}$$

**Verification:** $0.2686 + 0.5979 + 0.1335 = 1.0000$ ✓

**Step 2: Calculate Categorical Cross-Entropy Loss**

$$\text{CCE} = -\sum_{k=0}^{2} y_k \log(\hat{y}_k)$$

where $y_k$ is one-hot encoded true label: $y = [0, 1, 0]$

$$\text{CCE} = -[0 \cdot \log(0.2686) + 1 \cdot \log(0.5979) + 0 \cdot \log(0.1335)]$$
$$= -\log(0.5979) = -(-0.5148) = 0.5148$$

**Interpretation:** Lower loss is better. Loss of 0.515 indicates moderate confidence in correct class.

**Step 3: Identify predicted class**

Predicted class = $\arg\max_k P(y = k)$

- Class 0 (Negative): 26.86%
- Class 1 (Neutral): **59.79%** ← Maximum
- Class 2 (Positive): 13.35%

**Predicted class: 1 (Neutral)**

**Step 4: Check correctness**

- True label: 1 (Neutral)
- Predicted label: 1 (Neutral)
- **Prediction is CORRECT** ✓

**Answer:**
- Softmax probabilities: $[0.2686, 0.5979, 0.1335]$
- CCE Loss: $0.5148$
- Predicted class: 1 (Neutral)
- Correctness: **Yes, correct prediction**

---

### **Q.4(b) Mini-batch Softmax Code [5 Marks]**

**Complete the softmax training function:**

```python
import numpy as np

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    # Blank 1: Softmax return expression
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def train_softmax(X, Y, K, batch_size=32, lr=0.01, epochs=100):
    N, d = X.shape
    W = np.random.randn(d, K) * 0.01
    
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        
        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            X_batch, Y_batch = X[idx], Y[idx]
            
            # Blank 2: Logit calculation Z
            Z = np.dot(X_batch, W)
            
            Y_pred = softmax(Z)
            
            # Blank 3: Gradient calculation
            gradient = (1/len(X_batch)) * np.dot(X_batch.T, (Y_pred - Y_batch))
            
            # Blank 4: Weight update
            W = W - lr * gradient
    
    return W

def predict(X, W):
    Z = np.dot(X, W)
    # Blank 5: Final prediction (class index)
    return np.argmax(softmax(Z), axis=1)
```

**Solutions:**

- **Blank 1:** `exp_Z / np.sum(exp_Z, axis=1, keepdims=True)`
  - Normalizes each row (example) to sum to 1
  - `axis=1`: sum across classes (columns)
  - `keepdims=True`: maintains shape for broadcasting
  - Result shape: (batch_size, K)

- **Blank 2:** `np.dot(X_batch, W)`
  - Linear transformation: logits = features × weights
  - Shape: (batch_size, K) where K is number of classes
  - Each row contains logits for one example

- **Blank 3:** `(1/len(X_batch)) * np.dot(X_batch.T, (Y_pred - Y_batch))`
  - Gradient of cross-entropy loss for softmax
  - Formula: $\nabla_W J = \frac{1}{m} X^T (\hat{Y} - Y)$
  - Shape: (d, K) matching weight matrix
  - Y_batch is one-hot encoded

- **Blank 4:** `W - lr * gradient`
  - Standard gradient descent update
  - Reduces cross-entropy loss

- **Blank 5:** `np.argmax(softmax(Z), axis=1)`
  - Returns index of maximum probability for each example
  - `axis=1`: find max across classes
  - Output: array of class indices {0, 1, ..., K-1}

---

### **Q.4(c) Performance Evaluation [6 Marks]**

**Given confusion matrix for Cat, Dog, Bird classification (1000 total examples):**

```
              Predicted
           Cat  Dog  Bird
Actual Cat  320   25    5
       Dog   30  540   30
      Bird   10   40    0
```

**Part 1: Calculate Precision and Recall for "Bird" class [3 Marks]**

**Solution:**

**Identify components for Bird (class 2):**

- **True Positives (TP):** Actual Bird, Predicted Bird = 0
- **False Positives (FP):** Predicted Bird but not actually Bird
  - From Cat row: 5
  - From Dog row: 30
  - Total FP = 5 + 30 = 35

- **False Negatives (FN):** Actually Bird but not predicted Bird
  - From Bird row: 10 (predicted Cat) + 40 (predicted Dog) = 50

- **True Negatives (TN):** Not Bird, predicted not Bird
  - Cat-Cat: 320, Cat-Dog: 25
  - Dog-Cat: 30, Dog-Dog: 540
  - TN = 320 + 25 + 30 + 540 = 915

**Calculate metrics:**

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{0}{0 + 35} = 0.0$$

**Interpretation:** Of all examples predicted as Bird, 0% were actually Bird. Model never correctly predicts Bird!

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{0}{0 + 50} = 0.0$$

**Interpretation:** Of all actual Bird examples, 0% were detected. Model completely fails on Bird class!

**Part 2: Error pattern analysis [3 Marks]**

**Solution:**

**Given:** 40 out of 50 bird errors are misclassified as "Dog"

**What this suggests about learned features:**

1. **Feature confusion between Bird and Dog:**
   - Model has learned features that strongly overlap for these classes
   - Possible causes:
     - Fur/feather texture appears similar in images
     - Size: both might be medium-sized objects
     - Background: similar outdoor settings
     - Shape: certain bird poses (walking) resemble dogs

2. **Bird underrepresentation:**
   - Bird class likely has fewer training examples (class imbalance)
   - Model hasn't learned distinctive bird features (beaks, wings, flight posture)
   - Decision boundary is biased toward more frequent classes

3. **Specific feature problems:**
   - Missing key discriminative features:
     - Beak shape/color
     - Wing structure
     - Feather patterns
   - Model might rely on texture/background rather than shape

**Does 88% overall accuracy guarantee good per-class performance?**

**Calculate overall accuracy:**

$$\text{Accuracy} = \frac{320 + 540 + 0}{1000} = \frac{860}{1000} = 0.86 = 86\%$$

**Answer: NO - 86% overall accuracy hides Bird class failure:**

**Why this is misleading:**
1. **Class imbalance:** If Bird is only 5% of data, model can be 95% accurate by always predicting Cat/Dog
2. **Macro vs micro averaging:** Overall accuracy weighs by frequency, not fairness across classes
3. **Application requirements:** In wildlife monitoring, missing all birds is unacceptable

**Better metrics for imbalanced multiclass:**
- **Macro-averaged F1:** Average F1 across all classes equally
  - Cat F1: high
  - Dog F1: high
  - Bird F1: 0
  - Macro F1 = (F1_cat + F1_dog + F1_bird) / 3 → much lower

- **Per-class recall:** Ensures each class is adequately detected

- **Balanced accuracy:** $(Recall_{cat} + Recall_{dog} + Recall_{bird}) / 3$

**Recommendations to fix Bird detection:**
1. Collect more bird training data
2. Use class weighting in loss function
3. Data augmentation for bird class
4. Use focal loss (focuses on hard examples)
5. Add bird-specific features (wing detection, beak detection)

---

### **Q.4(d) Medical Imbalance Case Study [3 Marks]**

**Given: X-ray diagnosis for 5 conditions (highly imbalanced)**

- **Model A:** 92% Accuracy, 45% TB Recall
- **Model B:** 85% Accuracy, 78% TB Recall

**Solution:**

**Evaluation for clinical deployment:**

**Critical factors:**

1. **Clinical Consequences of Errors:**
   - **False Negative (missed TB):** Patient doesn't get treatment → disease spreads → potential death + community transmission
   - **False Positive:** Additional tests → inconvenience + cost, but no life risk
   - **TB is highly contagious:** Missing one case can infect dozens

2. **Accuracy vs Recall Trade-off:**
   - Model A: High accuracy (92%) but misses 55% of TB cases → **UNACCEPTABLE**
   - Model B: Lower accuracy (85%) but catches 78% of TB cases → **MUCH BETTER**
   - 7% accuracy drop is acceptable for 33% improvement in TB detection

3. **Class Imbalance Context:**
   - If TB is rare (1-2% of X-rays), even Model A's 92% accuracy could be achieved by always predicting "No TB"
   - Model B's willingness to sacrifice overall accuracy shows it's actually detecting TB

**Ethical Considerations:**

**Medical Ethics Principles:**
- **Primum non nocere** ("First, do no harm"): Missing deadly disease violates this
- **Justice:** TB disproportionately affects vulnerable populations; must detect it
- **Beneficence:** Benefit of catching TB outweighs cost of false alarms

**Regulatory Requirements:**
- Medical devices must meet **minimum sensitivity** (recall) thresholds
- For TB screening, typically require ≥80% sensitivity
- Model A would likely **not be approved** for clinical use

**Deployment Strategy:**

**Recommendation: Deploy Model B with two-stage screening**

**Primary Screening (Model B):**
- High sensitivity (78%) catches most TB cases
- Some false positives are acceptable

**Confirmatory Testing:**
- All Model B positives → expert radiologist review + sputum test
- Reduces false positives while maintaining high recall

**Cost-Benefit Analysis:**

Assume 10,000 X-rays, 100 TB cases (1% prevalence):

**Model A:**
- TB detected: 45
- TB missed: 55 → 55 × (treatment cost + transmission + potential death) = **$500K+ in societal cost**
- False positives: ~100

**Model B:**
- TB detected: 78
- TB missed: 22 → 22 × costs = **$200K+ in societal cost**
- False positives: ~250

**Even with more false positives, Model B saves $300K+ and prevents deaths**

**Answer:** **Deploy Model B** due to significantly higher TB recall (78% vs 45%), which is ethically imperative for a contagious, deadly disease. The 7% accuracy drop is acceptable as it reflects Model B's willingness to raise alarms rather than miss TB cases. Implement with two-stage workflow: Model B screening → expert confirmation for positives.

---

## **Q.5: Deep FeedForward Neural Network (DFNN)**

### **Q.5(a) Forward Propagation [6 Marks]**

**Given architecture:**
- **Input layer:** 2 inputs $(x_1, x_2)$
- **Hidden layer:** 2 neurons with ReLU activation
- **Output layer:** 1 neuron with Sigmoid activation

**Parameters:**
- $W^{[1]} = \begin{bmatrix} 0.5 & 0.3 \\ -0.2 & 0.8 \end{bmatrix}$ (shape: 2×2)
- $b^{[1]} = \begin{bmatrix} 0.1 \\ -0.3 \end{bmatrix}$ (shape: 2×1)
- $W^{[2]} = \begin{bmatrix} 1.0 \\ -0.5 \end{bmatrix}$ (shape: 2×1)
- $b^{[2]} = 0.2$ (scalar)

**Input:** $x = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$, True label: $y = 1$

**Solution:**

**Step (i): Calculate hidden layer pre-activations**

$$z^{[1]} = W^{[1]T} x + b^{[1]}$$

Note: Assuming column vectors and proper matrix multiplication:

$$z^{[1]}_1 = 0.5(1.0) + 0.3(2.0) + 0.1 = 0.5 + 0.6 + 0.1 = 1.2$$

$$z^{[1]}_2 = -0.2(1.0) + 0.8(2.0) + (-0.3) = -0.2 + 1.6 - 0.3 = 1.1$$

$$z^{[1]} = \begin{bmatrix} 1.2 \\ 1.1 \end{bmatrix}$$

**Step (ii): Calculate ReLU activations**

$$a^{[1]} = \text{ReLU}(z^{[1]}) = \max(0, z^{[1]})$$

$$a^{[1]}_1 = \max(0, 1.2) = 1.2$$

$$a^{[1]}_2 = \max(0, 1.1) = 1.1$$

$$a^{[1]} = \begin{bmatrix} 1.2 \\ 1.1 \end{bmatrix}$$

**Step (iii): Calculate output prediction**

$$z^{[2]} = W^{[2]T} a^{[1]} + b^{[2]}$$

$$z^{[2]} = 1.0(1.2) + (-0.5)(1.1) + 0.2 = 1.2 - 0.55 + 0.2 = 0.85$$

$$\hat{y} = \sigma(z^{[2]}) = \frac{1}{1 + e^{-0.85}} = \frac{1}{1 + 0.4274} = \frac{1}{1.4274} = 0.7007$$

**Interpretation:** Model predicts 70.07% probability of class 1

**Step (iv): Calculate Binary Cross-Entropy Loss**

$$\text{BCE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

Since $y = 1$:

$$\text{BCE} = -[1 \cdot \log(0.7007) + 0 \cdot \log(1-0.7007)]$$
$$= -\log(0.7007) = -(-0.3556) = 0.3556$$

**Answer:**
- Hidden pre-activations: $z^{[1]} = [1.2, 1.1]$
- ReLU activations: $a^{[1]} = [1.2, 1.1]$
- Output prediction: $\hat{y} = 0.7007$
- BCE Loss: $J = 0.3556$

---

### **Q.5(b) Python Implementation [5 Marks]**

**Complete the DFNN forward and backward propagation code:**

```python
import numpy as np

def relu(Z):
    # Blank 1: ReLU function
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    # Blank 2: Layer 1 activation (A1)
    A1 = relu(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2, (Z1, A1, Z2, A2)

def backward(X, Y, cache, W2):
    Z1, A1, Z2, A2 = cache
    N = X.shape[0]
    
    # Output layer gradient
    dZ2 = A2 - Y
    
    # Blank 3: Calculate dW2
    dW2 = (1/N) * np.dot(A1.T, dZ2)
    
    # Blank 4: Calculate dA1
    dA1 = np.dot(dZ2, W2.T)
    
    # Hidden layer gradient
    dZ1 = dA1 * (Z1 > 0)  # ReLU derivative
    
    # Blank 5: Calculate dW1
    dW1 = (1/N) * np.dot(X.T, dZ1)
    
    return dW1, dW2
```

**Solutions:**

- **Blank 1:** `np.maximum(0, Z)`
  - ReLU activation: $\text{ReLU}(z) = \max(0, z)$
  - Element-wise operation
  - Derivative: 1 if z > 0, else 0
  - Alternative: `Z * (Z > 0)`

- **Blank 2:** `relu(Z1)`
  - Apply ReLU activation to hidden layer pre-activations
  - Introduces non-linearity to network
  - Without this, entire network would be linear

- **Blank 3:** `(1/N) * np.dot(A1.T, dZ2)`
  - Gradient of weights in output layer
  - Formula: $\frac{\partial J}{\partial W^{[2]}} = \frac{1}{m} A^{[1]T} \delta^{[2]}$
  - Shape: same as W2

- **Blank 4:** `np.dot(dZ2, W2.T)`
  - Backpropagates error from output to hidden layer
  - Formula: $\delta^{[1]} = \delta^{[2]} W^{[2]T} \odot g'(Z^{[1]})$
  - Chain rule: how does J change with respect to A1?

- **Blank 5:** `(1/N) * np.dot(X.T, dZ1)`
  - Gradient of weights in hidden layer
  - Formula: $\frac{\partial J}{\partial W^{[1]}} = \frac{1}{m} X^T \delta^{[1]}$
  - Completes backpropagation through all layers

---

### **Q.5(c) Architecture Design [6 Marks]**

**Task:** Design a DFNN for 5-class sentiment analysis
- 1000 input features
- 50,000 training samples

**Solution:**

**1. Network Architecture Specification:**

**Proposed architecture:**

```
Input Layer:  1000 neurons (features)
           ↓
Hidden Layer 1: 512 neurons (ReLU)
           ↓
Dropout: 0.5 probability
           ↓
Hidden Layer 2: 256 neurons (ReLU)
           ↓
Dropout: 0.3 probability
           ↓
Hidden Layer 3: 128 neurons (ReLU)
           ↓
Output Layer: 5 neurons (Softmax)
```

**Rationale:**
- **Progressive dimensionality reduction:** 1000 → 512 → 256 → 128 → 5
  - Compresses high-dimensional features into meaningful representations
  - Each layer learns increasingly abstract features
- **Sufficient capacity:** 3 hidden layers can learn complex patterns in sentiment
- **Not too deep:** Avoid vanishing gradients with moderate depth

**2. Activation Functions:**

**Hidden Layers: ReLU (Rectified Linear Unit)**

**Why ReLU:**
- ✓ Prevents vanishing gradient (gradient = 1 for positive inputs)
- ✓ Computationally efficient: $\text{ReLU}(x) = \max(0, x)$
- ✓ Sparse activation: ~50% neurons are zero, reducing overfitting
- ✓ Works well in practice for text/NLP tasks

**Alternative considerations:**
- **Leaky ReLU:** $\max(0.01x, x)$ if concerned about "dying ReLU" problem
- **ELU/GELU:** Smoother variants, slightly better but more expensive

**Why NOT sigmoid/tanh in hidden layers:**
- ✗ Sigmoid/tanh saturate (gradients → 0 for large |x|)
- ✗ Vanishing gradient in deep networks
- ✗ Slower convergence

**Output Layer: Softmax**

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{5} e^{z_j}}$$

**Why Softmax:**
- ✓ Outputs valid probability distribution (sum to 1)
- ✓ Multi-class classification standard
- ✓ Pairs naturally with categorical cross-entropy loss

**3. Loss Function and Optimization:**

**Loss Function: Categorical Cross-Entropy**

$$J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{5} y_{ik} \log(\hat{y}_{ik})$$

**Why:**
- Standard for multi-class classification
- Penalizes confident wrong predictions more heavily
- Smooth gradient for optimization

**Optimization Algorithm: Mini-batch Gradient Descent with Adam**

**Mini-batch SGD:**
- **Batch size:** 64-128
- **Why:** 50,000 samples is too large for batch GD, too slow for pure SGD
- Sweet spot for GPU efficiency and gradient stability

**Adam Optimizer (Adaptive Moment Estimation):**
- Combines momentum + adaptive learning rates
- Learning rate: Start with 0.001
- Hyperparameters: β₁=0.9, β₂=0.999, ε=10⁻⁸

**Why Adam over alternatives:**
- ✓ Faster convergence than plain SGD
- ✓ Adaptive learning rates per parameter
- ✓ Robust to hyperparameter choices
- ✓ Industry standard for deep learning

**Training schedule:**
- Epochs: 50-100 with early stopping
- Learning rate decay: Reduce by 0.5 if validation loss plateaus for 5 epochs
- Early stopping: Patience of 10 epochs on validation loss

**4. Calculate Total Parameters:**

**Layer-by-layer calculation:**

**Hidden Layer 1:**
- Weights: $1000 \times 512 = 512,000$
- Biases: $512$
- **Total:** 512,512

**Hidden Layer 2:**
- Weights: $512 \times 256 = 131,072$
- Biases: $256$
- **Total:** 131,328

**Hidden Layer 3:**
- Weights: $256 \times 128 = 32,768$
- Biases: $128$
- **Total:** 32,896

**Output Layer:**
- Weights: $128 \times 5 = 640$
- Biases: $5$
- **Total:** 645

**Grand Total Parameters:**
$$512,512 + 131,328 + 32,896 + 645 = 677,381 \approx 677K \text{ parameters}$$

**5. Overfitting Risk Evaluation:**

**Sample-to-Parameter Ratio:**
$$\frac{50,000 \text{ samples}}{677,381 \text{ parameters}} \approx 73.8 \text{ samples per parameter}$$

**Assessment: MODERATE to LOW overfitting risk**

**Favorable factors:**
- ✓ 74 samples per parameter is quite good (typical threshold: >10)
- ✓ 50K samples is substantial for supervised learning
- ✓ High-dimensional input (1000 features) provides rich signal

**Risk factors:**
- ⚠ Deep networks can still memorize with sufficient capacity
- ⚠ If data is noisy or mislabeled, model might learn noise

**Mitigation strategies (CRITICAL to implement):**

1. **Dropout layers (already included):**
   - Layer 1-2: 50% dropout
   - Layer 2-3: 30% dropout
   - Randomly disables neurons during training → prevents co-adaptation

2. **L2 Regularization (Weight Decay):**
   - Add to loss: $\lambda \sum_{l} ||W^{[l]}||^2$
   - Typical λ: 0.0001-0.001
   - Penalizes large weights → smoother decision boundary

3. **Early Stopping:**
   - Monitor validation loss
   - Stop training when validation loss increases for 10 consecutive epochs
   - Prevents overfitting to training data

4. **Data Augmentation (for text):**
   - Synonym replacement
   - Back-translation
   - Random insertion/deletion
   - Increases effective dataset size

5. **Batch Normalization:**
   - Normalize activations within each mini-batch
   - Reduces internal covariate shift
   - Acts as regularizer

6. **Cross-validation:**
   - K-fold validation (K=5) to ensure model generalizes
   - Use hold-out test set (never seen during training)

**Monitoring during training:**
- Plot training vs validation loss curves
- If gap widens: overfitting → increase regularization
- If both high: underfitting → increase capacity or train longer

**Final Architecture Summary:**

```python
model = Sequential([
    Dense(512, activation='relu', input_shape=(1000,), 
          kernel_regularizer=l2(0.0001)),
    Dropout(0.5),
    BatchNormalization(),
    
    Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
    Dropout(0.3),
    BatchNormalization(),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
    
    Dense(5, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Expected Performance:**
- Training: 90-95% accuracy
- Validation: 85-92% accuracy (with proper regularization)
- Inference: ~1-2ms per sample (GPU) or 10-20ms (CPU)

---

## **Summary**

This question paper comprehensively tests understanding of:
- **Perceptrons:** Weight updates, decision boundaries, linear separability
- **Linear Regression:** Gradient descent, MSE loss, diagnostics
- **Logistic Regression:** Sigmoid function, binary classification, probability interpretation
- **Softmax Regression:** Multi-class classification, categorical cross-entropy
- **Deep Neural Networks:** Forward/backward propagation, architecture design
- **Evaluation:** Confusion matrices, precision/recall, class imbalance
- **Practical ML:** Overfitting, regularization, optimization algorithms

Key takeaways:
1. Always consider class imbalance when evaluating models
2. Accuracy can be misleading - use task-appropriate metrics
3. Deep learning requires careful regularization to prevent overfitting
4. Domain knowledge guides architecture and feature selection
5. Cost-benefit analysis is crucial for real-world deployment
