# Module 1: Introduction to Machine Learning

## What is Machine Learning?

**Machine Learning (ML)** is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Instead of following pre-programmed instructions, ML algorithms build mathematical models based on training data to make predictions or decisions.

```{admonition} Tom Mitchell's Definition
:class: note
"A computer program is said to learn from experience **E** with respect to some class of tasks **T** and performance measure **P**, if its performance at tasks in T, as measured by P, improves with experience E."

```

### Breaking Down the Definition

| Component | Description | Example |
|-----------|-------------|---------|
| **Task (T)** | What the system should do | Classify emails as spam/not spam |
| **Experience (E)** | Data the system learns from | Collection of labeled emails |
| **Performance (P)** | How we measure success | Accuracy of classification |

```{admonition} Spam Filter Example
:class: tip
- **Task**: Classify incoming emails as spam or not spam
- **Experience**: Database of 10,000 labeled emails
- **Performance**: 98% accuracy on new emails

The system "learns" by finding patterns in spam emails (certain words, sender patterns, etc.)

```

---

## Why Machine Learning?

### Traditional Programming vs Machine Learning

```
Traditional Programming:
    Input Data + Rules → Computer → Output

Machine Learning:
    Input Data + Output → Computer → Rules (Model)
```

```{admonition} Key Insight
:class: note
In traditional programming, humans write rules. In ML, the computer discovers rules from data!

```

### When to Use Machine Learning

✅ **Use ML when**:

1. **Rules are complex or unknown**: Face recognition, speech understanding
2. **Rules change frequently**: Stock market prediction, spam detection
3. **Data is abundant**: Customer behavior analysis
4. **Human expertise is scarce**: Medical diagnosis in remote areas

❌ **Don't use ML when**:

1. Rules are simple and well-defined (use traditional programming)
2. Insufficient data available
3. Explainability is critical and simple rules exist
4. Cost of errors is too high without human oversight

---

## Types of Machine Learning

### 1. Supervised Learning

```{admonition} Definition
:class: tip
Learning with **labeled** training data. The algorithm learns from input-output pairs to predict outputs for new inputs.

```

**How it works**:

```
Training Phase:
    Input Features (X) + Labels (Y) → Algorithm → Model

Prediction Phase:
    New Input (X') → Model → Predicted Output (Ŷ)
```

#### Classification vs Regression

| Aspect | Classification | Regression |
|--------|---------------|------------|
| **Output Type** | Discrete categories | Continuous values |
| **Question** | "What category?" | "How much/many?" |
| **Examples** | Spam detection, disease diagnosis | House price, temperature |
| **Evaluation** | Accuracy, F1-Score | MSE, RMSE, R² |

```{admonition} Classification Examples
:class: tip
- **Binary Classification**: Email spam detection (Spam/Not Spam)
- **Multi-class Classification**: Digit recognition (0-9)
- **Multi-label Classification**: Image tagging (can have multiple tags)

```

```{admonition} Regression Examples
:class: tip
- House price prediction based on features
- Stock price forecasting
- Temperature prediction
- Sales forecasting

```

**Common Supervised Learning Algorithms**:

| Algorithm | Type | Best For |
|-----------|------|----------|
| Linear Regression | Regression | Linear relationships |
| Logistic Regression | Classification | Binary classification |
| Decision Trees | Both | Interpretable models |
| Random Forest | Both | Complex patterns, ensemble |
| SVM | Both | High-dimensional data |
| Neural Networks | Both | Complex patterns, large data |

---

### 2. Unsupervised Learning

```{admonition} Definition
:class: tip
Learning from **unlabeled** data. The algorithm finds hidden patterns or structures without guidance.

```

**How it works**:

```
Training Phase:
    Input Data (X) only → Algorithm → Patterns/Clusters

Application:
    New Data → Model → Group Assignment/Reduced Features
```

#### Types of Unsupervised Learning

**a) Clustering**

- **Goal**: Group similar data points together
- **Output**: Cluster assignments

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| K-Means | Partition into K clusters | Spherical clusters |
| Hierarchical | Build tree of clusters | Unknown K, hierarchies |
| DBSCAN | Density-based clustering | Arbitrary shapes, outliers |

```{admonition} Clustering Applications
:class: tip
- Customer segmentation for marketing
- Image segmentation
- Anomaly/fraud detection
- Document grouping

```

**b) Dimensionality Reduction**

- **Goal**: Reduce number of features while preserving information
- **Output**: Lower-dimensional representation

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| PCA | Linear projection to max variance | Visualization, preprocessing |
| t-SNE | Non-linear, preserves local structure | Visualization |
| Autoencoders | Neural network-based | Complex patterns |

```{admonition} Dimensionality Reduction Applications
:class: tip
- Data visualization (reduce to 2D/3D)
- Feature extraction
- Noise reduction
- Data compression

```

**c) Association Rule Learning**

- **Goal**: Discover relationships between variables
- **Output**: Rules like "If A, then B"

```{admonition} Market Basket Analysis
:class: tip
- "Customers who buy bread also buy butter" (70% confidence)
- Used for product recommendations, store layout optimization

```

---

### 3. Reinforcement Learning

```{admonition} Definition
:class: tip
Learning through **interaction** with an environment. The agent takes actions and receives rewards/penalties.

```

**How it works**:

```
Agent → Action → Environment
         ↑           ↓
      Reward ← State Change
```

**Key Concepts**:

| Term | Description |
|------|-------------|
| **Agent** | The learner/decision maker |
| **Environment** | What the agent interacts with |
| **State** | Current situation |
| **Action** | What the agent can do |
| **Reward** | Feedback signal (positive/negative) |
| **Policy** | Strategy for choosing actions |

```{admonition} Reinforcement Learning Applications
:class: tip
- Game playing (AlphaGo, Chess engines)
- Robotics (walking, manipulation)
- Autonomous vehicles
- Recommendation systems
- Resource management

```

---

### Comparison of Learning Types

| Aspect | Supervised | Unsupervised | Reinforcement |
|--------|------------|--------------|---------------|
| **Labels** | Required | Not required | Rewards only |
| **Goal** | Predict output | Find patterns | Maximize reward |
| **Feedback** | Direct (labels) | None | Delayed (rewards) |
| **Example** | Spam detection | Customer clustering | Game playing |

```{admonition} Exam Tip
:class: tip
When asked to identify learning type, ask:

1. Is there a labeled output? → **Supervised**
2. Is there no output, just finding patterns? → **Unsupervised**
3. Is there an agent learning from rewards? → **Reinforcement**

```

---

## Machine Learning Workflow

### Complete Pipeline

```
1. Problem Definition
        ↓
2. Data Collection
        ↓
3. Data Preprocessing
        ↓
4. Exploratory Data Analysis (EDA)
        ↓
5. Feature Engineering
        ↓
6. Model Selection
        ↓
7. Training
        ↓
8. Evaluation
        ↓
9. Hyperparameter Tuning
        ↓
10. Deployment & Monitoring
```

### Step-by-Step Details

#### 1. Problem Definition
- Define the task clearly
- Identify success metrics
- Understand business requirements

#### 2. Data Collection
- Gather relevant data
- Ensure data quality and quantity
- Consider data sources and biases

#### 3. Data Preprocessing

```{admonition} Critical Step
:class: warning
Most ML projects spend 60-80% of time on data preprocessing!

```

**Common preprocessing steps**:

| Step | Description | Techniques |
|------|-------------|------------|
| **Missing Values** | Handle incomplete data | Remove, impute (mean/median/mode) |
| **Outliers** | Handle extreme values | Remove, cap, transform |
| **Encoding** | Convert categorical to numerical | One-hot, Label encoding |
| **Scaling** | Normalize feature ranges | Min-Max, Standardization |
| **Data Splitting** | Divide into train/val/test | Random split, stratified |

**Feature Scaling Methods**:

**Min-Max Normalization** (scales to [0, 1]):

$$
x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

**Standardization** (z-score, mean=0, std=1):

$$
x_{standardized} = \frac{x - \mu}{\sigma}
$$

```{admonition} When to Use Which
:class: note
- **Min-Max**: When you need bounded values (e.g., neural networks with sigmoid)
- **Standardization**: When data has outliers, or algorithm assumes normal distribution

```

#### 4. Feature Engineering

- Create new features from existing ones
- Select most relevant features
- Domain knowledge is crucial

```{admonition} Feature Engineering Examples
:class: tip
- **Date**: Extract day, month, year, day of week, is_weekend
- **Text**: Word count, sentiment score, TF-IDF
- **Combinations**: price_per_sqft = price / area

```

#### 5. Model Selection

Consider:
- Problem type (classification/regression)
- Data size and dimensionality
- Interpretability requirements
- Training time constraints

#### 6-8. Training and Evaluation

- Train model on training data
- Evaluate on validation/test data
- Use appropriate metrics

#### 9. Hyperparameter Tuning

- Grid Search: Try all combinations
- Random Search: Sample random combinations
- Bayesian Optimization: Smart search

#### 10. Deployment

- Deploy model to production
- Monitor performance
- Retrain as needed

---

## Overfitting vs Underfitting

### The Fundamental Tradeoff

```{admonition} Overfitting
:class: danger
Model is **too complex** - memorizes training data, fails on new data.

**Symptoms**:
- High training accuracy
- Low test accuracy
- Large gap between train and test performance

```

```{admonition} Underfitting
:class: danger
Model is **too simple** - cannot capture underlying patterns.

**Symptoms**:
- Low training accuracy
- Low test accuracy
- Both performances are poor

```

### Visual Understanding

```
Underfitting          Good Fit           Overfitting
(High Bias)          (Balanced)         (High Variance)

    *   *              *   *               *   *
  *       *          *       *           *       *
*           *      *           *       *     *     *
  --------         ~~~~~~~~~           ~~~~~~~~*~~~
(straight line)   (smooth curve)      (wiggly curve)
```

### Causes and Solutions

| Problem | Causes | Solutions |
|---------|--------|-----------|
| **Overfitting** | Model too complex | Regularization (L1, L2) |
| | Too many features | Feature selection |
| | Too little data | Get more data |
| | Training too long | Early stopping |
| **Underfitting** | Model too simple | Increase complexity |
| | Too few features | Add more features |
| | Too much regularization | Reduce regularization |
| | Training too short | Train longer |

### Detecting Overfitting/Underfitting

**Learning Curves**: Plot training and validation error vs. training set size

```
Overfitting:                    Underfitting:
Error                           Error
  |                               |
  |  Val Error                    |  Val Error ≈ Train Error
  |  --------                     |  --------
  |                               |
  |  Train Error                  |  
  |  --------                     |  --------
  └─────────────                  └─────────────
    Training Size                   Training Size
```

```{admonition} Exam Tip
:class: tip
- **High training error, high test error** → Underfitting
- **Low training error, high test error** → Overfitting
- **Low training error, low test error** → Good fit

```

---

## Bias-Variance Tradeoff

### Understanding Bias and Variance

**Bias**: Error from overly simplistic assumptions

- High bias → Model misses relevant relations → Underfitting
- Low bias → Model captures complex patterns

**Variance**: Error from sensitivity to small fluctuations in training data

- High variance → Model is too sensitive → Overfitting
- Low variance → Model is stable across datasets

### Mathematical Decomposition

Total Error = Bias² + Variance + Irreducible Error

$$
\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2
$$

Where:
- $\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$ (expected prediction - true function)
- $\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$ (variance of predictions)
- $\sigma^2$ = irreducible error (noise in data)

### The Tradeoff

| Model Complexity | Bias | Variance | Total Error |
|-----------------|------|----------|-------------|
| Very Simple | High | Low | High |
| Optimal | Medium | Medium | **Lowest** |
| Very Complex | Low | High | High |

```
Error
  |
  |  \         Total Error
  |   \       /
  |    \     /
  |     \___/  ← Optimal complexity
  |    /     \
  |   / Bias  \ Variance
  |  /         \
  └─────────────────────
    Simple → Complex
    Model Complexity
```

```{admonition} Goal
:class: tip
Find the sweet spot where **total error is minimized** - not too simple (high bias), not too complex (high variance).

```

---

## Training, Validation, and Test Sets

### Why Split Data?

```{admonition} Critical
:class: warning
Never evaluate your model on data it was trained on! This gives an overly optimistic estimate.

```

### Standard Split

```
Full Dataset
    │
    ├── Training Set (60-80%)
    │   └── Used to train the model
    │
    ├── Validation Set (10-20%)
    │   └── Used to tune hyperparameters
    │
    └── Test Set (10-20%)
        └── Used for final evaluation (only once!)
```

### Purpose of Each Set

| Set | Purpose | When Used | Can Modify Model? |
|-----|---------|-----------|-------------------|
| **Training** | Learn model parameters | During training | Yes |
| **Validation** | Tune hyperparameters, model selection | During development | Yes (indirectly) |
| **Test** | Final unbiased evaluation | At the end only | No |

### Cross-Validation

When data is limited, use **k-fold cross-validation**:

```
Fold 1: [Val][Train][Train][Train][Train]
Fold 2: [Train][Val][Train][Train][Train]
Fold 3: [Train][Train][Val][Train][Train]
Fold 4: [Train][Train][Train][Val][Train]
Fold 5: [Train][Train][Train][Train][Val]

Final Score = Average of all fold scores
```

**Benefits**:
- Uses all data for both training and validation
- More robust performance estimate
- Standard: 5-fold or 10-fold CV

```{admonition} Exam Tip
:class: tip
- **Training set**: Model learns from this
- **Validation set**: We tune hyperparameters using this
- **Test set**: Final evaluation, use only ONCE at the end

```

---

## Applications of Machine Learning

### By Domain

#### Healthcare
| Application | Type | Description |
|-------------|------|-------------|
| Disease Diagnosis | Classification | Predict disease from symptoms/images |
| Drug Discovery | Regression/Classification | Predict drug effectiveness |
| Medical Imaging | Classification | Detect tumors, abnormalities |
| Personalized Treatment | Regression | Predict optimal dosage |

#### Finance
| Application | Type | Description |
|-------------|------|-------------|
| Fraud Detection | Classification | Identify fraudulent transactions |
| Credit Scoring | Classification | Predict loan default risk |
| Stock Prediction | Regression | Predict stock prices |
| Algorithmic Trading | Reinforcement | Automated trading decisions |

#### E-commerce
| Application | Type | Description |
|-------------|------|-------------|
| Recommendation Systems | Classification | Suggest products |
| Price Optimization | Regression | Dynamic pricing |
| Customer Segmentation | Clustering | Group customers |
| Churn Prediction | Classification | Predict customer leaving |

#### Technology
| Application | Type | Description |
|-------------|------|-------------|
| Search Engines | Ranking | Order search results |
| Speech Recognition | Classification | Convert speech to text |
| Image Recognition | Classification | Identify objects |
| NLP | Various | Language understanding |

---

## Key Formulas and Concepts

### Quick Reference

| Concept | Formula/Definition |
|---------|---------------------|
| **Min-Max Scaling** | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ |
| **Standardization** | $x' = \frac{x - \mu}{\sigma}$ |
| **Total Error** | Bias² + Variance + Noise |
| **Overfitting** | Low train error, High test error |
| **Underfitting** | High train error, High test error |

---

## Common Exam Questions

```{admonition} Q1: Differentiate between supervised and unsupervised learning
:class: hint
**Supervised**: Uses labeled data, learns input-output mapping, used for classification/regression.

**Unsupervised**: Uses unlabeled data, finds hidden patterns, used for clustering/dimensionality reduction.

```

```{admonition} Q2: What is overfitting? How to prevent it?
:class: hint
**Overfitting**: Model memorizes training data, performs poorly on new data.

**Prevention**: Regularization, cross-validation, more data, simpler model, early stopping, dropout.

```

```{admonition} Q3: Explain bias-variance tradeoff
:class: hint
- **Bias**: Error from simplistic assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)
- **Tradeoff**: Reducing one often increases the other
- **Goal**: Find optimal model complexity minimizing total error

```

```{admonition} Q4: Why split data into train/validation/test?
:class: hint
- **Training**: Model learns parameters
- **Validation**: Tune hyperparameters, prevent overfitting
- **Test**: Final unbiased evaluation
- Prevents overfitting to test data, gives realistic performance estimate

```

---

## Important Points to Remember

✅ **Machine Learning**: Systems that learn from data without explicit programming

✅ **Supervised Learning**: Learn from labeled data (classification/regression)

✅ **Unsupervised Learning**: Find patterns in unlabeled data (clustering/dimensionality reduction)

✅ **Reinforcement Learning**: Learn through rewards and penalties

✅ **Overfitting**: Model too complex, memorizes training data

✅ **Underfitting**: Model too simple, can't learn patterns

✅ **Bias-Variance Tradeoff**: Balance between model complexity and generalization

✅ **Data Splitting**: Train (learn) → Validation (tune) → Test (evaluate)

✅ **Feature Scaling**: Normalize features for better algorithm performance

✅ **Cross-Validation**: Robust performance estimation using multiple folds

---

**Next**: [Module 2 - Supervised Learning](module2-supervised-learning.md)
