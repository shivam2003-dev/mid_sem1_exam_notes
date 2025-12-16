# Module 4: Unsupervised Learning

## Overview

Unsupervised learning finds hidden patterns in data **without labeled outputs**. This module covers clustering algorithms (K-Means, Hierarchical), dimensionality reduction (PCA), and association rules.

!!! info "Key Difference from Supervised Learning"
    - **Supervised**: "Here's the input and the correct answer, learn the pattern"
    - **Unsupervised**: "Here's the data, find interesting patterns on your own"

---

## Clustering

### What is Clustering?

**Clustering** groups similar data points together based on their features. The goal is to find natural groupings in data.

!!! success "Definition"
    Clustering partitions a dataset into groups (clusters) such that:
    
    - Points **within** a cluster are similar to each other
    - Points **across** clusters are dissimilar

### Applications of Clustering

| Domain | Application | Description |
|--------|-------------|-------------|
| **Marketing** | Customer Segmentation | Group customers by behavior for targeted campaigns |
| **Biology** | Gene Expression | Group genes with similar expression patterns |
| **Image Processing** | Image Segmentation | Separate foreground from background |
| **Security** | Anomaly Detection | Identify unusual patterns (fraud, intrusion) |
| **Social Media** | Community Detection | Find groups of related users |

### Key Concepts

| Term | Definition |
|------|------------|
| **Cluster** | Group of similar data points |
| **Centroid** | Center point of a cluster (mean of all points) |
| **Distance Metric** | How we measure similarity between points |
| **Inertia** | Sum of squared distances from points to their centroids |

---

## K-Means Clustering

### Algorithm Overview

**K-Means** partitions data into **K clusters** by minimizing the within-cluster variance.

!!! note "Key Idea"
    Iteratively assign points to nearest centroid, then update centroids to be the mean of assigned points.

### Algorithm Steps (Detailed)

```
K-Means Algorithm:

Input: Dataset X, number of clusters K
Output: K clusters with centroids

1. INITIALIZE: Randomly select K data points as initial centroids
   μ₁, μ₂, ..., μₖ

2. REPEAT until convergence:
   
   a. ASSIGNMENT STEP:
      For each data point xᵢ:
        - Calculate distance to all K centroids
        - Assign xᵢ to cluster with nearest centroid
        - cᵢ = argmin_k ||xᵢ - μₖ||²
   
   b. UPDATE STEP:
      For each cluster k:
        - Recalculate centroid as mean of all assigned points
        - μₖ = (1/|Cₖ|) Σ xᵢ for all xᵢ in cluster k

3. CONVERGENCE: Stop when centroids don't change (or change < threshold)

4. RETURN: Cluster assignments and final centroids
```

### Mathematical Formulation

**Objective Function** (Within-cluster sum of squares - WCSS):

$$
J = \sum_{i=1}^{m} \sum_{k=1}^{K} w_{ik} ||x^{(i)} - \mu_k||^2
$$

Where:
- $m$ = number of data points
- $K$ = number of clusters
- $w_{ik} = 1$ if point $i$ belongs to cluster $k$, else $0$
- $\mu_k$ = centroid of cluster $k$
- $||x^{(i)} - \mu_k||^2$ = squared Euclidean distance

**Goal**: Minimize $J$ (minimize total within-cluster variance)

### Distance Metrics

#### Euclidean Distance (Most Common)

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} = ||x - y||_2
$$

!!! tip "When to Use"
    Best for continuous data where all features have similar scales.

#### Manhattan Distance (L1)

$$
d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
$$

!!! tip "When to Use"
    Better when dealing with high-dimensional data or when outliers are present.

#### Cosine Similarity

$$
\text{similarity}(x, y) = \frac{x \cdot y}{||x|| \cdot ||y||} = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2} \cdot \sqrt{\sum_{i=1}^{n} y_i^2}}
$$

!!! tip "When to Use"
    Best for text data and high-dimensional sparse data (measures angle, not magnitude).

---

### Worked Example: K-Means Step by Step

!!! example "Problem"
    Cluster the following 2D points into K=2 clusters:
    
    | Point | x₁ | x₂ |
    |-------|----|----|
    | A | 1 | 1 |
    | B | 1.5 | 2 |
    | C | 3 | 4 |
    | D | 5 | 7 |
    | E | 3.5 | 5 |
    | F | 4.5 | 5 |

**Step 1: Initialize Centroids**

Randomly select K=2 points as initial centroids:
- $\mu_1 = A = (1, 1)$
- $\mu_2 = D = (5, 7)$

**Step 2: First Assignment Step**

Calculate distance from each point to both centroids:

| Point | Distance to μ₁ | Distance to μ₂ | Assigned Cluster |
|-------|----------------|----------------|------------------|
| A (1,1) | 0 | $\sqrt{(5-1)^2+(7-1)^2} = \sqrt{52} = 7.21$ | **Cluster 1** |
| B (1.5,2) | $\sqrt{0.5^2+1^2} = 1.12$ | $\sqrt{3.5^2+5^2} = 6.10$ | **Cluster 1** |
| C (3,4) | $\sqrt{2^2+3^2} = 3.61$ | $\sqrt{2^2+3^2} = 3.61$ | **Cluster 1** (tie, choose 1) |
| D (5,7) | $\sqrt{4^2+6^2} = 7.21$ | 0 | **Cluster 2** |
| E (3.5,5) | $\sqrt{2.5^2+4^2} = 4.72$ | $\sqrt{1.5^2+2^2} = 2.50$ | **Cluster 2** |
| F (4.5,5) | $\sqrt{3.5^2+4^2} = 5.32$ | $\sqrt{0.5^2+2^2} = 2.06$ | **Cluster 2** |

**Cluster 1**: {A, B, C}
**Cluster 2**: {D, E, F}

**Step 3: First Update Step**

Calculate new centroids:

$$
\mu_1 = \frac{1}{3}[(1,1) + (1.5,2) + (3,4)] = \left(\frac{5.5}{3}, \frac{7}{3}\right) = (1.83, 2.33)
$$

$$
\mu_2 = \frac{1}{3}[(5,7) + (3.5,5) + (4.5,5)] = \left(\frac{13}{3}, \frac{17}{3}\right) = (4.33, 5.67)
$$

**Step 4: Second Assignment Step**

Recalculate distances with new centroids and reassign...

(Continue until centroids don't change)

**Final Result**: 
- **Cluster 1**: {A, B, C} - Lower-left points
- **Cluster 2**: {D, E, F} - Upper-right points

---

### Choosing K: The Elbow Method

!!! question "How do we choose the optimal number of clusters K?"

**Elbow Method**:

1. Run K-Means for K = 1, 2, 3, ..., n
2. Calculate WCSS (Within-Cluster Sum of Squares) for each K
3. Plot K vs WCSS
4. Look for the "elbow" - where WCSS decreases sharply then levels off

```
WCSS
  |
  |\
  | \
  |  \
  |   \___________  ← Elbow at K=3
  |        
  └────────────────
    1  2  3  4  5  K
```

!!! tip "Exam Tip"
    The elbow point represents the optimal K - adding more clusters doesn't significantly reduce WCSS.

### Other Methods for Choosing K

| Method | Description |
|--------|-------------|
| **Silhouette Score** | Measures how similar points are to their own cluster vs other clusters |
| **Gap Statistic** | Compares WCSS to expected WCSS under null distribution |
| **Domain Knowledge** | Use prior knowledge (e.g., 3 customer types) |

### K-Means++ Initialization

!!! warning "Problem with Random Initialization"
    Random initialization can lead to poor local minima and inconsistent results.

**K-Means++ Algorithm**:

1. Choose first centroid randomly from data points
2. For each remaining centroid:
   - Calculate distance $D(x)$ from each point to nearest existing centroid
   - Choose next centroid with probability proportional to $D(x)^2$
3. Points far from existing centroids are more likely to be chosen

**Benefits**:
- More spread out initial centroids
- Faster convergence
- Better final clusters

### Advantages and Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| ✅ Simple and intuitive | ❌ Must specify K beforehand |
| ✅ Fast: O(n·K·iterations) | ❌ Sensitive to initialization |
| ✅ Scales to large datasets | ❌ Assumes spherical clusters |
| ✅ Guaranteed to converge | ❌ Sensitive to outliers |
| ✅ Works well for compact clusters | ❌ Struggles with varying cluster sizes |

!!! warning "Common Mistakes"
    1. **Not scaling features**: K-Means uses distance, so features with larger ranges dominate
    2. **Wrong K**: Always use elbow method or domain knowledge
    3. **Single run**: Always run multiple times with different initializations

---

## Hierarchical Clustering

### Overview

**Hierarchical Clustering** creates a tree-like structure (dendrogram) of clusters without pre-specifying the number of clusters.

### Types of Hierarchical Clustering

#### 1. Agglomerative (Bottom-Up) - Most Common

```
Start: Each point is its own cluster
       ↓
Merge closest clusters
       ↓
Repeat until one cluster remains
       ↓
Result: Dendrogram (tree structure)
```

#### 2. Divisive (Top-Down)

```
Start: All points in one cluster
       ↓
Split clusters
       ↓
Repeat until each point is its own cluster
       ↓
Result: Dendrogram
```

### Agglomerative Algorithm

```
Agglomerative Clustering:

1. INITIALIZE: Create n clusters (one per data point)

2. COMPUTE: Distance matrix between all pairs of clusters

3. REPEAT until one cluster remains:
   a. Find two closest clusters
   b. Merge them into one cluster
   c. Update distance matrix

4. RESULT: Dendrogram showing merge history
```

### Linkage Criteria

**How do we measure distance between clusters?**

#### 1. Single Linkage (Minimum)

$$
d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)
$$

- Distance = minimum distance between any two points
- **Pros**: Can find elongated clusters
- **Cons**: Chaining effect (long, stringy clusters)

#### 2. Complete Linkage (Maximum)

$$
d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)
$$

- Distance = maximum distance between any two points
- **Pros**: Produces compact, spherical clusters
- **Cons**: Sensitive to outliers

#### 3. Average Linkage (UPGMA)

$$
d(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)
$$

- Distance = average of all pairwise distances
- **Pros**: Balanced approach, less sensitive to outliers
- **Cons**: Computationally more expensive

#### 4. Centroid Linkage

$$
d(C_i, C_j) = d(\mu_i, \mu_j)
$$

- Distance = distance between cluster centroids
- **Pros**: Intuitive
- **Cons**: Can lead to inversions in dendrogram

#### 5. Ward's Method

$$
d(C_i, C_j) = \text{Increase in total within-cluster variance after merging}
$$

- Minimizes total within-cluster variance
- **Pros**: Produces compact, similar-sized clusters
- **Cons**: Assumes spherical clusters

### Dendrogram

!!! success "Definition"
    A **dendrogram** is a tree diagram showing the hierarchical relationship between clusters.

```
Height (Distance)
    |
  6 |     ┌───────────────┐
    |     │               │
  4 |   ┌─┴─┐           ┌─┴─┐
    |   │   │           │   │
  2 | ┌─┴─┐ │         ┌─┴─┐ │
    | │   │ │         │   │ │
  0 | A   B C         D   E F
    └─────────────────────────
```

**How to Read**:
- **Leaves** (bottom): Individual data points
- **Branches**: Clusters at different levels
- **Height**: Distance at which clusters merge
- **Horizontal cut**: Determines number of clusters

!!! tip "Choosing Number of Clusters"
    Draw a horizontal line at desired height - number of vertical lines it crosses = number of clusters.

### Worked Example: Hierarchical Clustering

!!! example "Problem"
    Cluster points A(0,0), B(1,0), C(4,0), D(5,0) using single linkage.

**Distance Matrix**:

|   | A | B | C | D |
|---|---|---|---|---|
| A | 0 | 1 | 4 | 5 |
| B | 1 | 0 | 3 | 4 |
| C | 4 | 3 | 0 | 1 |
| D | 5 | 4 | 1 | 0 |

**Step 1**: Minimum distance = 1 (A-B and C-D)
- Merge A and B → {A,B}
- Merge C and D → {C,D}

**Step 2**: Update distances (single linkage):
- d({A,B}, {C,D}) = min(d(A,C), d(A,D), d(B,C), d(B,D)) = min(4, 5, 3, 4) = 3

**Step 3**: Merge {A,B} and {C,D} at distance 3

**Dendrogram**:
```
Height
  3 |     ┌───────┐
    |     │       │
  1 |   ┌─┴─┐   ┌─┴─┐
    |   │   │   │   │
  0 |   A   B   C   D
```

### Comparison: K-Means vs Hierarchical

| Aspect | K-Means | Hierarchical |
|--------|---------|--------------|
| **K specification** | Required upfront | Not required |
| **Output** | Flat clusters | Dendrogram (tree) |
| **Complexity** | O(n·K·iterations) | O(n³) or O(n²log n) |
| **Scalability** | Good for large data | Better for small data |
| **Cluster shape** | Spherical | Any shape (with single linkage) |
| **Reproducibility** | Depends on initialization | Deterministic |

---

## Dimensionality Reduction

### Why Reduce Dimensions?

!!! warning "Curse of Dimensionality"
    As dimensions increase:
    
    - Data becomes sparse
    - Distance metrics become less meaningful
    - Computational cost increases exponentially
    - More data needed to avoid overfitting

**Benefits of Dimensionality Reduction**:

1. **Visualization**: Reduce to 2D/3D for plotting
2. **Noise Reduction**: Remove noisy features
3. **Computational Efficiency**: Faster training
4. **Avoid Overfitting**: Fewer features = simpler model
5. **Feature Extraction**: Create meaningful features

### Methods Overview

| Method | Type | Preserves |
|--------|------|-----------|
| **PCA** | Linear | Global variance |
| **t-SNE** | Non-linear | Local structure |
| **LDA** | Linear, Supervised | Class separability |
| **Autoencoders** | Non-linear | Learned representation |

---

## Principal Component Analysis (PCA)

### Key Idea

!!! success "PCA in One Sentence"
    PCA finds new axes (principal components) that capture the **maximum variance** in the data.

**Intuition**: 
- First PC: Direction of maximum variance
- Second PC: Direction of maximum variance perpendicular to first
- And so on...

### Mathematical Foundation

#### Step-by-Step Algorithm

**Step 1: Standardize the Data**

For each feature, compute z-score:

$$
z_i = \frac{x_i - \mu_i}{\sigma_i}
$$

!!! warning "Important"
    Always standardize before PCA! Otherwise, features with larger scales dominate.

**Step 2: Compute Covariance Matrix**

$$
\Sigma = \frac{1}{m-1} X^T X
$$

Where X is the centered data matrix (m samples × n features).

The covariance matrix shows how features vary together:
- Diagonal: Variance of each feature
- Off-diagonal: Covariance between features

**Step 3: Eigenvalue Decomposition**

Find eigenvalues $\lambda_1, \lambda_2, ..., \lambda_n$ and eigenvectors $v_1, v_2, ..., v_n$ such that:

$$
\Sigma v_i = \lambda_i v_i
$$

**Properties**:
- Eigenvalues are non-negative (covariance matrix is positive semi-definite)
- Eigenvectors are orthogonal
- Eigenvalue = variance explained by that component

**Step 4: Sort and Select**

Sort eigenvalues in descending order: $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_n$

Select top k eigenvectors (principal components) that capture desired variance.

**Step 5: Project Data**

$$
Y = X \cdot P_k
$$

Where:
- $X$ = original data (m × n)
- $P_k$ = matrix of top k eigenvectors (n × k)
- $Y$ = reduced data (m × k)

### Variance Explained

**Proportion of Variance Explained by component i**:

$$
\text{Variance Explained}_i = \frac{\lambda_i}{\sum_{j=1}^{n} \lambda_j}
$$

**Cumulative Variance Explained**:

$$
\text{Cumulative}_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{n} \lambda_j}
$$

!!! tip "Rule of Thumb"
    Choose k such that cumulative variance ≥ **95%** (or 90% for more compression).

### Choosing Number of Components

**Method 1: Scree Plot**

```
Eigenvalue
    |
    |*
    | *
    |  *
    |   *___*___*___*  ← Elbow
    └─────────────────
      1  2  3  4  5  k
```

Choose k at the "elbow" where eigenvalues drop sharply.

**Method 2: Cumulative Variance Threshold**

| k | Cumulative Variance |
|---|---------------------|
| 1 | 60% |
| 2 | 85% |
| 3 | **95%** ← Choose this |
| 4 | 99% |

**Method 3: Kaiser Criterion**

Keep components with eigenvalue > 1 (for standardized data).

### Worked Example: PCA

!!! example "Problem"
    Perform PCA on the following 2D data and reduce to 1D:
    
    | Point | x₁ | x₂ |
    |-------|----|----|
    | 1 | 2.5 | 2.4 |
    | 2 | 0.5 | 0.7 |
    | 3 | 2.2 | 2.9 |
    | 4 | 1.9 | 2.2 |
    | 5 | 3.1 | 3.0 |

**Step 1: Calculate means**

$$
\bar{x}_1 = \frac{2.5 + 0.5 + 2.2 + 1.9 + 3.1}{5} = 2.04
$$

$$
\bar{x}_2 = \frac{2.4 + 0.7 + 2.9 + 2.2 + 3.0}{5} = 2.24
$$

**Step 2: Center the data** (subtract means)

| Point | x₁ - 2.04 | x₂ - 2.24 |
|-------|-----------|-----------|
| 1 | 0.46 | 0.16 |
| 2 | -1.54 | -1.54 |
| 3 | 0.16 | 0.66 |
| 4 | -0.14 | -0.04 |
| 5 | 1.06 | 0.76 |

**Step 3: Compute covariance matrix**

$$
\Sigma = \begin{bmatrix} 0.616 & 0.615 \\ 0.615 & 0.716 \end{bmatrix}
$$

**Step 4: Find eigenvalues and eigenvectors**

Eigenvalues: $\lambda_1 = 1.284$, $\lambda_2 = 0.049$

Eigenvectors: $v_1 = [0.677, 0.736]$, $v_2 = [-0.736, 0.677]$

**Step 5: Variance explained**

- PC1: $\frac{1.284}{1.284 + 0.049} = 96.3\%$
- PC2: $\frac{0.049}{1.333} = 3.7\%$

**Conclusion**: First PC captures 96.3% of variance - we can reduce to 1D with minimal information loss!

### Properties of PCA

**Key Properties**:

1. **Orthogonality**: Principal components are uncorrelated
2. **Variance Maximization**: Each PC maximizes remaining variance
3. **Linear**: PCA finds linear combinations of original features
4. **Reversible**: Can reconstruct original data (with some loss)

**Limitations**:

| Limitation | Description |
|------------|-------------|
| **Linear only** | Cannot capture non-linear relationships |
| **Variance-based** | May not preserve class separability |
| **Sensitive to scaling** | Must standardize first |
| **Interpretability** | PCs are combinations of features |

---

## Association Rule Learning

### Introduction

**Association Rules** discover interesting relationships between variables in large datasets.

!!! example "Classic Example: Market Basket Analysis"
    "Customers who buy **bread** and **butter** also buy **milk**"
    
    Rule: {Bread, Butter} → {Milk}

### Key Concepts

#### Itemset

An **itemset** is a collection of items.
- {Bread} - 1-itemset
- {Bread, Butter} - 2-itemset
- {Bread, Butter, Milk} - 3-itemset

#### Support

**Support** measures how frequently an itemset appears in the dataset.

$$
\text{Support}(A) = \frac{\text{Number of transactions containing A}}{\text{Total number of transactions}}
$$

!!! example "Support Example"
    If 100 transactions and 30 contain {Bread, Milk}:
    
    Support({Bread, Milk}) = 30/100 = 0.30 = 30%

#### Confidence

**Confidence** measures how often the rule is true.

$$
\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)} = P(B|A)
$$

!!! example "Confidence Example"
    - Support({Bread}) = 50%
    - Support({Bread, Milk}) = 30%
    
    Confidence(Bread → Milk) = 30%/50% = 60%
    
    "60% of customers who buy bread also buy milk"

#### Lift

**Lift** measures how much more likely B is when A occurs, compared to B occurring independently.

$$
\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)} = \frac{P(B|A)}{P(B)}
$$

**Interpretation**:
- **Lift = 1**: A and B are independent
- **Lift > 1**: Positive correlation (A increases likelihood of B)
- **Lift < 1**: Negative correlation (A decreases likelihood of B)

!!! example "Lift Example"
    - Confidence(Bread → Milk) = 60%
    - Support(Milk) = 40%
    
    Lift = 60%/40% = 1.5
    
    "Customers who buy bread are 1.5× more likely to buy milk"

### Apriori Algorithm

!!! success "Apriori Principle"
    If an itemset is **infrequent**, all its supersets are also infrequent.
    
    Contrapositive: If an itemset is **frequent**, all its subsets are also frequent.

**Algorithm**:

```
Apriori Algorithm:

Input: Transaction database, minimum support, minimum confidence
Output: Association rules

1. Find all frequent 1-itemsets (items with support ≥ min_support)

2. For k = 2, 3, ... until no more frequent itemsets:
   a. Generate candidate k-itemsets from frequent (k-1)-itemsets
   b. Prune candidates with infrequent subsets
   c. Count support for remaining candidates
   d. Keep itemsets with support ≥ min_support

3. Generate rules from frequent itemsets:
   For each frequent itemset I:
     For each non-empty subset A of I:
       Rule: A → (I - A)
       If confidence ≥ min_confidence:
         Output rule
```

### Worked Example: Apriori

!!! example "Problem"
    Given transactions and min_support = 50%, min_confidence = 60%:
    
    | TID | Items |
    |-----|-------|
    | 1 | {Bread, Milk} |
    | 2 | {Bread, Butter, Milk} |
    | 3 | {Bread, Butter} |
    | 4 | {Milk, Eggs} |

**Step 1: Count 1-itemsets**

| Item | Count | Support |
|------|-------|---------|
| Bread | 3 | 75% ✓ |
| Milk | 3 | 75% ✓ |
| Butter | 2 | 50% ✓ |
| Eggs | 1 | 25% ✗ |

Frequent 1-itemsets: {Bread}, {Milk}, {Butter}

**Step 2: Generate and count 2-itemsets**

| Itemset | Count | Support |
|---------|-------|---------|
| {Bread, Milk} | 2 | 50% ✓ |
| {Bread, Butter} | 2 | 50% ✓ |
| {Milk, Butter} | 1 | 25% ✗ |

Frequent 2-itemsets: {Bread, Milk}, {Bread, Butter}

**Step 3: Generate 3-itemsets**

Candidate: {Bread, Milk, Butter}
- Subset {Milk, Butter} is infrequent → **Prune**

No frequent 3-itemsets.

**Step 4: Generate Rules**

From {Bread, Milk}:
- Bread → Milk: Confidence = 50%/75% = 67% ✓
- Milk → Bread: Confidence = 50%/75% = 67% ✓

From {Bread, Butter}:
- Bread → Butter: Confidence = 50%/75% = 67% ✓
- Butter → Bread: Confidence = 50%/50% = 100% ✓

**Final Rules** (confidence ≥ 60%):
1. Bread → Milk (67%)
2. Milk → Bread (67%)
3. Bread → Butter (67%)
4. Butter → Bread (100%)

---

## Key Formulas Summary

### K-Means

| Formula | Description |
|---------|-------------|
| $J = \sum_{i=1}^{m} \sum_{k=1}^{K} w_{ik} \|\|x^{(i)} - \mu_k\|\|^2$ | Objective function |
| $\mu_k = \frac{1}{\|C_k\|} \sum_{x \in C_k} x$ | Centroid update |
| $d(x, y) = \sqrt{\sum_i (x_i - y_i)^2}$ | Euclidean distance |

### PCA

| Formula | Description |
|---------|-------------|
| $\Sigma = \frac{1}{m-1} X^T X$ | Covariance matrix |
| $\Sigma v = \lambda v$ | Eigenvalue equation |
| $\frac{\lambda_i}{\sum_j \lambda_j}$ | Variance explained |
| $Y = X \cdot P_k$ | Projection |

### Association Rules

| Formula | Description |
|---------|-------------|
| $\text{Support}(A) = \frac{\text{Count}(A)}{N}$ | Support |
| $\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$ | Confidence |
| $\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$ | Lift |

---

## Common Exam Questions

!!! question "Q1: Explain K-Means algorithm with example"
    1. Initialize K centroids randomly
    2. Assign each point to nearest centroid
    3. Update centroids as mean of assigned points
    4. Repeat until convergence
    
    (Include numerical example with distance calculations)

!!! question "Q2: How to choose optimal K in K-Means?"
    **Elbow Method**: Plot WCSS vs K, choose K at the elbow where decrease slows down.
    
    **Silhouette Score**: Measures cluster quality, choose K with highest score.
    
    **Domain Knowledge**: Use prior knowledge about expected clusters.

!!! question "Q3: Explain PCA and its applications"
    PCA finds principal components (directions of maximum variance) to reduce dimensionality.
    
    **Steps**: Standardize → Covariance matrix → Eigendecomposition → Select top k → Project
    
    **Applications**: Visualization, noise reduction, feature extraction, preprocessing

!!! question "Q4: Calculate Support, Confidence, Lift"
    Given transactions, calculate:
    - Support = Count(itemset) / Total transactions
    - Confidence = Support(A∪B) / Support(A)
    - Lift = Confidence / Support(B)

---

## Important Points to Remember

✅ **K-Means**: Partition into K clusters, minimize within-cluster variance

✅ **Elbow Method**: Plot cost vs K, choose at elbow

✅ **K-Means++**: Better initialization, spread out centroids

✅ **Hierarchical**: Creates dendrogram, no need to specify K

✅ **Linkage**: Single (min), Complete (max), Average, Ward's

✅ **PCA**: Find directions of maximum variance, reduce dimensions

✅ **Variance Explained**: Choose k for 95% cumulative variance

✅ **Association Rules**: Support, Confidence, Lift

✅ **Apriori**: Frequent itemset mining using downward closure

✅ **Feature Scaling**: Critical for K-Means and PCA

---

**Previous**: [Module 3 - Classification & Evaluation](module3-classification-evaluation.md) | **Next**: [Module 5 - Decision Trees](module5-decision-trees.md)
