# Module 4: Unsupervised Learning

## Overview

Unsupervised learning finds hidden patterns in data without labeled outputs. This module covers clustering, dimensionality reduction, and association rules.

---

## Clustering

### Introduction

**Clustering** groups similar data points together. The goal is to find natural groupings in data.

**Applications**:
- Customer segmentation
- Image segmentation
- Anomaly detection
- Document clustering
- Market research

**Key Concepts**:
- **Cluster**: Group of similar data points
- **Centroid**: Center point of a cluster
- **Distance Metric**: How to measure similarity

---

## K-Means Clustering

### Algorithm

**K-Means** partitions data into $K$ clusters by minimizing within-cluster variance.

**Steps**:

1. **Initialize**: Randomly choose $K$ cluster centroids
   - Can use random data points or random positions

2. **Assignment Step**: Assign each data point to nearest centroid
   - Calculate distance to all centroids
   - Assign to closest one

3. **Update Step**: Recalculate centroids
   - New centroid = mean of all points in cluster

4. **Repeat**: Steps 2-3 until convergence
   - Centroids don't change (or change < threshold)
   - Maximum iterations reached

**Convergence**: When assignments don't change between iterations

### Mathematical Formulation

**Objective Function** (Within-cluster sum of squares):

$$J = \sum_{i=1}^{m} \sum_{k=1}^{K} w_{ik} ||x^{(i)} - \mu_k||^2$$

Where:
- $m$ = number of data points
- $K$ = number of clusters
- $w_{ik} = 1$ if point $i$ belongs to cluster $k$, else $0$
- $\mu_k$ = centroid of cluster $k$
- $||x^{(i)} - \mu_k||^2$ = squared distance from point to centroid

**Goal**: Minimize $J$

### Distance Metrics

**Euclidean Distance** (most common):
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

**Manhattan Distance**:
$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

**Cosine Similarity** (for high-dimensional data):
$$\text{similarity} = \frac{x \cdot y}{||x|| \cdot ||y||}$$

### Choosing K

**Methods**:

1. **Elbow Method**:
   - Plot $J$ (cost) vs $K$
   - Look for "elbow" where cost decreases sharply then plateaus
   - Choose $K$ at elbow

2. **Domain Knowledge**:
   - Use prior knowledge about data
   - Example: If segmenting customers into 3 groups, use $K = 3$

3. **Cross-Validation**:
   - Evaluate clustering quality for different $K$
   - Use metric like silhouette score

### Initialization

**Problem**: K-Means can converge to local minima

**Solutions**:

1. **Multiple Random Initializations**:
   - Run algorithm multiple times with different random starts
   - Choose result with lowest cost

2. **K-Means++** (Better initialization):
   - Choose first centroid randomly
   - Choose subsequent centroids far from existing ones
   - Reduces chance of poor local minima

### Advantages and Disadvantages

**Advantages**:
- ✅ Simple and fast
- ✅ Works well with spherical clusters
- ✅ Scales to large datasets

**Disadvantages**:
- ❌ Need to specify $K$ beforehand
- ❌ Sensitive to initialization
- ❌ Assumes spherical clusters
- ❌ Sensitive to outliers

### K-Means Algorithm Summary

```
1. Randomly initialize K centroids
2. Repeat until convergence:
   a. For each point, assign to nearest centroid
   b. For each cluster, update centroid (mean of points)
3. Return clusters and centroids
```

---

## Hierarchical Clustering

### Introduction

**Hierarchical Clustering** creates a tree of clusters (dendrogram) without pre-specifying number of clusters.

**Types**:

1. **Agglomerative** (Bottom-up):
   - Start with each point as its own cluster
   - Merge closest clusters iteratively
   - Continue until one cluster remains

2. **Divisive** (Top-down):
   - Start with all points in one cluster
   - Split clusters iteratively
   - Continue until each point is its own cluster

### Agglomerative Clustering Algorithm

**Steps**:

1. **Initialize**: Each data point is its own cluster

2. **Compute Distance Matrix**: Calculate distances between all clusters

3. **Merge Closest Clusters**: Combine two clusters with minimum distance

4. **Update Distance Matrix**: Recalculate distances to new cluster

5. **Repeat**: Steps 3-4 until one cluster remains

### Linkage Criteria

How to measure distance between clusters:

1. **Single Linkage** (Minimum):
   - Distance = minimum distance between any two points in clusters
   - Formula: $d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)$
   - Tends to create long, chain-like clusters

2. **Complete Linkage** (Maximum):
   - Distance = maximum distance between any two points
   - Formula: $d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)$
   - Tends to create compact, spherical clusters

3. **Average Linkage**:
   - Distance = average distance between all pairs
   - Formula: $d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)$
   - Balanced approach

4. **Centroid Linkage**:
   - Distance = distance between cluster centroids
   - Formula: $d(C_i, C_j) = d(\mu_i, \mu_j)$

### Dendrogram

**Definition**: Tree diagram showing cluster hierarchy

**How to Read**:
- **Leaves**: Individual data points
- **Branches**: Clusters at different levels
- **Height**: Distance at which clusters merge
- **Cut**: Horizontal line determines number of clusters

**To Get K Clusters**: Cut dendrogram at height that gives K clusters

### Advantages and Disadvantages

**Advantages**:
- ✅ No need to specify K
- ✅ Produces interpretable dendrogram
- ✅ Works with any distance metric

**Disadvantages**:
- ❌ Computationally expensive: O(n³) time complexity
- ❌ Sensitive to noise and outliers
- ❌ Once merged, clusters can't be split

---

## Dimensionality Reduction

### Introduction

**Dimensionality Reduction** reduces number of features while preserving important information.

**Goals**:
- Reduce computational cost
- Remove noise and redundancy
- Visualize high-dimensional data
- Prevent overfitting (curse of dimensionality)

**Methods**:
- **Feature Selection**: Choose subset of original features
- **Feature Extraction**: Create new features from original ones

---

## Principal Component Analysis (PCA)

### Introduction

**PCA** finds directions (principal components) of maximum variance in data and projects data onto these directions.

**Key Idea**: 
- First principal component: Direction of maximum variance
- Second principal component: Direction of maximum variance orthogonal to first
- And so on...

### Mathematical Foundation

**Steps**:

1. **Standardize Data**: Mean = 0, Std = 1
   $$z_i = \frac{x_i - \mu}{\sigma}$$

2. **Compute Covariance Matrix**:
   $$\Sigma = \frac{1}{m} X^T X$$

3. **Eigenvalue Decomposition**:
   $$\Sigma = P \Lambda P^T$$
   - $P$ = matrix of eigenvectors (principal components)
   - $\Lambda$ = diagonal matrix of eigenvalues

4. **Select Top k Components**:
   - Choose eigenvectors corresponding to largest eigenvalues
   - These capture most variance

5. **Project Data**:
   $$Y = X P_k$$
   - $P_k$ = first $k$ principal components
   - $Y$ = reduced dimension data

### Variance Explained

**Proportion of Variance Explained**:
$$\text{Variance Explained} = \frac{\lambda_i}{\sum_{j=1}^{n} \lambda_j}$$

**Cumulative Variance**:
$$\text{Cumulative} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{n} \lambda_j}$$

**Rule of Thumb**: Choose $k$ such that cumulative variance ≥ 0.95 (95% variance retained)

### Choosing Number of Components

**Methods**:

1. **Scree Plot**:
   - Plot eigenvalues vs component number
   - Look for "elbow" where eigenvalues drop sharply

2. **Variance Threshold**:
   - Keep components explaining ≥ threshold (e.g., 95%) variance

3. **Kaiser Criterion**:
   - Keep components with eigenvalue > 1

### Properties of PCA

**Properties**:
- Principal components are orthogonal (uncorrelated)
- First component captures maximum variance
- Components are linear combinations of original features
- Preserves global structure

**Limitations**:
- ❌ Assumes linear relationships
- ❌ Sensitive to feature scaling
- ❌ May not preserve local structure
- ❌ Interpretability can be lost

### Applications

- **Data Visualization**: Reduce to 2D/3D for plotting
- **Noise Reduction**: Remove components with low variance
- **Feature Extraction**: Create new features for ML models
- **Compression**: Reduce storage and computation

### PCA Algorithm Summary

```
1. Standardize the data (mean=0, std=1)
2. Compute covariance matrix
3. Find eigenvalues and eigenvectors
4. Sort by eigenvalues (descending)
5. Select top k eigenvectors
6. Project data onto selected components
```

---

## Association Rule Learning

### Introduction

**Association Rules** discover interesting relationships between variables in large datasets.

**Example**: Market Basket Analysis
- "If customer buys bread and butter, they also buy milk"
- Rule: {Bread, Butter} → {Milk}

### Key Concepts

**Itemset**: Set of items (e.g., {Bread, Butter})

**Support**: Frequency of itemset in dataset
$$\text{Support}(A) = \frac{\text{Count}(A)}{N}$$

**Confidence**: Probability that B occurs given A
$$\text{Confidence}(A \to B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)} = P(B|A)$$

**Lift**: How much more likely B is when A occurs
$$\text{Lift}(A \to B) = \frac{\text{Confidence}(A \to B)}{\text{Support}(B)} = \frac{P(B|A)}{P(B)}$$

**Interpretation**:
- Lift = 1: A and B independent
- Lift > 1: Positive correlation
- Lift < 1: Negative correlation

### Apriori Algorithm

**Principle**: If itemset is frequent, all its subsets are frequent

**Steps**:

1. **Find Frequent 1-itemsets**: Items with support ≥ minimum support

2. **Generate Candidate k-itemsets**: From frequent (k-1)-itemsets

3. **Prune**: Remove candidates with infrequent subsets

4. **Count Support**: For remaining candidates

5. **Filter**: Keep only frequent itemsets

6. **Repeat**: Until no more frequent itemsets

7. **Generate Rules**: From frequent itemsets with confidence ≥ minimum confidence

**Example**:
- Minimum Support = 2
- Minimum Confidence = 50%

**Transactions**:
1. {Bread, Milk}
2. {Bread, Butter, Milk}
3. {Bread, Eggs}
4. {Milk, Eggs}

**Frequent 1-itemsets**: {Bread: 3}, {Milk: 3}, {Butter: 1}, {Eggs: 2}

**Frequent 2-itemsets**: {Bread, Milk: 2}, {Bread, Eggs: 1}, {Milk, Eggs: 1}

**Rules**:
- {Bread} → {Milk}: Confidence = 2/3 = 67% ✓
- {Milk} → {Bread}: Confidence = 2/3 = 67% ✓

---

## Key Formulas Summary

### K-Means
- **Cost Function**: $J = \sum_{i=1}^{m} \sum_{k=1}^{K} w_{ik} ||x^{(i)} - \mu_k||^2$
- **Centroid Update**: $\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x$

### PCA
- **Covariance Matrix**: $\Sigma = \frac{1}{m} X^T X$
- **Variance Explained**: $\frac{\lambda_i}{\sum \lambda_j}$

### Association Rules
- **Support**: $\frac{\text{Count}(A)}{N}$
- **Confidence**: $\frac{\text{Support}(A \cup B)}{\text{Support}(A)}$
- **Lift**: $\frac{\text{Confidence}(A \to B)}{\text{Support}(B)}$

---

## Important Points to Remember

✅ **K-Means**: Partition into K clusters, minimize within-cluster variance

✅ **Hierarchical Clustering**: Creates dendrogram, no need to specify K

✅ **PCA**: Finds directions of maximum variance, reduces dimensions

✅ **Association Rules**: Discover relationships using support, confidence, lift

✅ **Choose K**: Elbow method, domain knowledge, cross-validation

✅ **Feature Scaling**: Important for K-Means and PCA

---

**Previous**: [Module 3 - Classification & Evaluation](module3-classification-evaluation.md) | **Next**: [Module 5 - Decision Trees](module5-decision-trees.md)

