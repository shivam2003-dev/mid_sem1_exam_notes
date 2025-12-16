# Module 6: Convolutional Neural Networks (CNN)

## Overview

This module covers Convolutional Neural Networks (CNNs), specialized neural networks for processing grid-like data such as images.

---

## Introduction to CNNs

### Why CNNs for Images?

**Problems with Fully Connected Networks**:
- Too many parameters for images (e.g., 1000×1000 image = 1M parameters per neuron!)
- Doesn't exploit spatial structure
- Translation sensitive

**CNN Advantages**:
- **Parameter Sharing**: Same filter used across image
- **Sparse Connectivity**: Each neuron connects to small region
- **Translation Invariance**: Can detect features anywhere in image

### Key Idea

**Local Receptive Fields**: Each neuron connects to a small local region of input, not all pixels.

---

## CNN Architecture Components

### 1. Convolutional Layer

**Operation**: Apply filters (kernels) to input

**Convolution Operation**:

$$
(f * g)(i, j) = \sum_{m} \sum_{n} f(m, n) \cdot g(i-m, j-n)
$$

**In CNN context**:

$$
\text{Output}(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \text{Input}(i+m, j+n) \cdot \text{Filter}(m, n) + b
$$

Where:
- $k$ = filter size (e.g., 3×3, 5×5)
- $b$ = bias term

**Example**: 3×3 Filter on 5×5 Image

```
Input Image          Filter          Output
[1 2 3 4 5]        [1 0 -1]        [ 0  2  2]
[6 7 8 9 0]    *   [1 0 -1]    =   [-2  0  2]
[1 2 3 4 5]        [1 0 -1]        [ 0  2  2]
[6 7 8 9 0]
[1 2 3 4 5]
```

**Output Size**:

$$
\text{Output Size} = \frac{\text{Input Size} - \text{Filter Size} + 2 \times \text{Padding}}{\text{Stride}} + 1
$$

**Parameters**:
- **Filter Size**: Typically 3×3 or 5×5
- **Number of Filters**: Depth of output feature map
- **Stride**: Step size (typically 1 or 2)
- **Padding**: Zero-padding around input (typically "same" or "valid")

### 2. Pooling Layer

**Purpose**: Reduce spatial dimensions, reduce parameters, provide translation invariance

**Types**:

**Max Pooling**:
$$
\text{Output}(i, j) = \max_{m,n \in \text{window}} \text{Input}(i+m, j+n)
$$

**Average Pooling**:
$$
\text{Output}(i, j) = \frac{1}{k^2} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \text{Input}(i+m, j+n)
$$

**Common Sizes**: 2×2 with stride 2 (reduces size by half)

**Example**: 2×2 Max Pooling

```
Input:              Output:
[1 3 2 4]          [3 4]
[2 1 3 2]    →     [2 3]
[1 2 3 1]
[2 3 1 2]
```

### 3. Fully Connected Layer

**Purpose**: Final classification/regression

**Structure**: Same as regular neural network layer

**Input**: Flattened feature maps from previous layers

---

## Complete CNN Architecture

### Typical Structure

```
Input Image (e.g., 224×224×3)
    ↓
Conv Layer 1 (e.g., 32 filters, 3×3) → ReLU
    ↓
Max Pooling (2×2)
    ↓
Conv Layer 2 (e.g., 64 filters, 3×3) → ReLU
    ↓
Max Pooling (2×2)
    ↓
Conv Layer 3 (e.g., 128 filters, 3×3) → ReLU
    ↓
Max Pooling (2×2)
    ↓
Flatten
    ↓
Fully Connected Layer 1 (e.g., 512 neurons) → ReLU
    ↓
Fully Connected Layer 2 (e.g., 256 neurons) → ReLU
    ↓
Output Layer (e.g., 10 classes) → Softmax
```

### Example: LeNet-5 (Simplified)

**Architecture**:
1. Conv: 6 filters, 5×5 → ReLU
2. Max Pool: 2×2
3. Conv: 16 filters, 5×5 → ReLU
4. Max Pool: 2×2
5. FC: 120 neurons → ReLU
6. FC: 84 neurons → ReLU
7. Output: 10 classes → Softmax

---

## Convolution Operation Details

### Stride

**Stride = 1** (default):
- Filter moves 1 pixel at a time
- More overlap, larger output

**Stride = 2**:
- Filter moves 2 pixels at a time
- Less overlap, smaller output (half size)

**Output Size with Stride**:

$$
\text{Output Height} = \left\lfloor \frac{H - F + 2P}{S} \right\rfloor + 1
$$

$$
\text{Output Width} = \left\lfloor \frac{W - F + 2P}{S} \right\rfloor + 1
$$

Where:
- $H, W$ = input height, width
- $F$ = filter size
- $P$ = padding
- $S$ = stride

### Padding

**Valid Padding** (no padding):
- Output size < Input size
- Formula: $\text{Output} = \text{Input} - \text{Filter} + 1$

**Same Padding** (zero padding):
- Output size = Input size (when stride = 1)
- Padding: $P = \frac{F-1}{2}$ (for odd filter sizes)

**Example**: 5×5 input, 3×3 filter
- Valid: Output = 3×3
- Same (P=1): Output = 5×5

---

## Backpropagation in CNNs

### Convolutional Layer Backpropagation

**Forward**:
$$
y_{i,j} = \sum_{m} \sum_{n} x_{i+m, j+n} \cdot w_{m,n} + b
$$

**Backward** (gradient w.r.t. filter):

$$
\frac{\partial J}{\partial w_{m,n}} = \sum_{i} \sum_{j} \frac{\partial J}{\partial y_{i,j}} \cdot x_{i+m, j+n}
$$

**Gradient w.r.t. input**:

$$
\frac{\partial J}{\partial x_{i,j}} = \sum_{m} \sum_{n} \frac{\partial J}{\partial y_{i-m, j-n}} \cdot w_{m,n}
$$

**Key Insight**: Backpropagation in convolution uses **correlation** (flipped convolution).

### Pooling Layer Backpropagation

**Max Pooling**:
- Gradient flows only to the maximum value in each window
- Other positions get zero gradient

**Average Pooling**:
- Gradient distributed equally to all positions in window

---

## Common CNN Architectures

### 1. LeNet-5 (1998)
- First successful CNN
- Handwritten digit recognition
- 5 layers

### 2. AlexNet (2012)
- Won ImageNet 2012
- 8 layers
- ReLU activation
- Dropout regularization

### 3. VGG (2014)
- Very deep (16-19 layers)
- Small 3×3 filters
- Simple architecture

### 4. ResNet (2015)
- Residual connections (skip connections)
- Very deep (50-152 layers)
- Solves vanishing gradient

### 5. Modern Architectures
- **Inception**: Multiple filter sizes
- **MobileNet**: Efficient for mobile
- **EfficientNet**: Balanced scaling

---

## Applications of CNNs

### Computer Vision

- **Image Classification**: Identify objects
- **Object Detection**: Locate and classify objects
- **Semantic Segmentation**: Pixel-level classification
- **Face Recognition**: Biometric identification

### Medical Imaging

- **X-ray Analysis**: Disease detection
- **MRI/CT Scan**: Tumor detection
- **Retinal Analysis**: Eye disease diagnosis

### Other Applications

- **Autonomous Vehicles**: Road sign recognition, obstacle detection
- **Security**: Surveillance, anomaly detection
- **Agriculture**: Crop monitoring, disease detection

---

## Key Formulas Summary

### Convolution

$$
\text{Output}(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \text{Input}(i+m, j+n) \cdot \text{Filter}(m, n) + b
$$

### Output Size

$$
\text{Output} = \left\lfloor \frac{\text{Input} - \text{Filter} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1
$$

### Max Pooling

$$
\text{Output}(i, j) = \max_{m,n \in \text{window}} \text{Input}(i+m, j+n)
$$

### Average Pooling

$$
\text{Output}(i, j) = \frac{1}{k^2} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \text{Input}(i+m, j+n)
$$

---

## Important Points to Remember

✅ **CNN**: Specialized for grid-like data (images)

✅ **Convolution**: Apply filters to detect features

✅ **Pooling**: Reduce spatial dimensions, provide invariance

✅ **Parameter Sharing**: Same filter used across image

✅ **Translation Invariance**: Can detect features anywhere

✅ **Architecture**: Conv → Pool → Conv → Pool → ... → FC → Output

✅ **Backpropagation**: Uses correlation (flipped convolution)

---

**Previous**: [Module 5 - Deep Feedforward Neural Networks](module5-dfnn.md) | **Back to**: [DNN Overview](index.md)

