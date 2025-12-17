# Deep Neural Networks (DNN) - Complete Revision Guide

Welcome to the Deep Neural Networks revision guide. This section covers all modules with detailed explanations, formulas, and important concepts.

## 📖 Modules Overview

1. **[Module 1: Introduction to Neural Networks](module1-introduction.md)**
    - Introduction to Neural Networks
    - Biological vs Artificial Neurons
    - History and Evolution
    - Applications of Neural Networks

2. **[Module 2: ANN & Perceptron](module2-ann-perceptron.md)**
    - Artificial Neural Networks (ANN)
    - Perceptron Model
    - Perceptron Learning Algorithm
    - Limitations of Perceptron

3. **[Module 3: Linear NN Regression](module3-linear-nn-regression.md)**
    - Linear Neural Networks for Regression
    - Forward Propagation
    - Backpropagation Algorithm
    - Gradient Computation

4. **[Module 4: Linear NN Classification](module4-linear-nn-classification.md)**
    - Linear Neural Networks for Classification
    - Activation Functions (Sigmoid, Tanh, ReLU)
    - Loss Functions for Classification
    - Multi-class Classification

5. **[Module 5: Deep Feedforward Neural Networks](module5-dfnn.md)**
    - Deep Feedforward Networks
    - Multi-layer Perceptron (MLP)
    - Backpropagation in Deep Networks
    - Vanishing/Exploding Gradients
    - Regularization Techniques

6. **[Module 6: Convolutional Neural Networks](module6-cnn.md)**
    - Introduction to CNNs
    - Convolutional Layers
    - Pooling Layers
    - CNN Architecture
    - Applications

## 📚 Solved Previous Year Papers

- [2024 MidSem Regular Paper - Solved](papers/2024-midsem-regular-solved.md)
- [2024 EndSem Regular Paper - Solved](papers/2024-endsem-regular-solved.md)
- [2023 MidSem Regular Paper - Solved](papers/2023-midsem-regular-solved.md)
- [2023 EndSem Regular Paper - Solved](papers/2023-endsem-regular-solved.md)
- [2022 MidSem Makeup Paper - Solved](papers/2022-midsem-makeup-solved.md)
- [2022 EndSem Regular Paper - Solved](papers/2022-endsem-regular-solved.md)

## 🎯 MidSem Important Topics

### Must Know Concepts

1. **Perceptron**
   - Perceptron Learning Algorithm
   - Perceptron Rule to design logic gates (AND, OR, NOT, XOR, XNOR)
   - Training rule step-by-step
   - Limitations of single-layer perceptron

2. **Forward Propagation**
   - How a neural network predicts output
   - Step-by-step forward pass
   - Computation graphs

3. **Backpropagation**
   - Learning the network
   - Backward differentiation
   - Computation graphs and gradients
   - Step-by-step algorithm

4. **Activation Functions**
   - Sigmoid, Tanh, ReLU
   - Derivatives of activation functions
   - When to use each

5. **Multi-Layer Perceptron (MLP)**
   - Solving XOR problem with MLP
   - Universal approximation theorem
   - Deep feedforward networks

### Important Note

```{admonition} Exam Strategy
:class: warning

**Make sure to go through all the slides and write answers or solutions step-by-step in the exam. Marks are given based on the steps, not just the final answer.**
```

### Key Formulas
- Perceptron Update Rule: $w_j := w_j + \alpha \cdot (y^{(i)} - \hat{y}^{(i)}) \cdot x_j^{(i)}$
- Forward Propagation: $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$, $a^{[l]} = g^{[l]}(z^{[l]})$
- Backpropagation: $\frac{\partial J}{\partial z^{[l]}} = (W^{[l+1]})^T \frac{\partial J}{\partial z^{[l+1]}} \odot g'^{[l]}(z^{[l]})$
- Activation Function Derivatives:
  - Sigmoid: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
  - Tanh: $\tanh'(z) = 1 - \tanh^2(z)$
  - ReLU: $\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$

### Study Resources

- Practice questions: DNN practice qs.pdf
- Past papers topic wise: DNN Past Papers
- Decision tree by Saurabh sir: Decision Trees.pdf
- Resources by Saurabh sir:
  - https://deeplearningwithpython.io/chapters/
  - https://mlu-explain.github.io/neural-networks/
  - https://poloclub.github.io/cnn-explainer/
- What is Connectionism? (Edward Thorndike's Connectionism)
- The Perceptron Explained
- Perceptron Learning Algorithm in Machine Learning | Neural Networks
- https://deeplearning.cs.cmu.edu/F22/document/slides/lec1.intro.pdf
- Perceptron Rule to design XOR Logic Gate Solved Example ANN Machine Learning by Mahesh Huddar
- https://medium.com/@stanleydukor/neural-representation-of-and-or-not-xor-and-xnor-logic-gates-perceptron-algorithm-b0275375fea1
- AND GATE Perceptron Training Rule | Artificial Neural Networks Machine Learning by Mahesh Huddar
- Multi Layer Perceptron | MLP Intuition
- Intro to Deep Learning Part 3: Multilayer Perceptron & XOR Problem
- https://deeplearning.cs.cmu.edu/F22/document/slides/lec2.universal.pdf
- Forward Propagation | How a neural network predicts output?
- 11-785, Fall 22 Lecture 5: Learning the network: Part 3 (1/2)
- https://deeplearning.cs.cmu.edu/F22/document/slides/lec5.learning.pdf
- Backpropagation in Deep Learning | Part 1 | The What?
- 11-785, Fall 22 Lecture 5: Learning the network: Part 3 (2/2)
- Neural Networks 6 Computation Graphs and Backward Differentiation
- Activation Functions In Neural Networks Explained | Deep Learning Tutorial

---

**Start with Module 1 and work through systematically!**

