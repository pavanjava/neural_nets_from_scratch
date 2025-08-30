# Chapter4: Neural Network Dataset Generator and Dense Layer Implementation

This repository contains two Python files that demonstrate fundamental concepts in neural network implementation: dataset generation and basic layer construction.

## 📁 File Overview

- `dataset_creator.py` - Generates synthetic spiral datasets for neural network training
- `dense_layer_implementation.py` - Implements a basic dense (fully connected) neural network layer

---

## 📊 Dataset Creator (`dataset_creator.py`)

This file contains three different approaches to generating spiral datasets, which are commonly used for testing neural network classification algorithms.

### Features

#### 1. Single Spiral Generation
- **Function**: `simple_spiral(n_points=1000, noise=0.1)`
- Generates a single spiral pattern with configurable noise
- Uses parametric equations with time parameter `t`
- Perfect for testing basic pattern recognition

#### 2. Multi-Label Spiral Generation
- **Function**: `generate_spiral_data(n_points=1000, noise=0.1, n_spirals=2)`
- Creates multiple interleaved spiral arms
- Each spiral represents a different class/label
- Adjustable number of spirals and noise levels
- Returns coordinates (x, y) and corresponding labels

#### 3. Standard Library Implementation
- Uses the `nnfs` (Neural Networks from Scratch) library
- **Function**: `spiral_data(samples=100, classes=3)`
- Provides a standardized, reproducible dataset
- Includes proper initialization for consistent results

### Usage Examples

```python
# Single spiral
x, y = simple_spiral(n_points=800, noise=0.3)

# Multi-class spiral
x, y, labels = generate_spiral_data(n_points=500, noise=0.2, n_spirals=3)

# Standard library approach
X, y = spiral_data(samples=100, classes=3)
```

### Visualizations
Each method includes matplotlib visualizations to display the generated patterns, making it easy to understand the data structure and distribution.

---

## 🧠 Dense Layer Implementation (`dense_layer_implementation.py`)

This file implements a fundamental building block of neural networks: the dense (fully connected) layer.

### Features

#### DenseLayer Class
- **Initialization**: `DenseLayer(n_inputs, n_neurons)`
    - `n_inputs`: Number of input features
    - `n_neurons`: Number of neurons in the layer
    - Weights initialized with small random values (0.01 * random normal)
    - Biases initialized with random normal distribution

#### Forward Pass
- **Method**: `forward(input)`
- Implements the core linear transformation: `output = input @ weights + biases`
- Uses NumPy's `dot` product for efficient matrix multiplication
- Returns the layer's output for the given input

### Mathematical Foundation

The dense layer performs the fundamental neural network operation:
```
output = X · W + b
```
Where:
- `X` is the input matrix (batch_size × n_inputs)
- `W` is the weight matrix (n_inputs × n_neurons)
- `b` is the bias vector (n_neurons,)

### Usage Example

```python
# Create a layer with 2 inputs and 3 neurons
layer = DenseLayer(2, 3)

# Generate test data
X, y = spiral_data(samples=100, classes=3)

# Forward pass
output = layer.forward(X)
print(f"Output shape: {output.shape}")  # (300, 3)
```

### Integration Example
The file demonstrates how the dense layer works with the spiral dataset:
- Takes 2D spiral coordinates as input
- Outputs 3 values (one for each class)
- Shows the raw output before activation functions

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy matplotlib nnfs
```

### Running the Code
1. **Dataset Generation**: Run `dataset_creator.py` to see various spiral patterns
2. **Layer Testing**: Run `dense_layer_implementation.py` to test the dense layer with spiral data

---

## 📚 Educational Value

- **Dataset Creator**: Demonstrates how synthetic datasets are created for machine learning research
- **Dense Layer**: Shows the core mathematical operations in neural networks
- **Integration**: Illustrates how data flows through network components

This codebase is ideal for students learning neural networks from scratch and understanding the mathematical foundations behind modern deep learning frameworks.