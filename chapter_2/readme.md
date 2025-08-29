# Neural Network Layers Implementation in NumPy

A step-by-step implementation of neural network layers using NumPy, demonstrating the progression from single neuron calculations to batch processing of multiple inputs.

## Overview

This repository contains Python code that demonstrates how to implement neural network layers from scratch using NumPy. The examples progress from basic single neuron operations to more complex batch processing scenarios commonly used in deep learning.

## Prerequisites

- Python 3.x
- NumPy library

```bash
pip install numpy
```

## Code Structure

### 1. Single Neuron Implementation

The first example demonstrates a basic neuron calculation:

```python
X = [0.8, -0.3, 1., 2.5]           # Input features
w = [0.92, 0.55, -0.73, -0.82]     # Weights
bias = -0.6                        # Bias term
output = np.dot(X, w) + bias       # Linear transformation
```

**Mathematical representation:**
```
output = x₁w₁ + x₂w₂ + x₃w₃ + x₄w₄ + bias
```

### 2. Single Layer with Multiple Neurons

The second example shows how to implement a layer with multiple neurons processing a single input:

```python
X = [1, 2, 3, 2.5]                 # Single input sample
w = np.array([[0.2, 0.8, -0.5, 1],      # Weights for neuron 1
              [0.5, -0.91, 0.26, -0.5],  # Weights for neuron 2
              [-0.26, -0.27, 0.17, 0.87]]) # Weights for neuron 3
bias = [2, 3, 0.5]                 # Bias for each neuron
outputs = np.dot(X, w.T) + bias    # Forward pass
```

**Key concepts:**
- Weight matrix transpose (`w.T`) enables proper matrix multiplication
- Each row in the weight matrix represents one neuron's parameters
- Output is a vector with one value per neuron

### 3. Batch Processing

The final example demonstrates batch processing - processing multiple input samples simultaneously:

```python
inputs = np.array([[1, 2, 3, 2.5],      # Sample 1
                   [2, 5, -1, 2.0],      # Sample 2
                   [-0.5, 2.2, 3.3, -0.8]]) # Sample 3

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])

bias = np.array([2.0, 3.0, 0.5])
outputs = np.dot(inputs, weights.T) + bias
```

**Matrix dimensions:**
- `inputs`: (3, 4) - 3 samples, 4 features each
- `weights`: (3, 4) - 3 neurons, 4 weights each
- `weights.T`: (4, 3) - transposed for matrix multiplication
- `outputs`: (3, 3) - 3 samples × 3 neurons

## Mathematical Foundation

### Forward Pass Formula
```
Y = XW^T + b
```

Where:
- `Y`: Output matrix (batch_size × num_neurons)
- `X`: Input matrix (batch_size × num_features)
- `W`: Weight matrix (num_neurons × num_features)
- `W^T`: Transposed weight matrix (num_features × num_neurons)
- `b`: Bias vector (num_neurons,)

### Broadcasting
NumPy automatically broadcasts the bias vector across all input samples, adding the same bias values to each sample's output.

## Key Learning Points

1. **Matrix Multiplication**: Understanding how `np.dot()` works with different matrix dimensions
2. **Transpose Operations**: Why we need `weights.T` for proper matrix multiplication
3. **Broadcasting**: How NumPy handles adding bias vectors to output matrices
4. **Batch Processing**: Efficiently processing multiple inputs simultaneously
5. **Vectorization**: Using NumPy operations instead of loops for better performance

## Usage

Run the code in a Jupyter notebook or Python script:

```python
python complex_layers_in_numpy.py
```

Each code block can be executed independently to see the progression from simple to complex neural network operations.

## Expected Output

- **Single Neuron**: Scalar output value
- **Single Layer**: Vector output (one value per neuron)
- **Batch Processing**: Matrix output (samples × neurons)