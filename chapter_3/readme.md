# Chapter 3: Multi-Layer Neural Network Forward Pass

This project demonstrates a simple forward pass implementation of a 3-layer neural network using NumPy. The network processes a batch of input samples through multiple layers to produce final outputs.

## Network Architecture

The neural network consists of:
- **Input Layer**: 4 features
- **Hidden Layer 1**: 3 nodes
- **Hidden Layer 2**: 3 nodes
- **Output Layer**: 2 nodes

```
Input (4) → Hidden1 (3) → Hidden2 (3) → Output (2)
```

## Data Structure

### Input Data
- **7 samples** with **4 features** each
- Sample format: `[x1, x2, x3, x4]`
- Includes both positive and negative values for diverse testing

### Weight Matrices
- **W1**: 3×4 matrix (Input → Hidden Layer 1)
- **W2**: 3×3 matrix (Hidden Layer 1 → Hidden Layer 2)
- **W3**: 2×3 matrix (Hidden Layer 2 → Output)

### Bias Vectors
- **b1**: 3-element bias for Hidden Layer 1
- **b2**: 3-element bias for Hidden Layer 2
- **b3**: 2-element bias for Output Layer

## Implementation

The code provides two approaches for computing the forward pass:

### 1. Direct Matrix Operations
```python
output_1 = np.dot(Inputs, W1.T) + b1
output_2 = np.dot(output_1, W2.T) + b2
output_3 = np.dot(output_2, W3.T) + b3
```

### 2. Loop-based Implementation
Uses a loop to iterate through weight matrices and bias vectors, making the code more scalable for networks with varying numbers of layers.

## Mathematical Formula

Each layer follows the formula:
```
Output = Input × Weights^T + Bias
```

Where:
- Input: Previous layer's output (or original input for first layer)
- Weights^T: Transposed weight matrix
- Bias: Bias vector for the current layer

## Key Features

- **Batch Processing**: Processes all 7 samples simultaneously using vectorized operations
- **No Activation Functions**: Uses linear transformations only (no ReLU, sigmoid, etc.)
- **NumPy Implementation**: Leverages efficient matrix operations for fast computation
- **Scalable Design**: Loop-based approach can easily handle networks with different architectures

## Output

The final output is a 7×2 matrix where each row represents the 2 output values for each of the 7 input samples.