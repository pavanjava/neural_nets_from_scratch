# Chapter5: Neural Network Activation Functions & NumPy Array Operations

This repository contains Python implementations demonstrating core neural network activation functions and essential NumPy array operations for deep learning.

## Files Overview

- **`activation_functions.py`** - Implements common activation functions and demonstrates their usage in a simple neural network
- **`array_rules_and_broadcasting.py`** - Explains NumPy array operations, axis manipulation, and broadcasting concepts

## Activation Functions (`activation_functions.ipynb`)

### What are Activation Functions?

Activation functions determine whether a neuron should be activated or not. They introduce non-linearity into neural networks, enabling them to learn complex patterns.

### Implemented Functions

#### 1. ReLU (Rectified Linear Unit)
```python
output = max(0, input)
```
- **Purpose**: Most popular activation function for hidden layers
- **Behavior**: Outputs the input if positive, otherwise outputs 0
- **Example**: Input `[-1, 2, -3, 4]` → Output `[0, 2, 0, 4]`

#### 2. Leaky ReLU
```python
output = max(0.01 * input, input)
```
- **Purpose**: Improved version of ReLU that prevents "dead neurons"
- **Behavior**: Allows small negative values (1% of input) instead of zero
- **Example**: Input `[-1, 2, -3, 4]` → Output `[-0.01, 2, -0.03, 4]`

#### 3. Softmax
```python
output = exp(input) / sum(exp(all_inputs))
```
- **Purpose**: Used in output layer for multi-class classification
- **Behavior**: Converts raw scores to probabilities (sum = 1)
- **Example**: Input `[1, 2, 3]` → Output `[0.09, 0.24, 0.67]`

#### 4. Tanh (Hyperbolic Tangent)
```python
output = (e^input - e^(-input)) / (e^input + e^(-input))
```
- **Purpose**: Alternative to sigmoid, outputs between -1 and 1
- **Behavior**: S-shaped curve centered at zero
- **Range**: (-1, 1)

#### 5. Sigmoid
```python
output = 1 / (1 + e^(-input))
```
- **Purpose**: Classic activation function, often used in binary classification
- **Behavior**: S-shaped curve that squashes input to (0, 1)
- **Range**: (0, 1)

### Neural Network Example

The code demonstrates a simple 2-layer neural network:
1. **Input Layer**: Takes 2D spiral data (100 samples, 3 classes)
2. **Hidden Layer**: 3 neurons with ReLU activation
3. **Output Layer**: 3 neurons with Softmax activation

```python
Input → Dense Layer → ReLU → Dense Layer → Softmax → Output
```

## NumPy Array Operations (`array_rules_and_broadcasting.ipynb`)

### Array Axis Operations

Understanding axes is crucial for neural network computations:

#### Axis Concept
- **axis=0**: Operations go DOWN the rows (column-wise)
- **axis=1**: Operations go ACROSS the columns (row-wise)

#### Example with Sales Data (3×4 matrix)
```python
sales_data = [[120, 150, 180, 200],   # Product A
              [90,  110, 130, 160],   # Product B  
              [200, 220, 240, 280]]   # Product C
```

**Sum Operations:**
- `np.sum(data, axis=0)`: Sums each column → `[410, 480, 550, 640]` (quarterly totals)
- `np.sum(data, axis=1)`: Sums each row → `[650, 490, 940]` (product totals)

**keepdims Parameter:**
- `keepdims=True`: Preserves original array dimensions
- `keepdims=False`: Reduces array dimensions

### Broadcasting

Broadcasting allows operations between arrays of different shapes without explicit loops.

#### Key Broadcasting Rules
1. **Scalar + Array**: Number gets applied to each element
2. **Compatible Shapes**: Arrays can broadcast if dimensions are compatible
3. **Automatic Expansion**: Smaller arrays get "stretched" to match larger ones

#### Broadcasting Examples

**1. Scalar Broadcasting**
```python
[1, 2, 3, 4] + 10 = [11, 12, 13, 14]
```

**2. 1D Array + 2D Array**
```python
[[1, 2, 3],     [10, 20, 30]     [[11, 22, 33],
 [4, 5, 6]]  +  [10, 20, 30]  =   [14, 25, 36]]
```

**3. Column Broadcasting**
```python
[[1, 2, 3],     [[100],     [[101, 102, 103],
 [4, 5, 6]]  +   [200]]  =   [204, 205, 206]]
```

**4. Real-World Example: Tax Calculation**
```python
# Prices for products in different stores
prices = [[10, 15, 20], [12, 18, 22]]
# Tax rates per product
tax_rates = [0.1, 0.15, 0.2]
# Broadcasting automatically applies each tax rate to corresponding products
final_prices = prices * (1 + tax_rates)
```

## Why These Concepts Matter

**Activation Functions:**
- Enable neural networks to learn non-linear patterns
- Each function has specific use cases and properties
- Critical for gradient flow during training

**NumPy Operations:**
- Essential for efficient neural network computations
- Broadcasting eliminates need for explicit loops
- Axis operations enable proper matrix manipulations
- Foundation for frameworks like TensorFlow and PyTorch

## Code Structure

Both files use Jupyter notebook cell structure (`#%%`) and include:
- Clear class-based implementations
- Practical examples with real data
- Step-by-step explanations
- Visual output demonstrations