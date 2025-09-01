# Chapter6: Cross Entropy Loss Functions

Cross entropy is a widely used loss function in machine learning, particularly for classification problems. It measures the difference between predicted probability distributions and true labels.

## Binary Cross Entropy (BCE)

### Formula

For a single sample:
```
BCE = -[y * log(p) + (1-y) * log(1-p)]
```

For multiple samples:
```
BCE = -(1/N) * Σ[y_i * log(p_i) + (1-y_i) * log(1-p_i)]
```

Where:
- `y` = true label (0 or 1)
- `p` = predicted probability (between 0 and 1)
- `N` = number of samples
- `i` = sample index

### When to Use BCE

**Use Binary Cross Entropy when:**
- You have a **binary classification** problem (2 classes only)
- Examples: spam/not spam, cat/dog, fraud/legitimate
- Output layer has 1 neuron with sigmoid activation
- Labels are encoded as 0 or 1

### Example
Predicting if an email is spam (1) or not spam (0):
- True label: y = 1 (spam)
- Predicted probability: p = 0.8
- BCE = -[1 * log(0.8) + 0 * log(0.2)] = -log(0.8) ≈ 0.223

## Categorical Cross Entropy (CCE)

### Formula

For a single sample:
```
CCE = -Σ(y_c * log(p_c))
```

For multiple samples:
```
CCE = -(1/N) * ΣΣ(y_ic * log(p_ic))
```

Where:
- `y_c` = true label for class c (1 if correct class, 0 otherwise)
- `p_c` = predicted probability for class c
- `C` = number of classes
- `N` = number of samples
- `i` = sample index, `c` = class index

### When to Use CCE

**Use Categorical Cross Entropy when:**
- You have a **multi-class classification** problem (3+ classes)
- Examples: image classification (cat/dog/bird), sentiment analysis (positive/negative/neutral)
- Output layer has C neurons (one per class) with softmax activation
- Labels are one-hot encoded

### Example
Classifying an image as cat/dog/bird:
- True label: [1, 0, 0] (cat)
- Predicted probabilities: [0.7, 0.2, 0.1]
- CCE = -[1*log(0.7) + 0*log(0.2) + 0*log(0.1)] = -log(0.7) ≈ 0.357

## Key Differences

| Aspect | Binary Cross Entropy | Categorical Cross Entropy |
|--------|---------------------|---------------------------|
| **Number of Classes** | 2 classes | 3+ classes |
| **Output Neurons** | 1 neuron | C neurons (one per class) |
| **Activation Function** | Sigmoid | Softmax |
| **Label Encoding** | Single value (0 or 1) | One-hot encoded vector |
| **Use Case** | Binary classification | Multi-class classification |



## Why Cross Entropy Works

Cross entropy loss has several advantages:
- **Probabilistic interpretation**: Directly optimizes for correct probability estimates
- **Gradient properties**: Provides strong gradients when predictions are wrong
- **Convex**: For linear models, guarantees global minimum
- **Well-calibrated**: Encourages confident correct predictions and uncertain incorrect ones

The logarithmic nature penalizes confident wrong predictions more heavily than uncertain wrong predictions, encouraging the model to be both accurate and well-calibrated.
checkout the notebook for more detailed implementation.