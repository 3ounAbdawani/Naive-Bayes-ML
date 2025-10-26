# Naive Bayes Classifier Implementation

This project demonstrates the implementation and application of Gaussian Naive Bayes classifiers for classification tasks using both scikit-learn and a custom implementation.

## Overview

The notebook contains two main implementations:
1. **Scikit-learn GaussianNB** - Using the built-in Gaussian Naive Bayes classifier
2. **Custom GaussianNaiveBayes** - A from-scratch implementation with additional visualization capabilities

## Features

### 1. Scikit-learn Implementation
- Uses `GaussianNB` from scikit-learn
- Applied to the Iris dataset
- Achieves 100% accuracy on the test set

### 2. Custom Implementation
- Built from scratch using NumPy and SciPy
- Applied to the Wine dataset
- Includes:
  - Data preprocessing (standardization)
  - Feature distribution visualization
  - Decision boundary plotting
  - Performance evaluation metrics

## Datasets

1. **Iris Dataset** (scikit-learn implementation)
   - 3 classes of iris flowers
   - 4 features (sepal length, sepal width, petal length, petal width)
   - 150 samples total

2. **Wine Dataset** (custom implementation)
   - 3 classes of wines
   - 13 features (alcohol, malic acid, ash, etc.)
   - 178 samples total

## Key Components

### Custom GaussianNaiveBayes Class
- `fit(X, y)`: Trains the model by calculating class priors and feature parameters
- `predict(X)`: Makes predictions using log probabilities
- `_calculate_likelihood()`: Computes Gaussian probabilities for features

### Visualization
- **Feature Distributions**: Plots Gaussian distributions for each feature across classes
- **Decision Boundaries**: 2D visualization of classification regions using first two features

## Results

Both implementations achieve excellent performance:
- **Iris Dataset**: 100% accuracy
- **Wine Dataset**: 100% accuracy with comprehensive classification report

## Requirements

```python
scikit-learn
numpy
scipy
matplotlib
