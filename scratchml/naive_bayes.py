"""
scratchml/naive_bayes.py

Naive Bayes Classifier built from scratch using NumPy.

Mathematical Concepts Covered:
1. Bayes Theorem: P(y|X) = (P(X|y) * P(y)) / P(X)
2. Naive Independence Assumption: P(X|y) = P(x_1|y) * P(x_2|y) * ... * P(x_n|y)
3. Log Likelihoods (to prevent underflow):
   log P(y|X) = log P(y) + sum(log P(x_i|y))
4. Gaussian Probability Density Function (for continuous features):
   P(x_i|y) = (1 / sqrt(2 * pi * var_y)) * exp(-(x_i - mean_y)^2 / (2 * var_y))

Generally Used Libraries:
- scikit-learn: sklearn.naive_bayes.GaussianNB
"""

import numpy as np

class GaussianNB:
    """
    Gaussian Naive Bayes Classifier for continuous features.
    """
    def __init__(self):
        self.classes = None
        self.means = {}
        self.variances = {}
        self.priors = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            # Extract samples belonging to class c
            X_c = X[y == c]
            
            # Math: P(y) = count(class_c) / total_count
            self.priors[c] = X_c.shape[0] / float(n_samples)
            
            # Math: Mean and Variance of each feature given class c
            # Line importance: Essential parameters for continuous Gaussian probability calculation.
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + 1e-9  # Add epsilon to prevent division by zero

        return self

    def _calculate_likelihood(self, class_idx: int, x: np.ndarray) -> np.ndarray:
        """
        Calculates the Gaussian Likelihood P(x_i | y) for each feature.
        """
        mean = self.means[class_idx]
        var = self.variances[class_idx]
        
        # Math: Gaussian PDF
        # Line importance: Standard normal distribution density calculation.
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_single(self, x: np.ndarray) -> int:
        posteriors = []

        for c in self.classes:
            # Math: log P(y)
            prior = np.log(self.priors[c])
            
            # Math: sum(log P(x_i|y))
            # Line importance: Log-sum-exp trick prevents floating-point underflow.
            likelihoods = self._calculate_likelihood(c, x)
            posterior = prior + np.sum(np.log(likelihoods + 1e-15))
            
            posteriors.append((posterior, c))

        # Return class with the maximum posterior probability
        return max(posteriors, key=lambda x: x[0])[1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Line importance: Processes each row in dataset.
        return np.array([self._predict_single(x) for x in X])
