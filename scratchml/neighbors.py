"""
scratchml/neighbors.py

K-Nearest Neighbors (KNN) Classifier and Regressor implemented from scratch using NumPy.

Mathematical Concepts Covered:
1. Distance Metrics:
   - Euclidean: d(p, q) = sqrt(sum((p_i - q_i)^2))
   - Manhattan: d(p, q) = sum(|p_i - q_i|)
2. Majority Voting (Classification): mode(nearest_k_labels)
3. Mean Aggregation (Regression): mean(nearest_k_targets)

Generally Used Libraries:
- scikit-learn: sklearn.neighbors.KNeighborsClassifier, sklearn.neighbors.KNeighborsRegressor
"""

import numpy as np

class KNN:
    """
    Base class for K-Nearest Neighbors.
    """
    def __init__(self, k: int = 3, metric: str = 'euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Line importance: KNN is a lazy learner; fitting simply stores training data.
        self.X_train = X
        self.y_train = y
        return self

    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.metric == 'euclidean':
            # Math: sqrt(sum((x1 - x2)^2))
            return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        elif self.metric == 'manhattan':
            # Math: sum(|x1 - x2|)
            return np.sum(np.abs(x1 - x2), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _predict_single(self, x: np.ndarray) -> float:
        # 1. Compute distances between test sample x and all training samples
        # Line importance: Vectorized distance calculation of 1 sample against the whole training set.
        distances = self._compute_distance(self.X_train, x.reshape(1, -1))

        # 2. Get indices of the k nearest neighbors
        # Line importance: np.argsort sorts distances and returns indices of smallest values.
        k_indices = np.argsort(distances)[:self.k]

        # 3. Retrieve labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # 4. Aggregate predictions (overridden by sub-classes)
        return self._aggregate(k_nearest_labels)

    def _aggregate(self, labels: np.ndarray) -> float:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Line importance: Processes each sample in test set X.
        return np.array([self._predict_single(x) for x in X])


class KNeighborsClassifier(KNN):
    """
    K-Nearest Neighbors Classifier.
    """
    def _aggregate(self, labels: np.ndarray) -> float:
        # Line importance: Majority voting.
        vals, counts = np.unique(labels, return_counts=True)
        return vals[np.argmax(counts)]


class KNeighborsRegressor(KNN):
    """
    K-Nearest Neighbors Regressor.
    """
    def _aggregate(self, labels: np.ndarray) -> float:
        # Line importance: Mean aggregation.
        return np.mean(labels)
