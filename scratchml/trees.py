"""
scratchml/trees.py

Decision Tree Classifier, Regressor, and Random Forest built entirely from scratch using NumPy and Pandas.

Mathematical Concepts Covered:
1. Entropy: H(S) = -sum(p_i * log2(p_i))
2. Gini Impurity: G(S) = 1 - sum(p_i^2)
3. Information Gain: IG(S, A) = H(S) - sum(|S_v| / |S| * H(S_v))
4. Variance Reduction (Regression): VR = Var(parent) - sum(|S_v| / |S| * Var(S_v))
5. Ensemble Bagging: Random sampling of data with replacement and feature bagging.

Generally Used Libraries:
- scikit-learn: sklearn.tree.DecisionTreeClassifier, sklearn.ensemble.RandomForestClassifier
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional

class Node:
    """
    Representation of a single decision node or leaf node in the Tree.
    """
    def __init__(self, feature: int = None, threshold: float = None, 
                 left: 'Node' = None, right: 'Node' = None, *, value: float = None):
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value for splitting
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Classification or regression target value if this is a leaf node

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTree:
    """
    Base class for Decision Tree Classifier and Regressor.
    """
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, 
                 n_features: int = None, criterion: str = 'gini'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features  # Used for feature bagging in Random Forest
        self.criterion = criterion
        self.root = None

    def _impurity(self, y: np.ndarray) -> float:
        """
        Computes the impurity of a node (Gini, Entropy, or Variance).
        """
        if len(y) == 0:
            return 0.0

        if self.criterion == 'variance':
            # Math: Var(y) = mean((y - mean_y)^2)
            # Line importance: Used for regression tree splits.
            return float(np.var(y))

        # Classification impurities
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        if self.criterion == 'gini':
            # Math: Gini = 1 - sum(p_i^2)
            # Line importance: Standard fast classification impurity metric.
            return 1.0 - np.sum(probabilities ** 2)

        elif self.criterion == 'entropy':
            # Math: Entropy = -sum(p_i * log2(p_i))
            # Line importance: Information theory measure of uncertainty.
            return -np.sum(probabilities * np.log2(probabilities + 1e-15))

        return 0.0

    def _split(self, X_column: np.ndarray, split_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits index masks based on a threshold.
        """
        # Line importance: Generates boolean masks to split data indices.
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _best_split(self, X: np.ndarray, y: np.ndarray, feature_idxs: np.ndarray) -> Tuple[int, float, float]:
        """
        Finds the optimal feature and threshold to split the dataset.
        """
        best_gain = -1.0
        split_idx, split_thresh = None, None
        parent_impurity = self._impurity(y)

        for feat_idx in feature_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for thresh in thresholds:
                # Splitting the data indices
                left_idxs, right_idxs = self._split(X_column, thresh)
                
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                # Math: Gain = Impurity(parent) - sum(|child|/|parent| * Impurity(child))
                n = len(y)
                n_l, n_r = len(left_idxs), len(right_idxs)
                e_l, e_r = self._impurity(y[left_idxs]), self._impurity(y[right_idxs])
                child_impurity = (n_l / n) * e_l + (n_r / n) * e_r

                gain = parent_impurity - child_impurity

                # Line importance: Maximize information gain or variance reduction.
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples, n_feats = X.shape

        # Stopping criteria
        # Line importance: Prevents overfitting by stopping growth when constraints are met.
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            
            # Leaf node value determination (Mean for regression, Mode for classification)
            if self.criterion == 'variance':
                leaf_val = float(np.mean(y))
            else:
                # Mode value
                vals, counts = np.unique(y, return_counts=True)
                leaf_val = float(vals[np.argmax(counts)])
            return Node(value=leaf_val)

        # Feature selection (Feature bagging support for Random Forest)
        feat_idxs = np.arange(n_feats)
        if self.n_features is not None:
            feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split
        best_feat, best_thresh, gain = self._best_split(X, y, feat_idxs)

        if gain <= 0.0 or best_feat is None:
            # Cannot split further, return leaf node
            if self.criterion == 'variance':
                leaf_val = float(np.mean(y))
            else:
                vals, counts = np.unique(y, return_counts=True)
                leaf_val = float(vals[np.argmax(counts)])
            return Node(value=leaf_val)

        # Recursive Tree construction
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        # Convert pandas DataFrames/Series to NumPy arrays for vectorized calculations
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_arr = y.values if isinstance(y, pd.Series) else np.array(y)
        self.root = self._build_tree(X_arr, y_arr)
        return self

    def _predict_row(self, node: Node, x: np.ndarray) -> float:
        if node.is_leaf:
            return node.value

        # Line importance: Route down the tree recursively based on split threshold.
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        return np.array([self._predict_row(self.root, x) for x in X_arr])


class DecisionTreeClassifier(DecisionTree):
    """
    Decision Tree Classifier using Gini or Entropy.
    """
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, 
                 n_features: int = None, criterion: str = 'gini'):
        super().__init__(min_samples_split=min_samples_split, max_depth=max_depth, 
                         n_features=n_features, criterion=criterion)


class DecisionTreeRegressor(DecisionTree):
    """
    Decision Tree Regressor using Variance Reduction.
    """
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, 
                 n_features: int = None):
        super().__init__(min_samples_split=min_samples_split, max_depth=max_depth, 
                         n_features=n_features, criterion='variance')


class RandomForestClassifier:
    """
    Random Forest Ensemble Classifier built from scratch.
    """
    def __init__(self, n_estimators: int = 10, max_depth: int = 10, 
                 min_samples_split: int = 2, max_features: str = 'sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees: List[DecisionTreeClassifier] = []

    def _bootstrap_samples(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a bootstrap sample (random sampling with replacement).
        """
        n_samples = X.shape[0]
        # Line importance: Standard bagging technique.
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_arr = y.values if isinstance(y, pd.Series) else np.array(y)
        n_features = X_arr.shape[1]

        # Determine feature bagging size
        if self.max_features == 'sqrt':
            n_sub_features = int(np.sqrt(n_features))
        else:
            n_sub_features = n_features

        self.trees = []
        for _ in range(self.n_estimators):
            # Line importance: Initialize base estimator with subset of features.
            tree = DecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=n_sub_features
            )
            # Bootstrap data
            X_b, y_b = self._bootstrap_samples(X_arr, y_arr)
            tree.fit(X_b, y_b)
            self.trees.append(tree)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        # Gather predictions from each decision tree
        # Shape: (n_estimators, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose to (n_samples, n_estimators) for row-by-row voting
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # Line importance: Majority voting across all trees.
        final_preds = []
        for row in tree_preds:
            vals, counts = np.unique(row, return_counts=True)
            final_preds.append(vals[np.argmax(counts)])
            
        return np.array(final_preds)
