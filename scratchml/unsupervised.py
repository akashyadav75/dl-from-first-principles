"""
scratchml/unsupervised.py

Industrial-grade Unsupervised Learning algorithms implemented from scratch using NumPy.

Mathematical Concepts Covered:
1. K-Means Clustering:
   - Distance Metric: Euclidean Distance d(p, q) = sqrt(sum((p_i - q_i)^2))
   - Centroid Update: C_k = (1 / |S_k|) * sum(x_i) for x_i in S_k
2. Principal Component Analysis (PCA):
   - Mean Centering: X_centered = X - Mean(X)
   - Covariance Matrix: Sigma = (1 / (N - 1)) * X_centered^T . X_centered
   - Eigendecomposition: Sigma . V = V . Lambda
   - Projection: X_reduced = X_centered . V_k

Generally Used Libraries:
- scikit-learn: sklearn.cluster.KMeans, sklearn.decomposition.PCA
"""

import numpy as np

class KMeans:
    """
    K-Means Clustering algorithm.
    """
    def __init__(self, k: int = 3, max_iters: int = 100, tol: float = 1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = []
        self.labels = None

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # Math: sqrt(sum((x1 - x2)^2))
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=-1))

    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape

        # Initialize centroids randomly from the dataset (K-Means++ style or simple random selection)
        # Line importance: Generates random distinct indices to seed the clusters.
        random_idxs = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idxs]

        for _ in range(self.max_iters):
            # 1. Assign samples to the closest centroids
            # Line importance: Vectorized distance calculation of each sample to each centroid.
            distances = np.array([self._euclidean_distance(X, centroid) for centroid in self.centroids])
            # Shape: (k, n_samples) -> Transpose to (n_samples, k) and find minimum index
            self.labels = np.argmin(distances.T, axis=1)

            # 2. Store old centroids for convergence check
            old_centroids = self.centroids.copy()

            # 3. Calculate new centroids by taking the mean of all samples assigned to each cluster
            # Line importance: Re-centering step.
            new_centroids = np.zeros((self.k, n_features))
            for cluster_idx in range(self.k):
                cluster_samples = X[self.labels == cluster_idx]
                if len(cluster_samples) > 0:
                    new_centroids[cluster_idx] = np.mean(cluster_samples, axis=0)
                else:
                    # Keep old centroid if no points are assigned to it
                    new_centroids[cluster_idx] = old_centroids[cluster_idx]

            self.centroids = new_centroids

            # 4. Check for convergence (if centroids move less than tolerance)
            centroid_shift = np.sum(self._euclidean_distance(old_centroids, self.centroids))
            if centroid_shift < self.tol:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Line importance: Assigns new/unseen data to the nearest trained centroid.
        distances = np.array([self._euclidean_distance(X, centroid) for centroid in self.centroids])
        return np.argmin(distances.T, axis=1)


class PCA:
    """
    Principal Component Analysis (PCA) for Dimensionality Reduction.
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        # 1. Center the data
        # Line importance: PCA is highly sensitive to mean shifts; centering is mandatory.
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Compute the Covariance Matrix
        # Math: Cov = (1 / (N - 1)) * X_centered^T * X_centered
        # Line importance: Measures linear correlation between all feature pairs.
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Compute Eigenvalues and Eigenvectors
        # Math: Cov * v = lambda * v
        # Line importance: Decomposes covariance matrix into orthogonal variance directions.
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvalues and eigenvectors in descending order
        # Line importance: Ensures first principal components explain the most variance.
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 5. Store the top principal components
        self.components = eigenvectors[:, :self.n_components]

        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # 1. Center the input data
        X_centered = X - self.mean
        # 2. Project data onto principal components
        # Math: X_reduced = X_centered . V_k
        # Line importance: Reduces dimensionality while preserving maximum variance.
        return np.dot(X_centered, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
