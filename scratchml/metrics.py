"""
scratchml/metrics.py

Performance Metrics for evaluating Machine Learning and Deep Learning models from scratch using NumPy.

Mathematical Concepts Covered:
1. Classification:
   - Confusion Matrix: TP, FP, TN, FN
   - Accuracy: (TP + TN) / (TP + TN + FP + FN)
   - Precision: TP / (TP + FP)
   - Recall (Sensitivity): TP / (TP + FN)
   - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
2. Regression:
   - R-squared (R2): 1 - (SS_res / SS_tot)
   - Mean Absolute Error (MAE): (1/N) * sum(|y_true - y_pred|)
   - Mean Squared Error (MSE): (1/N) * sum((y_true - y_pred)^2)

Generally Used Libraries:
- scikit-learn: sklearn.metrics (accuracy_score, classification_report, r2_score)
"""

import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Math: Correct / Total
    # Line importance: Simple classification ratio.
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes confusion matrix for binary classification.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Math: TP / (TP + FP)
    # Line importance: Measures quality of positive predictions.
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Math: TP / (TP + FN)
    # Line importance: Measures ability to find all positive samples.
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Math: 2 * (Precision * Recall) / (Precision + Recall)
    # Line importance: Harmonic mean of precision and recall.
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared coefficient of determination for Regression.
    """
    # Math: R2 = 1 - SS_res / SS_tot
    # Line importance: Measures proportion of variance explained by model.
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_total == 0:
        return 0.0
    return float(1.0 - (ss_residual / ss_total))
