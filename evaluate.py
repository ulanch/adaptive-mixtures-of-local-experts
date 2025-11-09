"""
Evaluation utilities for Mixture of Experts
Simple confusion matrix computation and pretty-printing
"""

import numpy as np


def compute_confusion_matrix(model, X, Y):
    """
    Compute confusion matrix for a trained model.
    Uses predict_class_combined which weights all experts by gating network.
    """
    y_true = np.argmax(Y, axis=1)
    y_pred = model.predict_class_combined(X)
    K = Y.shape[1]
    C = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    return C


def print_confusion_matrix(C, title, labels):
    """Print confusion matrix in a readable format"""
    print(f"\n{title} (rows=actual, cols=predicted):")
    header = "      " + "".join([f"{lab:>6s}" for lab in labels])
    print(header)
    print("-" * len(header))
    for i, lab in enumerate(labels):
        print(f"  {lab:>2s}  " + "".join([f"{C[i, j]:>6d}" for j in range(len(labels))]))
