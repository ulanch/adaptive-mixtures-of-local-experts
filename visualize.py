"""
Visualization utilities for Mixture of Experts
Generates decision boundary plots (Fig 2 in paper) and training curves
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for server environments
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_boundaries(
    model,
    X_train,
    y_train,
    vowel_labels,
    epoch,
    save_path,
    overlay_gate=False,
):
    """
    Plot expert decision boundaries overlaid on training data (replicates Fig 2).
    Each expert learns a linear decision surface. Optional: overlay gating network
    boundaries showing which expert is dominant in each region.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot training data points with different colors/markers per class
    palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']
    markers = ['o', 's', '^', 'D']
    y_idx = np.argmax(y_train, axis=1)
    for i, lab in enumerate(vowel_labels):
        mask = (y_idx == i)
        ax.scatter(
            X_train[mask, 0], X_train[mask, 1],
            c=palette[i], marker=markers[i], s=36, alpha=0.7,
            edgecolors='k', linewidths=0.3, label=lab
        )

    # Create grid for decision boundary contours
    pad = 0.2
    x_min, x_max = X_train[:, 0].min() - pad, X_train[:, 0].max() + pad
    y_min, y_max = X_train[:, 1].min() - pad, X_train[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict class for each grid point using weighted expert combination
    y_hat = model.predict_class_combined(grid)
    Z_class = y_hat.reshape(xx.shape)

    cmap = ListedColormap([
        (1.0, 0.92, 0.92),
        (0.92, 0.92, 1.0),
        (0.92, 1.0, 0.92),
        (0.96, 0.92, 1.0),
    ])
    levels = np.arange(-0.5, len(vowel_labels) + 0.5, 1)

    ax.contourf(
        xx, yy, Z_class,
        levels=levels, cmap=cmap, alpha=0.35, antialiased=False
    )
    ax.contour(
        xx, yy, Z_class,
        levels=levels, colors='k', linewidths=0.8, linestyles='-'
    )

    # Optionally overlay gating network boundaries (which expert is dominant)
    if overlay_gate:
        model.forward(grid)
        gate_dom = np.argmax(model.get_mixing_proportions(), axis=1).reshape(xx.shape)
        ax.contour(
            xx, yy, gate_dom,
            levels=np.arange(-0.5, model.num_experts + 0.5, 1),
            colors='k', linewidths=0.6, linestyles='--', alpha=0.35
        )

    ax.set_xlabel('F1 (kHz)')
    ax.set_ylabel('F2 (kHz)')
    ax.set_title(f'Mixture of Experts — Epoch {epoch}')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.25)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(train_losses, train_accs, test_accs, save_path, smooth=1):
    """
    Plot training curves: loss and accuracy over epochs.
    Optional smoothing via rolling window average for cleaner visualization.
    """
    def rolling_mean(arr, window):
        """Simple rolling window average for smoothing noisy curves"""
        if window <= 1 or len(arr) < window:
            return arr
        out = []
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            out.append(np.mean(arr[start:i+1]))
        return out
    
    if smooth > 1:
        train_losses = rolling_mean(train_losses, smooth)
        train_accs = rolling_mean(train_accs, smooth)
        test_accs = rolling_mean(test_accs, smooth)
    
    # Dual-axis plot: loss on left, accuracy on right
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_losses, label='Mixture NLL (train)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(train_accs, label='Train Acc (%)', linewidth=2, linestyle='--')
    ax2.plot(test_accs, label='Test Acc (%)', linewidth=2, linestyle=':')
    ax2.set_ylabel('Accuracy (%)')

    # Combine legends from both axes
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax2.legend(l1 + l2, lab1 + lab2, loc='best', frameon=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
