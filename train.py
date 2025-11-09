"""
Training script for Mixture of Experts on the PB-52 vowel task
(Jacobs, Jordan, Nowlan, Hinton, 1991)

Replicates the vowel discrimination experiments from Table 1 of the paper,
using formant data from Peterson & Barney (1952).
"""

import os
import argparse
import numpy as np

from moe import MixtureOfExperts
from data_loader import load_pb52_vowels
from visualize import plot_decision_boundaries, plot_training_curves
from evaluate import compute_confusion_matrix, print_confusion_matrix


def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    learning_rate,
    max_epochs=10000,
    target_mse=0.08,
    verbose=True,
    print_every=100,
    plot_callback=None,
    plot_every=500,
):
    """
    Generic training loop for both MoE and baseline backprop models.
    Trains until target MSE is reached or max epochs, tracking metrics.
    """
    train_losses = []
    train_accs = []
    test_accs = []
    train_mses = []
    
    converged = False
    epochs_ran = 0
    
    for epoch in range(1, max_epochs + 1):
        # Run one training step (batch gradient descent)
        if hasattr(model, 'compute_loss'):
            loss = model.train_step(X_train, y_train, learning_rate)
            train_losses.append(loss)
        else:
            model.train_step(X_train, y_train, learning_rate)
        
        train_acc = model.compute_accuracy(X_train, y_train) * 100.0
        test_acc = model.compute_accuracy(X_test, y_test) * 100.0
        train_mse = model.compute_squared_error(X_train, y_train)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_mses.append(train_mse)
        
        if verbose and (epoch == 1 or epoch % print_every == 0):
            if train_losses:
                print(f"Epoch {epoch:5d} | Loss {train_losses[-1]:.4f} | "
                      f"Train {train_acc:6.1f}% | Test {test_acc:6.1f}% | MSE {train_mse:.4f}")
            else:
                print(f"Epoch {epoch:5d} | Train {train_acc:6.1f}% | "
                      f"Test {test_acc:6.1f}% | MSE {train_mse:.4f}")
        
        if plot_callback is not None and (epoch == 1 or epoch % plot_every == 0):
            plot_callback(model, epoch)
        
        # Check convergence: paper uses MSE ≤ 0.08 as stopping criterion
        if train_mse <= target_mse:
            converged = True
            epochs_ran = epoch
            if verbose:
                print(f"\nReached target MSE ≤ {target_mse} at epoch {epoch}. Stopping.")
            break
        
        epochs_ran = epoch
    
    final_train_acc = model.compute_accuracy(X_train, y_train) * 100.0
    final_test_acc = model.compute_accuracy(X_test, y_test) * 100.0
    final_mse = model.compute_squared_error(X_train, y_train)
    
    return {
        "train_acc": float(final_train_acc),
        "test_acc": float(final_test_acc),
        "train_mse": float(final_mse),
        "epochs": int(epochs_ran),
        "converged": bool(converged),
        "history": {
            "loss": train_losses,
            "train_acc": train_accs,
            "test_acc": test_accs,
            "train_mse": train_mses,
        },
    }


def train_moe(
    data_path='data/pb52_dataset.csv',
    num_experts=4,
    learning_rate=0.002,
    max_epochs=2000,
    target_mse=0.08,
    sigma=2.0,
    tau=1.0,
    smooth=1,
    seed=42,
    save_dir='results',
    plot_every=500,
    make_plots=True,
    overlay_gate=False,
):
    """
    Complete training pipeline for MoE on PB-52 vowel discrimination.
    Loads data, trains model, generates visualizations and confusion matrices.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"{'MIXTURE OF EXPERTS TRAINING':^70}")
    print(f"{'='*70}")
    
    X_train, y_train, X_test, y_test, labels = load_pb52_vowels(
        data_path, vowels=['i', 'I', 'A', 'V'], train_speakers=50
    )
    
    class_dist = " | ".join([f"{lab}: {(np.argmax(y_train, 1) == i).sum()}" for i, lab in enumerate(labels)])
    print(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test | Classes: {class_dist}")
    print(f"Model: {num_experts} experts | LR: {learning_rate} | σ: {sigma} | τ: {tau}")
    print(f"{'─'*70}")
    model = MixtureOfExperts(
        input_dim=2,
        output_dim=y_train.shape[1],
        num_experts=num_experts,
        seed=seed,
        sigma=sigma,
        tau=tau
    )

    plot_callback = None
    if make_plots:
        def plot_fn(m, ep):
            plot_decision_boundaries(
                m, X_train, y_train, labels, ep,
                save_path=os.path.join(save_dir, f"decision_boundaries_e{ep}.png"),
                overlay_gate=overlay_gate,
            )
        plot_callback = plot_fn

    results = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        target_mse=target_mse,
        verbose=True,
        print_every=100,
        plot_callback=plot_callback,
        plot_every=plot_every,
    )

    print(f"\n{'='*70}")
    print(f"{'FINAL RESULTS':^70}")
    print(f"{'='*70}")
    print(f"Epochs: {results['epochs']:4d} | Train: {results['train_acc']:5.1f}% | "
          f"Test: {results['test_acc']:5.1f}% | MSE: {results['train_mse']:.4f}")
    print(f"{'─'*70}")

    # Show which experts are actually being used (p_i averaged over training set)
    model.forward(X_train)
    mix = model.get_mixing_proportions()
    avg_use = np.mean(mix, axis=0)
    print(f"\nExpert Usage: " + " | ".join([f"E{i}: {p:.3f}" for i, p in enumerate(avg_use)]))

    if make_plots:
        plot_decision_boundaries(
            model, X_train, y_train, labels, results['epochs'],
            save_path=os.path.join(save_dir, f"decision_boundaries_e{results['epochs']}.png"),
            overlay_gate=overlay_gate,
        )
        plot_training_curves(
            results['history']['loss'],
            results['history']['train_acc'],
            results['history']['test_acc'],
            save_path=os.path.join(save_dir, "training_curves.png"),
            smooth=smooth
        )

    log_path = os.path.join(save_dir, "training_log.tsv")
    with open(log_path, "w") as f:
        f.write("epoch\tloss\ttrain_acc\ttest_acc\ttrain_mse\n")
        for i in range(len(results['history']['train_acc'])):
            loss_str = f"{results['history']['loss'][i]:.6f}" if results['history']['loss'] else "N/A"
            f.write(f"{i+1}\t{loss_str}\t"
                    f"{results['history']['train_acc'][i]:.3f}\t"
                    f"{results['history']['test_acc'][i]:.3f}\t"
                    f"{results['history']['train_mse'][i]:.6f}\n")
    print(f"\nSaved logs and figures to: {save_dir}")

    # Show how responsibility is distributed across classes (task decomposition)
    _ = model.compute_loss(X_train, y_train)
    R = model.get_expert_responsibilities()
    y_idx = np.argmax(y_train, axis=1)
    print("\nTask Decomposition (avg responsibility per class):")
    header = "Class  " + "".join([f"  Expert{e}" for e in range(model.num_experts)])
    print(header)
    print("-" * len(header))
    for k, lab in enumerate(labels):
        rk = R[y_idx == k]
        means = [float(np.mean(rk[:, e])) for e in range(model.num_experts)]
        values = "".join(f"    {v:0.3f}" for v in means)
        print(f"  {lab:>2s}  {values}")

    Ctr = compute_confusion_matrix(model, X_train, y_train)
    Cte = compute_confusion_matrix(model, X_test,  y_test)
    print("\nConfusion Matrices:")
    print_confusion_matrix(Ctr, "Train", labels)
    print_confusion_matrix(Cte, "Test", labels)
    print("=" * 70)

    results["avg_mixing"] = avg_use.tolist()
    return model, results

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train Mixture of Experts on PB-52 vowels",
        conflict_handler="resolve",
    )
    p.add_argument("--data", type=str, default="data/pb52_dataset.csv",
                   help="Path to pb52_dataset.csv")
    p.add_argument("--experts", type=int, default=4,
                   help="Number of experts")
    p.add_argument("--lr", type=float, default=0.002,
                   help="Learning rate (fixed GD)")
    p.add_argument("--epochs", type=int, default=10000,
                   help="Max epochs")
    p.add_argument("--target-mse", type=float, default=0.08,
                   help="Stop training when train MSE <= this threshold")
    p.add_argument("--sigma", type=float, default=2.0,
                   help="Gaussian likelihood variance parameter")
    p.add_argument("--tau", type=float, default=1.0,
                   help="Gating network temperature parameter")
    p.add_argument("--smooth", type=int, default=1,
                   help="Smoothing window for training curves (1=no smoothing)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--out", type=str, default="results",
                   help="Output directory")
    p.add_argument("--plot-every", type=int, default=500,
                   help="Plot decision boundaries every N epochs")
    p.add_argument("--no-plots", action="store_true",
                   help="Disable plotting for speed")
    p.add_argument("--overlay-gate", action="store_true",
                   help="Overlay dashed gating boundaries on decision plots")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_moe(
        data_path=args.data,
        num_experts=args.experts,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        target_mse=args.target_mse,
        sigma=args.sigma,
        tau=args.tau,
        smooth=args.smooth,
        seed=args.seed,
        save_dir=args.out,
        plot_every=args.plot_every,
        make_plots=not args.no_plots,
        overlay_gate=args.overlay_gate,
    )