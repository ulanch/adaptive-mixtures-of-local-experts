"""
Experimental replication of Jacobs et al. (1991) Table 1 results.
Compares Mixture of Experts with standard backpropagation baselines
on the 4-class vowel discrimination task.
"""

import argparse
import numpy as np

from moe import MixtureOfExperts
from backprop import BackpropNetwork
from data_loader import load_pb52_vowels
from train import train_model


def run_trial(
    model_type: str = "moe",
    learning_rate: float = 0.05,
    num_experts: int = 4,
    hidden_units: int = 6,
    target_mse: float = 0.08,
    max_epochs: int = 2000,
    sigma: float = 2.0,
    tau: float = 1.0,
    seed: int | None = None,
    data_path: str = "data/pb52_dataset.csv",
    verbose: bool = False,
):
    """Run a single training trial for MoE or backprop baseline"""
    X_train, y_train, X_test, y_test, _ = load_pb52_vowels(
        data_path, vowels=["i", "I", "A", "V"], train_speakers=50
    )

    if model_type == "moe":
        model = MixtureOfExperts(2, y_train.shape[1], num_experts, seed=seed, sigma=sigma, tau=tau)
    elif model_type == "bp":
        model = BackpropNetwork(2, hidden_units, y_train.shape[1], seed=seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    results = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        target_mse=target_mse,
        verbose=verbose,
        print_every=100,
        plot_callback=None,
        plot_every=500,
    )

    return {
        "train_acc": results["train_acc"],
        "test_acc": results["test_acc"],
        "epochs": results["epochs"],
        "converged": results["converged"],
        "final_mse": results["train_mse"],
    }


def run_multiple_trials(
    model_type: str = "moe",
    learning_rate: float = 0.05,
    num_experts: int = 4,
    hidden_units: int = 6,
    num_trials: int = 25,
    target_mse: float = 0.08,
    max_epochs: int = 2000,
    sigma: float = 2.0,
    tau: float = 1.0,
    seed: int | None = 42,
    data_path: str = "data/pb52_dataset.csv",
):
    """
    Run multiple trials to replicate Table 1 statistics (mean epochs, std dev).
    Paper used 25 trials for each configuration.
    """
    label = (
        f"Mixture of Experts ({num_experts} experts)"
        if model_type == "moe"
        else f"Backprop ({hidden_units} hidden units)"
    )
    print(f"\n{'='*70}")
    print(f"Running {num_trials} trials: {label}")
    print(f"{'='*70}")
    print(f"LR: {learning_rate}", end="")
    if model_type == "moe":
        print(f" | Sigma: {sigma} | Tau: {tau}", end="")
    print(f" | Target MSE: {target_mse} | Max epochs: {max_epochs}")
    print()

    results = []
    for i in range(num_trials):
        trial_seed = (None if seed is None else seed + i)
        res = run_trial(
            model_type=model_type,
            learning_rate=learning_rate,
            num_experts=num_experts,
            hidden_units=hidden_units,
            target_mse=target_mse,
            max_epochs=max_epochs,
            sigma=sigma,
            tau=tau,
            seed=trial_seed,
            data_path=data_path,
            verbose=False,  # suppress all training output
        )
        results.append(res)
        
        # Progress bar
        bar_width = 40
        filled = int(bar_width * (i + 1) / num_trials)
        bar = '█' * filled + '░' * (bar_width - filled)
        status = '✓' if res['converged'] else '✗'
        
        print(f"\r[{bar}] {i+1}/{num_trials} {status} "
              f"Train: {res['train_acc']:5.1f}% | Test: {res['test_acc']:5.1f}% | "
              f"Epochs: {res['epochs']:4d}", end='', flush=True)
    
    print()  # newline after progress bar

    # Aggregate statistics across trials
    train_accs = np.array([r["train_acc"] for r in results], dtype=float)
    test_accs = np.array([r["test_acc"] for r in results], dtype=float)
    epochs = np.array([r["epochs"] for r in results], dtype=int)
    conv_mask = np.array([r["converged"] for r in results], dtype=bool)
    conv_epochs = epochs[conv_mask]

    # Compute mean/std only over converged trials (as in paper)
    if conv_epochs.size > 0:
        epochs_mean = float(np.mean(conv_epochs))
        epochs_std = float(np.std(conv_epochs, ddof=1)) if conv_epochs.size > 1 else 0.0
    else:
        epochs_mean = float(np.mean(epochs))
        epochs_std = float(np.std(epochs, ddof=1)) if epochs.size > 1 else 0.0

    print(f"\n{'─'*70}")
    sys_name = f"MoE-{num_experts}" if model_type == "moe" else f"BP-{hidden_units}"
    print(f"Results | {sys_name} | Converged: {int(np.sum(conv_mask))}/{num_trials}")
    print(f"{'─'*70}")
    print(f"Train Acc:  {np.mean(train_accs):5.1f}%  |  Test Acc:  {np.mean(test_accs):5.1f}%")
    print(f"Avg Epochs: {epochs_mean:5.0f}     |  Std Dev:   {epochs_std:5.0f}")
    print()

    return {
        "model_type": model_type,
        "num_experts": int(num_experts),
        "hidden_units": int(hidden_units),
        "learning_rate": float(learning_rate),
        "num_trials": int(num_trials),
        "converged": int(np.sum(conv_mask)),
        "train_acc_mean": float(np.mean(train_accs)),
        "test_acc_mean": float(np.mean(test_accs)),
        "epochs_mean": epochs_mean,
        "epochs_std": epochs_std,
    }


def run_table1_experiments(
    trials: int = 25,
    moe_lr: float = 0.002,
    bp_lr: float = 0.05,
    target_mse: float = 0.08,
    sigma: float = 2.0,
    tau: float = 1.0,
    max_epochs: int = 10000,
    seed: int | None = 42,
    data_path: str = "data/pb52_dataset.csv",
):
    """
    Replicate full Table 1 from paper: 4 experts, 8 experts, BP 6 hidden, BP 12 hidden.
    Paper reports MoE converges ~2x faster than backprop with comparable accuracy.
    """
    print(f"\n{'='*70}")
    print(f"{'TABLE 1 REPLICATION':^70}")
    print(f"{'Jacobs et al. (1991)':^70}")
    print(f"{'='*70}")
    print(f"MoE: lr={moe_lr}, σ={sigma}, τ={tau} | BP: lr={bp_lr} | Trials: {trials}")
    print(f"{'='*70}")

    configs = [
        ("moe", 4, moe_lr, None),
        ("moe", 8, moe_lr, None),
        ("bp", None, bp_lr, 6),
        ("bp", None, bp_lr, 12),
    ]

    all_results = []
    for model_type, num_experts, lr, hidden in configs:
        res = run_multiple_trials(
            model_type=model_type,
            learning_rate=lr,
            num_experts=(num_experts if num_experts is not None else 4),
            hidden_units=(hidden if hidden is not None else 6),
            num_trials=trials,
            target_mse=target_mse,
            max_epochs=max_epochs,
            sigma=sigma,
            tau=tau,
            seed=seed,
            data_path=data_path,
        )
        all_results.append(res)

    print(f"\n{'='*70}")
    print(f"{'SUMMARY TABLE':^70}")
    print(f"{'='*70}")
    print(f"{'System':>15} | {'Train%':>6} | {'Test%':>6} | {'Epochs':>7} | {'SD':>5}")
    print(f"{'─'*70}")
    for r in all_results:
        if r["model_type"] == "moe":
            name = f"{r['num_experts']} Experts"
        else:
            name = f"BP-{r['hidden_units']}"
        print(
            f"{name:>15} | "
            f"{r['train_acc_mean']:>6.1f} | "
            f"{r['test_acc_mean']:>6.1f} | "
            f"{r['epochs_mean']:>7.0f} | "
            f"{r['epochs_std']:>5.0f}"
        )
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Jacobs et al. (1991) Experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run")
    run_p.add_argument("--model", choices=["moe", "bp"], required=True)
    run_p.add_argument("--experts", type=int, default=4)
    run_p.add_argument("--hidden", type=int, default=6)
    run_p.add_argument("--lr", type=float, default=0.05)
    run_p.add_argument("--trials", type=int, default=3)
    run_p.add_argument("--epochs", type=int, default=10000)
    run_p.add_argument("--target-mse", type=float, default=0.08)
    run_p.add_argument("--sigma", type=float, default=2.0)
    run_p.add_argument("--tau", type=float, default=1.0)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--data", type=str, default="data/pb52_dataset.csv")

    tab_p = subparsers.add_parser("table1")
    tab_p.add_argument("--trials", type=int, default=25)
    tab_p.add_argument("--moe-lr", type=float, default=0.002)
    tab_p.add_argument("--bp-lr", type=float, default=0.05)
    tab_p.add_argument("--epochs", type=int, default=10000)
    tab_p.add_argument("--target-mse", type=float, default=0.08)
    tab_p.add_argument("--sigma", type=float, default=2.0)
    tab_p.add_argument("--tau", type=float, default=1.0)
    tab_p.add_argument("--seed", type=int, default=42)
    tab_p.add_argument("--data", type=str, default="data/pb52_dataset.csv")

    args = parser.parse_args()

    if args.command == "run":
        if args.model == "moe":
            run_multiple_trials(
                model_type="moe",
                learning_rate=args.lr,
                num_experts=args.experts,
                hidden_units=6,
                num_trials=args.trials,
                target_mse=args.target_mse,
                max_epochs=args.epochs,
                sigma=args.sigma,
                tau=args.tau,
                seed=args.seed,
                data_path=args.data,
            )
        else:
            run_multiple_trials(
                model_type="bp",
                learning_rate=args.lr,
                num_experts=4,
                hidden_units=args.hidden,
                num_trials=args.trials,
                target_mse=args.target_mse,
                max_epochs=args.epochs,
                sigma=2.0,
                tau=1.0,
                seed=args.seed,
                data_path=args.data,
            )

    elif args.command == "table1":
        run_table1_experiments(
            trials=args.trials,
            moe_lr=args.moe_lr,
            bp_lr=args.bp_lr,
            target_mse=args.target_mse,
            sigma=args.sigma,
            tau=args.tau,
            max_epochs=args.epochs,
            seed=args.seed,
            data_path=args.data,
        )


if __name__ == "__main__":
    main()