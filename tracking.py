from pathlib import Path
import os
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch


# Make Python load our locally compiled OpenSpiel module first.
# This build includes the universal_poker game we need.
sys.path.insert(
    0,
    r"C:\Users\PC\Documents\open_spiel\build\lib.win-amd64-cpython-313"
)

import pyspiel
from open_spiel.python.pytorch import deep_cfr


# ============================================================
# 3. POLICY LOSS LOGGER

def make_logged_learn_advantage(solver, history):
    # Save the original bound method before replacing it.
    original_method = solver._learn_advantage_network

    def logged_learn_advantage(player):
        loss = original_method(player)

        if loss is not None:
            history["iteration"].append(solver._iteration + 1)
            history["player"].append(player)
            history["advantage_loss"].append(float(loss))

        return loss

    return logged_learn_advantage

import numpy as np
import matplotlib.pyplot as plt

def plot_training(training_history, validation_history, validation_interval):
    """Plot advantage loss, policy losses, and validation return."""

    # --- Advantage loss averaged across players ---
    iterations = np.asarray(training_history.get("iteration", []), dtype=int)
    losses = np.asarray(training_history.get("advantage_loss", []), dtype=float)
    unique_iterations = np.unique(iterations)
    mean_losses = [float(np.mean(losses[iterations == iteration])) for iteration in unique_iterations]

    # --- Validation metrics ---
    validation_iterations = np.asarray(validation_history.get("iteration", []), dtype=int)
    validation_returns = np.asarray(validation_history.get("mean_return_bb", []), dtype=float)
    weighted_losses = np.asarray(validation_history.get("weighted_policy_loss", []), dtype=float)
    unweighted_losses = np.asarray(validation_history.get("unweighted_policy_loss", []), dtype=float)

    # Separate panels prevent one metric's scale from hiding another.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flat

    # 1. Advantage loss
    if len(unique_iterations) > 0:
        ax1.plot(unique_iterations, mean_losses, label="Mean advantage loss")
    ax1.set_title("Advantage Network")
    ax1.set_xlabel("Training iteration")
    ax1.set_ylabel("Advantage loss (MSE)")
    ax1.grid(True, alpha=0.3)

    # 2. Weighted policy loss
    if len(validation_iterations) > 0 and len(weighted_losses) == len(validation_iterations):
        ax2.plot(validation_iterations, weighted_losses, marker="o", label="Weighted policy MSE")
    ax2.set_title("Weighted Policy Loss")
    ax2.set_xlabel("Training iteration")
    ax2.set_ylabel("Weighted MSE")
    ax2.grid(True, alpha=0.3)

    # 3. Unweighted policy loss
    if len(validation_iterations) > 0 and len(unweighted_losses) == len(validation_iterations):
        ax3.plot(validation_iterations, unweighted_losses, marker="o", label="Unweighted policy MSE")
    ax3.set_title("Unweighted Policy Loss")
    ax3.set_xlabel("Training iteration")
    ax3.set_ylabel("Unweighted MSE")
    ax3.grid(True, alpha=0.3)

    # 4. Validation return
    if len(validation_iterations) > 0 and len(validation_returns) == len(validation_iterations):
        ax4.plot(validation_iterations, validation_returns, marker="o", label="Validation return")
    ax4.set_title("Policy Performance")
    ax4.set_xlabel("Training iteration")
    ax4.set_ylabel("Mean return (BB/hand)")
    ax4.axhline(0, linestyle="--", alpha=0.5)
    ax4.grid(True, alpha=0.3)

    # Use ticks aligned with the validation interval without overcrowding.
    all_iterations = np.concatenate((unique_iterations, validation_iterations))
    max_iteration = int(np.max(all_iterations)) if all_iterations.size else 0

    if validation_interval > 0 and max_iteration > 0:
        tick_multiplier = max(1, int(np.ceil(max_iteration / (validation_interval * 12))))
        tick_step = validation_interval * tick_multiplier
        ticks = np.arange(tick_step, max_iteration + 1, tick_step)
        for ax in axes.flat:
            ax.set_xticks(ticks)

    for ax in axes.flat:
        if ax.lines:
            ax.legend()

    fig.suptitle("Deep CFR Training and Validation")
    fig.tight_layout()

    # Save a distinct file for each checkpoint to avoid overwriting problems.
    output_path = Path(__file__).resolve().parent / f"training_iter_{max_iteration:04d}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Training plot saved to: {output_path}")

    return fig, axes

def print_network_config(input_size, output_size, ADVANTAGE_HIDDEN, POLICY_HIDDEN):
    print("\n========== NETWORK CONFIGURATION ==========")
    print(f"Input features: {input_size}")
    print(
        f"Policy network: {input_size} -> "
        f"{POLICY_HIDDEN} -> {output_size}"
    )
    print(f"Advantage network hidden layers: {ADVANTAGE_HIDDEN}")
    print("Policy output: one score/probability per action ID")
    print("===========================================\n")
