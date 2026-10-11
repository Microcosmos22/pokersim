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
    """Plot mean advantage loss and validation return in BB per hand."""

    # --- Average advantage loss across players for each iteration ---
    iterations = np.asarray(training_history["iteration"])
    losses = np.asarray(training_history["advantage_loss"])

    unique_iterations = np.unique(iterations)
    mean_losses = [
        np.mean(losses[iterations == iteration])
        for iteration in unique_iterations
    ]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    line1, = ax1.plot(
        unique_iterations,
        mean_losses,
        label="Mean advantage loss",
    )
    ax1.set_xlabel("Training iteration")
    ax1.set_ylabel("Advantage loss (MSE)")
    ax1.grid(True, alpha=0.3)

    # --- Validation return, using a separate y-axis ---
    validation_iterations = np.asarray(
        validation_history["iteration"]
    )
    validation_returns = np.asarray(
        validation_history["mean_return_bb"]
    )

    ax2 = ax1.twinx()

    lines = [line1]
    labels = [line1.get_label()]

    if len(validation_iterations) > 0:
        line2, = ax2.plot(
            validation_iterations,
            validation_returns,
            marker="o",
            label="Validation return",
        )
        lines.append(line2)
        labels.append(line2.get_label())

    ax2.set_ylabel("Mean return (BB/hand)")

    # Place x-axis ticks at the validation interval.
    max_iteration = max(
        np.max(unique_iterations) if len(unique_iterations) else 0,
        np.max(validation_iterations) if len(validation_iterations) else 0,
    )

    if validation_interval > 0:
        ticks = np.arange(
            validation_interval,
            max_iteration + 1,
            validation_interval,
        )
        if len(ticks) > 0:
            ax1.set_xticks(ticks)

    ax1.legend(lines, labels, loc="best")
    ax1.set_title("Deep CFR Training and Validation")
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "training.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Training plot saved to: {output_path}")

    return fig, ax1

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
