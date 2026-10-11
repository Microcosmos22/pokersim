
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

        print(
            f"Iteration {solver._iteration + 1:03d} | "
            f"Player {player} advantage loss: {loss}"
        )

        return loss

    return logged_learn_advantage

import numpy as np
import matplotlib.pyplot as plt


def plot_training(training_history):
    """Plot advantage loss averaged over players for each iteration.

    Returns:
        fig, ax: Matplotlib figure and axes, allowing validation results
                 to be added to the same plot.
    """
    iterations = np.array(training_history["iteration"])
    losses = np.array(training_history["advantage_loss"])

    unique_iterations = np.unique(iterations)

    mean_losses = [
        np.mean(losses[iterations == iteration])
        for iteration in unique_iterations
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(unique_iterations, mean_losses, label="Mean advantage loss")

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Advantage loss (MSE)")
    ax.set_title("Deep CFR Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    plt.show()

    return fig, ax



def print_network_config(input_size, output_size, ADVANTAGE_HIDDEN):
    print("\n========== NETWORK CONFIGURATION ==========")
    print(f"Input features: {input_size}")
    print(
        f"Policy network: {input_size} -> "
        f"{POLICY_HIDDEN} -> {output_size}"
    )
    print(f"Advantage network hidden layers: {ADVANTAGE_HIDDEN}")
    print("Policy output: one score/probability per action ID")
    print("===========================================\n")
