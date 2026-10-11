from tracking import *
import os
import random
import sys
import matplotlib.pyplot as plt
from tracking import make_logged_learn_advantage
import numpy as np
import torch

# Make Python load our locally compiled OpenSpiel module first
sys.path.insert(
    0,
    r"C:\Users\PC\Documents\open_spiel\build\lib.win-amd64-cpython-313"
)

import pyspiel
from open_spiel.python.pytorch import deep_cfr

GAME_PARAMS = ("universal_poker(""betting=nolimit,""numPlayers=6,""numRounds=4,""blind=100 50 0 0 0 0,""firstPlayer=2 1 1 1,""numSuits=4,""numRanks=13,""numHoleCards=2,""numBoardCards=0 3 1 1,""stack=20000 20000 20000 20000 20000 20000,""bettingAbstraction=fcpa"")")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# ============================================================
# 2. TRAINING HYPERPARAMETERS

NUM_ITERATIONS = 100
NUM_TRAVERSALS = 20

POLICY_TRAIN_STEPS = 100
ADVANTAGE_TRAIN_STEPS = 50

BATCH_SIZE = 128
MEMORY_CAPACITY = 10_000
POLICY_HIDDEN = [256, 256]
ADVANTAGE_HIDDEN = [128, 128]
SEED = 42
# Seed random-number generators to help make experiments reproducible.

CHECKPOINT_PATH = os.path.join("models", f"deepcfr_policy_it{NUM_ITERATIONS}_tr{NUM_TRAVERSALS}.pt")
# File where the trained policy and training metadata will be saved.

import collections

def validate_policy(policy_network, iteration):
    # Insert your existing bot creation and play_hand logic here.
    # Use one trained bot, one passive bot, and four random bots.
    # Calculate the trained bot's mean payoff in big blinds per hand.

    mean_return_bb = ...  # Calculate from the played hands.

    validation_history["iteration"].append(iteration)
    validation_history["mean_return_bb"].append(mean_return_bb)

    return mean_return_bb


def solve_with_validation(solver, validate_fn, validation_interval=20):
    advantage_losses = collections.defaultdict(list)

    for _ in range(solver._num_iterations):

        # 1. Collect traversal samples and train each player's
        #    advantage network, just as OpenSpiel normally does.
        for player in range(solver._num_players):
            for _ in range(solver._num_traversals):
                solver._traverse_game_tree(solver._root_node, player)

            if solver._reinitialize_advantage_networks:
                solver._reinitialize_advantage_network(player)

            loss = solver._learn_advantage_network(player)
            advantage_losses[player].append(loss)

        # Mark the iteration as completed.
        solver._iteration += 1
        completed_iteration = solver._iteration - 1

        # 2. Validate at the requested interval.
        if completed_iteration % validation_interval == 0:
            # Train/update the policy network using accumulated
            # strategy memory before evaluating it.
            policy_loss = solver._learn_strategy_network()

            # Evaluate the current policy against fixed opponents.
            mean_return_bb = validate_fn(
                solver._policy_network,
                completed_iteration,
            )

            print(
                f"Validation at iteration {completed_iteration} | "
                f"Policy loss: {policy_loss} | "
                f"Mean return: {mean_return_bb:.3f} BB/hand"
            )

    # Preserve OpenSpiel's final policy-training step.
    policy_loss = solver._learn_strategy_network()

    return solver._policy_network, advantage_losses, policy_loss

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = pyspiel.load_game(GAME_PARAMS)

    # Number of input features in OpenSpiel's information-state
    # encoding and number of distinct action IDs in the game.
    input_size = game.information_state_tensor_size()
    output_size = game.num_distinct_actions()

    print_network_config(input_size, output_size, ADVANTAGE_HIDDEN)

    # --------------------------------------------------------
    # 5C. INITIALIZE THE DEEP CFR SOLVER
    """input_size': 281, 'output_size': 4, 'policy_hidden_sizes': [256, 256], 'seed': 42, 'policy_state_dict': OrderedDict({'
    """
    solver = deep_cfr.DeepCFRSolver(game,policy_network_layers=POLICY_HIDDEN,advantage_network_layers=ADVANTAGE_HIDDEN,num_iterations=NUM_ITERATIONS,
        num_traversals=NUM_TRAVERSALS,learning_rate=1e-3,batch_size_advantage=BATCH_SIZE,batch_size_strategy=BATCH_SIZE,
        memory_capacity=MEMORY_CAPACITY,policy_network_train_steps=POLICY_TRAIN_STEPS,advantage_network_train_steps=ADVANTAGE_TRAIN_STEPS,
        reinitialize_advantage_networks=True,device=device,seed=SEED,print_nash_convs=False,
    )

    # -------------------------------------------------------- #
    # 5F. RUN TRAINING #########################################
    # -------------------------------------------------------- #
    training_history = {
        "iteration": [],
        "player": [],
        "advantage_loss": [],
        "loss_policy_MSE": [],
    }
    validation_history = {
        "iteration": [],
        "mean_return_bb": [],
    }

    solver._learn_advantage_network = make_logged_learn_advantage(solver, training_history)


    solve_with_validation(solver, validate_policy, validation_interval=20)

    print("\nTraining finished.")
    print("Final policy loss:", final_policy_loss)

    plot_training(training_history)

    # --------------------------------------------------------
    # 5G. SAVE THE TRAINED POLICY CHECKPOINT

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    # Save the policy weights and metadata needed to reconstruct a compatible model later. The advantage networks and solver
    # memories are not included in this policy checkpoint.
    checkpoint = {
        "game_params": GAME_PARAMS,
        "input_size": input_size,
        "output_size": output_size,
        "policy_hidden_sizes": POLICY_HIDDEN,
        "seed": SEED,
        # Learned neural-network parameters.
        "policy_state_dict": policy_net.state_dict()
    }

    torch.save(checkpoint, CHECKPOINT_PATH)
    print("Saved checkpoint:", CHECKPOINT_PATH)



if __name__ == "__main__":
    main()
