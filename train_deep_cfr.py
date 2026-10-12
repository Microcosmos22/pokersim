from tracking import *
from compare_bots import *
import os
import random
import sys
import matplotlib.pyplot as plt
from tracking import make_logged_learn_advantage
import numpy as np
import torch
from heuristicMC import HeuristicPokerBot
import inspect

# Make Python load our locally compiled OpenSpiel module first
sys.path.insert(
    0,
    r"C:\Users\PC\Documents\open_spiel\build\lib.win-amd64-cpython-313"
)

import pyspiel
from open_spiel.python.pytorch import deep_cfr

GAME_PARAMS = "universal_poker(betting=nolimit,numPlayers=4,numRounds=4,blind=2 1 0 0,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,stack=400 400 400 400,bettingAbstraction=fcpa)"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================================
# 2. TRAINING HYPERPARAMETERS
validation_interval = 5
VALIDATION_SEED = 42
VALIDATION_HANDS = 2000
BIG_BLIND = 2

NUM_ITERATIONS = 20000
NUM_TRAVERSALS = 100

POLICY_TRAIN_STEPS = 100
ADVANTAGE_TRAIN_STEPS = 50

BATCH_SIZE = 128
MEMORY_CAPACITY = 10_000
POLICY_HIDDEN = [256, 256]
ADVANTAGE_HIDDEN = [128, 128]
SEED = 42
# Seed random-number generators to help make experiments reproducible.

training_history = {"iteration": [],"player": [],
    "advantage_loss": [],"loss_policy_MSE": []}
validation_history = {"iteration": [], "mean_return_bb": [],  "weighted_policy_loss": [], "unweighted_policy_loss": []}

CHECKPOINT_PATH = os.path.join("models", f"deepcfr_policy_it{NUM_ITERATIONS}_tr{NUM_TRAVERSALS}.pt")
# File where the trained policy and training metadata will be saved.

import collections
def validate_policy(
    policy_network,
    iteration,
    game,
    device,
    validation_history,
):
    rng = np.random.default_rng(VALIDATION_SEED)
    returns_bb = []
    num_players = game.num_players()

    for hand in range(VALIDATION_HANDS):
        tested_seat = hand % num_players
        seed_base = VALIDATION_SEED + hand * num_players

        bots = [None] * num_players

        bots[tested_seat] = DeepCFRBot(
            player_id=tested_seat,
            policy_net=policy_network,
            device=device,
            seed=seed_base + tested_seat,
        )

        for seat in range(num_players):
            if seat == tested_seat:
                continue

            bots[seat] = HeuristicPokerBot(
                player_id=seat,
                seed=seed_base + seat,
                equity_samples=32,
            )

        result = play_hand(game, bots, rng)
        returns_bb.append(float(result[tested_seat]) / BIG_BLIND)

    mean_return_bb = float(np.mean(returns_bb))

    validation_history["iteration"].append(iteration)
    validation_history["mean_return_bb"].append(mean_return_bb)

    print(
        f"Validation | Iteration {iteration} | "
        f"{VALIDATION_HANDS} hands | "
        f"Mean return: {mean_return_bb:.3f} BB/hand"
    )

    return mean_return_bb

def solve_with_validation(game, solver, validate_fn, validation_history, training_history, validation_interval=2):
    advantage_losses = collections.defaultdict(list)

    for i in range(solver._num_iterations):
        print(f" Iteration {i}")
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
            policy_loss = solver._learn_strategy_network()

            validation_history["weighted_policy_loss"].append(solver._last_weighted_policy_loss)
            validation_history["unweighted_policy_loss"].append(solver._last_unweighted_policy_loss)

            print(f"Policy MSE | Weighted: {solver._last_weighted_policy_loss:.6f} | Unweighted: {solver._last_unweighted_policy_loss:.6f}")

            mean_return_bb = validate_fn(solver._policy_network,i,game,device,validation_history,)

            input_size = game.information_state_tensor_size()
            output_size = game.num_distinct_actions()
            checkpoint = {"game_params": GAME_PARAMS, "input_size": input_size, "output_size": output_size,
                "policy_hidden_sizes": POLICY_HIDDEN, "advantage_hidden_sizes": ADVANTAGE_HIDDEN, "seed": SEED,
                "policy_state_dict": solver._policy_network.state_dict(),
                "advantage_state_dicts": [net.state_dict() for net in solver._advantage_networks]}
            plot_training(training_history, validation_history, validation_interval)
            torch.save(checkpoint, CHECKPOINT_PATH)
            print("Saved checkpoint:", CHECKPOINT_PATH)

            print(
                f"Validation at iteration {completed_iteration} | "
                f"Policy loss: {policy_loss} | "
                f"Mean return: {mean_return_bb:.3f} BB/hand"
            )

    # Preserve OpenSpiel's final policy-training step.
    policy_loss = solver._learn_strategy_network()

    return solver._policy_network, advantage_losses, policy_loss

def main():

    game = pyspiel.load_game(GAME_PARAMS)
    input_size = game.information_state_tensor_size()
    output_size = game.num_distinct_actions()

    print_network_config(input_size, output_size, ADVANTAGE_HIDDEN, POLICY_HIDDEN)

    """input_size': 281, 'output_size': 4, 'policy_hidden_sizes': [256, 256], 'seed': 42, 'policy_state_dict': OrderedDict({'
    """
    solver = deep_cfr.DeepCFRSolver(game,policy_network_layers=POLICY_HIDDEN,advantage_network_layers=ADVANTAGE_HIDDEN,num_iterations=NUM_ITERATIONS,
        num_traversals=NUM_TRAVERSALS,learning_rate=1e-3,batch_size_advantage=BATCH_SIZE,batch_size_strategy=BATCH_SIZE,
        memory_capacity=MEMORY_CAPACITY,policy_network_train_steps=POLICY_TRAIN_STEPS,advantage_network_train_steps=ADVANTAGE_TRAIN_STEPS,
        reinitialize_advantage_networks=True,device=device,seed=SEED,print_nash_convs=False,
    )

    print("Deep CFR module:", deep_cfr.__file__)
    print("Method source:", inspect.getsourcefile(type(solver)._learn_strategy_network))
    print(inspect.getsource(type(solver)._learn_strategy_network))

    ###############################     LOAD
    model_name = f"deepcfr_policy_it20000_tr100BENCHMARK.pt"
    checkpoint = torch.load(
        os.path.join("models", model_name),
        map_location=device,
        weights_only=False,
    )
    print(f"Loading warmup model: {os.path.join("models", model_name)}")
    solver._policy_network.load_state_dict(
        checkpoint["policy_state_dict"]
    )

    # -------------------------------------------------------- #
    # 5F. RUN TRAINING #########################################
    # -------------------------------------------------------- #

    solver._learn_advantage_network = make_logged_learn_advantage(solver, training_history)


    policy_net, advantage_losses, final_policy_loss = solve_with_validation(game, solver, validate_policy, validation_history, training_history, validation_interval)

    print("\nTraining finished.")
    print("Final policy loss:", final_policy_loss)

    plot_training(training_history, validation_history, validation_interval)

    # --------------------------------------------------------
    # 5G. SAVE THE TRAINED POLICY CHECKPOINT

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    # Save the policy weights and metadata needed to reconstruct a compatible model later. The advantage networks and solver
    # memories are not included in this policy checkpoint.
    print([name for name in vars(solver) if "advantage_network" in name])
    checkpoint = {"game_params": GAME_PARAMS, "input_size": input_size, "output_size": output_size,
        "policy_hidden_sizes": POLICY_HIDDEN, "advantage_hidden_sizes": ADVANTAGE_HIDDEN, "seed": SEED,
        "policy_state_dict": solver._policy_network.state_dict(),
        "advantage_state_dicts": [net.state_dict() for net in solver._advantage_networks]}

    torch.save(checkpoint, CHECKPOINT_PATH)
    print("Saved checkpoint:", CHECKPOINT_PATH)



if __name__ == "__main__":
    main()
