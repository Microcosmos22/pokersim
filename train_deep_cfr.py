
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
# 1. DEFINE THE POKER ENVIRONMENT
# ============================================================
# OpenSpiel acts as the game simulator. It defines the rules,
# deals cards, tracks bets and determines which actions are legal.
#
# Deep CFR will train a strategy for this particular game setup.

GAME_PARAMS = (
    "universal_poker("
    "betting=nolimit,"                  # No-limit betting rules
    "numPlayers=6,"                     # Six players per hand
    "numRounds=4,"                      # Preflop, flop, turn and river
    "blind=100 50 0 0 0 0,"             # Blind configuration by position
    "firstPlayer=2 1 1 1,"              # First player to act in each round
    "numSuits=4,"                       # Four suits
    "numRanks=13,"                      # Thirteen ranks per suit
    "numHoleCards=2,"                   # Two private cards per player
    "numBoardCards=0 3 1 1,"            # 3 flop cards, turn and river
    "stack=20000 20000 20000 20000 20000 20000,"
    "bettingAbstraction=fcpa"           # Available betting-action abstraction
    ")"
)


# ============================================================
# 2. TRAINING HYPERPARAMETERS
# ============================================================
# These settings control how much training the Deep CFR solver
# performs and the size of its neural networks.
#
# The values below are deliberately small for an initial test.
# A successful run does not necessarily mean the bot is strong.

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


# ============================================================
# 3. POLICY LOSS LOGGER
# ============================================================
# MSE (mean squared error) measures the difference between
# the policy network's predictions and its training targets.
#
# This class records and prints the loss whenever the solver
# calls it. The loss measures fitting quality, not poker winnings.

class LoggingMSELoss(torch.nn.Module):
    """MSE loss that prints the loss for every policy update."""

    def __init__(self):
        super().__init__()
        self.step = 0
        self.losses = []  # Stores MSE values for plotting later.

    def forward(self, prediction, target):
        # Calculate the average squared prediction error.
        loss = torch.nn.functional.mse_loss(prediction, target)

        # Count and report this policy-training update.
        self.step += 1
        mse = loss.detach().item()
        self.losses.append(mse)

        print(
            f"Policy update {self.step:04d} | "
            f"MSE = {mse:.6f}"
        )

        return loss


# ============================================================
# 4. DIAGNOSTIC: INSPECT ONE POLICY DECISION
# ============================================================
# After training, this function checks how the policy network
# processes one sampled poker situation.
#
# It prints the encoded input, output dimensions and probabilities
# assigned to the legal actions. This is a diagnostic, not a
# measurement of how well the bot plays.

def show_encoding_example(game, policy_net, device):
    """Show one information-state vector and its network output."""

    # Create a fresh game at its initial state.
    state = game.new_initial_state()

    # Resolve chance events (such as initial card deals) until
    # a player can act. Chance events are sampled from the game's
    # specified probability distribution.
    while state.is_chance_node():
        outcomes, probabilities = zip(*state.chance_outcomes())
        action = int(np.random.choice(outcomes, p=probabilities))
        state.apply_action(action)

    # Identify the player whose decision this state represents.
    player = state.current_player()

    # Encode the information available to the acting player.
    # This is a feature vector, not the complete hidden game state.
    encoded = np.asarray(
        state.information_state_tensor(), dtype=np.float32
    )

    # Neural networks expect batches. Add a dimension so that
    # [features] becomes [1, features], representing one example.
    x = torch.tensor(
        encoded, dtype=torch.float32, device=device
    ).unsqueeze(0)

    # Switch to inference mode: no gradients or parameter updates
    # are needed for this diagnostic.
    policy_net.eval()
    with torch.inference_mode():
        output = policy_net(x)

    # The network has an output slot for each distinct action ID,
    # but only some actions are legal in this particular state.
    legal = state.legal_actions(player)
    probs = output[0, legal]

    # Clamp negative outputs to zero before normalizing the legal
    # action values. If all values are zero, use a uniform fallback.
    probs = probs.clamp(min=0)

    if probs.sum().item() > 0:
        probs = probs / probs.sum()
    else:
        probs = torch.full_like(probs, 1.0 / len(legal))

    # Print dimensions and the action probabilities for inspection.
    print("\n========== ENCODING DIAGNOSTIC ==========")
    print("Encoding source: state.information_state_tensor()")
    print("Player to act:", player)
    print("Encoded NumPy shape:", encoded.shape)
    print("Network input shape:", tuple(x.shape))
    print("Network output shape:", tuple(output.shape))
    print("Legal action count:", len(legal))
    print("Legal action probabilities:")

    for action, probability in zip(legal, probs.cpu().tolist()):
        print(
            f"  action={action:3d} | "
            f"probability={probability:.4f} | "
            f"{state.action_to_string(player, action)}"
        )

    print("=========================================\n")


# ============================================================
# 5. MAIN TRAINING PIPELINE
# ============================================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = pyspiel.load_game(GAME_PARAMS)

    # Number of input features in OpenSpiel's information-state
    # encoding and number of distinct action IDs in the game.
    input_size = game.information_state_tensor_size()
    output_size = game.num_distinct_actions()

    print("\n========== NETWORK CONFIGURATION ==========")
    print(f"Input features: {input_size}")
    print(
        f"Policy network: {input_size} -> "
        f"{POLICY_HIDDEN} -> {output_size}"
    )
    print(f"Advantage network hidden layers: {ADVANTAGE_HIDDEN}")
    print("Policy output: one score/probability per action ID")
    print("===========================================\n")

    # --------------------------------------------------------
    # 5C. INITIALIZE THE DEEP CFR SOLVER
    """input_size': 281, 'output_size': 4, 'policy_hidden_sizes': [256, 256], 'seed': 42, 'policy_state_dict': OrderedDict({'
    """
    solver = deep_cfr.DeepCFRSolver(game,policy_network_layers=POLICY_HIDDEN,advantage_network_layers=ADVANTAGE_HIDDEN,num_iterations=NUM_ITERATIONS,
        num_traversals=NUM_TRAVERSALS,learning_rate=1e-3,batch_size_advantage=BATCH_SIZE,batch_size_strategy=BATCH_SIZE,
        memory_capacity=MEMORY_CAPACITY,policy_network_train_steps=POLICY_TRAIN_STEPS,advantage_network_train_steps=ADVANTAGE_TRAIN_STEPS,
        reinitialize_advantage_networks=True,device=device,seed=SEED,print_nash_convs=False,
    )

    # --------------------------------------------------------
    # 5D. INSTRUMENT POLICY TRAINING
    # --------------------------------------------------------
    # Replace the policy loss function with our logging version,
    # if this OpenSpiel version exposes the expected attribute.
    # The solver must actually call this object for losses to print.

    policy_loss_logger = None

    if hasattr(solver, "_loss_policy"):
        policy_loss_logger = LoggingMSELoss()
        solver._loss_policy = policy_loss_logger
    else:
        print(
            "Note: this OpenSpiel version has no _loss_policy attribute; "
            "only the losses returned by solve() will be reported."
        )

    print("Policy logger exists:", policy_loss_logger is not None)

    if policy_loss_logger is not None:
        print("Number of recorded losses:", len(policy_loss_logger.losses))
        print("First 5 losses:", policy_loss_logger.losses[:5])

    # Plot policy MSE against the number of policy updates.
    if policy_loss_logger is not None and policy_loss_logger.losses:
        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, len(policy_loss_logger.losses) + 1),
            policy_loss_logger.losses
        )
        plt.xlabel("Policy training update")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title("Deep CFR Policy Training Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("policy_mse.png", dpi=150)
        plt.show()
    else:
        print("No policy MSE values were recorded.")

    # --------------------------------------------------------
    # 5E. INSTRUMENT ADVANTAGE NETWORK TRAINING
    # --------------------------------------------------------
    # Advantage networks estimate how actions compare with the
    # current strategy at a given information state.
    # Their training loss measures prediction error against
    # sampled advantage targets, not the player's winnings.

    # Keep a reference to the original solver method.
    original_learn_advantage = solver._learn_advantage_network

    # Wrap the method so we can log its returned loss.
    # The original method still performs the actual training.
    def logged_learn_advantage(player):
        loss = original_learn_advantage(player)

        print(
            f"Iteration {solver._iteration + 1:03d} | "
            f"Player {player} advantage loss: {loss}"
        )

        return loss

    # Replace the method with our logging wrapper.
    solver._learn_advantage_network = logged_learn_advantage

    # --------------------------------------------------------
    # 5F. RUN TRAINING
    # --------------------------------------------------------
    # solve() performs the solver's training procedure and returns:
    # 1. The trained policy network.
    # 2. Recorded advantage losses.
    # 3. The final policy-training loss.

    policy_net, advantage_losses, final_policy_loss = solver.solve()

    print("\nTraining finished.")
    print("Final policy loss:", final_policy_loss)

    # --------------------------------------------------------
    # 5G. SAVE THE TRAINED POLICY CHECKPOINT
    # --------------------------------------------------------

    # Create the models directory if it does not already exist.
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    # Save the policy weights and metadata needed to reconstruct
    # a compatible model later. The advantage networks and solver
    # memories are not included in this policy checkpoint.
    checkpoint = {
        "game_params": GAME_PARAMS,
        "input_size": input_size,
        "output_size": output_size,
        "policy_hidden_sizes": POLICY_HIDDEN,
        "seed": SEED,

        # Learned neural-network parameters.
        "policy_state_dict": policy_net.state_dict(),

        # Store the final policy loss as a regular Python number.
        "final_policy_loss": (
            float(final_policy_loss)
            if final_policy_loss is not None else None
        ),

        # Store advantage losses by player index.
        # Convert tensor-like values to ordinary numbers when possible.
        "advantage_losses": {
            str(player): [
                float(loss) if loss is not None else None
                for loss in losses
            ]
            for player, losses in advantage_losses.items()
        },
    }

    # Serialize the checkpoint to disk.
    torch.save(checkpoint, CHECKPOINT_PATH)



    print("Saved checkpoint:", CHECKPOINT_PATH)

    # --------------------------------------------------------
    # 5H. RUN THE POST-TRAINING DIAGNOSTIC
    # --------------------------------------------------------
    # Check that the saved policy's network architecture accepts
    # the information-state encoding and produces action outputs.
    show_encoding_example(game, policy_net, device)


# Run main() only when this file is executed directly,
# rather than when it is imported as a Python module.
if __name__ == "__main__":
    main()
