
import os
import random

import numpy as np
import torch
import pyspiel
from open_spiel.python.pytorch import deep_cfr


GAME_PARAMS = (
    "universal_poker("
    "betting=nolimit,"
    "numPlayers=2,"
    "numRounds=4,"
    "blind=100 50,"
    "firstPlayer=2 1 1 1,"
    "numSuits=4,"
    "numRanks=13,"
    "numHoleCards=2,"
    "numBoardCards=0 3 1 1,"
    "stack=20000 20000,"
    "bettingAbstraction=fcpa"
    ")"
)

# Small initial run; increase after verifying everything works.
NUM_ITERATIONS = 10
NUM_TRAVERSALS = 20
POLICY_TRAIN_STEPS = 100
ADVANTAGE_TRAIN_STEPS = 50
BATCH_SIZE = 128
MEMORY_CAPACITY = 10_000

POLICY_HIDDEN = [256, 256]
ADVANTAGE_HIDDEN = [128, 128]

SEED = 42
CHECKPOINT_PATH = os.path.join("models", "deepcfr_policy.pt")


class LoggingMSELoss(torch.nn.Module):
    """MSE loss that prints the loss for every policy update."""

    def __init__(self):
        super().__init__()
        self.step = 0

    def forward(self, prediction, target):
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.step += 1
        print(
            f"Policy update {self.step:04d} | "
            f"MSE = {loss.detach().item():.6f}"
        )
        return loss


def show_encoding_example(game, policy_net, device):
    """Show one information-state vector and its network output."""

    state = game.new_initial_state()

    # Sample chance events until a player can act.
    while state.is_chance_node():
        outcomes, probabilities = zip(*state.chance_outcomes())
        action = int(np.random.choice(outcomes, p=probabilities))
        state.apply_action(action)

    player = state.current_player()

    # OpenSpiel encodes the acting player's information state.
    encoded = np.asarray(
        state.information_state_tensor(), dtype=np.float32
    )

    # Add batch dimension: [features] -> [1, features].
    x = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)

    policy_net.eval()
    with torch.inference_mode():
        output = policy_net(x)

    legal = state.legal_actions(player)
    probs = output[0, legal]
    probs = probs.clamp(min=0)

    if probs.sum().item() > 0:
        probs = probs / probs.sum()
    else:
        probs = torch.full_like(probs, 1.0 / len(legal))

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


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Training device:", device)

    game = pyspiel.load_game(GAME_PARAMS)

    input_size = game.information_state_tensor_size()
    output_size = game.num_distinct_actions()

    print("\n========== NETWORK CONFIGURATION ==========")
    print(f"Input features: {input_size}")
    print(f"Policy network: {input_size} -> {POLICY_HIDDEN} -> {output_size}")
    print(f"Advantage network hidden layers: {ADVANTAGE_HIDDEN}")
    print("Policy output: one score/probability per action ID")
    print("===========================================\n")

    solver = deep_cfr.DeepCFRSolver(
        game,
        policy_network_layers=POLICY_HIDDEN,
        advantage_network_layers=ADVANTAGE_HIDDEN,
        num_iterations=NUM_ITERATIONS,
        num_traversals=NUM_TRAVERSALS,
        learning_rate=1e-3,
        batch_size_advantage=BATCH_SIZE,
        batch_size_strategy=BATCH_SIZE,
        memory_capacity=MEMORY_CAPACITY,
        policy_network_train_steps=POLICY_TRAIN_STEPS,
        advantage_network_train_steps=ADVANTAGE_TRAIN_STEPS,
        reinitialize_advantage_networks=True,
        device=device,
        seed=SEED,
        print_nash_convs=False,
    )

    # Instrument policy loss when this attribute exists in the installed version.
    if hasattr(solver, "_loss_policy"):
        solver._loss_policy = LoggingMSELoss()
    else:
        print(
            "Note: this OpenSpiel version has no _loss_policy attribute; "
            "only the losses returned by solve() will be reported."
        )

    # Log the advantage-training error returned for each player and iteration.
    original_learn_advantage = solver._learn_advantage_network

    def logged_learn_advantage(player):
        loss = original_learn_advantage(player)
        print(
            f"Iteration {solver._iteration + 1:03d} | "
            f"Player {player} advantage loss: {loss}"
        )
        return loss

    solver._learn_advantage_network = logged_learn_advantage

    # solve() returns: (policy network, advantage losses, final policy loss).
    policy_net, advantage_losses, final_policy_loss = solver.solve()

    print("\nTraining finished.")
    print("Final policy loss:", final_policy_loss)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    checkpoint = {
        "game_params": GAME_PARAMS,
        "input_size": input_size,
        "output_size": output_size,
        "policy_hidden_sizes": POLICY_HIDDEN,
        "seed": SEED,
        "policy_state_dict": policy_net.state_dict(),
        "final_policy_loss": (
            float(final_policy_loss)
            if final_policy_loss is not None else None
        ),
        "advantage_losses": {
            str(player): [
                float(loss) if loss is not None else None
                for loss in losses
            ]
            for player, losses in advantage_losses.items()
        },
    }

    torch.save(checkpoint, CHECKPOINT_PATH)
    print("Saved checkpoint:", CHECKPOINT_PATH)

    show_encoding_example(game, policy_net, device)


if __name__ == "__main__":
    main()
