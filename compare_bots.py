
import os
import sys
import numpy as np
import torch
sys.path.insert(
    0,
    r"C:\Users\PC\Documents\open_spiel\build\lib.win-amd64-cpython-313"
)

import pyspiel
from open_spiel.python.pytorch import deep_cfr


CHECKPOINT_PATH = os.path.join("models", "deepcfr_policy.pt")
CHECKPOINT_PATH_1 = os.path.join("models", "deepcfr_policy_it100_tr20.pt")
NUM_HANDS = 100
SEED = 123


class DeepCFRBot:
    def __init__(self, player_id, policy_net, device, seed=0):
        self.player_id = player_id
        self.policy_net = policy_net
        self.device = device
        self.rng = np.random.default_rng(seed)

    def step(self, state):
        legal_actions = state.legal_actions(self.player_id)

        # 1. Encode the current information state.
        encoded = np.asarray(
            state.information_state_tensor(), dtype=np.float32
        )
        x = torch.tensor(
            encoded, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # 2. Run inference.
        self.policy_net.eval()
        with torch.inference_mode():
            output = self.policy_net(x)[0].cpu().numpy()

        # 3. Keep only legal actions.
        # Output index is the OpenSpiel action ID.
        probabilities = np.array(
            [max(0.0, float(output[action])) for action in legal_actions],
            dtype=np.float64,
        )

        # 4. Normalize and sample an action.
        total = probabilities.sum()
        if not np.isfinite(total) or total <= 0:
            probabilities = np.full(
                len(legal_actions), 1.0 / len(legal_actions)
            )
        else:
            probabilities /= total

        return int(self.rng.choice(legal_actions, p=probabilities))


class PassiveBot:
    """Check/call if possible; otherwise fold."""

    def __init__(self, player_id):
        self.player_id = player_id

    def step(self, state):
        legal = state.legal_actions(self.player_id)
        descriptions = [
            (a, state.action_to_string(self.player_id, a).lower())
            for a in legal
        ]

        for action, description in descriptions:
            if "check" in description or "call" in description:
                return int(action)

        for action, description in descriptions:
            if "fold" in description:
                return int(action)

        return int(legal[0])


def load_policy(CHECKPOINT_PATH):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    game = pyspiel.load_game(checkpoint["game_params"])

    # Reconstruct the same MLP architecture before loading its weights.
    policy_net = deep_cfr.MLP(input_size=checkpoint["input_size"],hidden_sizes=checkpoint["policy_hidden_sizes"],
        output_size=checkpoint["output_size"],final_activation=torch.nn.Softmax(dim=-1),seed=checkpoint["seed"],
    ).to(device)

    policy_net.load_state_dict(checkpoint["policy_state_dict"])
    policy_net.eval()

    print("Loaded:", CHECKPOINT_PATH)
    print("Device:", device)
    print(
        f"Network: {checkpoint['input_size']} -> "
        f"{checkpoint['policy_hidden_sizes']} -> "
        f"{checkpoint['output_size']}"
    )

    return game, policy_net, device


def play_hand(game, bots, rng):
    state = game.new_initial_state()

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probabilities = zip(*state.chance_outcomes())
            action = int(rng.choice(outcomes, p=probabilities))
        else:
            player = state.current_player()
            action = bots[player].step(state)

        state.apply_action(action)

    return state.returns()


def compare(game, trained_bots, device, opponent_type):
    rng = np.random.default_rng(SEED)
    wins = losses = ties = 0
    returns = []

    for hand in range(NUM_HANDS):
        # Alternate which trained bot is evaluated.
        deep_seat = hand % 2

        # Start with six seats.
        bots = [None] * 6

        # Seat the two trained bots.
        bots[0] = trained_bots[0]
        bots[1] = trained_bots[1]

        # One passive bot.
        bots[2] = PassiveBot(2)

        # Three random bots.
        for seat in range(3, 6):
            bots[seat] = pyspiel.make_uniform_random_bot(
                seat,
                SEED + 10_000 * hand + seat
            )

        # Play only after all six seats have been assigned.
        result = play_hand(game, bots, rng)

        value = float(result[deep_seat])
        returns.append(value)

        if value > 0:
            wins += 1
        elif value < 0:
            losses += 1
        else:
            ties += 1

        if (hand + 1) % 10 == 0:
            print(f"{opponent_type}: {hand + 1}/{NUM_HANDS} hands")

    print(f"\nDeep CFR vs {opponent_type}")
    print(f"Wins: {wins} | Losses: {losses} | Ties: {ties}")

    decisive = wins + losses
    print(
        "Win rate (excluding ties): "
        f"{wins / decisive:.1%}" if decisive
        else "Win rate: n/a"
    )
    print(f"Mean return per hand: {np.mean(returns):.2f}")
    print(f"Total return: {np.sum(returns):.2f}")


def main():
    game, policy_net, device = load_policy(CHECKPOINT_PATH)
    game, policy_net1, device = load_policy(CHECKPOINT_PATH_1)

    deep_bot = DeepCFRBot(0, policy_net, device, seed=SEED + hand)
    deep_bot1 = DeepCFRBot(1, policy_net1, device, seed=SEED + hand)

    compare(game, [deep_bot, deep_bot1], device, "passive")


if __name__ == "__main__":
    main()
