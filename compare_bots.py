
import os

import numpy as np
import torch
import pyspiel
from open_spiel.python.pytorch import deep_cfr


CHECKPOINT_PATH = os.path.join("models", "deepcfr_policy.pt")
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


def load_policy():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} not found. Run train_deep_cfr.py first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    game = pyspiel.load_game(checkpoint["game_params"])

    # Reconstruct the same MLP architecture before loading its weights.
    policy_net = deep_cfr.MLP(
        input_size=checkpoint["input_size"],
        hidden_sizes=checkpoint["policy_hidden_sizes"],
        output_size=checkpoint["output_size"],
        final_activation=torch.nn.Softmax(dim=-1),
        seed=checkpoint["seed"],
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


def compare(game, policy_net, device, opponent_type):
    rng = np.random.default_rng(SEED)
    wins = losses = ties = 0
    returns = []

    for hand in range(NUM_HANDS):
        deep_seat = hand % 2
        opponent_seat = 1 - deep_seat

        deep_bot = DeepCFRBot(
            deep_seat, policy_net, device, seed=SEED + hand
        )

        if opponent_type == "random":
            opponent = pyspiel.make_uniform_random_bot(
                opponent_seat, SEED + 10_000 + hand
            )
        elif opponent_type == "passive":
            opponent = PassiveBot(opponent_seat)
        else:
            raise ValueError(opponent_type)

        bots = [None, None]
        bots[deep_seat] = deep_bot
        bots[opponent_seat] = opponent

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
        f"{wins / decisive:.1%}" if decisive else "Win rate: n/a"
    )
    print(f"Mean return per hand: {np.mean(returns):.2f}")
    print(f"Total return: {np.sum(returns):.2f}")


def main():
    game, policy_net, device = load_policy()
    compare(game, policy_net, device, "random")
    compare(game, policy_net, device, "passive")


if __name__ == "__main__":
    main()
