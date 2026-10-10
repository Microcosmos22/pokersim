
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


class PassiveBot:
    def __init__(self, player_id):
        self.player_id = player_id

    def step(self, state):
        actions = state.legal_actions()
        named = [(a, state.action_to_string(a).lower()) for a in actions]

        for preferred in ("check", "call"):
            matches = [a for a, name in named if preferred in name]
            if matches:
                return matches[0]

        folds = [a for a, name in named if "fold" in name]
        return folds[0] if folds else actions[0]


class DeepCFRBot:
    def __init__(self, player_id, policy, seed=42):
        self.player_id = player_id
        self.policy = policy
        self.rng = random.Random(seed)

    def step(self, state):
        actions = state.legal_actions()

        # Get the learned action probabilities for this information state.
        probabilities = self.policy.action_probabilities(
            state, self.player_id
        )

        weights = [
            max(0.0, float(probabilities.get(action, 0.0)))
            for action in actions
        ]

        total = sum(weights)

        # Fallback if the policy returns no usable probabilities.
        if total <= 0:
            return self.rng.choice(actions)

        return self.rng.choices(actions, weights=weights, k=1)[0]


def play_one_hand(game, bots, rng):
    state = game.new_initial_state()

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probabilities = zip(*outcomes)
            action = rng.choices(
                actions, weights=probabilities, k=1
            )[0]
        else:
            player = state.current_player()
            action = bots[player].step(state)

        state.apply_action(action)

    return state.returns()


def train_deep_cfr(game, seed=42):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training Deep CFR on {device}...")

    solver = deep_cfr.DeepCFRSolver(
        game,
        policy_network_layers=(64,),
        advantage_network_layers=(64,),
        num_iterations=10,
        num_traversals=20,
        learning_rate=1e-3,
        batch_size_advantage=128,
        batch_size_strategy=128,
        memory_capacity=10000,
        policy_network_train_steps=100,
        advantage_network_train_steps=50,
        reinitialize_advantage_networks=True,
        device=device,
        seed=seed,
        print_nash_convs=False,
    )

    result = solver.solve()
    policy = result[0]  # Extract the average policy from the returned tuple
    print("Deep CFR training complete.")
    print("Policy type:", type(policy))
    return policy


def evaluate_match(
    game, policy, opponent_name, num_hands=500, seed=100
):
    rng = random.Random(seed)

    deep_bots = [
        DeepCFRBot(0, policy, seed + 10),
        DeepCFRBot(1, policy, seed + 11),
    ]

    random_bots = [
        pyspiel.make_uniform_random_bot(0, seed + 20),
        pyspiel.make_uniform_random_bot(1, seed + 21),
    ]

    opponent_returns = []
    deep_returns = []

    for hand in range(num_hands):
        # Alternate positions to reduce positional bias.
        deep_id = hand % 2
        opponent_id = 1 - deep_id

        if opponent_name == "random":
            opponent = random_bots[opponent_id]
        else:
            opponent = PassiveBot(opponent_id)

        bots = [None, None]
        bots[deep_id] = deep_bots[deep_id]
        bots[opponent_id] = opponent

        returns = play_one_hand(game, bots, rng)

        deep_returns.append(returns[deep_id])
        opponent_returns.append(returns[opponent_id])

    deep_returns = np.asarray(deep_returns)
    opponent_returns = np.asarray(opponent_returns)

    print(f"\nDeep CFR vs {opponent_name.title()} Bot")
    print(f"Hands: {num_hands}")
    print(f"Deep CFR mean return: {deep_returns.mean():.2f}")
    print(f"Opponent mean return: {opponent_returns.mean():.2f}")
    print(f"Deep CFR total return: {deep_returns.sum():.0f}")
    print(f"Opponent total return: {opponent_returns.sum():.0f}")
    print(f"Deep CFR winning hands: {(deep_returns > 0).sum()}")
    print(f"Opponent winning hands: {(deep_returns < 0).sum()}")
    print(f"Tied hands: {(deep_returns == 0).sum()}")


def main():
    game = pyspiel.load_game(GAME_PARAMS)

    policy = train_deep_cfr(game)

    evaluate_match(game, policy, "random", num_hands=500, seed=100)
    evaluate_match(game, policy, "passive", num_hands=500, seed=200)


if __name__ == "__main__":
    main()
