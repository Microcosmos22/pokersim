
import random
import numpy as np
import pyspiel

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


def play_one_hand(game, bots, rng):
    state = game.new_initial_state()

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probabilities = zip(*outcomes)
            action = rng.choices(actions, weights=probabilities, k=1)[0]
        else:
            action = bots[state.current_player()].step(state)

        state.apply_action(action)

    return state.returns()


def main(num_hands=1000, seed=42):
    game = pyspiel.load_game(GAME_PARAMS)
    rng = random.Random(seed)

    bots = [
        PassiveBot(0),
        pyspiel.make_uniform_random_bot(1, seed + 1),
    ]

    returns = np.array([
        play_one_hand(game, bots, rng)
        for _ in range(num_hands)
    ])

    print(f"Hands: {num_hands}")
    print(f"Player 0 mean return: {returns[:, 0].mean():.2f}")
    print(f"Player 1 mean return: {returns[:, 1].mean():.2f}")
    print(f"Player 0 total return: {returns[:, 0].sum():.0f}")
    print(f"Player 1 total return: {returns[:, 1].sum():.0f}")
    print(f"Player 0 wins: {(returns[:, 0] > 0).sum()}")
    print(f"Player 1 wins: {(returns[:, 1] > 0).sum()}")
    print(f"Ties: {(returns[:, 0] == 0).sum()}")


if __name__ == "__main__":
    main()
