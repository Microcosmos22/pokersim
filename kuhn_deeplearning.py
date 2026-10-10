
import torch
import pyspiel

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.pytorch import deep_cfr


def main():
    game = pyspiel.load_game("kuhn_poker")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Game: {game}")
    print(f"Device: {device}")

    solver = deep_cfr.DeepCFRSolver(
        game,
        policy_network_layers=(64,),
        advantage_network_layers=(64,),
        num_iterations=1000,
        num_traversals=20,
        learning_rate=1e-3,
        batch_size_advantage=128,
        batch_size_strategy=128,
        memory_capacity=10000,
        policy_network_train_steps=100,
        advantage_network_train_steps=50,
        reinitialize_advantage_networks=True,
        device=device,
        seed=42,
        print_nash_convs=False,
    )

    print("Starting Deep CFR training...")
    _, advantage_losses, policy_loss = solver.solve()
    print("Training complete.")

    average_policy = policy.tabular_policy_from_callable(
        game, solver.action_probabilities
    )
    spiel_policy = policy.python_policy_to_pyspiel_policy(average_policy)

    nash_conv = pyspiel.nash_conv(game, spiel_policy)
    values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2
    )

    print(f"\nNashConv: {nash_conv:.6f}")
    print(f"Player 0 expected value: {values[0]:.6f}")
    print(f"Player 1 expected value: {values[1]:.6f}")
    print(f"Final policy loss: {policy_loss}")
    print("Advantage buffer sizes:",
          [len(buf) for buf in solver.advantage_buffers])
    print("Strategy buffer size:", len(solver.strategy_buffer))

    print("\nLearned policy:")
    for state, probabilities in zip(
        average_policy.states_per_player,
        average_policy.action_probability_array,
    ):
        for info_state, probs in zip(state, probabilities):
            print(f"{info_state}: {probs}")


if __name__ == "__main__":
    main()
