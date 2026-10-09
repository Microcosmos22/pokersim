import time
from treys import Card, Evaluator
from rich.console import Console

# FIXED: Correctly instantiate a Console object instance
console = Console()

def get_poker_strength(rank: int) -> float:
    """Transforms rank (1 to 7462) into a 0.0 to 1.0 float."""
    rank = max(1, min(7462, rank))
    return (7462 - rank) / (7462 - 1)

def get_color_and_label(strength: float) -> tuple[str, str]:
    """Returns a terminal color tag and short label based on strength."""
    if strength >= 0.75:
        return "bold green", "MONSTER"
    elif strength >= 0.40:
        return "bold yellow", "MADE HAND"
    else:
        return "bold red", "WEAK/AIR"

def draw_poker_bar(rank: int, width: int = 30):
    """Calculates, colors, and draws a single-line terminal progress bar."""
    strength = get_poker_strength(rank)
    color, label = get_color_and_label(strength)

    # Calculate visual blocks
    filled_length = int(width * strength)
    empty_length = width - filled_length

    # Construct the visual bar asset
    bar = "█" * filled_length + "░" * empty_length
    percentage = strength * 100

    # Prints dynamically on the active terminal line
    console.print(
        f"\rRank: [cyan]{rank:<4}[/cyan] |[{color}]{bar}[/{color}]| "
        f"[bold]{percentage:>5.1f}%[/bold] ({label:<9})",
        end=""
    )



# Initialize the evaluator
evaluator = Evaluator()


while True:

    user_input = input("Enter cards: ")
    hand1 = [Card.new(user_input[i:i+2]) for i in range(0, 4, 2)]

    board1 = [Card.new(user_input[i:i+2]) for i in range(4, len(user_input), 2)]

    score1 = evaluator.evaluate(board1, hand1)


    # --- EXECUTION / SIMULATION ---
    console.print("[bold underline]Live Poker Hand Evaluation:[/bold underline]\n")

    # Display Scenario 1
    draw_poker_bar(score1)
    time.sleep(2.0)  # Pause for 2 seconds so you can see the first hand value

    # Move to a new line for the next hand state, just like dealing a new stage
    console.print()
