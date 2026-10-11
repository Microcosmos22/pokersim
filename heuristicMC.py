
import random
import re
from collections import Counter, defaultdict


# CardSet strings use rank followed by suit, e.g. AsKd.
_RANKS = "23456789TJQKA"
_RANK_VALUE = {rank: i + 2 for i, rank in enumerate(_RANKS)}
_RANK_VALUE["10"] = 10

_SUITS = "cdhs"
_SUIT_VALUE = {suit: i for i, suit in enumerate(_SUITS)}

_CARD_PATTERN = re.compile(
    r"(10|[2-9TJQKA])([cdhs])",
    flags=re.IGNORECASE,
)


def _parse_cards(text):
    """Parse a string such as 'AsKd' or '2c 3h 4s'."""
    cards = []
    for rank_text, suit_text in _CARD_PATTERN.findall(text or ""):
        rank = _RANK_VALUE[rank_text.upper()]
        suit = _SUIT_VALUE[suit_text.lower()]
        cards.append((rank, suit))
    return cards


def _field(text, name):
    """Extract a field like [Pot: 500] from an OpenSpiel state string."""
    match = re.search(
        rf"\[{re.escape(name)}:\s*([^\]]*)\]",
        text,
    )
    return match.group(1).strip() if match else None


def _straight_high(ranks):
    """Return the highest rank of a straight, or 0 if none exists."""
    values = set(ranks)

    # Ace can be low in A-2-3-4-5.
    if 14 in values:
        values.add(1)

    for high in range(14, 4, -1):
        if all(high - offset in values for offset in range(5)):
            return high

    return 0


def _evaluate_7(cards):
    """
    Return a comparable poker hand strength tuple.

    Higher tuples are stronger. Expects seven distinct cards, represented
    as (rank, suit) pairs, with ranks 2 through 14.
    """
    rank_counts = Counter(rank for rank, _ in cards)

    suit_ranks = defaultdict(list)
    for rank, suit in cards:
        suit_ranks[suit].append(rank)

    # 8: Straight flush.
    for ranks in suit_ranks.values():
        if len(ranks) >= 5:
            high = _straight_high(ranks)
            if high:
                return (8, high)

    # 7: Four of a kind.
    quads = sorted(
        (rank for rank, count in rank_counts.items() if count == 4),
        reverse=True,
    )
    if quads:
        kicker = max(
            rank for rank in rank_counts if rank != quads[0]
        )
        return (7, quads[0], kicker)

    # 6: Full house. A second set of three can serve as the pair.
    trips = sorted(
        (rank for rank, count in rank_counts.items() if count >= 3),
        reverse=True,
    )
    if trips:
        pairs = sorted(
            (
                rank
                for rank, count in rank_counts.items()
                if rank != trips[0] and count >= 2
            ),
            reverse=True,
        )
        if pairs:
            return (6, trips[0], pairs[0])

    # 5: Flush.
    flushes = [
        tuple(sorted(ranks, reverse=True)[:5])
        for ranks in suit_ranks.values()
        if len(ranks) >= 5
    ]
    if flushes:
        return (5, *max(flushes))

    # 4: Straight.
    straight = _straight_high(rank_counts.keys())
    if straight:
        return (4, straight)

    # 3: Three of a kind.
    if trips:
        kickers = sorted(
            (
                rank
                for rank in rank_counts
                if rank != trips[0]
            ),
            reverse=True,
        )
        return (3, trips[0], *kickers[:2])

    # 2: Two pair.
    pairs = sorted(
        (rank for rank, count in rank_counts.items() if count >= 2),
        reverse=True,
    )
    if len(pairs) >= 2:
        kicker = max(
            rank
            for rank in rank_counts
            if rank not in pairs[:2]
        )
        return (2, pairs[0], pairs[1], kicker)

    # 1: One pair.
    if pairs:
        kickers = sorted(
            (rank for rank in rank_counts if rank != pairs[0]),
            reverse=True,
        )
        return (1, pairs[0], *kickers[:3])

    # 0: High card.
    high_cards = sorted(rank_counts, reverse=True)[:5]
    return (0, *high_cards)


def _estimate_equity(hole_cards, board_cards, num_opponents, rng, samples):
    """
    Estimate the fraction of the pot the hero would win at showdown
    against uniformly random opponent hands.

    Future community cards are sampled when the board is incomplete.
    """
    if num_opponents <= 0:
        return 1.0

    used = set(hole_cards + board_cards)
    deck = [
        (rank, suit)
        for rank in range(2, 15)
        for suit in range(4)
        if (rank, suit) not in used
    ]

    future_board_count = 5 - len(board_cards)
    cards_needed = 2 * num_opponents + future_board_count

    if cards_needed > len(deck):
        return 0.5

    total_share = 0.0

    for _ in range(samples):
        drawn = rng.sample(deck, cards_needed)

        opponent_hands = [
            [drawn[2 * i], drawn[2 * i + 1]]
            for i in range(num_opponents)
        ]

        final_board = board_cards + drawn[2 * num_opponents:]

        hero_rank = _evaluate_7(hole_cards + final_board)

        opponent_ranks = [
            _evaluate_7(hand + final_board)
            for hand in opponent_hands
        ]

        best_rank = max([hero_rank] + opponent_ranks)

        if hero_rank == best_rank:
            tied_winners = (
                1 + sum(rank == best_rank for rank in opponent_ranks)
            )
            total_share += 1.0 / tied_winners

    return total_share / max(1, samples)


def _action_kind(description):
    """Classify the action strings used by OpenSpiel universal_poker."""
    text = description.lower().replace("-", "").replace(" ", "")

    if "fold" in text:
        return "fold"
    if "allin" in text:
        return "allin"
    if "call" in text or "check" in text:
        return "call"
    if "bet" in text or "raise" in text or "halfpot" in text:
        return "raise"

    return "other"


def _history_profile(state, hero_id):
    """
    Replay the public action history to identify folded players and
    count aggressive actions by opponents who remain in the hand.
    """
    game = state.get_game()
    num_players = game.num_players()

    folded = set()
    aggression_by_player = [0] * num_players

    try:
        replay = game.new_initial_state()

        for action in state.history():
            action = int(action)

            if replay.is_chance_node():
                replay.apply_action(action)
                continue

            actor = replay.current_player()
            description = replay.action_to_string(actor, action)
            kind = _action_kind(description)

            if kind == "fold":
                folded.add(actor)
            elif kind in ("raise", "allin") and actor != hero_id:
                aggression_by_player[actor] += 1

            replay.apply_action(action)

        opponents = [
            p for p in range(num_players)
            if p != hero_id and p not in folded
        ]

        aggression = sum(aggression_by_player[p] for p in opponents)
        return opponents, aggression

    except Exception:
        # Fallback if this OpenSpiel build cannot replay the history.
        opponents = [p for p in range(num_players) if p != hero_id]
        return opponents, 0


class HeuristicPokerBot:
    """
    Equity-aware heuristic opponent for universal_poker with two-card
    Texas Hold'em.

    Decisions are stochastic. The bot estimates showdown equity against
    random holdings, adds a margin for pot odds and opponent aggression,
    and uses occasional value raises and semi-bluffs.
    """

    def __init__(self, player_id, seed=0, equity_samples=48):
        self.player_id = int(player_id)
        self.rng = random.Random(seed)
        self.equity_samples = max(1, int(equity_samples))

    def _choose_raise(
        self,
        bet_actions,
        allin_actions,
        equity,
        round_id,
        pot,
        effective_stack,
    ):
        """Prefer a pot bet; use all-in selectively."""
        spr = effective_stack / max(1.0, pot)

        can_shove_for_value = (
            spr <= 1.2 and equity >= 0.83
        ) or (
            round_id == 3 and spr <= 4.0 and equity >= 0.96
        )

        if allin_actions and can_shove_for_value:
            if self.rng.random() < 0.45:
                return self.rng.choice(allin_actions)

        if bet_actions:
            return self.rng.choice(bet_actions)

        if allin_actions and equity >= 0.90:
            return self.rng.choice(allin_actions)

        return None

    def step(self, state):
        legal_actions = [int(a) for a in state.legal_actions(self.player_id)]

        if not legal_actions:
            raise RuntimeError("HeuristicPokerBot received no legal actions.")

        action_descriptions = {
            action: state.action_to_string(self.player_id, action).lower()
            for action in legal_actions
        }

        kinds = {
            action: _action_kind(description)
            for action, description in action_descriptions.items()
        }

        fold_actions = [a for a in legal_actions if kinds[a] == "fold"]
        call_actions = [a for a in legal_actions if kinds[a] == "call"]
        bet_actions = [a for a in legal_actions if kinds[a] == "raise"]
        allin_actions = [a for a in legal_actions if kinds[a] == "allin"]

        call_action = call_actions[0] if call_actions else None
        fold_action = fold_actions[0] if fold_actions else None

        # OpenSpiel's information-state string contains only this player's
        # private cards plus the public board, money and betting sequence.
        info = state.information_state_string(self.player_id)

        hole_cards = _parse_cards(_field(info, "Private"))
        board_cards = _parse_cards(_field(info, "Public"))

        round_match = re.search(r"\[Round\s+(\d+)\]", info)
        round_id = int(round_match.group(1)) if round_match else 0

        # Use OpenSpiel's public state structure for the actual pot and
        # current player contributions. Never use opponent private cards.
        try:
            data = state.to_struct().to_dict()
        except Exception:
            data = {}

        if len(hole_cards) != 2:
            raise RuntimeError(
                "Could not parse the bot's private cards. "
                f"Information state was: {info}"
            )

        pot = data.get("pot_size", 0) or 0

        # Fallback for builds that do not expose the state structure.
        if pot <= 0:
            pot_text = _field(info, "Pot")
            pot = float(pot_text) if pot_text else 1.0

        pot = max(1.0, float(pot))

        contributions = data.get("player_contributions", [])

        if not contributions:
            observation = state.observation_string(self.player_id)
            contribution_text = _field(observation, "PlayerContribution")
            if contribution_text:
                contributions = [
                    float(x)
                    for x in re.findall(r"-?\d+", contribution_text)
                ]

        if len(contributions) == state.get_game().num_players():
            current_bet = max(float(x) for x in contributions)
            to_call = max(
                0.0,
                current_bet - float(contributions[self.player_id]),
            )
        else:
            to_call = 0.0

        money_text = _field(info, "Money")
        money = (
            [float(x) for x in re.findall(r"-?\d+", money_text)]
            if money_text else []
        )

        if len(money) == state.get_game().num_players():
            to_call = min(to_call, max(0.0, money[self.player_id]))
        else:
            money = []

        opponents, opponent_aggression = _history_profile(
            state, self.player_id
        )
        num_opponents = len(opponents)

        equity = _estimate_equity(
            hole_cards=hole_cards,
            board_cards=board_cards,
            num_opponents=num_opponents,
            rng=self.rng,
            samples=self.equity_samples,
        )

        if money:
            remaining_stacks = [
                max(0.0, money[p])
                for p in opponents
                if money[p] > 0
            ]
            if money[self.player_id] > 0:
                remaining_stacks.append(money[self.player_id])

            effective_stack = (
                min(remaining_stacks) if remaining_stacks else 0.0
            )
        else:
            effective_stack = 10.0 * pot

        # Facing a bet: compare estimated equity with pot odds, while
        # requiring extra equity against aggressive betting lines.
        if to_call > 0:
            pot_odds = to_call / (pot + to_call)

            range_margin = (
                0.035
                + min(0.12, 0.03 * opponent_aggression)
                + 0.015 * max(0, num_opponents - 1)
            )
            required_equity = min(0.98, pot_odds + range_margin)

            if equity < required_equity:
                # Occasionally defend against very small bets.
                cheap_call = to_call <= 0.05 * pot
                near_threshold = equity >= required_equity - 0.07

                if (
                    call_action is not None
                    and cheap_call
                    and near_threshold
                    and self.rng.random() < 0.20
                ):
                    return call_action

                if fold_action is not None:
                    return fold_action
                return call_action if call_action is not None else legal_actions[0]

            # Raise some very strong hands instead of always calling.
            raise_threshold = max(0.72, required_equity + 0.18)

            if (
                equity >= raise_threshold
                and (bet_actions or allin_actions)
            ):
                raise_probability = min(
                    0.50,
                    0.15 + 1.2 * (equity - raise_threshold),
                )

                if self.rng.random() < raise_probability:
                    action = self._choose_raise(
                        bet_actions,
                        allin_actions,
                        equity,
                        round_id,
                        pot,
                        effective_stack,
                    )
                    if action is not None:
                        return action

            if call_action is not None:
                return call_action

            return legal_actions[0]

        # No bet to call: value-bet strong equity and occasionally bluff
        # with marginal equity.
        threshold = 0.52 if round_id == 0 else 0.62
        threshold += 0.02 * max(0, num_opponents - 1)
        threshold += min(0.06, 0.02 * opponent_aggression)

        if bet_actions or allin_actions:
            should_bet = equity >= threshold
            occasional_bluff = (
                equity >= threshold - 0.12
                and self.rng.random() < 0.08
            )

            if should_bet:
                bet_probability = min(
                    0.78,
                    0.18 + 1.5 * (equity - threshold),
                )
                should_bet = self.rng.random() < bet_probability

            if should_bet or occasional_bluff:
                action = self._choose_raise(
                    bet_actions,
                    allin_actions,
                    equity,
                    round_id,
                    pot,
                    effective_stack,
                )
                if action is not None:
                    return action

        # Check if available.
        if call_action is not None:
            return call_action

        if fold_action is not None:
            return fold_action

        return legal_actions[0]
