"""
12口弹珠机最佳出卡策略模块
Optimal strategy module for the 12-slot pinball machine.

Rules:
- Each play costs T small marbles (minimum bet = T).
- After pressing the button the multiplier is revealed (2x/4x/6x/8x/10x).
- Typical slot distribution per multiplier:
    2x  -> 4 slots lit   (P_win ≈ 4/12)
    4x  -> 3 slots lit   (P_win ≈ 3/12)
    6x  -> 2 slots lit   (P_win ≈ 2/12)
    8x  -> 1 slot lit    (P_win ≈ 1/12)
    10x -> 1 slot lit    (P_win ≈ 1/12)
- Before shooting you may add more marbles up to a total of 99.
- A win returns (multiplier × total_bet) marbles and min(multiplier × total_bet, J) score cards.
"""

import math
from typing import Dict, List, Optional, Tuple

# Number of physical slots on the machine
NUM_SLOTS = 12

# Maximum total marbles per play
MAX_BET = 99

# Default multiplier -> number of lit slots mapping
DEFAULT_MULTIPLIER_SLOTS: Dict[int, int] = {
    2: 4,
    4: 3,
    6: 2,
    8: 1,
    10: 1,
}


class PinballStrategy:
    """
    Strategy advisor for the 12-slot pinball machine.

    It maintains a running estimate of the physical landing probability of the
    marble across all 12 slots and uses that estimate—together with the current
    multiplier and the set of lit slots—to recommend the optimal number of
    marbles to bet each round.

    Parameters
    ----------
    T : int
        Base cost per play in small marbles (minimum bet, 1 ≤ T ≤ 99).
    J : int
        Maximum number of score cards awarded per winning play (J ≥ 1).
    priority : str
        ``'cards'``   – maximise score-card yield per marble spent.
        ``'marbles'`` – maximise marble return (minimise losses / maximise EV).
    multiplier_slots : dict, optional
        Mapping of multiplier value to the number of lit slots.  Defaults to
        ``DEFAULT_MULTIPLIER_SLOTS``.
    """

    def __init__(
        self,
        T: int,
        J: int,
        priority: str = "cards",
        multiplier_slots: Optional[Dict[int, int]] = None,
    ) -> None:
        if T < 1 or T > MAX_BET:
            raise ValueError(f"T must be between 1 and {MAX_BET}, got {T}")
        if J < 1:
            raise ValueError(f"J must be at least 1, got {J}")
        if priority not in ("cards", "marbles"):
            raise ValueError("priority must be 'cards' or 'marbles'")

        self.T = T
        self.J = J
        self.priority = priority
        self.multiplier_slots: Dict[int, int] = (
            multiplier_slots if multiplier_slots is not None else dict(DEFAULT_MULTIPLIER_SLOTS)
        )

        # Landing history: count of times the marble landed in each slot
        self._landing_counts: List[int] = [0] * NUM_SLOTS
        self._total_plays: int = 0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def total_plays(self) -> int:
        """Total number of plays recorded so far."""
        return self._total_plays

    def get_landing_probs(self) -> List[float]:
        """
        Return the estimated probability distribution for where the marble lands.

        Before any data have been collected a uniform distribution is returned.
        """
        if self._total_plays == 0:
            return [1.0 / NUM_SLOTS] * NUM_SLOTS
        return [c / self._total_plays for c in self._landing_counts]

    def record_landing(self, slot: int) -> None:
        """
        Record which slot the marble landed in after a play.

        Parameters
        ----------
        slot : int
            0-indexed slot number (0 to NUM_SLOTS-1).
        """
        if not (0 <= slot < NUM_SLOTS):
            raise ValueError(f"slot must be between 0 and {NUM_SLOTS - 1}, got {slot}")
        self._landing_counts[slot] += 1
        self._total_plays += 1

    # ------------------------------------------------------------------
    # Core strategy calculations
    # ------------------------------------------------------------------

    def win_probability(self, lit_slots: List[int]) -> float:
        """
        Estimate the probability of winning given a set of lit slots.

        Parameters
        ----------
        lit_slots : list of int
            0-indexed indices of the currently lit slots.
        """
        probs = self.get_landing_probs()
        return sum(probs[s] for s in lit_slots if 0 <= s < NUM_SLOTS)

    def optimal_bet(self, multiplier: int, lit_slots: List[int]) -> int:
        """
        Recommend the optimal total number of marbles to bet this round.

        Parameters
        ----------
        multiplier : int
            Reward multiplier shown after pressing the button.
        lit_slots : list of int
            0-indexed indices of the currently lit slots.

        Returns
        -------
        int
            Total marbles to commit (between T and MAX_BET inclusive).
        """
        p_win = self.win_probability(lit_slots)

        if self.priority == "marbles":
            return self._bet_for_marbles(multiplier, p_win)
        return self._bet_for_cards(multiplier)

    def recommend(self, multiplier: int, lit_slots: List[int]) -> dict:
        """
        Return a full recommendation dict for the current play.

        Parameters
        ----------
        multiplier : int
            Reward multiplier shown after pressing the button.
        lit_slots : list of int
            0-indexed indices of the currently lit slots.

        Returns
        -------
        dict with keys:
            multiplier, lit_slots, win_probability, optimal_bet,
            expected_marble_return, expected_score_cards, marble_roi
        """
        p_win = self.win_probability(lit_slots)
        bet = self.optimal_bet(multiplier, lit_slots)
        expected_marbles = multiplier * bet * p_win
        expected_cards = p_win * min(multiplier * bet, self.J)
        roi = expected_marbles / bet if bet > 0 else 0.0

        return {
            "multiplier": multiplier,
            "lit_slots": lit_slots,
            "win_probability": round(p_win, 4),
            "optimal_bet": bet,
            "expected_marble_return": round(expected_marbles, 2),
            "expected_score_cards": round(expected_cards, 4),
            "marble_roi": round(roi, 4),
        }

    # ------------------------------------------------------------------
    # Analytical helpers (static / no instance state needed)
    # ------------------------------------------------------------------

    @staticmethod
    def expected_value_table(
        T: int,
        J: int,
        multiplier_slots: Optional[Dict[int, int]] = None,
    ) -> List[dict]:
        """
        Build an expected-value analysis table for each multiplier,
        assuming a uniform landing distribution (i.e., no historical data).

        Useful for understanding the baseline characteristics of the machine.
        """
        ms = multiplier_slots if multiplier_slots is not None else DEFAULT_MULTIPLIER_SLOTS
        rows = []
        for mult, n_lit in sorted(ms.items()):
            p_win = n_lit / NUM_SLOTS

            # --- Marble priority: bet T (minimum) ---
            ev_marbles_per_marble = mult * p_win

            # --- Card priority: bet ceil(J / mult) clamped to [T, MAX_BET] ---
            n_card = max(T, min(MAX_BET, math.ceil(J / mult)))
            cards_per_marble = p_win * min(mult * n_card, J) / n_card if n_card else 0

            rows.append(
                {
                    "multiplier": mult,
                    "lit_slots": n_lit,
                    "p_win": round(p_win, 4),
                    "marble_ev_ratio": round(ev_marbles_per_marble, 4),
                    "card_optimal_bet": n_card,
                    "cards_per_marble": round(cards_per_marble, 4),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bet_for_marbles(self, multiplier: int, p_win: float) -> int:
        """
        Marble-priority bet calculation.

        Bet the maximum (99) only when the expected return exceeds the input.
        Otherwise bet the minimum (T) to limit losses.
        """
        ev_ratio = multiplier * p_win
        if ev_ratio > 1.0:
            return MAX_BET
        return self.T

    def _bet_for_cards(self, multiplier: int) -> int:
        """
        Card-priority bet calculation.

        Score-cards per marble = p_win × min(mult × N, J) / N.
        When N is large enough that mult × N ≥ J the rate becomes
        p_win × J / N, which *decreases* in N, so we want the smallest N
        that reaches the cap: N* = ceil(J / mult).

        Clamp N* to [T, MAX_BET].  Note: this calculation is independent of
        the win probability because the optimal N is determined solely by the
        cap constraint and the multiplier.
        """
        n_optimal = math.ceil(self.J / multiplier)
        return max(self.T, min(MAX_BET, n_optimal))
