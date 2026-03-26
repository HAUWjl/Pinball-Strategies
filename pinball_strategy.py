"""
12口弹珠机最佳出卡策略模块
Optimal strategy module for the 12-slot pinball machine.

Rules:
- The machine requires a minimum of MIN_BET (5) small marbles to start each play.
- After pressing the button the multiplier is revealed (2x/4x/6x/8x/10x).
- Typical slot distribution per multiplier:
    2x  -> 4 slots lit   (P_win ≈ 4/12)
    4x  -> 3 slots lit   (P_win ≈ 3/12)
    6x  -> 2 slots lit   (P_win ≈ 2/12)
    8x  -> 1 slot lit    (P_win ≈ 1/12)
    10x -> 1 slot lit    (P_win ≈ 1/12)
- Before shooting you may add more marbles up to a total of 99.
- A win returns (multiplier × total_bet) marbles and
  min(floor(multiplier × total_bet / T), J) score cards,
  where T is the machine's score-card divisor (typically 20–50).
"""

import math
from typing import Dict, List, Optional, Tuple

# Number of physical slots on the machine
NUM_SLOTS = 12

# Absolute minimum marbles required to start a play
MIN_BET = 5

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
        Score-card divisor: every T returned marbles (bet × multiplier) yield
        one score card.  Typically 20–50 (must be ≥ 1).
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
        prior_weight: float = 24.0,
        confidence_threshold: float = 0.0,
        max_bet: int = MAX_BET,
    ) -> None:
        if T < 1:
            raise ValueError(f"T must be at least 1, got {T}")
        if J < 1:
            raise ValueError(f"J must be at least 1, got {J}")
        if priority not in ("cards", "marbles"):
            raise ValueError("priority must be 'cards' or 'marbles'")
        if prior_weight < 0:
            raise ValueError(f"prior_weight must be >= 0, got {prior_weight}")
        if confidence_threshold < 0:
            raise ValueError(f"confidence_threshold must be >= 0, got {confidence_threshold}")
        if not (MIN_BET <= max_bet <= MAX_BET):
            raise ValueError(f"max_bet must be between {MIN_BET} and {MAX_BET}, got {max_bet}")

        self.T = T
        self.J = J
        self.priority = priority
        self.max_bet = max_bet
        self.multiplier_slots: Dict[int, int] = (
            multiplier_slots if multiplier_slots is not None else dict(DEFAULT_MULTIPLIER_SLOTS)
        )
        self.prior_weight = prior_weight
        self.confidence_threshold = confidence_threshold

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

        Uses Bayesian smoothing (Dirichlet prior) so that a small number of
        observations does not dominate the estimate.  The prior assumes a
        uniform distribution with total weight ``prior_weight`` (split equally
        across all slots).  With ``prior_weight=0`` this reduces to pure
        frequency counting.

        .. math::

            P(\\text{slot}_i)
            = \\frac{\\alpha + \\text{count}_i}
                   {N_{\\alpha} + \\text{total\\_plays}}

        where :math:`\\alpha = \\text{prior\\_weight} / N_{\\text{slots}}`.
        """
        alpha = self.prior_weight / NUM_SLOTS
        total = self.prior_weight + self._total_plays
        if total == 0:
            return [1.0 / NUM_SLOTS] * NUM_SLOTS
        return [(alpha + c) / total for c in self._landing_counts]

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
        unique_slots = set(s for s in lit_slots if 0 <= s < NUM_SLOTS)
        return sum(probs[s] for s in unique_slots)

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
            Total marbles to commit (between MIN_BET and MAX_BET inclusive).
        """
        p_win = self.win_probability(lit_slots)

        if self.confidence_threshold > 0:
            # Adaptive mode: scale bet based on observation confidence
            if self.priority == "marbles":
                return self._bet_for_marbles_adaptive(multiplier, p_win)
            return self._bet_for_cards_adaptive(multiplier, p_win)

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
        expected_cards = p_win * min(multiplier * bet // self.T, self.J)
        expected_marbles_rounded = round(expected_marbles, 2)
        roi = expected_marbles_rounded / bet if bet > 0 else 0.0

        return {
            "multiplier": multiplier,
            "lit_slots": lit_slots,
            "win_probability": round(p_win, 4),
            "optimal_bet": bet,
            "expected_marble_return": expected_marbles_rounded,
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
        max_bet: int = MAX_BET,
    ) -> List[dict]:
        """
        Build an expected-value analysis table for each multiplier,
        assuming a uniform landing distribution (i.e., no historical data).

        Useful for understanding the baseline characteristics of the machine.
        """
        mb = max(MIN_BET, min(MAX_BET, max_bet))
        ms = multiplier_slots if multiplier_slots is not None else DEFAULT_MULTIPLIER_SLOTS
        rows = []
        for mult, n_lit in sorted(ms.items()):
            p_win = n_lit / NUM_SLOTS

            # --- Marble priority: bet MIN_BET (minimum) ---
            ev_marbles_per_marble = mult * p_win

            # --- Card priority: find the bet with best cards-per-marble ---
            n_card = max(MIN_BET, min(mb, math.ceil(T * J / mult)))
            best_eff = 0.0
            for k in range(1, J + 1):
                n = max(MIN_BET, math.ceil(k * T / mult))
                if n > mb:
                    break
                eff = k / n
                if eff > best_eff:
                    best_eff = eff
                    n_card = n
            cards_per_marble = p_win * min(mult * n_card // T, J) / n_card if n_card else 0

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

        Bet the maximum (max_bet) only when the expected return exceeds the input.
        Otherwise bet the minimum (MIN_BET) to limit losses.
        """
        ev_ratio = multiplier * p_win
        if ev_ratio > 1.0:
            return self.max_bet
        return MIN_BET

    def _bet_for_cards(self, multiplier: int) -> int:
        """
        Card-priority bet calculation.

        score_cards(N) = min(floor(N × mult / T), J).
        Due to floor-division rounding, the highest cards-per-marble
        efficiency is not always at the J-card cap.  We check every
        card tier k = 1 … J and pick the bet with the best k/N ratio.
        Complexity is O(J), which is negligible.
        """
        best_n = max(MIN_BET, min(self.max_bet, math.ceil(self.T * self.J / multiplier)))
        best_eff = 0.0
        for k in range(1, self.J + 1):
            n = max(MIN_BET, math.ceil(k * self.T / multiplier))
            if n > self.max_bet:
                break
            eff = k / n
            if eff > best_eff:
                best_eff = eff
                best_n = n
        return best_n

    # ------------------------------------------------------------------
    # Adaptive (confidence-aware) betting
    # ------------------------------------------------------------------

    def _confidence(self) -> float:
        """Return a 0→1 confidence measure based on observations so far."""
        return self._total_plays / (self._total_plays + self.confidence_threshold)

    def _bet_for_cards_adaptive(self, multiplier: int, p_win: float) -> int:
        """
        Confidence-aware card-priority bet.

        Key insight: when mult × p_win < 1 (negative EV, which is most rounds),
        small step-aligned bets (ceil(T/mult)) are equally or MORE card-efficient
        per marble than large bets, because large bets suffer floor-rounding
        waste (e.g., bet 99 at 2x gives floor(198/20)=9 cards, but bet 90
        gives floor(180/20)=9 cards too — same cards, 9 fewer marbles).

        Strategy:
        - Negative EV: always bet the minimum step-aligned amount (1 card on win).
          This also provides cheap exploration data.
        - Positive EV detected: ramp toward MAX_BET based on confidence.
          Higher confidence → we trust the positive-EV signal more → bet bigger.
        """
        # Minimum bet that earns exactly 1 card on a win
        n_floor = max(MIN_BET, math.ceil(self.T / multiplier))

        ev_ratio = multiplier * p_win

        if ev_ratio <= 1.0:
            # Negative EV: small bet is most card-efficient per marble
            return n_floor

        # Positive EV: worth betting big (we gain marbles AND cards).
        # Scale with confidence to avoid overcommitting on noisy estimates.
        conf = self._confidence()
        bet = n_floor + round(conf * (self.max_bet - n_floor))
        return max(MIN_BET, min(self.max_bet, bet))

    def _bet_for_marbles_adaptive(self, multiplier: int, p_win: float) -> int:
        """
        Confidence-aware marble-priority bet.

        When confidence is low, require a stronger EV signal before betting big.
        As confidence grows, the threshold drops to the standard EV > 1.0.
        """
        ev_ratio = multiplier * p_win
        conf = self._confidence()

        # Required EV threshold: 1.5 at zero confidence → 1.0 at full confidence
        threshold = 1.0 + 0.5 * (1 - conf)

        if ev_ratio <= 1.0:
            return MIN_BET
        if ev_ratio >= threshold:
            return self.max_bet

        # Between 1.0 and threshold: scale proportionally
        fraction = (ev_ratio - 1.0) / max(0.01, threshold - 1.0) * conf
        return max(MIN_BET, min(self.max_bet, MIN_BET + round(fraction * (self.max_bet - MIN_BET))))
