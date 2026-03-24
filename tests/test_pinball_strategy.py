"""
Unit tests for the PinballStrategy module.
"""

import math
import pytest

from pinball_strategy import (
    NUM_SLOTS,
    MAX_BET,
    DEFAULT_MULTIPLIER_SLOTS,
    PinballStrategy,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def strategy_cards():
    """T=5, J=20, priority='cards'"""
    return PinballStrategy(T=5, J=20, priority="cards")


@pytest.fixture()
def strategy_marbles():
    """T=5, J=20, priority='marbles'"""
    return PinballStrategy(T=5, J=20, priority="marbles")


# ── Construction ───────────────────────────────────────────────────────────────

class TestConstruction:
    def test_valid_construction(self):
        s = PinballStrategy(T=3, J=10, priority="cards")
        assert s.T == 3
        assert s.J == 10
        assert s.priority == "cards"

    def test_default_priority_is_cards(self):
        s = PinballStrategy(T=1, J=1)
        assert s.priority == "cards"

    def test_invalid_T_zero(self):
        with pytest.raises(ValueError):
            PinballStrategy(T=0, J=10)

    def test_invalid_T_too_large(self):
        with pytest.raises(ValueError):
            PinballStrategy(T=100, J=10)

    def test_invalid_J(self):
        with pytest.raises(ValueError):
            PinballStrategy(T=5, J=0)

    def test_invalid_priority(self):
        with pytest.raises(ValueError):
            PinballStrategy(T=5, J=10, priority="unknown")


# ── Landing probability ────────────────────────────────────────────────────────

class TestLandingProbability:
    def test_uniform_before_any_data(self, strategy_cards):
        probs = strategy_cards.get_landing_probs()
        assert len(probs) == NUM_SLOTS
        for p in probs:
            assert abs(p - 1.0 / NUM_SLOTS) < 1e-9

    def test_probs_sum_to_one(self, strategy_cards):
        for slot in [0, 3, 7, 11]:
            strategy_cards.record_landing(slot)
        probs = strategy_cards.get_landing_probs()
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_single_landing_updates_distribution(self):
        s = PinballStrategy(T=5, J=20)
        for _ in range(10):
            s.record_landing(0)  # always slot 0
        probs = s.get_landing_probs()
        assert probs[0] == pytest.approx(1.0)
        for i in range(1, NUM_SLOTS):
            assert probs[i] == pytest.approx(0.0)

    def test_invalid_slot_raises(self, strategy_cards):
        with pytest.raises(ValueError):
            strategy_cards.record_landing(-1)
        with pytest.raises(ValueError):
            strategy_cards.record_landing(NUM_SLOTS)

    def test_total_plays_increments(self, strategy_cards):
        assert strategy_cards.total_plays == 0
        strategy_cards.record_landing(0)
        assert strategy_cards.total_plays == 1
        strategy_cards.record_landing(5)
        assert strategy_cards.total_plays == 2


# ── Win probability ────────────────────────────────────────────────────────────

class TestWinProbability:
    def test_uniform_win_prob_matches_lit_slots_fraction(self, strategy_cards):
        # No history → uniform distribution
        for mult, n_lit in DEFAULT_MULTIPLIER_SLOTS.items():
            lit = list(range(n_lit))
            expected = n_lit / NUM_SLOTS
            assert strategy_cards.win_probability(lit) == pytest.approx(expected)

    def test_skewed_win_prob(self):
        s = PinballStrategy(T=1, J=10)
        # Record landings only in slot 0
        for _ in range(100):
            s.record_landing(0)
        # If only slot 0 is lit, p_win ≈ 1.0
        assert s.win_probability([0]) == pytest.approx(1.0)
        # If no slot 0, p_win ≈ 0.0
        assert s.win_probability([1, 2, 3]) == pytest.approx(0.0)

    def test_empty_lit_slots_zero_probability(self, strategy_cards):
        assert strategy_cards.win_probability([]) == pytest.approx(0.0)


# ── Optimal bet (marbles priority) ────────────────────────────────────────────

class TestOptimalBetMarbles:
    def test_bet_max_when_ev_positive(self):
        """A very high p_win forces EV > 1 for any non-trivial multiplier."""
        s = PinballStrategy(T=5, J=20, priority="marbles")
        # Force p_win = 1.0 (marble always falls in slot 0)
        for _ in range(100):
            s.record_landing(0)
        bet = s.optimal_bet(multiplier=2, lit_slots=[0])
        assert bet == MAX_BET

    def test_bet_minimum_when_ev_negative(self, strategy_marbles):
        # Uniform distribution: 2x → EV = (4/12)*2 = 2/3 < 1 → bet T
        bet = strategy_marbles.optimal_bet(multiplier=2, lit_slots=[0, 1, 2, 3])
        assert bet == strategy_marbles.T

    def test_4x_multiplier_ev_equals_one(self, strategy_marbles):
        # 4x, 3 lit slots: EV = (3/12)*4 = 1.0 → bet T (not strictly > 1)
        bet = strategy_marbles.optimal_bet(multiplier=4, lit_slots=[0, 1, 2])
        assert bet == strategy_marbles.T

    def test_bet_clamped_to_T(self, strategy_marbles):
        bet = strategy_marbles.optimal_bet(multiplier=2, lit_slots=[0, 1, 2, 3])
        assert bet >= strategy_marbles.T

    def test_bet_clamped_to_max(self):
        s = PinballStrategy(T=1, J=5, priority="marbles")
        for _ in range(100):
            s.record_landing(0)  # p_win = 1 for slot 0
        bet = s.optimal_bet(multiplier=2, lit_slots=[0])
        assert bet <= MAX_BET


# ── Optimal bet (cards priority) ──────────────────────────────────────────────

class TestOptimalBetCards:
    def test_optimal_bet_formula(self):
        """
        For card priority, optimal_bet = max(T, min(99, ceil(J / mult))).
        """
        T, J = 5, 20
        s = PinballStrategy(T=T, J=J, priority="cards")
        for mult in DEFAULT_MULTIPLIER_SLOTS:
            expected = max(T, min(MAX_BET, math.ceil(J / mult)))
            got = s.optimal_bet(mult, list(range(DEFAULT_MULTIPLIER_SLOTS[mult])))
            assert got == expected, f"mult={mult}: expected {expected}, got {got}"

    def test_card_bet_at_least_T(self):
        s = PinballStrategy(T=10, J=1, priority="cards")
        # ceil(1/10) = 1 < T=10, so clamp to T
        bet = s.optimal_bet(10, [0])
        assert bet == 10

    def test_card_bet_at_most_max(self):
        s = PinballStrategy(T=1, J=9999, priority="cards")
        # ceil(9999/2) = 5000 > 99 → clamp to 99
        bet = s.optimal_bet(2, [0, 1, 2, 3])
        assert bet == MAX_BET

    def test_j_divisible_by_mult(self):
        s = PinballStrategy(T=1, J=20, priority="cards")
        # 2x: ceil(20/2)=10
        assert s.optimal_bet(2, [0, 1, 2, 3]) == 10
        # 4x: ceil(20/4)=5
        assert s.optimal_bet(4, [0, 1, 2]) == 5
        # 10x: ceil(20/10)=2
        assert s.optimal_bet(10, [0]) == 2


# ── Recommend ─────────────────────────────────────────────────────────────────

class TestRecommend:
    def test_recommend_keys(self, strategy_cards):
        rec = strategy_cards.recommend(2, [0, 1, 2, 3])
        required_keys = {
            "multiplier", "lit_slots", "win_probability",
            "optimal_bet", "expected_marble_return",
            "expected_score_cards", "marble_roi",
        }
        assert required_keys.issubset(rec.keys())

    def test_recommend_marble_return_consistency(self, strategy_cards):
        """expected_marble_return = multiplier * bet * p_win"""
        rec = strategy_cards.recommend(4, [0, 1, 2])
        expected = rec["multiplier"] * rec["optimal_bet"] * rec["win_probability"]
        assert rec["expected_marble_return"] == pytest.approx(expected, rel=1e-3)

    def test_recommend_score_cards_capped(self, strategy_cards):
        """expected_score_cards never exceeds J * p_win"""
        rec = strategy_cards.recommend(10, [0])
        # max cards = J, so expected ≤ J
        assert rec["expected_score_cards"] <= strategy_cards.J + 1e-9

    def test_recommend_win_probability_range(self, strategy_cards):
        rec = strategy_cards.recommend(6, [0, 1])
        assert 0.0 <= rec["win_probability"] <= 1.0

    def test_recommend_optimal_bet_in_range(self, strategy_cards):
        rec = strategy_cards.recommend(8, [0])
        assert strategy_cards.T <= rec["optimal_bet"] <= MAX_BET


# ── Expected-value table ──────────────────────────────────────────────────────

class TestExpectedValueTable:
    def test_table_has_all_multipliers(self):
        rows = PinballStrategy.expected_value_table(T=5, J=20)
        mults = {r["multiplier"] for r in rows}
        assert mults == set(DEFAULT_MULTIPLIER_SLOTS.keys())

    def test_4x_and_6x_break_even(self):
        rows = PinballStrategy.expected_value_table(T=5, J=20)
        for r in rows:
            if r["multiplier"] in (4, 6):
                assert r["marble_ev_ratio"] == pytest.approx(1.0)

    def test_2x_and_8x_negative_ev(self):
        rows = PinballStrategy.expected_value_table(T=5, J=20)
        for r in rows:
            if r["multiplier"] in (2, 8):
                assert r["marble_ev_ratio"] < 1.0

    def test_card_optimal_bet_formula(self):
        T, J = 5, 20
        rows = PinballStrategy.expected_value_table(T=T, J=J)
        for r in rows:
            expected = max(T, min(MAX_BET, math.ceil(J / r["multiplier"])))
            assert r["card_optimal_bet"] == expected


# ── Custom multiplier_slots ───────────────────────────────────────────────────

class TestCustomMultiplierSlots:
    def test_custom_slots_accepted(self):
        custom = {3: 6, 5: 3, 9: 1}
        s = PinballStrategy(T=2, J=10, multiplier_slots=custom)
        # Should not raise; use the custom mapping
        rec = s.recommend(3, list(range(6)))
        assert rec["multiplier"] == 3

    def test_ev_table_respects_custom_slots(self):
        custom = {3: 6}
        rows = PinballStrategy.expected_value_table(T=1, J=6, multiplier_slots=custom)
        assert len(rows) == 1
        assert rows[0]["multiplier"] == 3
        assert rows[0]["lit_slots"] == 6
