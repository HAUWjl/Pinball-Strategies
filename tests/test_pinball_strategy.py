"""
Unit tests for the PinballStrategy module.
"""

import math
import pytest

from pinball_strategy import (
    NUM_SLOTS,
    MIN_BET,
    MAX_BET,
    DEFAULT_MULTIPLIER_SLOTS,
    PinballStrategy,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def strategy_cards():
    """T=20 (score-card divisor), J=100, priority='cards'"""
    return PinballStrategy(T=20, J=100, priority="cards")


@pytest.fixture()
def strategy_marbles():
    """T=20 (score-card divisor), J=100, priority='marbles'"""
    return PinballStrategy(T=20, J=100, priority="marbles")


# ── Construction ───────────────────────────────────────────────────────────────

class TestConstruction:
    def test_valid_construction(self):
        s = PinballStrategy(T=20, J=10, priority="cards")
        assert s.T == 20
        assert s.J == 10
        assert s.priority == "cards"

    def test_default_priority_is_cards(self):
        s = PinballStrategy(T=20, J=1)
        assert s.priority == "cards"

    def test_invalid_T_zero(self):
        with pytest.raises(ValueError):
            PinballStrategy(T=0, J=10)

    def test_invalid_T_negative(self):
        with pytest.raises(ValueError):
            PinballStrategy(T=-1, J=10)

    def test_t_above_99_is_valid(self):
        # T is the score-card divisor, not a bet, so values > 99 are allowed
        s = PinballStrategy(T=100, J=10)
        assert s.T == 100

    def test_invalid_J(self):
        with pytest.raises(ValueError):
            PinballStrategy(T=20, J=0)

    def test_invalid_priority(self):
        with pytest.raises(ValueError):
            PinballStrategy(T=20, J=10, priority="unknown")


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
        # With prior_weight=0 (pure frequency), 10 landings in slot 0 → P(0)=1
        s = PinballStrategy(T=20, J=100, prior_weight=0)
        for _ in range(10):
            s.record_landing(0)  # always slot 0
        probs = s.get_landing_probs()
        assert probs[0] == pytest.approx(1.0)
        for i in range(1, NUM_SLOTS):
            assert probs[i] == pytest.approx(0.0)

    def test_bayesian_smoothing_dampens_single_observation(self):
        """With default prior_weight=24, a single observation barely moves probs."""
        s = PinballStrategy(T=20, J=100)  # default prior_weight=24
        s.record_landing(0)
        probs = s.get_landing_probs()
        # P(slot 0) = (2 + 1) / (24 + 1) = 3/25 = 0.12
        assert probs[0] == pytest.approx(3.0 / 25.0)
        # Other slots: (2 + 0) / 25 = 0.08
        for i in range(1, NUM_SLOTS):
            assert probs[i] == pytest.approx(2.0 / 25.0)

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
        # Use prior_weight=0 to test pure-frequency skew
        s = PinballStrategy(T=20, J=10, prior_weight=0)
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
        s = PinballStrategy(T=20, J=100, priority="marbles", prior_weight=0)
        # Force p_win = 1.0 (marble always falls in slot 0)
        for _ in range(100):
            s.record_landing(0)
        bet = s.optimal_bet(multiplier=2, lit_slots=[0])
        assert bet == MAX_BET

    def test_bet_minimum_when_ev_negative(self, strategy_marbles):
        # Uniform distribution: 2x → EV = (4/12)*2 = 2/3 < 1 → bet MIN_BET
        bet = strategy_marbles.optimal_bet(multiplier=2, lit_slots=[0, 1, 2, 3])
        assert bet == MIN_BET

    def test_4x_multiplier_ev_equals_one(self, strategy_marbles):
        # 4x, 3 lit slots: EV = (3/12)*4 = 1.0 → bet MIN_BET (not strictly > 1)
        bet = strategy_marbles.optimal_bet(multiplier=4, lit_slots=[0, 1, 2])
        assert bet == MIN_BET

    def test_bet_clamped_to_MIN_BET(self, strategy_marbles):
        bet = strategy_marbles.optimal_bet(multiplier=2, lit_slots=[0, 1, 2, 3])
        assert bet >= MIN_BET

    def test_bet_clamped_to_max(self):
        s = PinballStrategy(T=20, J=100, priority="marbles", prior_weight=0)
        for _ in range(100):
            s.record_landing(0)  # p_win = 1 for slot 0
        bet = s.optimal_bet(multiplier=2, lit_slots=[0])
        assert bet <= MAX_BET


# ── Optimal bet (cards priority) ──────────────────────────────────────────────

class TestOptimalBetCards:
    def test_optimal_bet_formula(self):
        """
        For card priority, optimal_bet = max(MIN_BET, min(99, ceil(T*J/mult))).
        """
        T, J = 20, 100
        s = PinballStrategy(T=T, J=J, priority="cards")
        for mult in DEFAULT_MULTIPLIER_SLOTS:
            expected = max(MIN_BET, min(MAX_BET, math.ceil(T * J / mult)))
            got = s.optimal_bet(mult, list(range(DEFAULT_MULTIPLIER_SLOTS[mult])))
            assert got == expected, f"mult={mult}: expected {expected}, got {got}"

    def test_card_bet_at_least_MIN_BET(self):
        # ceil(T*J/mult) < MIN_BET → clamp to MIN_BET
        s = PinballStrategy(T=1, J=1, priority="cards")
        # ceil(1*1/10)=1 < MIN_BET=5 → should return MIN_BET
        bet = s.optimal_bet(10, [0])
        assert bet == MIN_BET

    def test_card_bet_at_most_max(self):
        s = PinballStrategy(T=20, J=9999, priority="cards")
        # ceil(20*9999/2)=99990 > 99 → clamp to 99
        bet = s.optimal_bet(2, [0, 1, 2, 3])
        assert bet == MAX_BET

    def test_card_bet_typical_machine(self):
        # T=20, J=10: optimal N per multiplier
        s = PinballStrategy(T=20, J=10, priority="cards")
        # 2x: ceil(20*10/2)=100 → clamped to 99
        assert s.optimal_bet(2, [0, 1, 2, 3]) == 99
        # 4x: ceil(20*10/4)=50
        assert s.optimal_bet(4, [0, 1, 2]) == 50
        # 10x: ceil(20*10/10)=20
        assert s.optimal_bet(10, [0]) == 20


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
        assert MIN_BET <= rec["optimal_bet"] <= MAX_BET

    def test_recommend_score_cards_uses_T_divisor(self):
        """score cards = floor(bet * mult / T), capped at J"""
        T, J = 20, 10
        s = PinballStrategy(T=T, J=J, priority="marbles", prior_weight=0)
        # Force p_win = 1 so expected_cards = actual cards per win
        for _ in range(100):
            s.record_landing(0)
        # With marble priority and p_win=1 (EV>1), bet=MAX_BET=99
        rec = s.recommend(multiplier=4, lit_slots=[0])
        expected_cards = min(99 * 4 // T, J)  # min(floor(396/20), 10) = min(19, 10) = 10
        assert rec["expected_score_cards"] == pytest.approx(expected_cards, rel=1e-3)


# ── Expected-value table ──────────────────────────────────────────────────────

class TestExpectedValueTable:
    def test_table_has_all_multipliers(self):
        rows = PinballStrategy.expected_value_table(T=20, J=100)
        mults = {r["multiplier"] for r in rows}
        assert mults == set(DEFAULT_MULTIPLIER_SLOTS.keys())

    def test_4x_and_6x_break_even(self):
        rows = PinballStrategy.expected_value_table(T=20, J=100)
        for r in rows:
            if r["multiplier"] in (4, 6):
                assert r["marble_ev_ratio"] == pytest.approx(1.0)

    def test_2x_and_8x_negative_ev(self):
        rows = PinballStrategy.expected_value_table(T=20, J=100)
        for r in rows:
            if r["multiplier"] in (2, 8):
                assert r["marble_ev_ratio"] < 1.0

    def test_card_optimal_bet_formula(self):
        T, J = 20, 100
        rows = PinballStrategy.expected_value_table(T=T, J=J)
        for r in rows:
            expected = max(MIN_BET, min(MAX_BET, math.ceil(T * J / r["multiplier"])))
            assert r["card_optimal_bet"] == expected


# ── Custom multiplier_slots ───────────────────────────────────────────────────

class TestCustomMultiplierSlots:
    def test_custom_slots_accepted(self):
        custom = {3: 6, 5: 3, 9: 1}
        s = PinballStrategy(T=20, J=10, multiplier_slots=custom)
        # Should not raise; use the custom mapping
        rec = s.recommend(3, list(range(6)))
        assert rec["multiplier"] == 3

    def test_ev_table_respects_custom_slots(self):
        custom = {3: 6}
        rows = PinballStrategy.expected_value_table(T=20, J=6, multiplier_slots=custom)
        assert len(rows) == 1
        assert rows[0]["multiplier"] == 3
        assert rows[0]["lit_slots"] == 6


# ── V2 Adaptive Strategy Tests ─────────────────────────────────────────────────

class TestAdaptiveV2:
    """Tests for the V2 adaptive betting (negative EV → n_floor, positive EV → ramp)."""

    def test_confidence_threshold_zero_uses_original(self):
        """ct=0 should use the original (non-adaptive) strategy."""
        s = PinballStrategy(T=20, J=10, priority="cards", confidence_threshold=0)
        bet = s.optimal_bet(2, [0, 1, 2, 3])
        # Original: ceil(20*10/2) = 100 → clamped to 99
        assert bet == MAX_BET

    def test_negative_ev_always_bets_floor(self):
        """With negative EV, V2 should always bet n_floor regardless of confidence."""
        s = PinballStrategy(T=20, J=10, priority="cards", confidence_threshold=5)
        # Train with 1000 obs on slot 0, then test lit slots that EXCLUDE slot 0
        for _ in range(1000):
            s.record_landing(0)
        # p_win for [4,5,6,7] ≈ 4*0.002 ≈ 0.008, ev = 2*0.008 << 1
        bet = s.optimal_bet(2, [4, 5, 6, 7])
        assert bet == 10  # n_floor = ceil(20/2)

    def test_positive_ev_ramps_with_confidence(self):
        """With positive EV and high confidence, V2 should bet near MAX."""
        s = PinballStrategy(T=20, J=10, priority="cards", confidence_threshold=5)
        for _ in range(1000):
            s.record_landing(0)
        # 8x with lit=[0,1]: p ≈ 0.98, ev = 7.84 >> 1.0, conf ≈ 1.0
        bet = s.optimal_bet(8, [0, 1])
        assert bet == MAX_BET

    def test_early_rounds_bet_floor_for_negative_ev(self):
        """At 0 observations, uniform prior gives ev ≤ 1 → n_floor."""
        s = PinballStrategy(T=20, J=10, priority="cards", confidence_threshold=50)
        assert s.optimal_bet(2, [0, 1, 2, 3]) == 10  # ceil(20/2)
        assert s.optimal_bet(4, [0, 1, 2]) == 5       # ceil(20/4)

    def test_n_floor_values_by_multiplier(self):
        """Verify n_floor = max(MIN_BET, ceil(T/mult)) for each multiplier."""
        s = PinballStrategy(T=20, J=10, priority="cards", confidence_threshold=100)
        assert s.optimal_bet(2, [0, 1, 2, 3]) == 10   # ceil(20/2) = 10
        assert s.optimal_bet(4, [0, 1, 2]) == 5        # ceil(20/4) = 5
        assert s.optimal_bet(6, [0, 1]) == 5            # ceil(20/6) = 4 → MIN 5
        assert s.optimal_bet(8, [0, 1]) == 5            # ceil(20/8) = 3 → MIN 5
        assert s.optimal_bet(10, [0]) == 5              # ceil(20/10) = 2 → MIN 5

    def test_j5_negative_ev_same_as_j10(self):
        """n_floor is independent of J — J=5 and J=10 give the same negative-EV bets."""
        s5 = PinballStrategy(T=20, J=5, priority="cards", confidence_threshold=50)
        s10 = PinballStrategy(T=20, J=10, priority="cards", confidence_threshold=50)
        for mult, lit in [(2, [0,1,2,3]), (4, [0,1,2]), (6, [0,1]), (10, [0])]:
            assert s5.optimal_bet(mult, lit) == s10.optimal_bet(mult, lit)

    def test_j5_original_strategy_uses_lower_bets(self):
        """Original (ct=0) with J=5 bets less than J=10 due to lower card cap."""
        s5 = PinballStrategy(T=20, J=5, priority="cards", confidence_threshold=0)
        s10 = PinballStrategy(T=20, J=10, priority="cards", confidence_threshold=0)
        # J=5, 2x: ceil(100/2) = 50; J=10, 2x: ceil(200/2) = 100 → 99
        assert s5.optimal_bet(2, [0, 1, 2, 3]) == 50
        assert s10.optimal_bet(2, [0, 1, 2, 3]) == 99
        # J=5, 4x: ceil(100/4) = 25; J=10, 4x: ceil(200/4) = 50
        assert s5.optimal_bet(4, [0, 1, 2]) == 25
        assert s10.optimal_bet(4, [0, 1, 2]) == 50

    def test_j5_positive_ev_still_ramps_to_max(self):
        """With J=5 under positive EV, V2 still ramps to MAX_BET for marble generation."""
        s = PinballStrategy(T=20, J=5, priority="cards", confidence_threshold=5)
        for _ in range(1000):
            s.record_landing(0)
        # 8x lit=[0,1] → strong positive EV, high confidence → MAX_BET
        bet = s.optimal_bet(8, [0, 1])
        assert bet == MAX_BET
