"""
Unit tests for the simulation_test module.
"""

import pytest
from simulation_test import (
    normalize_probs,
    roll_multiplier,
    roll_lit_slots,
    roll_landing,
    run_simulation,
    MULTIPLIER_PROBS,
    DEFAULT_HOLE_PROBS,
)
from pinball_strategy import NUM_SLOTS, DEFAULT_MULTIPLIER_SLOTS
import random


class TestNormalizeProbs:
    def test_already_normalized(self):
        probs = [1.0 / 12] * 12
        result = normalize_probs(probs)
        assert abs(sum(result) - 1.0) < 1e-9

    def test_unnormalized(self):
        probs = [2.0] * 12
        result = normalize_probs(probs)
        assert abs(sum(result) - 1.0) < 1e-9
        assert all(abs(p - 1.0 / 12) < 1e-9 for p in result)


class TestRollMultiplier:
    def test_returns_valid_multiplier(self):
        rng = random.Random(42)
        for _ in range(200):
            m = roll_multiplier(rng)
            assert m in MULTIPLIER_PROBS

    def test_distribution_roughly_matches(self):
        """大量采样后，各倍数出现频率应接近设定概率。"""
        rng = random.Random(123)
        counts = {m: 0 for m in MULTIPLIER_PROBS}
        n = 100_000
        for _ in range(n):
            counts[roll_multiplier(rng)] += 1
        for m, expected_p in MULTIPLIER_PROBS.items():
            actual_p = counts[m] / n
            assert abs(actual_p - expected_p) < 0.02, (
                f"倍数{m}x: 期望{expected_p:.3f}, 实际{actual_p:.3f}"
            )


class TestRollLitSlots:
    def test_correct_count(self):
        rng = random.Random(42)
        for mult, n_lit in DEFAULT_MULTIPLIER_SLOTS.items():
            slots = roll_lit_slots(mult, rng)
            assert len(slots) == n_lit

    def test_slots_in_range(self):
        rng = random.Random(42)
        for mult in DEFAULT_MULTIPLIER_SLOTS:
            slots = roll_lit_slots(mult, rng)
            assert all(0 <= s < NUM_SLOTS for s in slots)

    def test_no_duplicates(self):
        rng = random.Random(42)
        for mult in DEFAULT_MULTIPLIER_SLOTS:
            slots = roll_lit_slots(mult, rng)
            assert len(set(slots)) == len(slots)


class TestRollLanding:
    def test_returns_valid_slot(self):
        rng = random.Random(42)
        probs = normalize_probs(DEFAULT_HOLE_PROBS)
        for _ in range(200):
            s = roll_landing(probs, rng)
            assert 0 <= s < NUM_SLOTS

    def test_distribution_roughly_matches(self):
        rng = random.Random(456)
        probs = normalize_probs(DEFAULT_HOLE_PROBS)
        counts = [0] * NUM_SLOTS
        n = 100_000
        for _ in range(n):
            counts[roll_landing(probs, rng)] += 1
        for i in range(NUM_SLOTS):
            actual_p = counts[i] / n
            assert abs(actual_p - probs[i]) < 0.02, (
                f"槽{i+1}: 期望{probs[i]:.3f}, 实际{actual_p:.3f}"
            )


class TestRunSimulation:
    def test_basic_run_completes(self):
        """模拟应能正常完成并返回所有必需字段。"""
        result = run_simulation(
            initial_marbles=1000, consume_target=100,
            T=20, J=10, priority="cards",
            hole_probs=DEFAULT_HOLE_PROBS, seed=42, verbose=False,
        )
        assert result["total_rounds"] > 0
        assert result["total_marbles_spent"] > 0
        assert len(result["estimated_probs"]) == NUM_SLOTS
        assert len(result["real_probs"]) == NUM_SLOTS
        assert result["mae"] >= 0

    def test_deterministic_with_seed(self):
        """相同种子应产生完全相同的结果。"""
        r1 = run_simulation(
            initial_marbles=5000, consume_target=200,
            T=20, J=10, priority="cards",
            hole_probs=DEFAULT_HOLE_PROBS, seed=999, verbose=False,
        )
        r2 = run_simulation(
            initial_marbles=5000, consume_target=200,
            T=20, J=10, priority="cards",
            hole_probs=DEFAULT_HOLE_PROBS, seed=999, verbose=False,
        )
        assert r1["total_rounds"] == r2["total_rounds"]
        assert r1["wins"] == r2["wins"]
        assert r1["total_marbles_spent"] == r2["total_marbles_spent"]
        assert r1["estimated_probs"] == r2["estimated_probs"]

    def test_more_consume_more_accuracy(self):
        """更多的消耗（采样次数更多）应导致更准确的拟合。"""
        r_small = run_simulation(
            initial_marbles=10000, consume_target=200,
            T=20, J=10, priority="cards",
            hole_probs=DEFAULT_HOLE_PROBS, seed=42, verbose=False,
        )
        r_large = run_simulation(
            initial_marbles=50000, consume_target=5000,
            T=20, J=10, priority="cards",
            hole_probs=DEFAULT_HOLE_PROBS, seed=42, verbose=False,
        )
        # 更大消耗应有更低的MAE（或至少不会大幅恶化）
        assert r_large["mae"] < r_small["mae"] + 0.01

    def test_probs_sum_to_one(self):
        result = run_simulation(
            initial_marbles=5000, consume_target=300,
            T=20, J=10, priority="cards",
            hole_probs=DEFAULT_HOLE_PROBS, seed=42, verbose=False,
        )
        assert abs(sum(result["estimated_probs"]) - 1.0) < 1e-9

    def test_marble_priority(self):
        """marbles优先级也应能正常运行。"""
        result = run_simulation(
            initial_marbles=5000, consume_target=300,
            T=20, J=10, priority="marbles",
            hole_probs=DEFAULT_HOLE_PROBS, seed=42, verbose=False,
        )
        assert result["total_rounds"] > 0

    def test_net_consumed_reaches_target(self):
        """净消耗应达到或超过目标Y。"""
        result = run_simulation(
            initial_marbles=10000, consume_target=500,
            T=20, J=10, priority="cards",
            hole_probs=DEFAULT_HOLE_PROBS, seed=42, verbose=False,
        )
        assert result["net_consumed"] >= 500
        assert result["marbles_remaining"] <= 10000 - 500

    def test_uniform_probs_converge(self):
        """均匀概率下，估计应接近1/12。"""
        uniform = [1.0] * 12
        result = run_simulation(
            initial_marbles=50000, consume_target=3000,
            T=20, J=10, priority="cards",
            hole_probs=uniform, seed=42, verbose=False,
        )
        for p in result["estimated_probs"]:
            assert abs(p - 1.0 / 12) < 0.05
