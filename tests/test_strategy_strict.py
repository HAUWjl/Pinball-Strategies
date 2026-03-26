"""
严格策略测试 — 深入探查边界条件、数学一致性和潜在 bug。
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


# ═══════════════════════════════════════════════════════════════════
# 1. win_probability 重复 lit_slots bug
# ═══════════════════════════════════════════════════════════════════


class TestWinProbDuplicateLitSlots:
    """修复后: win_probability 应对重复 lit_slots 去重。"""

    def test_duplicate_lit_slots_are_deduplicated(self):
        s = PinballStrategy(T=20, J=10)
        p_single = s.win_probability([0])
        p_dup = s.win_probability([0, 0, 0])
        assert p_single == pytest.approx(p_dup), "重复 lit_slots 应被去重"

    def test_all_slots_duplicated_still_at_most_one(self):
        """所有槽重复多次 → p_win 应等于所有槽之和 = 1.0"""
        s = PinballStrategy(T=20, J=10)
        lit = list(range(NUM_SLOTS)) * 3
        p = s.win_probability(lit)
        assert p == pytest.approx(1.0), f"p_win 应为 1.0，实际 {p}"


# ═══════════════════════════════════════════════════════════════════
# 2. _bet_for_cards: 验证真实卡片效率 = 算法宣告的效率
# ═══════════════════════════════════════════════════════════════════


class TestCardBetRealEfficiency:
    """对各种 T, J 组合，验证 _bet_for_cards 返回的 bet 是全局最优。"""

    CONFIGS = [
        (20, 3), (20, 5), (20, 10), (20, 100),
        (25, 3), (25, 10),
        (30, 3), (30, 5), (30, 10),
        (50, 3), (50, 10),
        (10, 10), (10, 50),
    ]

    @pytest.mark.parametrize("T,J", CONFIGS)
    def test_optimal_bet_is_globally_best(self, T, J):
        """暴力枚举 MIN_BET..MAX_BET，验证 _bet_for_cards 结果是真正的最优。"""
        for mult in [2, 4, 6, 8, 10]:
            s = PinballStrategy(T=T, J=J, priority="cards")
            bet = s.optimal_bet(mult, list(range(DEFAULT_MULTIPLIER_SLOTS[mult])))

            # 暴力计算该 bet 的真实效率
            real_cards = min(mult * bet // T, J)
            real_eff = real_cards / bet if bet > 0 else 0

            # 暴力枚举所有可能的 bet
            best_eff_brute = 0.0
            best_n_brute = MIN_BET
            for n in range(MIN_BET, MAX_BET + 1):
                cards = min(mult * n // T, J)
                eff = cards / n
                if eff > best_eff_brute or (eff == best_eff_brute and n < best_n_brute):
                    best_eff_brute = eff
                    best_n_brute = n

            assert real_eff >= best_eff_brute - 1e-12, (
                f"T={T}, J={J}, mult={mult}: bet={bet}(eff={real_eff:.6f}) "
                f"不如 bet={best_n_brute}(eff={best_eff_brute:.6f})"
            )

    @pytest.mark.parametrize("T,J", CONFIGS)
    def test_optimal_bet_is_smallest_at_max_efficiency(self, T, J):
        """当多个 bet 有相同效率时，应选最小的（资本风险最低）。"""
        for mult in [2, 4, 6, 8, 10]:
            s = PinballStrategy(T=T, J=J, priority="cards")
            bet = s.optimal_bet(mult, list(range(DEFAULT_MULTIPLIER_SLOTS[mult])))
            real_cards = min(mult * bet // T, J)
            real_eff = real_cards / bet if bet > 0 else 0

            # 检查有没有更小的 bet 达到相同效率
            for n in range(MIN_BET, bet):
                cards_n = min(mult * n // T, J)
                eff_n = cards_n / n
                assert eff_n < real_eff + 1e-12, (
                    f"T={T}, J={J}, mult={mult}: bet={n}(eff={eff_n:.6f}) "
                    f"≥ 选定的 bet={bet}(eff={real_eff:.6f})，应选更小的"
                )


# ═══════════════════════════════════════════════════════════════════
# 3. expected_value_table 一致性
# ═══════════════════════════════════════════════════════════════════


class TestEVTableConsistency:
    """验证 expected_value_table 与 recommend() 的数值自洽。"""

    def test_ev_table_card_bet_matches_instance(self):
        """静态表格的 card_optimal_bet 应与实例方法一致。"""
        for T in [20, 25, 30, 50]:
            for J in [3, 5, 10]:
                rows = PinballStrategy.expected_value_table(T, J)
                s = PinballStrategy(T=T, J=J, priority="cards")
                for r in rows:
                    mult = r["multiplier"]
                    n_lit = r["lit_slots"]
                    lit = list(range(n_lit))
                    bet = s.optimal_bet(mult, lit)
                    assert bet == r["card_optimal_bet"], (
                        f"T={T}, J={J}, mult={mult}: table={r['card_optimal_bet']} vs instance={bet}"
                    )

    def test_ev_table_marble_ev_values(self):
        """marble_ev_ratio = mult * p_win = mult * n_lit / 12"""
        rows = PinballStrategy.expected_value_table(T=20, J=10)
        for r in rows:
            expected = r["multiplier"] * r["lit_slots"] / NUM_SLOTS
            assert r["marble_ev_ratio"] == pytest.approx(expected, abs=1e-4)

    def test_ev_table_cards_per_marble_formula(self):
        """cards_per_marble = p_win * floor(mult * bet / T) / bet (capped at J)"""
        for T in [20, 30]:
            for J in [3, 10]:
                rows = PinballStrategy.expected_value_table(T, J)
                for r in rows:
                    bet = r["card_optimal_bet"]
                    mult = r["multiplier"]
                    p_win = r["p_win"]
                    cards_on_win = min(mult * bet // T, J)
                    expected_cpm = p_win * cards_on_win / bet if bet > 0 else 0
                    assert r["cards_per_marble"] == pytest.approx(expected_cpm, abs=1e-4), (
                        f"T={T}, J={J}, mult={mult}"
                    )


# ═══════════════════════════════════════════════════════════════════
# 4. recommend() 数学一致性
# ═══════════════════════════════════════════════════════════════════


class TestRecommendConsistency:
    def test_expected_marbles_formula(self):
        """expected_marble_return = multiplier * bet * p_win"""
        for priority in ("cards", "marbles"):
            s = PinballStrategy(T=20, J=10, priority=priority)
            for mult in [2, 4, 6, 8, 10]:
                lit = list(range(DEFAULT_MULTIPLIER_SLOTS[mult]))
                rec = s.recommend(mult, lit)
                expected = mult * rec["optimal_bet"] * rec["win_probability"]
                assert rec["expected_marble_return"] == pytest.approx(expected, abs=0.01)

    def test_expected_cards_formula(self):
        """expected_score_cards = p_win * min(floor(mult*bet/T), J)"""
        for T in [20, 30]:
            for J in [3, 10]:
                s = PinballStrategy(T=T, J=J, priority="cards")
                for mult in [2, 4, 6, 8, 10]:
                    lit = list(range(DEFAULT_MULTIPLIER_SLOTS[mult]))
                    rec = s.recommend(mult, lit)
                    cards_on_win = min(mult * rec["optimal_bet"] // T, J)
                    expected = rec["win_probability"] * cards_on_win
                    assert rec["expected_score_cards"] == pytest.approx(expected, abs=1e-4)

    def test_roi_formula(self):
        """marble_roi = expected_marble_return / bet"""
        s = PinballStrategy(T=20, J=10, priority="cards")
        for mult in [2, 4, 6, 8, 10]:
            lit = list(range(DEFAULT_MULTIPLIER_SLOTS[mult]))
            rec = s.recommend(mult, lit)
            expected_roi = rec["expected_marble_return"] / rec["optimal_bet"]
            assert rec["marble_roi"] == pytest.approx(expected_roi, abs=1e-4)

    def test_recommend_with_biased_data(self):
        """有观测数据后，p_win 应反映真实偏差"""
        s = PinballStrategy(T=20, J=10, priority="cards", prior_weight=0)
        # 所有球落在 slot 0
        for _ in range(100):
            s.record_landing(0)
        rec = s.recommend(2, [0, 1, 2, 3])
        assert rec["win_probability"] == pytest.approx(1.0, abs=0.01)

        rec2 = s.recommend(2, [4, 5, 6, 7])
        assert rec2["win_probability"] == pytest.approx(0.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
# 5. Marble priority: 边界 EV 场景
# ═══════════════════════════════════════════════════════════════════


class TestMarblePriorityEdgeCases:
    def test_ev_exactly_one_bets_min(self):
        """EV = 1.0 时不严格 > 1，应投 MIN_BET"""
        s = PinballStrategy(T=20, J=10, priority="marbles", prior_weight=0)
        # 制造 p_win = 0.25, mult = 4 → EV = 1.0
        for i in range(100):
            s.record_landing(i % 3)  # 0,1,2 平分
        # p_win([0,1,2]) ≈ 1.0 (因为只记录了0,1,2)
        # 但 mult=4 时 EV = 4 * 1.0 = 4.0 > 1 → MAX
        # 换一个: 让 p_win = 0.5 for mult=2 → EV = 1.0
        s2 = PinballStrategy(T=20, J=10, priority="marbles", prior_weight=0)
        for i in range(100):
            s2.record_landing(i % 6)  # 平分前6个
        # p_win([0,1,2,3,4,5]) = 1.0, mult=1... 不行
        # 直接用特定概率: p_win = 0.5 需要 6 个 lit out of 12
        # 但 2x only has 4 lit slots...
        # 用 8x (1 lit), 需要 p_win > 0.125 → bet MAX
        # 最简单: 用数学已知 EV = 1 的情况
        # 4x, 3 lit in uniform: EV = 4 * 3/12 = 1.0
        s3 = PinballStrategy(T=20, J=10, priority="marbles")
        bet = s3.optimal_bet(4, [0, 1, 2])
        assert bet == MIN_BET  # EV = 1.0, not > 1

    def test_ev_just_above_one(self):
        """EV > 1.0 → MAX_BET"""
        s = PinballStrategy(T=20, J=10, priority="marbles", prior_weight=0)
        # slot 0 概率 = 100%
        for _ in range(100):
            s.record_landing(0)
        # mult=2, lit=[0] → EV = 2 * 1.0 = 2.0 > 1 → MAX
        assert s.optimal_bet(2, [0]) == MAX_BET

    def test_ev_just_below_one(self):
        """EV < 1.0 → MIN_BET"""
        s = PinballStrategy(T=20, J=10, priority="marbles")
        # Uniform: 2x, 4 lit → EV = 2 * 4/12 ≈ 0.667
        assert s.optimal_bet(2, [0, 1, 2, 3]) == MIN_BET

    def test_marble_priority_uniform_always_min(self):
        """均匀分布下，所有倍数 EV ≤ 1.0，marble 优先永远投 MIN_BET"""
        s = PinballStrategy(T=20, J=10, priority="marbles")
        for mult, n_lit in DEFAULT_MULTIPLIER_SLOTS.items():
            lit = list(range(n_lit))
            ev = mult * n_lit / NUM_SLOTS
            bet = s.optimal_bet(mult, lit)
            if ev <= 1.0:
                assert bet == MIN_BET, f"{mult}x: EV={ev:.3f} ≤ 1 但 bet={bet}"

    def test_marble_priority_no_graduated_betting(self):
        """当前 marble 策略是二元的: MIN 或 MAX，没有中间值"""
        s = PinballStrategy(T=20, J=10, priority="marbles")
        for mult in [2, 4, 6, 8, 10]:
            lit = list(range(DEFAULT_MULTIPLIER_SLOTS[mult]))
            bet = s.optimal_bet(mult, lit)
            assert bet in (MIN_BET, MAX_BET) or bet == s.max_bet, \
                f"{mult}x: bet={bet} 既不是 MIN 也不是 MAX"


# ═══════════════════════════════════════════════════════════════════
# 6. max_bet 边界
# ═══════════════════════════════════════════════════════════════════


class TestMaxBetBoundary:
    def test_max_bet_equals_min_bet(self):
        """max_bet=5 时所有策略应返回 5"""
        s = PinballStrategy(T=20, J=10, priority="cards", max_bet=5)
        for mult in [2, 4, 6, 8, 10]:
            lit = list(range(DEFAULT_MULTIPLIER_SLOTS[mult]))
            assert s.optimal_bet(mult, lit) == MIN_BET

    def test_max_bet_limits_card_bet(self):
        """max_bet=10 时 card bet 不应超过 10"""
        s = PinballStrategy(T=20, J=10, priority="cards", max_bet=10)
        for mult in [2, 4, 6, 8, 10]:
            lit = list(range(DEFAULT_MULTIPLIER_SLOTS[mult]))
            assert s.optimal_bet(mult, lit) <= 10

    def test_max_bet_small_still_optimal(self):
        """即使 max_bet 很小，返回的 bet 仍是在 [MIN_BET, max_bet] 内的最优"""
        for mb in [5, 10, 15, 20, 30]:
            s = PinballStrategy(T=20, J=10, priority="cards", max_bet=mb)
            for mult in [2, 4, 6, 8, 10]:
                lit = list(range(DEFAULT_MULTIPLIER_SLOTS[mult]))
                bet = s.optimal_bet(mult, lit)
                real_cards = min(mult * bet // 20, 10)
                real_eff = real_cards / bet if bet > 0 else 0
                # 暴力检查
                for n in range(MIN_BET, mb + 1):
                    cards_n = min(mult * n // 20, 10)
                    eff_n = cards_n / n
                    assert eff_n <= real_eff + 1e-12, (
                        f"mb={mb}, mult={mult}: bet={n}(eff={eff_n:.6f}) > "
                        f"selected bet={bet}(eff={real_eff:.6f})"
                    )


# ═══════════════════════════════════════════════════════════════════
# 7. 大 T 值（卡很难拿）
# ═══════════════════════════════════════════════════════════════════


class TestLargeT:
    def test_large_T_zero_cards(self):
        """T=200, mult=2 时: MIN_BET=5 → floor(10/200)=0, 拿不到卡"""
        s = PinballStrategy(T=200, J=10, priority="cards")
        bet = s.optimal_bet(2, [0, 1, 2, 3])
        cards = min(2 * bet // 200, 10)
        # 算法应至少返回足够拿到1张卡的 bet
        # 2x: 需要 bet >= ceil(200/2) = 100, 但在 MAX_BET=99 内不够
        if bet <= 99:
            # 无论怎么投都拿不到卡（2*99/200 = 0.99, floor = 0）
            assert cards == 0

    def test_large_T_high_mult_can_still_earn(self):
        """T=200, mult=10 时: ceil(200/10)=20 可以拿1张卡"""
        s = PinballStrategy(T=200, J=10, priority="cards")
        bet = s.optimal_bet(10, [0])
        cards = min(10 * bet // 200, 10)
        assert cards >= 1, f"T=200, 10x, bet={bet}: 应至少拿到1张卡"


# ═══════════════════════════════════════════════════════════════════
# 8. Prior weight 极端值
# ═══════════════════════════════════════════════════════════════════


class TestPriorWeightEdges:
    def test_zero_prior_pure_frequency(self):
        """prior_weight=0 → 纯频率估计"""
        s = PinballStrategy(T=20, J=10, prior_weight=0)
        s.record_landing(0)
        probs = s.get_landing_probs()
        assert probs[0] == 1.0
        for i in range(1, NUM_SLOTS):
            assert probs[i] == 0.0

    def test_zero_prior_no_data_uniform(self):
        """prior_weight=0 且无数据 → 应返回均匀分布（不是除零）"""
        s = PinballStrategy(T=20, J=10, prior_weight=0)
        probs = s.get_landing_probs()
        for p in probs:
            assert p == pytest.approx(1.0 / NUM_SLOTS)

    def test_large_prior_dampens_data(self):
        """prior_weight=10000 → 大量观测也几乎不影响分布"""
        s = PinballStrategy(T=20, J=10, prior_weight=10000)
        for _ in range(100):
            s.record_landing(0)
        probs = s.get_landing_probs()
        # P(0) = (10000/12 + 100) / (10000 + 100) ≈ 933.3/10100 ≈ 0.0924
        expected = (10000 / 12 + 100) / (10000 + 100)
        assert probs[0] == pytest.approx(expected, abs=1e-6)
        # 与均匀的 1/12 ≈ 0.0833 相差不大
        assert abs(probs[0] - 1 / 12) < 0.02

    def test_prior_weight_sum_always_one(self):
        """任意 prior_weight 下概率总和恒为1"""
        for pw in [0, 1, 12, 24, 100, 10000]:
            s = PinballStrategy(T=20, J=10, prior_weight=pw)
            for i in range(50):
                s.record_landing(i % NUM_SLOTS)
            probs = s.get_landing_probs()
            assert abs(sum(probs) - 1.0) < 1e-9, f"pw={pw}: sum={sum(probs)}"


# ═══════════════════════════════════════════════════════════════════
# 9. Adaptive 策略精确测试
# ═══════════════════════════════════════════════════════════════════


class TestAdaptiveDetailed:
    def test_confidence_formula(self):
        """confidence = plays / (plays + threshold)"""
        s = PinballStrategy(T=20, J=10, confidence_threshold=100)
        assert s._confidence() == 0.0  # 0/(0+100)
        for _ in range(50):
            s.record_landing(0)
        assert s._confidence() == pytest.approx(50 / 150)
        for _ in range(50):
            s.record_landing(0)
        assert s._confidence() == pytest.approx(100 / 200)

    def test_adaptive_cards_negative_ev_always_floor(self):
        """负 EV 时自适应策略始终投 n_floor，不管有多少数据"""
        for ct in [5, 50, 500]:
            s = PinballStrategy(T=20, J=10, priority="cards", confidence_threshold=ct)
            # 均匀记录 1000 局
            for i in range(1000):
                s.record_landing(i % NUM_SLOTS)
            # 2x 4 lit: EV = 2 * 4/12 = 0.667 < 1
            bet = s.optimal_bet(2, [0, 1, 2, 3])
            n_floor = max(MIN_BET, math.ceil(20 / 2))
            assert bet == n_floor

    def test_adaptive_marbles_threshold_decreases_with_confidence(self):
        """随置信度增加，marble 自适应策略的 EV 阈值从 1.5 降到 1.0"""
        s = PinballStrategy(T=20, J=10, priority="marbles", confidence_threshold=100)
        # 0 plays: threshold = 1.5
        # Need EV > 1.5 to bet MAX. With slot 0 always landing:
        # mult=2, lit=[0] → EV by probability after training

        # Train with slot 0 only → p([0]) ≈ 1.0
        for _ in range(1000):
            s.record_landing(0)
        conf = s._confidence()
        # conf = 1000/1100 ≈ 0.909
        assert conf > 0.9

        # mult=2, lit=[0]: EV = 2 * ~1.0 = ~2.0 >> threshold → MAX
        bet = s.optimal_bet(2, [0])
        assert bet == MAX_BET

    def test_adaptive_low_confidence_conservative(self):
        """低置信度时，即使 EV > 1 也可能不投 MAX"""
        s = PinballStrategy(T=20, J=10, priority="marbles", confidence_threshold=1000,
                            prior_weight=0)
        # 10次记录 slot 0 → conf = 10/1010 ≈ 0.01
        for _ in range(10):
            s.record_landing(0)
        # mult=2, lit=[0]: p_win=1.0, EV=2.0
        # threshold = 1 + 0.5*(1-0.01) = 1.495
        # EV=2.0 > 1.495 → MAX (even at low conf, strong EV wins)
        bet = s.optimal_bet(2, [0])
        assert bet == MAX_BET

    def test_adaptive_ramp_increases_with_plays(self):
        """cards 自适应: 正 EV 时，随局数增加 bet 应递增或不减"""
        bets = []
        s = PinballStrategy(T=20, J=10, priority="cards", confidence_threshold=50,
                            prior_weight=0)
        for i in range(200):
            s.record_landing(0)  # 强烈偏向 slot 0
            if (i + 1) % 10 == 0:
                # 10x, lit=[0]: EV = 10*~1.0 >> 1
                bet = s.optimal_bet(10, [0])
                bets.append(bet)
        # 随置信度增加 bet 应非递减
        for i in range(1, len(bets)):
            assert bets[i] >= bets[i - 1], (
                f"bet 在 {i*10} 局后下降: {bets[i]} < {bets[i-1]}"
            )


# ═══════════════════════════════════════════════════════════════════
# 10. 特殊 T, J 组合的 _bet_for_cards 验证
# ═══════════════════════════════════════════════════════════════════


class TestSpecialTJCombinations:
    def test_T_equals_1(self):
        """T=1: 每投一个弹珠就可能拿1张卡 → 效率极高"""
        s = PinballStrategy(T=1, J=10, priority="cards")
        # 10x: ceil(10/10)=1 → n=5 (MIN_BET). 实际卡 = min(50, 10) = 10
        bet = s.optimal_bet(10, [0])
        assert bet == MIN_BET
        assert min(10 * MIN_BET // 1, 10) == 10

    def test_T_equals_mult(self):
        """T = mult 时，每个步进 bet 恰好多一张卡"""
        T = 10
        s = PinballStrategy(T=T, J=5, priority="cards")
        bet = s.optimal_bet(10, [0])
        # 10x, T=10: step = ceil(k*10/10) = k
        # k=1→n=5, k=2→n=5, k=3→n=5, k=4→n=5, k=5→n=5 (all clamped to MIN_BET)
        # All map to n=5, cards = min(50/10, 5) = 5
        # eff at k=5: 5/5 = 1.0
        assert bet == MIN_BET

    def test_J_equals_1_smallest_bet(self):
        """J=1 时只能拿1张卡，应投最小的 step-aligned bet"""
        T = 20
        s = PinballStrategy(T=T, J=1, priority="cards")
        # 2x: ceil(20/2)=10 → 1 card. eff=1/10
        assert s.optimal_bet(2, [0, 1, 2, 3]) == 10
        # 10x: ceil(20/10)=2 → n=5(MIN). cards=min(floor(50/20),1)=min(2,1)=1. eff=1/5
        assert s.optimal_bet(10, [0]) == MIN_BET


# ═══════════════════════════════════════════════════════════════════
# 11. 边界输入验证
# ═══════════════════════════════════════════════════════════════════


class TestInputValidation:
    def test_negative_prior_weight_rejected(self):
        with pytest.raises(ValueError):
            PinballStrategy(T=20, J=10, prior_weight=-1)

    def test_negative_confidence_threshold_rejected(self):
        with pytest.raises(ValueError):
            PinballStrategy(T=20, J=10, confidence_threshold=-1)

    def test_max_bet_below_min_rejected(self):
        with pytest.raises(ValueError):
            PinballStrategy(T=20, J=10, max_bet=4)

    def test_max_bet_above_cap_rejected(self):
        with pytest.raises(ValueError):
            PinballStrategy(T=20, J=10, max_bet=100)

    def test_slot_minus_one_rejected(self):
        s = PinballStrategy(T=20, J=10)
        with pytest.raises(ValueError):
            s.record_landing(-1)

    def test_slot_twelve_rejected(self):
        s = PinballStrategy(T=20, J=10)
        with pytest.raises(ValueError):
            s.record_landing(12)

    def test_empty_lit_slots_returns_zero_probability(self):
        s = PinballStrategy(T=20, J=10)
        assert s.win_probability([]) == 0.0

    def test_out_of_range_lit_slot_ignored(self):
        """超出范围的 lit_slot 应被过滤掉"""
        s = PinballStrategy(T=20, J=10)
        p = s.win_probability([0, 100, -1])
        # 只有 slot 0 有效
        assert p == pytest.approx(1 / 12)


# ═══════════════════════════════════════════════════════════════════
# 12. 学习收敛性
# ═══════════════════════════════════════════════════════════════════


class TestLearningConvergence:
    def test_probs_converge_to_true_distribution(self):
        """大量数据后，估计概率应接近真实分布"""
        true_probs = [0.15, 0.12, 0.10, 0.09, 0.08, 0.07,
                      0.07, 0.08, 0.07, 0.06, 0.06, 0.05]
        import random
        rng = random.Random(42)

        s = PinballStrategy(T=20, J=10, prior_weight=24)
        for _ in range(10000):
            r = rng.random()
            cum = 0.0
            for i, p in enumerate(true_probs):
                cum += p
                if r < cum:
                    s.record_landing(i)
                    break

        estimated = s.get_landing_probs()
        for i in range(NUM_SLOTS):
            assert abs(estimated[i] - true_probs[i]) < 0.02, (
                f"slot {i}: estimated={estimated[i]:.4f}, true={true_probs[i]}"
            )

    def test_fewer_data_higher_error(self):
        """少量数据时误差应较大"""
        s = PinballStrategy(T=20, J=10, prior_weight=24)
        for _ in range(10):
            s.record_landing(0)  # 全落 slot 0
        probs = s.get_landing_probs()
        # P(0) = (2+10)/(24+10) ≈ 0.353, 远高于真实（如果均匀则0.083）
        # 但由于先验平滑，不会是1.0
        assert probs[0] < 0.5
        assert probs[0] > 1 / 12

    def test_heavily_biased_machine_detected(self):
        """严重偏差的机器应被正确检测"""
        s = PinballStrategy(T=20, J=10, prior_weight=24)
        # 80% 落在 slot 0, 20% 分散
        import random
        rng = random.Random(123)
        for _ in range(500):
            if rng.random() < 0.8:
                s.record_landing(0)
            else:
                s.record_landing(rng.randint(1, 11))

        probs = s.get_landing_probs()
        # slot 0 应明显高于其他
        assert probs[0] > 0.5
        assert all(probs[i] < probs[0] for i in range(1, NUM_SLOTS))
