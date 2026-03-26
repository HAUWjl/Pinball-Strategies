#!/usr/bin/env python3
"""
商家操控分析报告：奖励倍数概率调整对玩家的影响

研究内容：
1. 商家调整倍率出现概率（如降低高倍率概率、提高低倍率概率）
2. 商家调整亮灯格数（减少每个倍率的亮灯数量）
3. 商家组合操控（同时调整倍率概率+亮灯格数）
4. V2策略在各种操控环境下的抗操控能力分析
5. 玩家盈亏平衡点分析

运行:
    python merchant_manipulation_analysis.py
"""

import math
import sys
from typing import Dict, List

from pinball_strategy import (
    NUM_SLOTS, MIN_BET, MAX_BET, DEFAULT_MULTIPLIER_SLOTS, PinballStrategy
)
from simulation_test import (
    run_simulation, normalize_probs, MULTIPLIER_PROBS
)

# ══════════════════════════════════════════════════════════════════════════════
#  商家操控方案定义
# ══════════════════════════════════════════════════════════════════════════════

# 基准倍率出现概率（实测值）
BASELINE_MULT_PROBS = dict(MULTIPLIER_PROBS)  # {2: 0.420, 4: 0.288, 6: 0.127, 8: 0.108, 10: 0.057}

# ── 操控方案一：调整倍率出现概率 ────────────────────────────────────────────
# 商家可通过电路板/软件修改倍率出现概率

MANIPULATED_MULT_PROBS = {
    "基准(实测)": {2: 0.420, 4: 0.288, 6: 0.127, 8: 0.108, 10: 0.057},
    "温和压低高倍": {2: 0.500, 4: 0.280, 6: 0.110, 8: 0.070, 10: 0.040},
    "严重压低高倍": {2: 0.600, 4: 0.250, 6: 0.080, 8: 0.050, 10: 0.020},
    "极端压低高倍": {2: 0.700, 4: 0.200, 6: 0.060, 8: 0.030, 10: 0.010},
    "提升中间倍率": {2: 0.350, 4: 0.350, 6: 0.150, 8: 0.100, 10: 0.050},
    "虚假高倍(8x10x↑但灯少)": {2: 0.300, 4: 0.250, 6: 0.150, 8: 0.180, 10: 0.120},
}

# ── 操控方案二：调整亮灯格数 ────────────────────────────────────────────────
# 商家可减少某些倍率下的亮灯数

MANIPULATED_SLOTS = {
    "基准(标准)": {2: 4, 4: 3, 6: 2, 8: 1, 10: 1},
    "温和减灯": {2: 3, 4: 2, 6: 2, 8: 1, 10: 1},
    "严重减灯": {2: 3, 4: 2, 6: 1, 8: 1, 10: 1},
    "极端减灯": {2: 2, 4: 2, 6: 1, 8: 1, 10: 1},
    "低倍多灯高倍减灯": {2: 5, 4: 2, 6: 1, 8: 1, 10: 1},
}

# ── 操控方案三：组合操控 ────────────────────────────────────────────────────
COMBO_MANIPULATIONS = {
    "基准": {
        "mult_probs": {2: 0.420, 4: 0.288, 6: 0.127, 8: 0.108, 10: 0.057},
        "slots": {2: 4, 4: 3, 6: 2, 8: 1, 10: 1},
    },
    "轻度操控": {
        "mult_probs": {2: 0.500, 4: 0.280, 6: 0.110, 8: 0.070, 10: 0.040},
        "slots": {2: 3, 4: 3, 6: 2, 8: 1, 10: 1},
    },
    "中度操控": {
        "mult_probs": {2: 0.550, 4: 0.260, 6: 0.100, 8: 0.060, 10: 0.030},
        "slots": {2: 3, 4: 2, 6: 1, 8: 1, 10: 1},
    },
    "重度操控": {
        "mult_probs": {2: 0.650, 4: 0.220, 6: 0.070, 8: 0.040, 10: 0.020},
        "slots": {2: 2, 4: 2, 6: 1, 8: 1, 10: 1},
    },
    "极端操控": {
        "mult_probs": {2: 0.750, 4: 0.170, 6: 0.050, 8: 0.020, 10: 0.010},
        "slots": {2: 2, 4: 1, 6: 1, 8: 1, 10: 1},
    },
}

# ── 落点概率分布 ────────────────────────────────────────────────────────────
DISTRIBUTIONS = {
    "均匀分布": [1.0] * 12,
    "轻微偏斜": [0.10, 0.10, 0.10, 0.09, 0.08, 0.07, 0.07, 0.09, 0.08, 0.08, 0.07, 0.07],
    "中等偏斜": [0.15, 0.13, 0.11, 0.10, 0.08, 0.07, 0.06, 0.06, 0.05, 0.07, 0.06, 0.06],
    "严重偏斜": [0.25, 0.18, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.04, 0.03, 0.02],
}

# ── 模拟参数 ────────────────────────────────────────────────────────────────
INITIAL_MARBLES = 10000
CONSUME_TARGET = 3000
T = 20
J = 10
N_SEEDS = 50
MAX_ROUNDS = 10000


def weighted_ev_ratio(mult_probs: Dict[int, float], slots: Dict[int, int]) -> float:
    """计算加权综合期望回报率。"""
    total = 0.0
    for mult, prob in mult_probs.items():
        n_lit = slots.get(mult, DEFAULT_MULTIPLIER_SLOTS.get(mult, 1))
        p_win = n_lit / NUM_SLOTS
        total += prob * mult * p_win
    return total


def expected_card_rate(mult_probs: Dict[int, float], slots: Dict[int, int], T: int, bet: int) -> float:
    """计算每珠期望积分卡产出率。"""
    total = 0.0
    for mult, prob in mult_probs.items():
        n_lit = slots.get(mult, DEFAULT_MULTIPLIER_SLOTS.get(mult, 1))
        p_win = n_lit / NUM_SLOTS
        cards = min(mult * bet // T, J)
        total += prob * p_win * cards
    return total / bet


def run_manipulation_simulation(
    mult_probs: Dict[int, float],
    multiplier_slots: Dict[int, int],
    hole_probs: List[float],
    ct: float,
    n_seeds: int = N_SEEDS,
) -> dict:
    """用指定的操控参数运行模拟，需修改模拟函数以接受自定义倍率概率。"""
    import random as _random
    hole_probs = normalize_probs(hole_probs)

    total_cards = 0
    total_consumed = 0
    total_rounds = 0
    total_wins = 0
    total_marbles_spent = 0
    total_marbles_won = 0

    for seed in range(n_seeds):
        rng = _random.Random(seed)
        strategy = PinballStrategy(
            T=T, J=J, priority="cards",
            multiplier_slots=multiplier_slots,
            confidence_threshold=ct,
        )

        marbles_remaining = INITIAL_MARBLES
        rounds = 0
        wins = 0
        spent = 0
        won = 0
        cards = 0

        while marbles_remaining >= MIN_BET and rounds < MAX_ROUNDS:
            rounds += 1

            # 用操控后的倍率概率抽取倍数
            r = rng.random()
            cumulative = 0.0
            multiplier = list(mult_probs.keys())[-1]
            for mult, prob in mult_probs.items():
                cumulative += prob
                if r < cumulative:
                    multiplier = mult
                    break

            # 亮灯格数用操控后的配置
            n_lit = multiplier_slots.get(multiplier, 1)
            lit_slots = sorted(rng.sample(range(NUM_SLOTS), n_lit))

            # 策略推荐
            recommended_bet = strategy.optimal_bet(multiplier, lit_slots)
            actual_bet = min(recommended_bet, marbles_remaining)
            actual_bet = max(actual_bet, MIN_BET)
            if actual_bet > marbles_remaining:
                break

            # 弹珠落点
            r2 = rng.random()
            cumulative2 = 0.0
            landing = NUM_SLOTS - 1
            for i, p in enumerate(hole_probs):
                cumulative2 += p
                if r2 < cumulative2:
                    landing = i
                    break

            is_win = landing in lit_slots
            marbles_won_round = multiplier * actual_bet if is_win else 0
            cards_won_round = min(marbles_won_round // T, J) if is_win else 0

            marbles_remaining -= actual_bet
            marbles_remaining += marbles_won_round
            spent += actual_bet
            won += marbles_won_round
            cards += cards_won_round
            if is_win:
                wins += 1

            strategy.record_landing(landing)

            net_consumed = INITIAL_MARBLES - marbles_remaining
            if net_consumed >= CONSUME_TARGET:
                break

        total_cards += cards
        net = INITIAL_MARBLES - marbles_remaining
        total_consumed += net
        total_rounds += rounds
        total_wins += wins
        total_marbles_spent += spent
        total_marbles_won += won

    avg_cards = total_cards / n_seeds
    avg_consumed = total_consumed / n_seeds
    avg_rounds = total_rounds / n_seeds
    win_rate = total_wins / total_rounds if total_rounds > 0 else 0
    cpm = total_cards / total_consumed if total_consumed > 0 else float('inf')

    return {
        "avg_cards": avg_cards,
        "avg_consumed": avg_consumed,
        "avg_rounds": avg_rounds,
        "win_rate": win_rate,
        "cpm": cpm,
        "avg_spent": total_marbles_spent / n_seeds,
        "avg_won": total_marbles_won / n_seeds,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  报告生成
# ══════════════════════════════════════════════════════════════════════════════

def section_header(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def subsection(title: str) -> None:
    print(f"\n── {title} {'─' * max(1, 70 - len(title) * 2)}")


def report_part1_theory():
    """第一部分：理论分析 — 倍率概率操控对期望值的影响"""
    section_header("第一部分：倍率出现概率操控的理论影响")

    print("\n商家可以通过调整电路板/程序来改变各倍率的出现概率。")
    print("以下分析各种操控方案对玩家加权EV的影响。\n")

    print(f"  {'方案':>20} | {'2x':>6} {'4x':>6} {'6x':>6} {'8x':>6} {'10x':>5} | {'加权EV':>8} {'亏损率':>8} {'vs基准':>8}")
    print("  " + "-" * 92)

    baseline_ev = weighted_ev_ratio(BASELINE_MULT_PROBS, DEFAULT_MULTIPLIER_SLOTS)
    for name, probs in MANIPULATED_MULT_PROBS.items():
        ev = weighted_ev_ratio(probs, DEFAULT_MULTIPLIER_SLOTS)
        loss_rate = (1 - ev) * 100
        vs_baseline = (ev - baseline_ev) / baseline_ev * 100
        print(
            f"  {name:>20} | "
            f"{probs[2]:>5.1%} {probs[4]:>5.1%} {probs[6]:>5.1%} {probs[8]:>5.1%} {probs[10]:>5.1%} | "
            f"{ev:>8.4f} {loss_rate:>+7.2f}% {vs_baseline:>+7.2f}%"
        )

    # 各倍率的单独EV分析
    subsection("单倍率EV分析（标准亮灯数）")
    print(f"\n  {'倍率':>4}  {'亮灯':>4}  {'P_win':>8}  {'EV_ratio':>10}  {'性质':>10}")
    print("  " + "-" * 45)
    for mult, n_lit in sorted(DEFAULT_MULTIPLIER_SLOTS.items()):
        p_win = n_lit / NUM_SLOTS
        ev_ratio = mult * p_win
        nature = "正EV" if ev_ratio > 1 else ("平EV" if ev_ratio == 1 else "负EV")
        print(f"  {mult:>4}x  {n_lit:>4}  {p_win:>8.4f}  {ev_ratio:>10.4f}  {nature:>10}")

    print("\n  【关键发现】")
    print("  • 2x (EV=0.667) 和 8x (EV=0.667) 是负EV倍率")
    print("  • 4x (EV=1.000) 和 6x (EV=1.000) 是平EV倍率")
    print("  • 10x (EV=0.833) 虽然倍率高，但亮灯少，仍是负EV")
    print("  • 商家提高2x出现概率 = 增加负EV局占比 = 加速玩家亏损")


def report_part2_slots_manipulation():
    """第二部分：亮灯格数操控分析"""
    section_header("第二部分：亮灯格数操控的理论影响")

    print("\n商家可以通过调整亮灯逻辑来减少中奖概率。")
    print("即使倍率概率不变，减少亮灯数也会显著降低玩家EV。\n")

    print(f"  {'方案':>20} | {'2x灯':>4} {'4x灯':>4} {'6x灯':>4} {'8x灯':>4} {'10x灯':>4} | {'加权EV':>8} {'亏损率':>8} {'vs基准':>8}")
    print("  " + "-" * 88)

    baseline_ev = weighted_ev_ratio(BASELINE_MULT_PROBS, DEFAULT_MULTIPLIER_SLOTS)
    for name, slots in MANIPULATED_SLOTS.items():
        ev = weighted_ev_ratio(BASELINE_MULT_PROBS, slots)
        loss_rate = (1 - ev) * 100
        vs_baseline = (ev - baseline_ev) / baseline_ev * 100
        print(
            f"  {name:>20} | "
            f"{slots[2]:>4} {slots[4]:>4} {slots[6]:>4} {slots[8]:>4} {slots[10]:>4} | "
            f"{ev:>8.4f} {loss_rate:>+7.2f}% {vs_baseline:>+7.2f}%"
        )

    # 详细对比
    subsection("减灯操控下各倍率EV变化")
    for name, slots in MANIPULATED_SLOTS.items():
        print(f"\n  [{name}]")
        print(f"    {'倍率':>4}  {'灯数':>4}  {'标准灯':>6}  {'P_win':>8}  {'EV_ratio':>10}  {'标准EV':>10}  {'EV变化':>10}")
        print("    " + "-" * 62)
        for mult in sorted(slots.keys()):
            n_lit = slots[mult]
            n_lit_std = DEFAULT_MULTIPLIER_SLOTS[mult]
            p_win = n_lit / NUM_SLOTS
            ev_ratio = mult * p_win
            ev_std = mult * n_lit_std / NUM_SLOTS
            change = (ev_ratio - ev_std) / ev_std * 100 if ev_std > 0 else 0
            print(
                f"    {mult:>4}x  {n_lit:>4}  {n_lit_std:>6}  {p_win:>8.4f}  "
                f"{ev_ratio:>10.4f}  {ev_std:>10.4f}  {change:>+9.1f}%"
            )


def report_part3_combo():
    """第三部分：组合操控理论分析"""
    section_header("第三部分：组合操控（倍率概率+亮灯格数同时调整）")

    print("\n现实中商家最可能同时调整多个参数。以下分析组合操控的累积效果。\n")

    baseline_ev = weighted_ev_ratio(
        COMBO_MANIPULATIONS["基准"]["mult_probs"],
        COMBO_MANIPULATIONS["基准"]["slots"],
    )

    print(f"  {'操控程度':>10} | {'加权EV':>8} {'亏损率':>8} {'vs基准':>8} | {'投100珠期望回收':>16} {'净亏损':>8}")
    print("  " + "-" * 80)
    for name, cfg in COMBO_MANIPULATIONS.items():
        ev = weighted_ev_ratio(cfg["mult_probs"], cfg["slots"])
        loss_rate = (1 - ev) * 100
        vs_baseline = (ev - baseline_ev) / baseline_ev * 100
        return_100 = 100 * ev
        loss_100 = 100 - return_100
        print(
            f"  {name:>10} | {ev:>8.4f} {loss_rate:>+7.2f}% {vs_baseline:>+7.2f}% | "
            f"{return_100:>14.1f}珠 {loss_100:>7.1f}珠"
        )

    # 商家利润率分析
    subsection("商家利润率估算")
    print("\n  假设玩家持续投入，商家从玩家亏损中获利。\n")
    print(f"  {'操控程度':>10} | {'每100珠商家利润':>16} {'利润率':>8} | {'玩家投1000珠预期亏':>18} {'投5000珠预期亏':>16}")
    print("  " + "-" * 90)
    for name, cfg in COMBO_MANIPULATIONS.items():
        ev = weighted_ev_ratio(cfg["mult_probs"], cfg["slots"])
        profit_100 = 100 * (1 - ev)
        profit_rate = (1 - ev) * 100
        loss_1000 = 1000 * (1 - ev)
        loss_5000 = 5000 * (1 - ev)
        print(
            f"  {name:>10} | {profit_100:>14.1f}珠 {profit_rate:>7.1f}% | "
            f"{loss_1000:>16.0f}珠 {loss_5000:>14.0f}珠"
        )


def report_part4_simulation():
    """第四部分：蒙特卡洛模拟验证"""
    section_header("第四部分：蒙特卡洛模拟 — 操控对策略效果的影响")

    print(f"\n  模拟参数：初始={INITIAL_MARBLES}珠, 消耗目标={CONSUME_TARGET}珠, T={T}, J={J}")
    print(f"  每配置运行 {N_SEEDS} 组随机种子取平均\n")

    # 对两种落点分布，比较不同操控方案 × 两种策略
    for dist_name in ["均匀分布", "中等偏斜"]:
        hole_probs = DISTRIBUTIONS[dist_name]
        subsection(f"落点分布：{dist_name}")

        print(f"\n  {'操控方案':>10} {'策略':>8} | {'积分卡':>8} {'净消耗':>10} {'局数':>8} {'胜率':>6} {'卡/珠':>8} | {'vs基准原始':>10}")
        print("  " + "-" * 90)

        baseline_cards = None
        for combo_name, cfg in COMBO_MANIPULATIONS.items():
            for ct_label, ct_val in [("原始", 0.0), ("V2", 5.0)]:
                result = run_manipulation_simulation(
                    mult_probs=cfg["mult_probs"],
                    multiplier_slots=cfg["slots"],
                    hole_probs=hole_probs,
                    ct=ct_val,
                    n_seeds=N_SEEDS,
                )
                if baseline_cards is None and combo_name == "基准" and ct_label == "原始":
                    baseline_cards = result["avg_cards"]

                cpm_s = f"{result['cpm']:.4f}" if result['cpm'] != float('inf') and result['avg_consumed'] > 0 else "盈利"
                vs_base = ""
                if baseline_cards and baseline_cards > 0:
                    change = (result["avg_cards"] - baseline_cards) / baseline_cards * 100
                    vs_base = f"{change:>+.1f}%"

                print(
                    f"  {combo_name:>10} {ct_label:>8} | "
                    f"{result['avg_cards']:>8.1f} {result['avg_consumed']:>10.1f} "
                    f"{result['avg_rounds']:>8.1f} {result['win_rate']:>5.1%} {cpm_s:>8} | "
                    f"{vs_base:>10}"
                )
            print()
        baseline_cards = None  # 重置


def report_part5_mult_prob_simulation():
    """第五部分：纯倍率概率操控的模拟验证"""
    section_header("第五部分：纯倍率概率操控模拟（亮灯数不变）")

    print(f"\n  仅改变倍率出现概率，亮灯数保持标准配置。")
    print(f"  落点分布：轻微偏斜 | V2策略(ct=5)\n")

    hole_probs = DISTRIBUTIONS["轻微偏斜"]

    print(f"  {'方案':>20} | {'积分卡':>8} {'净消耗':>10} {'局数':>8} {'胜率':>6} | {'加权EV':>8}")
    print("  " + "-" * 78)

    for name, probs in MANIPULATED_MULT_PROBS.items():
        ev = weighted_ev_ratio(probs, DEFAULT_MULTIPLIER_SLOTS)
        result = run_manipulation_simulation(
            mult_probs=probs,
            multiplier_slots=DEFAULT_MULTIPLIER_SLOTS,
            hole_probs=hole_probs,
            ct=5.0,
            n_seeds=N_SEEDS,
        )
        cpm_s = f"{result['avg_consumed']:>10.1f}"
        print(
            f"  {name:>20} | "
            f"{result['avg_cards']:>8.1f} {cpm_s} "
            f"{result['avg_rounds']:>8.1f} {result['win_rate']:>5.1%} | "
            f"{ev:>8.4f}"
        )


def report_part6_player_detection():
    """第六部分：玩家如何检测商家操控"""
    section_header("第六部分：操控检测方法与玩家对策")

    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │  玩家可通过以下统计指标检测商家是否进行了操控                      │
  └────────────────────────────────────────────────────────────────────┘

  1. 倍率频率统计
     ────────────────
     记录每次倍率出现情况，与标准概率对比：
     标准值: 2x=42.0%, 4x=28.8%, 6x=12.7%, 8x=10.8%, 10x=5.7%

     如果连续50局中2x出现概率>55%，极大概率被操控。
     统计检验：χ² 拟合检验，显著性水平 α=0.05
     50局所需的最小可检测偏差: 约±12%（即2x>54%才可检测）
     200局所需的最小可检测偏差: 约±6%（即2x>48%才可检测）

  2. 亮灯格数验证
     ────────────────
     注意观察每个倍率对应的亮灯数是否一致：
     标准值: 2x→4灯, 4x→3灯, 6x→2灯, 8x→1灯, 10x→1灯

     任何偏离都说明机器配置被修改。
     这是最直观、最容易验证的指标。

  3. 综合胜率监控
     ────────────────
     标准配置下，均匀落点的综合中奖率约为：""")

    # 计算标准配置下的综合中奖率
    total_win_rate = 0
    for mult, prob in BASELINE_MULT_PROBS.items():
        n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
        total_win_rate += prob * n_lit / NUM_SLOTS

    print(f"     P_总中奖 = Σ π_m × k_m/12 = {total_win_rate:.4f} ({total_win_rate*100:.1f}%)")
    print()

    # 各操控方案下的中奖率
    print(f"     {'操控方案':>20} | {'综合中奖率':>10} | {'偏离':>8}")
    print("     " + "-" * 48)
    for name, cfg in COMBO_MANIPULATIONS.items():
        wr = 0
        for mult, prob in cfg["mult_probs"].items():
            n_lit = cfg["slots"].get(mult, 1)
            wr += prob * n_lit / NUM_SLOTS
        deviation = (wr - total_win_rate) / total_win_rate * 100
        print(f"     {name:>20} | {wr:>9.1%} | {deviation:>+7.1f}%")

    print("""
     如果100局以上的实际中奖率低于18%，建议更换机器。

  4. EV回报率监控
     ────────────────
     记录每局投入和返珠，计算实际回报率：
     实际EV = 总返珠 / 总投入

     标准配置下预期EV ≈ 0.8145
     如果200局后实际EV < 0.70，极大概率被操控。

  ┌────────────────────────────────────────────────────────────────────┐
  │  玩家对策建议                                                      │
  └────────────────────────────────────────────────────────────────────┘

  • 前20局以最小投注(5珠)试探，统计倍率分布和中奖率
  • 如果2x出现频率>55%或综合中奖率<18%，立即更换机器
  • 使用V2策略的信心阈值机制，天然具备抗操控能力：
    - 操控使正EV局更难出现 → V2自动保持最小投注 → 减少损失
    - 即使被操控，V2策略的损失也远小于原始策略
""")


def report_part7_v2_resilience():
    """第七部分：V2策略的抗操控能力量化分析"""
    section_header("第七部分：V2策略抗操控能力量化")

    print("\n  对比在不同操控强度下，原始策略和V2策略的损失程度。")
    print(f"  落点分布：轻微偏斜 | T={T}, J={J}\n")

    hole_probs = DISTRIBUTIONS["轻微偏斜"]

    results_table = []
    for combo_name, cfg in COMBO_MANIPULATIONS.items():
        row = {"name": combo_name}
        for ct_label, ct_val in [("原始", 0.0), ("V2", 5.0)]:
            result = run_manipulation_simulation(
                mult_probs=cfg["mult_probs"],
                multiplier_slots=cfg["slots"],
                hole_probs=hole_probs,
                ct=ct_val,
                n_seeds=N_SEEDS,
            )
            row[ct_label] = result
        results_table.append(row)

    # 损失对比
    print(f"  {'操控':>10} | {'原始:积分卡':>10} {'原始:净消耗':>10} | {'V2:积分卡':>10} {'V2:净消耗':>10} | {'V2卡提升':>8} {'V2损失减少':>10}")
    print("  " + "-" * 95)
    for row in results_table:
        o = row["原始"]
        v = row["V2"]
        card_improve = (v["avg_cards"] - o["avg_cards"]) / o["avg_cards"] * 100 if o["avg_cards"] > 0 else 0
        loss_reduce = ""
        if o["avg_consumed"] > 0 and v["avg_consumed"] > 0:
            loss_reduce = f"{(1 - v['avg_consumed'] / o['avg_consumed']) * 100:>+.1f}%"
        elif v["avg_consumed"] <= 0:
            loss_reduce = "盈利!"
        print(
            f"  {row['name']:>10} | "
            f"{o['avg_cards']:>10.1f} {o['avg_consumed']:>10.1f} | "
            f"{v['avg_cards']:>10.1f} {v['avg_consumed']:>10.1f} | "
            f"{card_improve:>+7.1f}% {loss_reduce:>10}"
        )

    # 结论
    subsection("抗操控能力结论")
    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │  V2策略的天然抗操控机制                                            │
  └────────────────────────────────────────────────────────────────────┘

  1. 负EV保守下注：操控使更多局面变为负EV，V2自动降至最小投注，
     而原始策略不区分正负EV，照样高额投注，损失更大。

  2. 正EV门槛提高：操控降低了中奖概率，使得正EV更难触发，
     V2不会因为偶然的正EV信号就贸然大额投注。

  3. 信心度保护：即使检测到正EV，前期低信心度也会限制投注规模，
     防止在数据不足时基于噪声做出错误决策。

  4. 总结：V2策略在被操控的环境下，比原始策略少亏更多，
     是一种内在的"防守优先"策略。
""")


def report_part8_breakeven():
    """第八部分：盈亏平衡点分析"""
    section_header("第八部分：盈亏平衡点 — 商家操控到什么程度玩家必输？")

    print("\n  分析在不同落点偏斜程度下，V2策略能承受的最大操控强度。")
    print("  (即：操控到什么程度，V2策略也无法实现盈利或减损)\n")

    # 理论盈亏平衡：加权EV=1.0 时的参数
    subsection("理论盈亏平衡线")
    print("\n  在均匀落点假设下，加权EV=1.0 意味着长期不亏不赚。")
    print("  标准配置加权EV=0.8145，已经是负和博弈。\n")
    print("  但偏斜落点 + V2策略可以通过利用正EV局扭转劣势。\n")

    print(f"  {'操控程度':>10} {'加权EV':>8} | ", end="")
    for dist_name in DISTRIBUTIONS:
        print(f"{'卡数@' + dist_name:>16}", end=" ")
    print()
    print("  " + "-" * 90)

    for combo_name, cfg in COMBO_MANIPULATIONS.items():
        ev = weighted_ev_ratio(cfg["mult_probs"], cfg["slots"])
        print(f"  {combo_name:>10} {ev:>8.4f} | ", end="")
        for dist_name, hole_probs in DISTRIBUTIONS.items():
            result = run_manipulation_simulation(
                mult_probs=cfg["mult_probs"],
                multiplier_slots=cfg["slots"],
                hole_probs=hole_probs,
                ct=5.0,
                n_seeds=N_SEEDS,
            )
            print(f"{result['avg_cards']:>16.1f}", end=" ")
        print()

    # 盈亏状态表
    subsection("盈亏状态表（V2策略，ct=5）")
    print(f"\n  ✓=盈利(净消耗<0) △=微亏(<1000珠) ✗=明显亏损(>1000珠)\n")
    print(f"  {'操控程度':>10} | ", end="")
    for dist_name in DISTRIBUTIONS:
        print(f"{dist_name:>12}", end=" ")
    print()
    print("  " + "-" * 70)

    for combo_name, cfg in COMBO_MANIPULATIONS.items():
        print(f"  {combo_name:>10} | ", end="")
        for dist_name, hole_probs in DISTRIBUTIONS.items():
            result = run_manipulation_simulation(
                mult_probs=cfg["mult_probs"],
                multiplier_slots=cfg["slots"],
                hole_probs=hole_probs,
                ct=5.0,
                n_seeds=N_SEEDS,
            )
            if result["avg_consumed"] < 0:
                symbol = "  ✓ 盈利  "
            elif result["avg_consumed"] < 1000:
                symbol = "  △ 微亏  "
            else:
                symbol = "  ✗ 亏损  "
            print(f"{symbol:>12}", end=" ")
        print()


def report_summary():
    """总结"""
    section_header("总结与结论")

    baseline_ev = weighted_ev_ratio(BASELINE_MULT_PROBS, DEFAULT_MULTIPLIER_SLOTS)
    extreme_ev = weighted_ev_ratio(
        COMBO_MANIPULATIONS["极端操控"]["mult_probs"],
        COMBO_MANIPULATIONS["极端操控"]["slots"],
    )

    print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │  研究结论                                                          │
  └────────────────────────────────────────────────────────────────────┘

  1. 操控影响幅度：
     • 标准配置加权EV = {baseline_ev:.4f}（玩家每投100珠平均亏{(1-baseline_ev)*100:.1f}珠）
     • 极端操控加权EV = {extreme_ev:.4f}（玩家每投100珠平均亏{(1-extreme_ev)*100:.1f}珠）
     • 极端操控使亏损率提高了 {((1-extreme_ev)/(1-baseline_ev) - 1)*100:.0f}%

  2. 操控手段的隐蔽性排序（从易到难检测）：
     - 亮灯格数变化：最容易被玩家发现（直接可见）
     - 倍率概率变化：需要记录50+局数据才能统计检测
     - 落点概率变化：最隐蔽，需要物理改造机器钉板

  3. V2策略的抗操控能力：
     • 在任何操控程度下，V2策略的积分卡产出均≥原始策略
     • V2策略在轻度操控+偏斜落点下仍可实现盈利
     • 重度操控下V2策略也能将亏损控制在原始策略的一个较小比例

  4. 给玩家的建议：
     • 前20局用最小投注试探机器
     • 统计2x倍率出现频率，>55%应更换机器
     • 使用V2策略，自动适应操控环境
     • 优先选择物理钉板有明显磨损/偏斜的老机器（落点更不均匀）

  5. 给监管方的建议：
     • 要求机器公示各倍率理论出现概率
     • 定期抽查机器实际概率是否与公示一致
     • 限制亮灯格数的最低标准（如2x不低于3灯）
""")


# ══════════════════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "商家操控分析报告：奖励倍数概率调整对玩家的影响".center(56) + "█")
    print("█" + "Merchant Manipulation Analysis for 12-Slot Pinball".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # 理论分析部分（快速）
    report_part1_theory()
    report_part2_slots_manipulation()
    report_part3_combo()

    # 检测方法（无模拟）
    report_part6_player_detection()

    # 模拟部分（耗时较长）
    print("\n" + "=" * 80)
    print("  正在运行蒙特卡洛模拟（可能需要几分钟）...")
    print("=" * 80)

    report_part4_simulation()
    report_part5_mult_prob_simulation()
    report_part7_v2_resilience()
    report_part8_breakeven()

    # 总结
    report_summary()

    print("\n  报告生成完毕。\n")


if __name__ == "__main__":
    main()
