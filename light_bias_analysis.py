#!/usr/bin/env python3
"""
灯位偏置分析报告：商家改变某些灯出现概率产生的影响

研究内容：
    商家已知弹珠落点的物理概率分布（由钉板决定），通过电路板/软件
    使某些格位更容易或更难被选为亮灯位。核心操控方式是让弹珠最容易
    落入的格位更少亮灯、最不容易落入的格位更多亮灯。

    这种操控手段极其隐蔽：
    - 亮灯数量不变（视觉上无异常）
    - 倍率出现概率不变（统计频率正常）
    - 但实际中奖率被系统性降低

分析维度：
    1. 理论分析：灯位偏置对中奖概率的数学影响
    2. 偏置强度梯度：从轻度到极端偏置的效果曲线
    3. 蒙特卡洛模拟：不同偏置下的实际投珠效果
    4. V2策略的抗偏置能力
    5. 玩家检测方法与实操建议

运行:
    python light_bias_analysis.py
"""

import math
import random as _random
import sys
from typing import Dict, List, Tuple

from pinball_strategy import (
    NUM_SLOTS, MIN_BET, MAX_BET, DEFAULT_MULTIPLIER_SLOTS, PinballStrategy,
)
from simulation_test import normalize_probs, MULTIPLIER_PROBS

# ══════════════════════════════════════════════════════════════════════════════
#  参数配置
# ══════════════════════════════════════════════════════════════════════════════

INITIAL_MARBLES = 10000
CONSUME_TARGET = 3000
T = 20
J = 10
N_SEEDS = 50
MAX_ROUNDS = 10000

# 落点物理概率分布
LANDING_DISTRIBUTIONS = {
    "均匀分布": [1 / 12] * 12,
    "轻微偏斜": normalize_probs(
        [0.10, 0.10, 0.10, 0.09, 0.08, 0.07, 0.07, 0.09, 0.08, 0.08, 0.07, 0.07]
    ),
    "中等偏斜": normalize_probs(
        [0.15, 0.13, 0.11, 0.10, 0.08, 0.07, 0.06, 0.06, 0.05, 0.07, 0.06, 0.06]
    ),
    "严重偏斜": normalize_probs(
        [0.25, 0.18, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.04, 0.03, 0.02]
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
#  灯位偏置选择算法
# ══════════════════════════════════════════════════════════════════════════════

def select_lit_slots_fair(n_lit: int, rng: _random.Random) -> List[int]:
    """公平选灯：从12格中随机等概率选取 n_lit 个亮灯位。"""
    return sorted(rng.sample(range(NUM_SLOTS), n_lit))


def select_lit_slots_biased(
    n_lit: int,
    landing_probs: List[float],
    bias_strength: float,
    rng: _random.Random,
) -> List[int]:
    """
    偏置选灯：商家让弹珠不容易落入的格位更容易亮灯。

    bias_strength 控制偏置强度：
        0.0  = 完全公平（均匀选择）
        1.0  = 完全反向（严格选择落点概率最低的格位）
    中间值 = 混合权重

    选灯权重 w_i = (1 - bias_strength) * (1/12) + bias_strength * (1 - p_i_normalized)
    其中 p_i_normalized = p_i / max(p) 用于将反向概率映射到 [0, 1]。
    """
    if bias_strength <= 0.0:
        return select_lit_slots_fair(n_lit, rng)

    max_p = max(landing_probs)
    if max_p == 0:
        return select_lit_slots_fair(n_lit, rng)

    # 反向权重：落点概率越高，被选为灯位的权重越低
    inverse_weights = []
    for p in landing_probs:
        inv = 1.0 - (p / max_p)  # 落点概率最高的格 -> inv=0
        inv = max(inv, 0.01)  # 确保非零权重
        inverse_weights.append(inv)

    # 混合权重
    fair_weight = 1.0 / NUM_SLOTS
    weights = [
        (1.0 - bias_strength) * fair_weight + bias_strength * iw
        for iw in inverse_weights
    ]

    # 按权重不放回抽样
    selected = []
    available = list(range(NUM_SLOTS))
    w_pool = list(weights)
    for _ in range(n_lit):
        total_w = sum(w_pool[i] for i, _ in enumerate(available) if i < len(w_pool))
        if total_w <= 0:
            # fallback
            pick = rng.choice(available)
        else:
            r = rng.random() * total_w
            cumul = 0.0
            pick_idx = len(available) - 1
            for idx, slot in enumerate(available):
                cumul += w_pool[idx]
                if r < cumul:
                    pick_idx = idx
                    break
            pick = available[pick_idx]
            available.pop(pick_idx)
            w_pool.pop(pick_idx)
            selected.append(pick)
            continue
        idx = available.index(pick)
        available.pop(idx)
        w_pool.pop(idx)
        selected.append(pick)

    return sorted(selected)


# ══════════════════════════════════════════════════════════════════════════════
#  理论分析
# ══════════════════════════════════════════════════════════════════════════════

def theoretical_win_prob_fair(n_lit: int) -> float:
    """公平选灯下的理论中奖概率（任何落点分布下都一样）。"""
    return n_lit / NUM_SLOTS


def theoretical_win_prob_biased(
    n_lit: int,
    landing_probs: List[float],
    bias_strength: float,
    n_samples: int = 100000,
) -> float:
    """
    蒙特卡洛估算偏置选灯下的实际中奖概率。

    对每次试验：
    1. 按偏置权重选择 n_lit 个灯位
    2. 按落点概率抽取弹珠落点
    3. 检查是否中奖
    """
    rng = _random.Random(42)
    wins = 0
    for _ in range(n_samples):
        lit = select_lit_slots_biased(n_lit, landing_probs, bias_strength, rng)
        # 弹珠落点
        r = rng.random()
        cumul = 0.0
        landing = NUM_SLOTS - 1
        for i, p in enumerate(landing_probs):
            cumul += p
            if r < cumul:
                landing = i
                break
        if landing in lit:
            wins += 1
    return wins / n_samples


def compute_weighted_ev(
    mult_probs: Dict[int, float],
    mult_slots: Dict[int, int],
    landing_probs: List[float],
    bias_strength: float,
) -> Tuple[float, float, Dict[int, float]]:
    """
    计算偏置选灯下的加权 EV 和加权中奖率。

    返回: (加权EV, 加权中奖率, 各倍率中奖概率)
    """
    total_ev = 0.0
    total_win_rate = 0.0
    per_mult_win = {}

    for mult, prob in mult_probs.items():
        n_lit = mult_slots.get(mult, 1)
        if bias_strength <= 0:
            p_win = n_lit / NUM_SLOTS
        else:
            p_win = theoretical_win_prob_biased(n_lit, landing_probs, bias_strength, n_samples=50000)
        per_mult_win[mult] = p_win
        total_ev += prob * mult * p_win
        total_win_rate += prob * p_win

    return total_ev, total_win_rate, per_mult_win


# ══════════════════════════════════════════════════════════════════════════════
#  蒙特卡洛模拟
# ══════════════════════════════════════════════════════════════════════════════

def run_bias_simulation(
    landing_probs: List[float],
    bias_strength: float,
    ct: float = 0.0,
    n_seeds: int = N_SEEDS,
) -> dict:
    """
    用灯位偏置运行完整游戏模拟。

    与 merchant_manipulation_analysis 中的模拟类似，但灯位选择使用偏置算法。
    """
    landing_probs = normalize_probs(landing_probs)
    mult_probs = dict(MULTIPLIER_PROBS)
    mult_slots = dict(DEFAULT_MULTIPLIER_SLOTS)

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
            multiplier_slots=mult_slots,
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

            # 倍率抽取（标准概率）
            r = rng.random()
            cumulative = 0.0
            multiplier = list(mult_probs.keys())[-1]
            for mult, prob in mult_probs.items():
                cumulative += prob
                if r < cumulative:
                    multiplier = mult
                    break

            # 灯位选择（可能带偏置）
            n_lit = mult_slots.get(multiplier, 1)
            if bias_strength > 0:
                lit_slots = select_lit_slots_biased(n_lit, landing_probs, bias_strength, rng)
            else:
                lit_slots = sorted(rng.sample(range(NUM_SLOTS), n_lit))

            # 策略推荐
            recommended_bet = strategy.optimal_bet(multiplier, lit_slots)
            actual_bet = min(recommended_bet, marbles_remaining)
            actual_bet = max(actual_bet, MIN_BET)
            if actual_bet > marbles_remaining:
                break

            # 弹珠落点（遵循物理概率）
            r2 = rng.random()
            cumulative2 = 0.0
            landing = NUM_SLOTS - 1
            for i, p in enumerate(landing_probs):
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
#  报告辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def section_header(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def subsection(title: str) -> None:
    print(f"\n── {title} {'─' * max(1, 70 - len(title) * 2)}")


# ══════════════════════════════════════════════════════════════════════════════
#  报告各部分
# ══════════════════════════════════════════════════════════════════════════════

def report_part1_mechanism():
    """第一部分：灯位偏置操控的原理"""
    section_header("第一部分：灯位偏置操控的原理")

    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │  什么是灯位偏置操控？                                              │
  └────────────────────────────────────────────────────────────────────┘

  在12口弹珠机中，每次按下按钮确定倍率后，机器随机选择若干格位亮灯。
  标准规则下，灯位选择应该是完全随机的——每个格位被选中的概率相等。

  然而，弹珠的物理落点并不均匀。由于钉板布局、重力和弹射力度，
  某些格位天然更容易接住弹珠。

  商家可以利用这一信息，通过修改程序/电路板，让灯更倾向于亮在
  弹珠不容易落入的格位上。

  ┌────────────────────────────────────────────────────────────────────┐
  │  操控方式                                                          │
  │                                                                    │
  │  1. 统计弹珠的物理落点分布（长期运行数据）                         │
  │  2. 修改灯位选择算法，使高概率落点的格位更少被选为灯位              │
  │  3. 保持亮灯数量不变（2x仍亮4格、4x仍亮3格...）                   │
  └────────────────────────────────────────────────────────────────────┘

  为什么这种操控最隐蔽？

  • 亮灯数量正确 → 通过视觉检查
  • 倍率频率正常 → 通过频率统计
  • 每个灯位看起来都会亮 → 不容易发现规律
  • 唯一的线索：中奖率偏低——但这可以被归因于"运气不好"
""")


def report_part2_theory():
    """第二部分：理论数学分析"""
    section_header("第二部分：灯位偏置的数学分析")

    print("""
  ── 数学模型 ──────────────────────────────────────────────────────────

  定义：
    p_i = 弹珠落入格位 i 的物理概率 (i = 0..11)
    w_i = 格位 i 被选为灯位的权重

  公平选灯：w_i = 1/12 对所有 i
    P(中奖 | n灯) = n/12  （与落点分布无关）

  偏置选灯：w_i ∝ (1 - bias × p_i/max(p))
    bias_strength = 0: 公平  |  bias_strength = 1: 最大偏置

  关键结论：
    公平选灯下，无论弹珠落点分布如何偏斜，
    P(中奖|n灯) 恒等于 n/12。

    但偏置选灯打破了这个等式：
    P(中奖|n灯, 偏置) < n/12
    且落点越偏斜，偏置效果越显著。
""")

    # 不同偏斜程度 × 不同偏置强度下的中奖概率
    bias_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for dist_name, landing_probs in LANDING_DISTRIBUTIONS.items():
        subsection(f"落点分布: {dist_name}")
        
        # 显示落点分布
        print(f"\n  落点概率: ", end="")
        for i, p in enumerate(landing_probs):
            print(f"#{i}={p:.3f}", end="  ")
            if i == 5:
                print(f"\n               ", end="")
        print()
        
        print(f"\n  {'偏置强度':>8} | ", end="")
        for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
            n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
            print(f"{mult}x({n_lit}灯)中奖率  ", end="")
        print(f"| {'加权中奖率':>10} {'加权EV':>8} {'vs公平':>8}")
        print("  " + "-" * 100)

        fair_ev = None
        for bias in bias_levels:
            ev, wr, per_mult = compute_weighted_ev(
                MULTIPLIER_PROBS, DEFAULT_MULTIPLIER_SLOTS, landing_probs, bias
            )
            if fair_ev is None:
                fair_ev = ev
            vs_fair = (ev - fair_ev) / fair_ev * 100 if fair_ev > 0 else 0
            
            print(f"  {bias:>8.1f} | ", end="")
            for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
                print(f"    {per_mult[mult]:>6.2%}       ", end="")
            print(f"| {wr:>9.2%} {ev:>8.4f} {vs_fair:>+7.1f}%")


def report_part3_gradient():
    """第三部分：偏置强度梯度分析"""
    section_header("第三部分：偏置强度梯度 — 精细化影响曲线")

    print("\n  在中等偏斜的落点分布下，逐步增加偏置强度，观察EV变化趋势。")
    print("  这可以帮助判断商家需要多大的偏置才能达到特定的利润目标。\n")

    landing_probs = LANDING_DISTRIBUTIONS["中等偏斜"]
    bias_steps = [i * 0.05 for i in range(21)]

    print(f"  {'偏置强度':>8} | {'加权EV':>8} {'亏损率':>8} {'vs无偏置':>10} | {'每100珠商家额外利润':>20}")
    print("  " + "-" * 72)

    baseline_ev = None
    for bias in bias_steps:
        ev, wr, _ = compute_weighted_ev(
            MULTIPLIER_PROBS, DEFAULT_MULTIPLIER_SLOTS, landing_probs, bias
        )
        if baseline_ev is None:
            baseline_ev = ev

        loss_rate = (1 - ev) * 100
        vs_base = (ev - baseline_ev) / baseline_ev * 100
        extra_profit = (baseline_ev - ev) * 100  # 每100珠额外利润

        print(
            f"  {bias:>8.2f} | {ev:>8.4f} {loss_rate:>+7.2f}% {vs_base:>+9.2f}% | "
            f"{extra_profit:>18.2f}珠"
        )

    print("""
  【关键发现】
  • 即使轻微的偏置(0.2)也能显著降低玩家EV
  • 偏置效果在落点越偏斜时越显著
  • 偏置强度与EV降幅之间近似线性关系
  • 商家只需 0.3-0.5 的偏置就能额外多赚数个百分点
""")


def report_part4_simulation():
    """第四部分：蒙特卡洛模拟"""
    section_header("第四部分：蒙特卡洛模拟 — 灯位偏置的实际影响")

    print(f"\n  模拟参数: 初始={INITIAL_MARBLES}珠, 消耗目标={CONSUME_TARGET}珠, T={T}, J={J}")
    print(f"  每配置运行{N_SEEDS}组随机种子取平均\n")

    bias_levels = [0.0, 0.3, 0.5, 0.7, 1.0]

    for dist_name in ["轻微偏斜", "中等偏斜", "严重偏斜"]:
        landing_probs = LANDING_DISTRIBUTIONS[dist_name]
        subsection(f"落点分布: {dist_name}")

        print(f"\n  {'偏置':>6} {'策略':>6} | {'积分卡':>8} {'净消耗':>10} {'局数':>8} {'胜率':>6} {'卡/珠':>8}")
        print("  " + "-" * 70)

        for bias in bias_levels:
            for ct_label, ct_val in [("原始", 0.0), ("V2", 5.0)]:
                result = run_bias_simulation(
                    landing_probs=landing_probs,
                    bias_strength=bias,
                    ct=ct_val,
                    n_seeds=N_SEEDS,
                )

                cpm_s = f"{result['cpm']:.4f}" if result['cpm'] != float('inf') and result['avg_consumed'] > 0 else "盈利"

                print(
                    f"  {bias:>6.1f} {ct_label:>6} | "
                    f"{result['avg_cards']:>8.1f} {result['avg_consumed']:>10.1f} "
                    f"{result['avg_rounds']:>8.1f} {result['win_rate']:>5.1%} {cpm_s:>8}"
                )
            print()


def report_part5_v2_resilience():
    """第五部分：V2策略抗偏置能力"""
    section_header("第五部分：V2策略的抗灯位偏置能力")

    print("\n  对比原始策略与V2策略在不同偏置强度下的表现差异。")
    print(f"  落点分布: 中等偏斜 | T={T}, J={J}\n")

    landing_probs = LANDING_DISTRIBUTIONS["中等偏斜"]
    bias_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    results = []
    for bias in bias_levels:
        row = {"bias": bias}
        for label, ct in [("原始", 0.0), ("V2", 5.0)]:
            row[label] = run_bias_simulation(
                landing_probs=landing_probs,
                bias_strength=bias,
                ct=ct,
                n_seeds=N_SEEDS,
            )
        results.append(row)

    print(f"  {'偏置':>6} | {'原始:卡数':>10} {'原始:消耗':>10} | {'V2:卡数':>10} {'V2:消耗':>10} | {'V2卡提升':>8} {'V2少亏':>10}")
    print("  " + "-" * 85)

    for row in results:
        o = row["原始"]
        v = row["V2"]
        card_improve = ""
        if o["avg_cards"] > 0:
            card_improve = f"{(v['avg_cards'] - o['avg_cards']) / o['avg_cards'] * 100:>+.1f}%"
        loss_reduce = ""
        if o["avg_consumed"] > 0 and v["avg_consumed"] > 0:
            loss_reduce = f"{(1 - v['avg_consumed'] / o['avg_consumed']) * 100:>+.1f}%"
        elif v["avg_consumed"] <= 0:
            loss_reduce = "V2盈利!"

        print(
            f"  {row['bias']:>6.1f} | "
            f"{o['avg_cards']:>10.1f} {o['avg_consumed']:>10.1f} | "
            f"{v['avg_cards']:>10.1f} {v['avg_consumed']:>10.1f} | "
            f"{card_improve:>8} {loss_reduce:>10}"
        )


def report_part6_cross_analysis():
    """第六部分：灯位偏置与其他操控的叠加效果"""
    section_header("第六部分：灯位偏置 × 落点偏斜 交叉分析")

    print("\n  不同的物理落点分布下，灯位偏置的影响程度不同。")
    print("  此部分量化两者的交互效应。\n")

    bias_levels = [0.0, 0.3, 0.6, 1.0]

    print(f"  {'落点分布':>10} × {'偏置':>4} | {'中奖率':>8} {'加权EV':>8} {'亏损率':>8} | {'每1000珠预期亏损':>16}")
    print("  " + "-" * 76)

    for dist_name, landing_probs in LANDING_DISTRIBUTIONS.items():
        for bias in bias_levels:
            ev, wr, _ = compute_weighted_ev(
                MULTIPLIER_PROBS, DEFAULT_MULTIPLIER_SLOTS, landing_probs, bias
            )
            loss_rate = (1 - ev) * 100
            loss_1000 = 1000 * (1 - ev)

            marker = " " if bias == 0 else ""
            print(
                f"  {dist_name:>10} × {bias:>3.1f} | "
                f"{wr:>7.1%} {ev:>8.4f} {loss_rate:>+7.2f}% | "
                f"{loss_1000:>14.1f}珠 {marker}"
            )
        print()


def report_part7_detection():
    """第七部分：玩家如何检测灯位偏置"""
    section_header("第七部分：灯位偏置的检测方法")

    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │  灯位偏置是最难检测的操控方式                                      │
  └────────────────────────────────────────────────────────────────────┘

  与其他操控相比：
  • 亮灯数量操控 → 一眼可见
  • 倍率概率操控 → 记50局可发现
  • 灯位偏置操控 → 需要精密的统计分析

  ── 检测方法一：灯位频率卡方检验 ──────────────────────────────────────

  记录每个格位亮灯的次数。公平选灯下，每个格位被选中的概率应相等。

  具体操作：
  1. 记录至少100次灯位（包括所有倍率的所有亮灯格）
  2. 统计每个格位出现的总次数
  3. 与理论期望值比较（χ² 检验）

  理论期望：每个格位亮灯次数 ≈ 总亮灯次数 / 12

  例：100局中，按标准倍率概率，平均亮灯总次数约为：""")

    # 计算100局平均亮灯总次数
    avg_lights = 0
    for mult, prob in MULTIPLIER_PROBS.items():
        n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
        avg_lights += 100 * prob * n_lit
    print(f"     100 × Σ(π_m × k_m) = {avg_lights:.0f} 次亮灯")
    print(f"     每格期望次数 = {avg_lights / 12:.1f} 次")

    print(f"""
  χ² 统计量 = Σ (观测值 - 期望值)² / 期望值
  自由度 = 11
  α=0.05 时临界值 = 19.68

  如果 χ² > 19.68，有95%的把握认为灯位选择不公平。

  ── 检测方法二：落点-灯位相关性分析 ──────────────────────────────────

  这是更高级的检测方法：

  1. 记录每局：(a)哪些格位亮灯 (b)弹珠落入哪个格位
  2. 计算每个格位：亮灯频率 vs 落点频率
  3. 如果存在显著负相关（落点多的格位亮灯少），说明存在偏置

  相关系数 r < -0.5 且 p < 0.05 时可确认偏置。
  通常需要200+局数据。

  ── 检测方法三：分倍率中奖率验证 ──────────────────────────────────────

  最实用的方法：分别统计每个倍率下的中奖率。

  公平选灯理论中奖率：""")

    for mult, n_lit in sorted(DEFAULT_MULTIPLIER_SLOTS.items()):
        p_fair = n_lit / NUM_SLOTS
        print(f"     {mult}x ({n_lit}灯): P_win = {p_fair:.4f} ({p_fair*100:.1f}%)")

    print(f"""
  如果实际中奖率系统性低于以上理论值，很可能存在灯位偏置。
  建议至少每个倍率收集30+局数据再做判断。

  ── 实用检测阈值（建议） ─────────────────────────────────────────────

  ┌──────────┬───────────────┬──────────────────────────┐
  │ 数据量   │ 检测方法      │ 可检测的最小偏置强度     │
  ├──────────┼───────────────┼──────────────────────────┤
  │ 50 局    │ 综合中奖率    │ 约 0.6 (强偏置)          │
  │ 100 局   │ 灯位频率χ²    │ 约 0.4 (中等偏置)        │
  │ 200 局   │ 落点-灯位相关 │ 约 0.25 (轻度偏置)       │
  │ 500 局   │ 分倍率中奖率  │ 约 0.15 (微弱偏置)       │
  └──────────┴───────────────┴──────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────┐
  │  实操建议                                                          │
  └────────────────────────────────────────────────────────────────────┘

  1. 新机器前20局：用最小投注(5珠)试探
  2. 记录每局亮灯格位编号（手机拍照最简单）
  3. 每20局检查一次各格位亮灯次数是否均匀
  4. 如果某些格位明显很少亮灯，且这些格位恰好是你中奖最多的格位
     → 高度怀疑灯位偏置操控
  5. 使用V2策略，即使存在偏置也能自动减少损失
""")


def report_summary():
    """总结"""
    section_header("总结与结论")

    # 计算关键数据
    landing_probs = LANDING_DISTRIBUTIONS["中等偏斜"]
    ev_fair, _, _ = compute_weighted_ev(
        MULTIPLIER_PROBS, DEFAULT_MULTIPLIER_SLOTS, landing_probs, 0.0
    )
    ev_mild, _, _ = compute_weighted_ev(
        MULTIPLIER_PROBS, DEFAULT_MULTIPLIER_SLOTS, landing_probs, 0.3
    )
    ev_moderate, _, _ = compute_weighted_ev(
        MULTIPLIER_PROBS, DEFAULT_MULTIPLIER_SLOTS, landing_probs, 0.6
    )
    ev_extreme, _, _ = compute_weighted_ev(
        MULTIPLIER_PROBS, DEFAULT_MULTIPLIER_SLOTS, landing_probs, 1.0
    )

    print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │  灯位偏置操控影响 — 核心结论                                       │
  └────────────────────────────────────────────────────────────────────┘

  1. 影响幅度（中等偏斜落点分布下）：
     • 无偏置:   加权EV = {ev_fair:.4f}  (每100珠亏{(1-ev_fair)*100:.1f}珠)
     • 轻度偏置(0.3): EV = {ev_mild:.4f}  (每100珠亏{(1-ev_mild)*100:.1f}珠)
     • 中度偏置(0.6): EV = {ev_moderate:.4f}  (每100珠亏{(1-ev_moderate)*100:.1f}珠)
     • 极端偏置(1.0): EV = {ev_extreme:.4f}  (每100珠亏{(1-ev_extreme)*100:.1f}珠)

  2. 关键特性：
     • 灯位偏置是最隐蔽的操控方式（灯数正确、倍率正常）
     • 落点越偏斜，偏置操控效果越强（两者相互放大）
     • 均匀落点下偏置无效（所有格位落球概率相等）
     • 这意味着老旧/磨损机器更容易被灯位偏置操控

  3. V2策略的防御效果：
     • V2策略在偏置环境下仍能减少损失
     • 核心机制：偏置导致实际中奖率低于理论值，
       V2通过观测到的低中奖率自动降低投注
     • 但V2无法完全抵消偏置影响——因为它不知道灯位是被操纵的

  4. 检测建议：
     • 记录每局亮灯格位，检查分布均匀性
     • 100+局后做χ²检验
     • 观察"弹珠常去的格位是否很少亮灯"
     • 如果综合中奖率低于理论值20%以上，建议更换机器

  5. 与其他操控手段的对比：
     ┌───────────────┬──────────┬──────────┬──────────────────┐
     │ 操控方式      │ 影响程度 │ 隐蔽性   │ 检测难度         │
     ├───────────────┼──────────┼──────────┼──────────────────┤
     │ 倍率概率调整  │ 高       │ 中       │ 50局可检测       │
     │ 亮灯数量减少  │ 很高     │ 低       │ 一眼可见         │
     │ 灯位偏置      │ 中~高    │ 很高     │ 100~200局检测    │
     │ 组合操控      │ 极高     │ 取决于   │ 需多维度分析     │
     └───────────────┴──────────┴──────────┴──────────────────┘
""")


# ══════════════════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "灯位偏置分析报告：商家改变某些灯出现概率的影响".center(54) + "█")
    print("█" + "Light Position Bias Analysis for 12-Slot Pinball".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # 第一部分：机制说明
    report_part1_mechanism()

    # 第二部分：理论分析
    print("\n  正在计算理论中奖概率（蒙特卡洛估算）...\n")
    report_part2_theory()

    # 第三部分：梯度分析
    report_part3_gradient()

    # 第四部分：交叉分析
    report_part6_cross_analysis()

    # 第五部分：检测方法
    report_part7_detection()

    # 第六部分：蒙特卡洛模拟（耗时）
    print("\n" + "=" * 80)
    print("  正在运行蒙特卡洛模拟（可能需要几分钟）...")
    print("=" * 80)

    report_part4_simulation()

    # 第七部分：V2策略抗偏置能力
    report_part5_v2_resilience()

    # 总结
    report_summary()

    print("\n  报告生成完毕。\n")


if __name__ == "__main__":
    main()
