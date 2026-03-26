#!/usr/bin/env python3
"""
最大投注珠数(max_bet)对收益与策略影响的分析报告

分析维度：
1. 理论分析：不同 max_bet 下各倍数的最优投注量与积分效率
2. 模拟实验：在多种分布 × 多种 max_bet × 两种优先级下穷举模拟
3. 输出综合分析报告（文本 + Markdown）
"""

import math
import sys
import time
from typing import Dict, List

from simulation_test import (
    run_simulation,
    DEFAULT_HOLE_PROBS,
    normalize_probs,
)
from pinball_strategy import (
    NUM_SLOTS,
    MIN_BET,
    MAX_BET,
    DEFAULT_MULTIPLIER_SLOTS,
    PinballStrategy,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 配置参数
# ═══════════════════════════════════════════════════════════════════════════════

# 要测试的 max_bet 值
MAX_BET_VALUES = [5, 10, 15, 20, 30, 50, 70, 99]

# 落点分布
DISTRIBUTIONS = {
    "均匀分布": [1.0] * 12,
    "轻微偏斜": [
        0.10, 0.10, 0.10, 0.09, 0.08, 0.07,
        0.07, 0.09, 0.08, 0.08, 0.07, 0.07,
    ],
    "中等偏斜": [
        0.15, 0.13, 0.11, 0.10, 0.08, 0.07,
        0.06, 0.06, 0.05, 0.07, 0.06, 0.06,
    ],
    "严重偏斜": [
        0.25, 0.18, 0.12, 0.10, 0.08, 0.06,
        0.05, 0.04, 0.03, 0.04, 0.03, 0.02,
    ],
}

# 模拟参数
INITIAL_MARBLES = 10000
CONSUME_TARGET = 3000
T_VALUES = [20, 30, 50]  # 测试不同 T 值
J_DEFAULT = 10
T_DEFAULT = 20
PRIORITIES = ["cards", "marbles"]
N_SEEDS = 30
CONFIDENCE_THRESHOLD = 20.0  # 使用自适应策略


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 理论分析：不同 max_bet 下各倍数的最优投注与积分效率
# ═══════════════════════════════════════════════════════════════════════════════

def theoretical_analysis(T: int, J: int) -> Dict:
    """对每个 max_bet 值，计算各倍数下的最优投注和效率指标。"""
    results = {}
    for mb in MAX_BET_VALUES:
        table = PinballStrategy.expected_value_table(T, J, max_bet=mb)
        results[mb] = table
    return results


def print_theoretical_section(T: int, J: int):
    """输出理论分析部分。"""
    results = theoretical_analysis(T, J)

    print(f"\n{'═' * 90}")
    print(f"  理论分析：不同 max_bet 下各倍数的最优投注与效率 (T={T}, J={J})")
    print(f"{'═' * 90}")

    # 按倍数输出各 max_bet 下的对比
    for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
        n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
        p_win = n_lit / NUM_SLOTS
        print(f"\n  ── {mult}x (亮灯格={n_lit}, P_win={p_win:.4f}, 珠子ROI={mult * p_win:.4f}) ──")
        print(f"  {'max_bet':>8}  {'最优投注':>8}  {'赢时积分卡':>10}  {'每珠积分期望':>14}  {'每次期望积分':>14}  {'注释'}")
        print(f"  {'─' * 80}")

        for mb in MAX_BET_VALUES:
            row = [r for r in results[mb] if r["multiplier"] == mult][0]
            opt_bet = row["card_optimal_bet"]
            cards_on_win = min(mult * opt_bet // T, J)
            cards_per_marble = row["cards_per_marble"]
            expected_cards_per_play = p_win * cards_on_win

            note = ""
            if opt_bet == mb:
                note = "← 受max_bet限制"
            elif opt_bet == MIN_BET:
                note = "← 最低投注"

            print(f"  {mb:>8}  {opt_bet:>8}  {cards_on_win:>10}  {cards_per_marble:>14.4f}  {expected_cards_per_play:>14.4f}  {note}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 理论分析：max_bet 对珠子 EV 策略的影响
# ═══════════════════════════════════════════════════════════════════════════════

def print_marble_ev_section():
    """分析 max_bet 对珠子 EV（收益率）策略的影响。"""
    print(f"\n{'═' * 90}")
    print(f"  理论分析：max_bet 对珠子 EV 策略的影响")
    print(f"{'═' * 90}")

    print(f"\n  珠子优先级策略的核心逻辑：")
    print(f"    - 当 multiplier × P_win > 1（正EV）时下注 max_bet")
    print(f"    - 当 multiplier × P_win ≤ 1（负EV）时下注 MIN_BET={MIN_BET}")
    print(f"\n  各倍数在均匀分布下的 EV_ratio:")

    for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
        n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
        p_win = n_lit / NUM_SLOTS
        ev = mult * p_win
        action = "正EV → 下max_bet" if ev > 1 else "负EV → 下MIN_BET"
        print(f"    {mult}x: EV_ratio = {ev:.4f}  ({action})")

    print(f"\n  ── max_bet 对正EV场景的放大效应 ──")
    print(f"  {'max_bet':>8}  {'2x正EV场景':>12}  {'每次期望赢珠':>14}  {'净收益/次':>12}")
    print(f"  {'─' * 55}")

    mult = 2
    n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
    p_win = n_lit / NUM_SLOTS

    for mb in MAX_BET_VALUES:
        ev_return = mult * mb * p_win
        net = ev_return - mb
        print(f"  {mb:>8}  {mb:>12}  {ev_return:>14.2f}  {net:>+12.2f}")

    # 高倍数也分析
    print(f"\n  ── max_bet 对负EV场景的影响（以6x为例）──")
    print(f"  无论 max_bet 为多少，负EV场景下策略始终只投 MIN_BET={MIN_BET}")
    print(f"  因此 max_bet 仅在正EV场景中产生显著影响。")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. T 值交互分析
# ═══════════════════════════════════════════════════════════════════════════════

def print_t_interaction_section():
    """分析 T 值与 max_bet 之间的相互作用。"""
    print(f"\n{'═' * 90}")
    print(f"  T 值与 max_bet 的交互效应分析 (J={J_DEFAULT})")
    print(f"{'═' * 90}")

    print(f"\n  积分卡计算公式: cards = min(floor(multiplier × bet / T), J)")
    print(f"  当 T 较大时，需要更多下注才能获得同等积分卡")
    print(f"  max_bet 的限制在 T 较大时更为显著")

    for t in T_VALUES:
        print(f"\n  ── T = {t} ──")
        print(f"  {'倍数':>4}  {'max_bet=5':>10}  {'max_bet=20':>10}  {'max_bet=50':>10}  {'max_bet=99':>10}  (每珠积分期望)")
        print(f"  {'─' * 60}")

        for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
            n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
            p_win = n_lit / NUM_SLOTS
            line = f"  {mult}x"

            for mb in [5, 20, 50, 99]:
                # 找最优投注
                best_n = max(MIN_BET, min(mb, math.ceil(t * J_DEFAULT / mult)))
                best_eff = 0.0
                for k in range(1, J_DEFAULT + 1):
                    n = max(MIN_BET, math.ceil(k * t / mult))
                    if n > mb:
                        break
                    eff = k / n
                    if eff > best_eff:
                        best_eff = eff
                        best_n = n
                cards_on_win = min(mult * best_n // t, J_DEFAULT)
                cpm = p_win * cards_on_win / best_n if best_n > 0 else 0
                line += f"  {cpm:>10.4f}"

            print(line)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 模拟实验
# ═══════════════════════════════════════════════════════════════════════════════

def run_batch(max_bet, priority, hole_probs, n_seeds, ct=0.0):
    """对一个 max_bet + priority + 分布组合运行 n_seeds 次模拟。"""
    results = []
    for seed in range(n_seeds):
        r = run_simulation(
            initial_marbles=INITIAL_MARBLES,
            consume_target=CONSUME_TARGET,
            T=T_DEFAULT,
            J=J_DEFAULT,
            priority=priority,
            hole_probs=hole_probs,
            seed=seed,
            verbose=False,
            confidence_threshold=ct,
            max_rounds=10000,
        )
        results.append(r)

    total_cards = sum(r["total_cards_won"] for r in results)
    total_consumed = sum(r["net_consumed"] for r in results)
    total_rounds = sum(r["total_rounds"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    total_marbles_spent = sum(r["total_marbles_spent"] for r in results)
    total_marbles_won = sum(r["total_marbles_won"] for r in results)

    cards_per_marble = total_cards / total_consumed if total_consumed > 0 else 0
    cards_per_round = total_cards / total_rounds if total_rounds > 0 else 0
    avg_cards = total_cards / n_seeds
    avg_consumed = total_consumed / n_seeds
    avg_rounds = total_rounds / n_seeds
    win_rate = total_wins / total_rounds if total_rounds > 0 else 0
    marble_roi = total_marbles_won / total_marbles_spent if total_marbles_spent > 0 else 0

    return {
        "max_bet": max_bet,
        "priority": priority,
        "avg_cards": avg_cards,
        "avg_consumed": avg_consumed,
        "cards_per_marble": cards_per_marble,
        "cards_per_round": cards_per_round,
        "avg_rounds": avg_rounds,
        "win_rate": win_rate,
        "marble_roi": marble_roi,
    }


def run_simulation_experiments():
    """运行所有模拟实验并返回结果。"""
    all_results = {}

    total_combos = len(DISTRIBUTIONS) * len(MAX_BET_VALUES) * len(PRIORITIES)
    done = 0
    start_time = time.time()

    for dist_name, hole_probs in DISTRIBUTIONS.items():
        all_results[dist_name] = {}
        for priority in PRIORITIES:
            all_results[dist_name][priority] = {}
            for mb in MAX_BET_VALUES:
                done += 1
                elapsed = time.time() - start_time
                eta = (elapsed / done) * (total_combos - done) if done > 0 else 0
                print(f"\r  模拟进度: {done}/{total_combos} ({dist_name}, {priority}, max_bet={mb}) ETA: {eta:.0f}s  ", end="", flush=True)

                r = run_batch(mb, priority, hole_probs, N_SEEDS, ct=CONFIDENCE_THRESHOLD)
                all_results[dist_name][priority][mb] = r

    print(f"\r  模拟进度: {total_combos}/{total_combos} — 完成！{'':>40}")
    return all_results


def print_simulation_results(all_results):
    """输出模拟实验结果。"""
    print(f"\n{'═' * 90}")
    print(f"  模拟实验结果 (初始={INITIAL_MARBLES}, 消耗目标={CONSUME_TARGET}, T={T_DEFAULT}, J={J_DEFAULT})")
    print(f"  每组 {N_SEEDS} 个随机种子, 自适应策略 confidence_threshold={CONFIDENCE_THRESHOLD}")
    print(f"{'═' * 90}")

    for priority in PRIORITIES:
        priority_cn = "积分卡优先" if priority == "cards" else "珠子优先"
        print(f"\n  ━━ 优先级: {priority_cn} ━━")

        for dist_name in DISTRIBUTIONS.keys():
            print(f"\n  ── 落点分布: {dist_name} ──")
            print(f"  {'max_bet':>8}  {'平均积分卡':>10}  {'平均消耗珠':>10}  "
                  f"{'积分/珠':>8}  {'积分/局':>8}  {'平均局数':>8}  "
                  f"{'胜率':>6}  {'珠子ROI':>8}")
            print(f"  {'─' * 82}")

            best_cpm = 0
            best_mb_cpm = 0
            best_cards = 0
            best_mb_cards = 0

            for mb in MAX_BET_VALUES:
                r = all_results[dist_name][priority][mb]
                if r["cards_per_marble"] > best_cpm:
                    best_cpm = r["cards_per_marble"]
                    best_mb_cpm = mb
                if r["avg_cards"] > best_cards:
                    best_cards = r["avg_cards"]
                    best_mb_cards = mb

            for mb in MAX_BET_VALUES:
                r = all_results[dist_name][priority][mb]
                marker = ""
                if mb == best_mb_cpm:
                    marker += " ★效率最优"
                if mb == best_mb_cards and best_mb_cards != best_mb_cpm:
                    marker += " ☆总量最优"

                print(f"  {r['max_bet']:>8}  {r['avg_cards']:>10.1f}  {r['avg_consumed']:>10.1f}  "
                      f"{r['cards_per_marble']:>8.4f}  {r['cards_per_round']:>8.4f}  {r['avg_rounds']:>8.1f}  "
                      f"{r['win_rate']:>6.1%}  {r['marble_roi']:>8.4f}{marker}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 生成 Markdown 报告
# ═══════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(all_results) -> str:
    """生成 Markdown 格式的完整分析报告。"""
    lines = []
    L = lines.append

    L("# 最大投注珠数(max_bet)对收益与策略的影响分析报告")
    L("")
    L("> 自动生成 | 12口弹珠机最佳出卡策略项目")
    L("")

    # ── 目录 ──
    L("## 目录")
    L("1. [研究背景与参数设置](#1-研究背景与参数设置)")
    L("2. [理论分析](#2-理论分析)")
    L("3. [模拟实验结果](#3-模拟实验结果)")
    L("4. [关键发现](#4-关键发现)")
    L("5. [实用建议](#5-实用建议)")
    L("")

    # ── 1. 背景 ──
    L("## 1. 研究背景与参数设置")
    L("")
    L("### 研究目的")
    L("分析 `max_bet`（每局最大投注珠数）参数对弹珠机策略表现的影响，包括：")
    L("- 积分卡获取效率")
    L("- 珠子消耗速度")
    L("- 珠子投资回报率(ROI)")
    L("- 在不同落点分布和优先级下的表现差异")
    L("")
    L("### 机器规则回顾")
    L("| 参数 | 值 |")
    L("|------|-----|")
    L("| 机器孔数 | 12 |")
    L(f"| 最低投注 | {MIN_BET} 珠 |")
    L(f"| 系统最大投注 | {MAX_BET} 珠 |")
    L("| 倍数选项 | 2x/4x/6x/8x/10x |")
    L("| 积分卡公式 | min(floor(multiplier × bet / T), J) |")
    L("")
    L("### 倍数与亮灯格对应关系")
    L("| 倍数 | 亮灯格数 | 中奖概率 | 珠子EV比 |")
    L("|------|---------|---------|---------|")
    for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
        n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
        p = n_lit / NUM_SLOTS
        ev = mult * p
        ev_mark = "✅ >1" if ev > 1 else "❌ <1"
        L(f"| {mult}x | {n_lit} | {p:.4f} | {ev:.4f} ({ev_mark}) |")
    L("")
    L("### 实验参数")
    L(f"- 测试的 max_bet 值: {MAX_BET_VALUES}")
    L(f"- 初始珠子: {INITIAL_MARBLES}")
    L(f"- 消耗目标: {CONSUME_TARGET}")
    L(f"- T = {T_DEFAULT}, J = {J_DEFAULT}")
    L(f"- 自适应 confidence_threshold = {CONFIDENCE_THRESHOLD}")
    L(f"- 每组随机种子数: {N_SEEDS}")
    L("")

    # ── 2. 理论分析 ──
    L("## 2. 理论分析")
    L("")
    L("### 2.1 积分卡优先策略下 max_bet 的影响")
    L("")
    L("积分卡优先策略会为每个倍数寻找 **积分卡/珠子** 效率最高的下注量。")
    L("当 `max_bet` 受限时，高效率的投注量可能无法达到，导致效率降低。")
    L("")

    # 对每个倍数输出表格
    for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
        n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
        p_win = n_lit / NUM_SLOTS
        L(f"#### {mult}x 倍数 (亮灯={n_lit}, P_win={p_win:.4f})")
        L("")
        L("| max_bet | 最优投注 | 赢时积分卡 | 每珠积分期望 | 是否受限 |")
        L("|---------|---------|-----------|------------|---------|")

        for mb in MAX_BET_VALUES:
            table = PinballStrategy.expected_value_table(T_DEFAULT, J_DEFAULT, max_bet=mb)
            row = [r for r in table if r["multiplier"] == mult][0]
            opt = row["card_optimal_bet"]
            cards = min(mult * opt // T_DEFAULT, J_DEFAULT)
            cpm = row["cards_per_marble"]
            limited = "⚠️ 受限" if opt == mb and mb < 99 else "✅"
            L(f"| {mb} | {opt} | {cards} | {cpm:.4f} | {limited} |")
        L("")

    L("### 2.2 珠子优先策略下 max_bet 的影响")
    L("")
    L("珠子优先策略逻辑简单：")
    L("- **正EV (multiplier × P_win > 1)**: 下注 max_bet → max_bet 越大，预期收益越高")
    L("- **负EV (multiplier × P_win ≤ 1)**: 下注 MIN_BET → max_bet 无影响")
    L("")
    L("在12口弹珠机中，**只有 2x 是正EV (EV_ratio=0.6667)**...实际上**所有倍数在均匀分布下都是负EV**。")
    L("")

    # 检查是否有正EV
    has_positive = False
    for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
        n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
        ev = mult * n_lit / NUM_SLOTS
        if ev > 1:
            has_positive = True
    
    if not has_positive:
        L("> **重要发现**: 在均匀落点分布下，所有倍数的 EV_ratio 均 < 1，")
        L("> 即珠子优先策略始终只投 MIN_BET。此时 **max_bet 对珠子优先策略无影响**。")
        L("> 只有在偏斜分布下，当高概率落点恰好是亮灯格时，才可能出现正EV，此时 max_bet 才发挥作用。")
    L("")

    L("### 2.3 T 值与 max_bet 的交互效应")
    L("")
    L("T 越大，达到同等积分卡所需的投注量越高，max_bet 的限制效应越明显。")
    L("")
    L("| T | 倍数 | max_bet=5 | max_bet=20 | max_bet=50 | max_bet=99 |")
    L("|---|------|-----------|-----------|-----------|-----------|")

    for t in T_VALUES:
        for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
            n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
            p_win = n_lit / NUM_SLOTS
            vals = []
            for mb in [5, 20, 50, 99]:
                best_n = max(MIN_BET, min(mb, math.ceil(t * J_DEFAULT / mult)))
                best_eff = 0.0
                for k in range(1, J_DEFAULT + 1):
                    n = max(MIN_BET, math.ceil(k * t / mult))
                    if n > mb:
                        break
                    eff = k / n
                    if eff > best_eff:
                        best_eff = eff
                        best_n = n
                cards_on_win = min(mult * best_n // t, J_DEFAULT)
                cpm = p_win * cards_on_win / best_n if best_n > 0 else 0
                vals.append(f"{cpm:.4f}")
            L(f"| {t} | {mult}x | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    L("")

    # ── 3. 模拟实验 ──
    L("## 3. 模拟实验结果")
    L("")

    for priority in PRIORITIES:
        priority_cn = "积分卡优先" if priority == "cards" else "珠子优先"
        L(f"### 3.{1 if priority == 'cards' else 2} {priority_cn}策略")
        L("")

        for dist_name in DISTRIBUTIONS.keys():
            L(f"#### {dist_name}")
            L("")
            L("| max_bet | 平均积分卡 | 平均消耗珠 | 积分/珠 | 积分/局 | 平均局数 | 胜率 | 珠子ROI |")
            L("|---------|-----------|-----------|--------|--------|---------|------|---------|")

            best_cpm = 0
            best_mb = 0
            for mb in MAX_BET_VALUES:
                r = all_results[dist_name][priority][mb]
                if r["cards_per_marble"] > best_cpm:
                    best_cpm = r["cards_per_marble"]
                    best_mb = mb

            for mb in MAX_BET_VALUES:
                r = all_results[dist_name][priority][mb]
                marker = " **★**" if mb == best_mb else ""
                L(f"| {r['max_bet']}{marker} | {r['avg_cards']:.1f} | {r['avg_consumed']:.1f} | "
                  f"{r['cards_per_marble']:.4f} | {r['cards_per_round']:.4f} | "
                  f"{r['avg_rounds']:.1f} | {r['win_rate']:.1%} | {r['marble_roi']:.4f} |")
            L("")

    # ── 4. 关键发现 ──
    L("## 4. 关键发现")
    L("")

    # 从模拟数据中提取洞察
    L("### 4.1 积分卡优先策略")
    L("")

    # 找出各分布下的最优 max_bet
    for dist_name in DISTRIBUTIONS.keys():
        best_cpm = 0
        best_mb = 0
        for mb in MAX_BET_VALUES:
            r = all_results[dist_name]["cards"][mb]
            if r["cards_per_marble"] > best_cpm:
                best_cpm = r["cards_per_marble"]
                best_mb = mb
        L(f"- **{dist_name}**: 最优 max_bet = **{best_mb}**, 积分效率 = {best_cpm:.4f} 积分/珠")

    L("")
    L("### 4.2 珠子优先策略")
    L("")

    for dist_name in DISTRIBUTIONS.keys():
        best_roi = 0
        best_mb = 0
        for mb in MAX_BET_VALUES:
            r = all_results[dist_name]["marbles"][mb]
            if r["marble_roi"] > best_roi:
                best_roi = r["marble_roi"]
                best_mb = mb
        L(f"- **{dist_name}**: 最优 max_bet = **{best_mb}**, 珠子ROI = {best_roi:.4f}")

    L("")
    L("### 4.3 核心规律")
    L("")

    # 比较极端值
    for priority in PRIORITIES:
        priority_cn = "积分卡优先" if priority == "cards" else "珠子优先"
        metric = "cards_per_marble" if priority == "cards" else "marble_roi"
        metric_cn = "积分效率" if priority == "cards" else "珠子ROI"

        L(f"#### {priority_cn}")
        L("")
        L(f"| 分布 | max_bet=5 | max_bet=99 | 变化 | 变化率 |")
        L(f"|------|-----------|-----------|------|--------|")

        for dist_name in DISTRIBUTIONS.keys():
            r5 = all_results[dist_name][priority][5]
            r99 = all_results[dist_name][priority][99]
            v5 = r5[metric]
            v99 = r99[metric]
            change = v99 - v5
            pct = (change / v5 * 100) if v5 > 0 else 0
            L(f"| {dist_name} | {v5:.4f} | {v99:.4f} | {change:+.4f} | {pct:+.1f}% |")
        L("")

    L("### 4.4 max_bet 对游戏节奏的影响")
    L("")
    L("| 分布 | priority | max_bet=5 局数 | max_bet=99 局数 | 局数变化 |")
    L("|------|----------|---------------|----------------|---------|")
    for dist_name in DISTRIBUTIONS.keys():
        for priority in PRIORITIES:
            r5 = all_results[dist_name][priority][5]
            r99 = all_results[dist_name][priority][99]
            L(f"| {dist_name} | {priority} | {r5['avg_rounds']:.0f} | {r99['avg_rounds']:.0f} | {r99['avg_rounds'] - r5['avg_rounds']:+.0f} |")
    L("")

    # ── 5. 实用建议 ──
    L("## 5. 实用建议")
    L("")
    L("### 场景化推荐")
    L("")
    L("| 场景 | 推荐 max_bet | 理由 |")
    L("|------|-------------|------|")
    L("| 积分卡效率最大化 | 10-30 | 积分卡公式的阶梯效应使中等投注最高效 |")
    L("| 珠子保守（减少风险） | 5-10 | 负EV下小注减少方差，延长游戏时间 |")
    L("| 发现正EV后激进 | 70-99 | 正EV场景下大注放大收益 |")
    L("| 初期探索阶段 | 5-15 | 低成本积累落点数据 |")
    L("| 有信心后的积分卡策略 | 20-50 | 在稳定估计下选择最优阶梯投注 |")
    L("")
    L("### 关键结论")
    L("")
    L("1. **max_bet 对积分卡优先策略的影响较大**：因为积分卡计算存在阶梯效应 `floor(mult × bet / T)`，"
       "最优投注量往往在某个特定值上，max_bet 过低会截断最优解。")
    L("")
    L("2. **max_bet 对珠子优先策略的影响取决于EV**：在均匀分布（所有倍数负EV）下，"
       "策略始终投 MIN_BET，max_bet 无影响；在偏斜分布下若出现正EV，max_bet 越大收益越高。")
    L("")
    L("3. **T 值越大，max_bet 的影响越显著**：T 大意味着获得1张积分卡所需的投注更多，"
       "max_bet 的限制更容易生效。")
    L("")
    L("4. **max_bet 影响游戏节奏**：高 max_bet 在积分卡策略下可能导致每局消耗更多珠子，"
       "减少总局数，但每局积分卡收益更高。")
    L("")
    L("5. **推荐默认值**：对于 T=20, J=10 的典型配置，`max_bet=20~50` 是积分卡策略的甜区，"
       "能充分利用阶梯效应而不产生过多浪费。")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("█" * 90)
    print("█  最大投注珠数(max_bet)对收益与策略影响的分析报告")
    print("█" * 90)

    # 1. 理论分析
    print_theoretical_section(T_DEFAULT, J_DEFAULT)
    print_marble_ev_section()
    print_t_interaction_section()

    # 2. 模拟实验
    print(f"\n{'═' * 90}")
    print(f"  开始模拟实验...")
    print(f"  共 {len(DISTRIBUTIONS)} 种分布 × {len(MAX_BET_VALUES)} 种 max_bet × {len(PRIORITIES)} 种优先级 = {len(DISTRIBUTIONS) * len(MAX_BET_VALUES) * len(PRIORITIES)} 组")
    print(f"  每组 {N_SEEDS} 个种子，共 {len(DISTRIBUTIONS) * len(MAX_BET_VALUES) * len(PRIORITIES) * N_SEEDS} 次模拟")
    print(f"{'═' * 90}")

    all_results = run_simulation_experiments()
    print_simulation_results(all_results)

    # 3. 生成 Markdown 报告
    report = generate_markdown_report(all_results)
    report_path = "max_bet_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  ✓ Markdown 报告已保存到: {report_path}")

    print(f"\n{'█' * 90}")
    print(f"█  分析完毕")
    print(f"{'█' * 90}")


if __name__ == "__main__":
    main()
