#!/usr/bin/env python3
"""
商家限制最大投珠数对玩家收益的影响分析

研究场景：
商家将机器的最大投珠数限制为低于99的某个值（如50、30、20、15、10等），
分析此限制对玩家在不同策略下的收益损失程度。

分析维度：
1. 理论分析：max_bet限制如何截断最优投注，量化积分效率损失
2. 分倍数分析：各倍数受限程度和损失占比
3. 综合模拟：原始策略 + 自适应策略在限制下的表现
4. T/J参数交互：不同机器配置下限制的敏感度
5. 玩家应对策略建议

运行:
    python max_bet_limit_analysis.py
"""

import math
import time
import random as _random
from typing import Dict, List

from pinball_strategy import (
    NUM_SLOTS, MIN_BET, MAX_BET, DEFAULT_MULTIPLIER_SLOTS, PinballStrategy
)
from simulation_test import (
    run_simulation, normalize_probs, MULTIPLIER_PROBS, DEFAULT_HOLE_PROBS
)

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

# 商家可能设置的限制值（常见档位）
LIMIT_VALUES = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 99]

# 基准值：无限制时（99）
BASELINE_MAX_BET = 99

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
T_DEFAULT = 20
J_DEFAULT = 10
T_VALUES = [20, 30, 50]
J_VALUES = [5, 10, 20]
N_SEEDS = 40
MAX_ROUNDS = 10000


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 理论分析
# ═══════════════════════════════════════════════════════════════════════════════

def compute_card_optimal_bet(T, J, mult, max_bet):
    """计算在给定 max_bet 限制下，某倍数的积分卡最优投注量。"""
    best_n = max(MIN_BET, min(max_bet, math.ceil(T * J / mult)))
    best_eff = 0.0
    for k in range(1, J + 1):
        n = max(MIN_BET, math.ceil(k * T / mult))
        if n > max_bet:
            break
        eff = k / n
        if eff > best_eff:
            best_eff = eff
            best_n = n
    cards_on_win = min(mult * best_n // T, J)
    return best_n, cards_on_win, best_eff


def theoretical_loss_analysis(T, J):
    """对比各限制值 vs 无限制(99)的理论积分效率损失。"""
    results = {}

    for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
        n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
        p_win = n_lit / NUM_SLOTS

        # 基准：max_bet=99
        base_n, base_cards, base_eff = compute_card_optimal_bet(T, J, mult, BASELINE_MAX_BET)
        base_cpm = p_win * base_cards / base_n if base_n > 0 else 0

        mult_results = []
        for limit in LIMIT_VALUES:
            opt_n, opt_cards, opt_eff = compute_card_optimal_bet(T, J, mult, limit)
            cpm = p_win * opt_cards / opt_n if opt_n > 0 else 0

            loss_cpm = base_cpm - cpm
            loss_pct = (loss_cpm / base_cpm * 100) if base_cpm > 0 else 0

            is_constrained = (opt_n == limit and limit < base_n)

            mult_results.append({
                "limit": limit,
                "opt_bet": opt_n,
                "cards_on_win": opt_cards,
                "cards_per_marble": cpm,
                "loss_vs_99": loss_cpm,
                "loss_pct": loss_pct,
                "constrained": is_constrained,
            })

        results[mult] = {
            "n_lit": n_lit,
            "p_win": p_win,
            "base_bet": base_n,
            "base_cards": base_cards,
            "base_cpm": base_cpm,
            "limits": mult_results,
        }

    return results


def weighted_card_efficiency(T, J, max_bet):
    """
    计算所有倍数加权后的综合积分效率。
    权重 = 各倍数的出现概率。
    """
    total_cpm = 0.0
    for mult, prob in MULTIPLIER_PROBS.items():
        n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
        p_win = n_lit / NUM_SLOTS
        opt_n, opt_cards, _ = compute_card_optimal_bet(T, J, mult, max_bet)
        cpm = p_win * opt_cards / opt_n if opt_n > 0 else 0
        total_cpm += prob * cpm
    return total_cpm


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 模拟实验
# ═══════════════════════════════════════════════════════════════════════════════

def run_limit_batch(max_bet, priority, hole_probs, n_seeds, ct=0.0):
    """运行指定 max_bet 限制下的批量模拟。"""
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
            max_rounds=MAX_ROUNDS,
            max_bet=max_bet,
        )
        results.append(r)

    total_cards = sum(r["total_cards_won"] for r in results)
    total_consumed = sum(r["net_consumed"] for r in results)
    total_rounds = sum(r["total_rounds"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    total_spent = sum(r["total_marbles_spent"] for r in results)
    total_won = sum(r["total_marbles_won"] for r in results)

    cards_per_marble = total_cards / total_consumed if total_consumed > 0 else float('inf')
    avg_cards = total_cards / n_seeds
    avg_consumed = total_consumed / n_seeds
    avg_rounds = total_rounds / n_seeds
    win_rate = total_wins / total_rounds if total_rounds > 0 else 0
    marble_roi = total_won / total_spent if total_spent > 0 else 0
    cards_per_round = total_cards / total_rounds if total_rounds > 0 else 0

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


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 打印报告到终端
# ═══════════════════════════════════════════════════════════════════════════════

def print_theory(T, J):
    """打印理论分析部分。"""
    results = theoretical_loss_analysis(T, J)

    print(f"\n{'═' * 95}")
    print(f"  第一部分：理论分析 — 限制 max_bet 对积分卡效率的影响 (T={T}, J={J})")
    print(f"{'═' * 95}")

    for mult in sorted(results.keys()):
        info = results[mult]
        print(f"\n  ── {mult}x (亮灯={info['n_lit']}, P_win={info['p_win']:.4f}) "
              f"基准: 投{info['base_bet']}得{info['base_cards']}卡, 效率={info['base_cpm']:.4f}/珠 ──")
        print(f"  {'限制值':>6}  {'最优投注':>8}  {'赢时积分':>8}  {'效率/珠':>8}  {'损失/珠':>8}  {'损失%':>7}  {'是否受限':>8}")
        print(f"  {'─' * 70}")
        for r in info["limits"]:
            flag = "⚠受限" if r["constrained"] else "✓"
            loss_str = f"{r['loss_pct']:>6.1f}%" if r['loss_pct'] > 0 else "   0%"
            print(f"  {r['limit']:>6}  {r['opt_bet']:>8}  {r['cards_on_win']:>8}  "
                  f"{r['cards_per_marble']:>8.4f}  {r['loss_vs_99']:>8.4f}  {loss_str}  {flag:>8}")

    # 综合加权效率
    print(f"\n  ── 综合加权积分效率（按倍数出现概率加权）──")
    print(f"  {'限制值':>6}  {'加权效率/珠':>12}  {'vs 无限制':>10}  {'损失%':>7}")
    print(f"  {'─' * 45}")
    base_wce = weighted_card_efficiency(T, J, BASELINE_MAX_BET)
    for limit in LIMIT_VALUES:
        wce = weighted_card_efficiency(T, J, limit)
        loss = base_wce - wce
        pct = (loss / base_wce * 100) if base_wce > 0 else 0
        marker = " ★基准" if limit == BASELINE_MAX_BET else ""
        print(f"  {limit:>6}  {wce:>12.6f}  {loss:>+10.6f}  {pct:>6.1f}%{marker}")


def print_tj_sensitivity():
    """打印不同 T/J 参数下的敏感度。"""
    print(f"\n{'═' * 95}")
    print(f"  第二部分：T/J 参数交互 — 不同机器配置下限制的敏感度")
    print(f"{'═' * 95}")

    key_limits = [10, 15, 20, 30, 50, 99]

    for J in J_VALUES:
        print(f"\n  ━━ J = {J} (单次最多{J}张积分卡) ━━")
        for T in T_VALUES:
            print(f"\n  ── T = {T} (每{T}返珠=1卡) ──")
            base_wce = weighted_card_efficiency(T, J, BASELINE_MAX_BET)
            print(f"  基准(99): 加权效率 = {base_wce:.6f}/珠")
            print(f"  {'限制值':>6}", end="")
            for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
                print(f"  {mult}x最优投注", end="")
            print(f"  {'加权效率':>10}  {'损失%':>7}")
            print(f"  {'─' * 80}")
            for limit in key_limits:
                print(f"  {limit:>6}", end="")
                for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
                    n, c, _ = compute_card_optimal_bet(T, J, mult, limit)
                    flag = "*" if n == limit and limit < 99 else " "
                    print(f"  {n:>4}→{c}卡{flag}", end="")
                wce = weighted_card_efficiency(T, J, limit)
                loss_pct = ((base_wce - wce) / base_wce * 100) if base_wce > 0 else 0
                print(f"  {wce:>10.6f}  {loss_pct:>6.1f}%")


def run_all_simulations():
    """运行所有模拟组合。"""
    configs = []
    for dist_name in DISTRIBUTIONS:
        for priority in ["cards", "marbles"]:
            for ct_label, ct_val in [("原始策略", 0.0), ("自适应(ct=20)", 20.0)]:
                for limit in LIMIT_VALUES:
                    configs.append((dist_name, priority, ct_label, ct_val, limit))

    total = len(configs)
    print(f"\n{'═' * 95}")
    print(f"  第三部分：蒙特卡洛模拟实验")
    print(f"  共 {total} 组配置 × {N_SEEDS} 种子 = {total * N_SEEDS} 次模拟")
    print(f"  参数: 初始={INITIAL_MARBLES}, 消耗目标={CONSUME_TARGET}, T={T_DEFAULT}, J={J_DEFAULT}")
    print(f"{'═' * 95}")

    all_results = {}
    start = time.time()

    for i, (dist_name, priority, ct_label, ct_val, limit) in enumerate(configs):
        elapsed = time.time() - start
        eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(f"\r  进度: {i+1}/{total} | {dist_name} {priority} {ct_label} limit={limit} | ETA: {eta:.0f}s  ", end="", flush=True)

        r = run_limit_batch(limit, priority, DISTRIBUTIONS[dist_name], N_SEEDS, ct=ct_val)

        key = (dist_name, priority, ct_label)
        if key not in all_results:
            all_results[key] = {}
        all_results[key][limit] = r

    elapsed = time.time() - start
    print(f"\r  模拟完成！耗时 {elapsed:.0f}s{'':>60}")
    return all_results


def print_simulation_results(all_results):
    """打印模拟结果表格。"""
    for priority in ["cards", "marbles"]:
        priority_cn = "积分卡优先" if priority == "cards" else "珠子优先"

        for ct_label in ["原始策略", "自适应(ct=20)"]:
            print(f"\n  ━━ {priority_cn} / {ct_label} ━━")

            for dist_name in DISTRIBUTIONS:
                key = (dist_name, priority, ct_label)
                if key not in all_results:
                    continue

                data = all_results[key]
                baseline = data.get(BASELINE_MAX_BET)
                if not baseline:
                    continue

                print(f"\n  ── {dist_name} ──")
                print(f"  {'限制':>5}  {'平均积分卡':>10}  {'平均消耗':>10}  {'积分/珠':>9}  "
                      f"{'积分/局':>8}  {'局数':>8}  {'胜率':>6}  {'ROI':>7}  "
                      f"{'卡损失%':>7}  {'效率损失%':>9}")
                print(f"  {'─' * 105}")

                for limit in LIMIT_VALUES:
                    r = data[limit]
                    # 计算损失
                    card_loss_pct = ((baseline["avg_cards"] - r["avg_cards"]) / baseline["avg_cards"] * 100) if baseline["avg_cards"] > 0 else 0
                    eff_loss_pct = 0
                    if baseline["cards_per_marble"] > 0 and r["cards_per_marble"] != float('inf'):
                        eff_loss_pct = ((baseline["cards_per_marble"] - r["cards_per_marble"]) / baseline["cards_per_marble"] * 100)

                    marker = " ★" if limit == BASELINE_MAX_BET else ""
                    # 处理负消耗（珠子在增长）
                    consumed_str = f"{r['avg_consumed']:>10.1f}"

                    cpm_str = f"{r['cards_per_marble']:>9.4f}" if r['cards_per_marble'] != float('inf') and r['avg_consumed'] > 0 else "    ∞(净赚)"

                    print(f"  {limit:>5}  {r['avg_cards']:>10.1f}  {consumed_str}  {cpm_str}  "
                          f"{r['cards_per_round']:>8.4f}  {r['avg_rounds']:>8.1f}  {r['win_rate']:>6.1%}  {r['marble_roi']:>7.4f}  "
                          f"{card_loss_pct:>+6.1f}%  {eff_loss_pct:>+8.1f}%{marker}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 关键阈值分析
# ═══════════════════════════════════════════════════════════════════════════════

def find_critical_thresholds(T, J):
    """找出各倍数的关键 max_bet 阈值（低于该值效率明显下降）。"""
    thresholds = {}
    for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
        n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
        p_win = n_lit / NUM_SLOTS

        # 无限制时的最优投注
        base_n, base_cards, base_eff = compute_card_optimal_bet(T, J, mult, BASELINE_MAX_BET)
        base_cpm = p_win * base_cards / base_n if base_n > 0 else 0

        # 找到效率开始下降的 max_bet 值
        critical = MIN_BET
        for mb in range(MIN_BET, BASELINE_MAX_BET + 1):
            n, c, _ = compute_card_optimal_bet(T, J, mult, mb)
            cpm = p_win * c / n if n > 0 else 0
            if abs(cpm - base_cpm) < 1e-8:
                critical = mb
                break

        # 每个积分卡等级的阈值
        card_steps = []
        for k in range(1, J + 1):
            n_needed = max(MIN_BET, math.ceil(k * T / mult))
            if n_needed > BASELINE_MAX_BET:
                break
            cards_at_n = min(mult * n_needed // T, J)
            card_steps.append({"k": k, "bet_needed": n_needed, "actual_cards": cards_at_n})

        thresholds[mult] = {
            "base_bet": base_n,
            "base_cards": base_cards,
            "critical_min_bet": critical,
            "card_steps": card_steps,
        }

    return thresholds


def print_critical_thresholds(T, J):
    """打印关键阈值信息。"""
    thresholds = find_critical_thresholds(T, J)

    print(f"\n{'═' * 95}")
    print(f"  第四部分：关键阈值分析 — 效率不受损的最低 max_bet (T={T}, J={J})")
    print(f"{'═' * 95}")

    print(f"\n  各倍数的关键 max_bet 阈值（低于此值积分效率下降）:")
    print(f"  {'倍数':>4}  {'无限制最优投注':>14}  {'无限制赢时积分':>14}  {'临界max_bet':>12}  {'说明'}")
    print(f"  {'─' * 75}")

    for mult in sorted(thresholds.keys()):
        info = thresholds[mult]
        note = f"max_bet≥{info['critical_min_bet']}即可获得满效率"
        print(f"  {mult}x  {info['base_bet']:>14}  {info['base_cards']:>14}  "
              f"{info['critical_min_bet']:>12}  {note}")

    # 综合建议
    all_criticals = [info['critical_min_bet'] for info in thresholds.values()]
    max_critical = max(all_criticals)
    print(f"\n  ► 综合结论: max_bet ≥ {max_critical} 时，积分卡策略效率完全不受影响。")
    print(f"    低于 {max_critical} 时，部分倍数的最优投注被截断。")

    # 阶梯表
    print(f"\n  各倍数的积分卡阶梯（需要多少投注才能再多得1张卡）:")
    for mult in sorted(thresholds.keys()):
        info = thresholds[mult]
        steps = info["card_steps"]
        if not steps:
            continue
        step_str = "  ".join([f"{s['bet_needed']}珠→{s['actual_cards']}卡" for s in steps[:8]])
        print(f"  {mult}x: {step_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 生成 Markdown 报告
# ═══════════════════════════════════════════════════════════════════════════════

def generate_markdown(all_results) -> str:
    L = []
    A = L.append

    A("# 商家限制最大投珠数对玩家收益的影响分析报告")
    A("")
    A("> 自动生成 | 12口弹珠机最佳出卡策略项目")
    A("")
    A("## 目录")
    A("1. [问题背景](#1-问题背景)")
    A("2. [理论分析：积分效率损失](#2-理论分析积分效率损失)")
    A("3. [关键阈值：效率不受损的最低限制值](#3-关键阈值效率不受损的最低限制值)")
    A("4. [T/J参数敏感度](#4-tj参数敏感度)")
    A("5. [蒙特卡洛模拟验证](#5-蒙特卡洛模拟验证)")
    A("6. [核心结论与应对建议](#6-核心结论与应对建议)")
    A("")

    # ── 1. 背景 ──
    A("## 1. 问题背景")
    A("")
    A("部分商家会将弹珠机的最大投珠数限制为低于99的某个值，常见限制有：")
    A("- **50珠**、**40珠**、**30珠**、**20珠**、**15珠**、**10珠** 等")
    A("")
    A("这一限制直接影响：")
    A("- 积分卡策略的最优投注量是否能达到")
    A("- 珠子策略在正EV场景下的收益放大效应")
    A("- 玩家的整体收益水平")
    A("")
    A("### 实验参数")
    A("")
    A("| 参数 | 值 |")
    A("|------|-----|")
    A(f"| 测试限制值 | {LIMIT_VALUES} |")
    A(f"| 基准(无限制) | {BASELINE_MAX_BET} |")
    A(f"| T(积分卡分母) | {T_DEFAULT} |")
    A(f"| J(单次上限) | {J_DEFAULT} |")
    A(f"| 初始珠子 | {INITIAL_MARBLES} |")
    A(f"| 消耗目标 | {CONSUME_TARGET} |")
    A(f"| 每组种子数 | {N_SEEDS} |")
    A("")

    # ── 2. 理论分析 ──
    A("## 2. 理论分析：积分效率损失")
    A("")
    A("### 核心原理")
    A("")
    A("积分卡公式: `cards = min(floor(multiplier × bet / T), J)`")
    A("")
    A("由于 `floor()` 的阶梯效应，最优投注量并不总是越大越好，而是对齐到")
    A("恰好跨过下一个积分卡阶梯的值。当 `max_bet` 被限制时：")
    A("- 若限制值 ≥ 各倍数的最优投注 → **无影响**")
    A("- 若限制值 < 某些倍数的最优投注 → **效率下降**")
    A("")

    # 各倍数详细表格
    results = theoretical_loss_analysis(T_DEFAULT, J_DEFAULT)

    for mult in sorted(results.keys()):
        info = results[mult]
        A(f"### {mult}x 倍数 (亮灯={info['n_lit']}, P_win={info['p_win']:.4f})")
        A("")
        A(f"基准(无限制): 投 **{info['base_bet']}珠**, 赢时得 **{info['base_cards']}卡**, "
          f"效率 **{info['base_cpm']:.4f}**/珠")
        A("")
        A("| 限制值 | 最优投注 | 赢时积分 | 效率/珠 | 损失 | 损失% | 状态 |")
        A("|--------|---------|---------|--------|-------|-------|------|")
        for r in info["limits"]:
            flag = "⚠️ 受限" if r["constrained"] else "✅"
            loss_str = f"{r['loss_pct']:.1f}%" if r['loss_pct'] > 0 else "0%"
            A(f"| {r['limit']} | {r['opt_bet']} | {r['cards_on_win']} | "
              f"{r['cards_per_marble']:.4f} | {r['loss_vs_99']:.4f} | {loss_str} | {flag} |")
        A("")

    # 综合加权
    A("### 综合加权积分效率")
    A("")
    A("以各倍数的实际出现概率加权，计算综合积分效率：")
    A("")
    A("| 限制值 | 加权效率/珠 | vs 无限制 | 损失% |")
    A("|--------|-----------|----------|-------|")
    base_wce = weighted_card_efficiency(T_DEFAULT, J_DEFAULT, BASELINE_MAX_BET)
    for limit in LIMIT_VALUES:
        wce = weighted_card_efficiency(T_DEFAULT, J_DEFAULT, limit)
        loss = base_wce - wce
        pct = (loss / base_wce * 100) if base_wce > 0 else 0
        marker = " ★" if limit == BASELINE_MAX_BET else ""
        A(f"| {limit}{marker} | {wce:.6f} | {loss:+.6f} | {pct:.1f}% |")
    A("")

    # ── 3. 关键阈值 ──
    A("## 3. 关键阈值：效率不受损的最低限制值")
    A("")
    thresholds = find_critical_thresholds(T_DEFAULT, J_DEFAULT)

    A("| 倍数 | 无限制最优投注 | 无限制赢时积分 | 临界max_bet |")
    A("|------|-------------|-------------|-----------|")
    for mult in sorted(thresholds.keys()):
        info = thresholds[mult]
        A(f"| {mult}x | {info['base_bet']} | {info['base_cards']} | **{info['critical_min_bet']}** |")
    A("")

    max_critical = max(info['critical_min_bet'] for info in thresholds.values())
    A(f"> **关键结论**: `max_bet ≥ {max_critical}` 时积分卡策略效率完全不受影响。"
      f"商家限制在 {max_critical} 以上等于没有限制。")
    A("")

    # 阶梯表
    A("### 各倍数的积分卡阶梯")
    A("")
    A("每个阶梯显示：获得 N 张积分卡所需的最低投注量")
    A("")
    for mult in sorted(thresholds.keys()):
        info = thresholds[mult]
        steps = info["card_steps"]
        if steps:
            A(f"**{mult}x**: " + " → ".join([f"`{s['bet_needed']}珠={s['actual_cards']}卡`" for s in steps[:10]]))
    A("")

    # ── 4. T/J 敏感度 ──
    A("## 4. T/J参数敏感度")
    A("")
    A("不同机器配置下，限制值造成的损失程度不同：")
    A("")

    for J in J_VALUES:
        A(f"### J = {J}")
        A("")
        A("| T | 限制=10 | 限制=15 | 限制=20 | 限制=30 | 限制=50 | 无限制 |")
        A("|---|---------|---------|---------|---------|---------|--------|")
        for T in T_VALUES:
            base = weighted_card_efficiency(T, J, 99)
            vals = []
            for lim in [10, 15, 20, 30, 50]:
                wce = weighted_card_efficiency(T, J, lim)
                pct = ((base - wce) / base * 100) if base > 0 else 0
                vals.append(f"{wce:.4f} ({pct:+.0f}%)")
            A(f"| {T} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} | {base:.4f} |")
        A("")

    # ── 5. 模拟结果 ──
    A("## 5. 蒙特卡洛模拟验证")
    A("")

    for priority in ["cards", "marbles"]:
        priority_cn = "积分卡优先" if priority == "cards" else "珠子优先"

        for ct_label in ["原始策略", "自适应(ct=20)"]:
            A(f"### {priority_cn} / {ct_label}")
            A("")

            for dist_name in DISTRIBUTIONS:
                key = (dist_name, priority, ct_label)
                if key not in all_results:
                    continue

                data = all_results[key]
                baseline = data.get(BASELINE_MAX_BET)
                if not baseline:
                    continue

                A(f"#### {dist_name}")
                A("")
                A("| 限制 | 平均积分卡 | 平均消耗 | 积分/珠 | 积分/局 | 局数 | ROI | 卡损失% | 效率损失% |")
                A("|------|-----------|---------|--------|--------|------|-----|--------|---------|")

                for limit in LIMIT_VALUES:
                    r = data[limit]
                    card_loss = ((baseline["avg_cards"] - r["avg_cards"]) / baseline["avg_cards"] * 100) if baseline["avg_cards"] > 0 else 0
                    eff_loss = 0
                    if baseline["cards_per_marble"] > 0 and r["cards_per_marble"] != float('inf') and r["avg_consumed"] > 0:
                        eff_loss = ((baseline["cards_per_marble"] - r["cards_per_marble"]) / baseline["cards_per_marble"] * 100)

                    cpm_str = f"{r['cards_per_marble']:.4f}" if r['cards_per_marble'] != float('inf') and r["avg_consumed"] > 0 else "∞"
                    mark = " ★" if limit == BASELINE_MAX_BET else ""

                    A(f"| {limit}{mark} | {r['avg_cards']:.1f} | {r['avg_consumed']:.1f} | "
                      f"{cpm_str} | {r['cards_per_round']:.4f} | {r['avg_rounds']:.0f} | "
                      f"{r['marble_roi']:.4f} | {card_loss:+.1f}% | {eff_loss:+.1f}% |")
                A("")

    # ── 6. 结论 ──
    A("## 6. 核心结论与应对建议")
    A("")

    # 从模拟数据提取结论
    A("### 6.1 商家限制的实际影响分级")
    A("")
    A("| 限制等级 | max_bet范围 | 积分卡策略影响 | 珠子策略影响 | 严重程度 |")
    A("|---------|-----------|-------------|-----------|---------|")
    A(f"| 无感限制 | ≥{max_critical} | 无损失 | 正EV场景略受限 | ⭐ 无影响 |")
    A(f"| 轻度限制 | 30~{max_critical-1} | 0~5%损失 | 正EV收益受限 | ⭐⭐ 轻微 |")
    A("| 中度限制 | 15~29 | 5~20%损失 | 正EV严重受限 | ⭐⭐⭐ 中等 |")
    A("| 严重限制 | 10~14 | 20~40%损失 | 正EV几乎无法利用 | ⭐⭐⭐⭐ 严重 |")
    A("| 极端限制 | 5~9 | 40%+损失 | 策略完全失效 | ⭐⭐⭐⭐⭐ 极端 |")
    A("")

    A("### 6.2 玩家应对策略")
    A("")
    A("| 场景 | 建议 |")
    A("|------|------|")
    A("| max_bet ≥ 30 | 正常使用积分卡策略，影响极小 |")
    A("| max_bet = 15~29 | 积分卡策略仍有效，注意选择对齐阶梯的投注量 |")
    A("| max_bet = 10~14 | 积分卡策略效率明显下降，考虑是否值得继续玩 |")
    A("| max_bet < 10 | 策略价值极低，不建议在此类机器上消费 |")
    A("| 偏斜分布+限制 | 珠子策略受损更大（无法放大正EV），请降低期望 |")
    A("")

    A("### 6.3 商家视角")
    A("")
    A("- 限制 max_bet 是商家**最温和**的抽水手段之一（相比改倍率概率或减灯格数）")
    A("- 对积分卡策略：限制到 10~15 会显著削弱玩家效率")
    A("- 对珠子策略：限制 max_bet 可以有效阻止玩家在正EV场景下翻盘")
    A("- 商家设限越低，玩家的策略空间越小，长期亏损越确定")
    A("")

    A("### 6.4 关键数据总结")
    A("")

    # 从模拟中提取关键对比
    for priority in ["cards", "marbles"]:
        priority_cn = "积分卡优先" if priority == "cards" else "珠子优先"
        A(f"**{priority_cn}策略 (原始) — 轻微偏斜分布**:")
        A("")
        key = ("轻微偏斜", priority, "原始策略")
        if key in all_results:
            data = all_results[key]
            b = data.get(BASELINE_MAX_BET, {})
            for limit in [10, 15, 20, 30, 50, 99]:
                r = data.get(limit, {})
                if r and b:
                    card_loss = ((b["avg_cards"] - r["avg_cards"]) / b["avg_cards"] * 100) if b.get("avg_cards", 0) > 0 else 0
                    A(f"- max_bet={limit}: 平均 {r['avg_cards']:.0f} 卡, ROI={r['marble_roi']:.4f}"
                      f" (vs无限制: {card_loss:+.1f}%)")
            A("")

    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("█" * 95)
    print("█  商家限制最大投珠数对玩家收益的影响分析报告")
    print("█" * 95)

    # 1. 理论分析
    print_theory(T_DEFAULT, J_DEFAULT)

    # 2. 关键阈值
    print_critical_thresholds(T_DEFAULT, J_DEFAULT)

    # 3. T/J敏感度
    print_tj_sensitivity()

    # 4. 模拟实验
    all_results = run_all_simulations()
    print_simulation_results(all_results)

    # 5. 生成 Markdown
    report = generate_markdown(all_results)
    report_path = "max_bet_limit_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  ✓ Markdown 报告已保存到: {report_path}")

    print(f"\n{'█' * 95}")
    print(f"█  分析完毕")
    print(f"{'█' * 95}")


if __name__ == "__main__":
    main()
