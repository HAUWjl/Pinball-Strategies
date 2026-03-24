#!/usr/bin/env python3
"""
策略对比批量测试

对比原始策略与不同 confidence_threshold 的自适应策略，
在多种落点概率分布下运行大量模拟，找出最优参数。
"""

import sys
from simulation_test import run_simulation, DEFAULT_HOLE_PROBS, normalize_probs
from pinball_strategy import NUM_SLOTS

# ── 测试用落点概率分布 ───────────────────────────────────────────────────────

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

# ── 要测试的 confidence_threshold 值 ────────────────────────────────────────

THRESHOLDS = [0, 5, 10, 20, 50, 120, 200]

# ── 测试参数 ────────────────────────────────────────────────────────────────

INITIAL_MARBLES = 10000
CONSUME_TARGET = 3000
T = 20
J = 10
PRIORITY = "cards"
N_SEEDS = 50  # 每个配置跑多少个随机种子


def run_batch(threshold, hole_probs, n_seeds):
    """对一个threshold+分布组合运行n_seeds次模拟，返回汇总统计。"""
    results = []
    for seed in range(n_seeds):
        r = run_simulation(
            initial_marbles=INITIAL_MARBLES,
            consume_target=CONSUME_TARGET,
            T=T, J=J, priority=PRIORITY,
            hole_probs=hole_probs,
            seed=seed,
            verbose=False,
            confidence_threshold=float(threshold),
            max_rounds=10000,
        )
        results.append(r)

    total_cards = sum(r["total_cards_won"] for r in results)
    total_consumed = sum(r["net_consumed"] for r in results)
    total_rounds = sum(r["total_rounds"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    avg_mae = sum(r["mae"] for r in results) / n_seeds

    cards_per_marble = total_cards / total_consumed if total_consumed > 0 else float('inf')
    cards_per_round = total_cards / total_rounds if total_rounds > 0 else 0
    avg_cards = total_cards / n_seeds
    avg_consumed = total_consumed / n_seeds
    avg_rounds = total_rounds / n_seeds
    win_rate = total_wins / total_rounds if total_rounds > 0 else 0

    return {
        "threshold": threshold,
        "avg_cards": avg_cards,
        "avg_consumed": avg_consumed,
        "cards_per_marble": cards_per_marble,
        "cards_per_round": cards_per_round,
        "avg_rounds": avg_rounds,
        "win_rate": win_rate,
        "avg_mae": avg_mae,
    }


def main():
    print("=" * 90)
    print("  弹珠机策略对比测试")
    print(f"  初始={INITIAL_MARBLES} | 消耗目标={CONSUME_TARGET} | T={T}, J={J} | "
          f"优先级={PRIORITY} | 每组{N_SEEDS}个种子")
    print("=" * 90)

    all_results = {}

    for dist_name, hole_probs in DISTRIBUTIONS.items():
        print(f"\n{'─' * 90}")
        print(f"  落点分布: {dist_name}")
        norm_probs = normalize_probs(hole_probs)
        max_p = max(norm_probs)
        min_p = min(norm_probs)
        print(f"  概率范围: {min_p:.4f} ~ {max_p:.4f}")
        print(f"{'─' * 90}")
        print(f"  {'阈值':>6}  {'平均积分卡':>10}  {'平均净消耗':>10}  {'卡/珠效率':>10}  "
              f"{'卡/轮效率':>10}  {'平均局数':>8}  {'胜率':>6}  {'MAE':>8}")
        print(f"  {'─' * 92}")

        dist_results = []
        for threshold in THRESHOLDS:
            stats = run_batch(threshold, hole_probs, N_SEEDS)
            dist_results.append(stats)

            label = "原始" if threshold == 0 else f"ct={threshold}"
            cpm_str = f"{stats['cards_per_marble']:>10.4f}" if stats['cards_per_marble'] != float('inf') else "  盈利∞ "
            print(f"  {label:>6}  {stats['avg_cards']:>10.1f}  {stats['avg_consumed']:>10.1f}  "
                  f"{cpm_str}  {stats['cards_per_round']:>10.4f}  {stats['avg_rounds']:>8.1f}  "
                  f"{stats['win_rate']:>6.1%}  {stats['avg_mae']:>8.4f}")

        # 找出最优: 使用总积分卡作为主指标
        best = max(dist_results, key=lambda x: x["avg_cards"])
        baseline = dist_results[0]  # threshold=0
        if best["threshold"] == 0:
            print(f"  → 最优: 原始策略")
        else:
            improvement = (best["avg_cards"] - baseline["avg_cards"]) / baseline["avg_cards"] * 100
            profitable = best['avg_consumed'] <= 0
            extra = " (策略盈利!)" if profitable else ""
            print(f"  → 最优: ct={best['threshold']} "
                  f"(积分卡 +{improvement:.1f}%, "
                  f"{best['avg_cards']:.1f} vs {baseline['avg_cards']:.1f}){extra}")

        all_results[dist_name] = dist_results

    # ── 总汇总 ────────────────────────────────────────────────────────────
    print(f"\n{'═' * 90}")
    print("  总汇总: 各分布下的最优 confidence_threshold")
    print(f"{'═' * 90}")
    print(f"  {'分布':>12}  {'最优阈值':>8}  {'平均积分卡':>10}  {'对比原始':>8}  {'平均局数':>8}")
    print(f"  {'─' * 58}")
    for dist_name, results in all_results.items():
        best = max(results, key=lambda x: x["avg_cards"])
        baseline = results[0]
        improvement = (best["avg_cards"] - baseline["avg_cards"]) / baseline["avg_cards"] * 100
        label = "原始" if best["threshold"] == 0 else f"ct={best['threshold']}"
        print(f"  {dist_name:>12}  {label:>8}  {best['avg_cards']:>10.1f}  {improvement:>+7.1f}%  {best['avg_rounds']:>8.1f}")

    print()


if __name__ == "__main__":
    main()
