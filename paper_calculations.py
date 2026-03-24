#!/usr/bin/env python3
"""
论文验证计算脚本
生成论文中需要的所有数值结果。
"""
import math
import json
from pinball_strategy import (
    NUM_SLOTS, MIN_BET, MAX_BET, DEFAULT_MULTIPLIER_SLOTS, PinballStrategy
)
from simulation_test import (
    run_simulation, normalize_probs, MULTIPLIER_PROBS
)

print("=" * 80)
print("  第一部分：基础数学模型验证")
print("=" * 80)

# 1. 各倍率下的基本数学特性
print("\n1. 均匀分布假设下各倍率期望值：")
print(f"  {'倍率':>4}  {'亮灯':>4}  {'P_win':>8}  {'EV_ratio':>10}  {'净EV(投5)':>10}  {'净EV(投99)':>10}")
print("  " + "-" * 55)
for mult, n_lit in sorted(DEFAULT_MULTIPLIER_SLOTS.items()):
    p_win = n_lit / NUM_SLOTS
    ev_ratio = mult * p_win
    net_ev_5 = 5 * (ev_ratio - 1)
    net_ev_99 = 99 * (ev_ratio - 1)
    print(f"  {mult:>4}x  {n_lit:>4}  {p_win:>8.4f}  {ev_ratio:>10.4f}  {net_ev_5:>+10.2f}  {net_ev_99:>+10.2f}")

# 2. 综合期望值（加权各倍率出现概率）
print("\n2. 加权综合期望值 (各倍率按出现概率加权)：")
weighted_ev = 0
for mult, prob_mult in MULTIPLIER_PROBS.items():
    n_lit = DEFAULT_MULTIPLIER_SLOTS[mult]
    p_win = n_lit / NUM_SLOTS
    ev_ratio = mult * p_win
    weighted_ev += prob_mult * ev_ratio
print(f"  综合 EV_ratio = {weighted_ev:.6f}")
print(f"  投1珠的平均期望回报 = {weighted_ev:.4f} 珠 (每投1珠平均亏损 {1-weighted_ev:.4f} 珠)")

# 3. floor()取整浪费分析
print("\n3. floor()取整造成的浪费分析 (T=20)：")
T = 20
print(f"  {'倍率':>4}  {'投注':>4}  {'返珠':>6}  {'积分卡':>6}  {'浪费珠':>6}  {'卡效率':>10}")
print("  " + "-" * 50)
for mult in [2, 4, 6]:
    for bet in [5, 10, 20, 50, 90, 99]:
        returned = mult * bet
        cards = min(returned // T, 10)
        used_for_cards = cards * T
        wasted = returned - used_for_cards  # 不产生积分卡的返珠
        efficiency = cards / bet if bet > 0 else 0
        print(f"  {mult:>4}x  {bet:>4}  {returned:>6}  {cards:>6}  {wasted:>6}  {efficiency:>10.4f}")
    print()

# 4. n_floor计算
print("4. 最小步进量 n_floor 计算 (T=20)：")
print(f"  {'倍率':>4}  {'n_floor':>8}  {'返珠':>6}  {'积分卡':>6}  {'卡效率':>10}")
print("  " + "-" * 42)
for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
    n_floor = max(MIN_BET, math.ceil(T / mult))
    returned = mult * n_floor
    cards = min(returned // T, 10)
    eff = cards / n_floor
    print(f"  {mult:>4}x  {n_floor:>8}  {returned:>6}  {cards:>6}  {eff:>10.4f}")

# 5. 原始策略 n_cap 计算 (T=20, J=10)
print("\n5. 原始策略 n_cap = ceil(T*J/mult) 计算 (T=20, J=10)：")
J = 10
print(f"  {'倍率':>4}  {'n_cap':>8}  {'返珠':>6}  {'积分卡':>6}  {'卡效率':>10}")
print("  " + "-" * 42)
for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
    n_cap = max(MIN_BET, min(MAX_BET, math.ceil(T * J / mult)))
    returned = mult * n_cap
    cards = min(returned // T, J)
    eff = cards / n_cap
    print(f"  {mult:>4}x  {n_cap:>8}  {returned:>6}  {cards:>6}  {eff:>10.4f}")

# 6. n_floor vs n_cap 效率对比
print("\n6. 关键对比：n_floor vs n_cap 的卡效率 (T=20, J=10)：")
print(f"  {'倍率':>4}  | {'n_floor':>8} {'卡数':>4} {'效率':>8} | {'n_cap':>8} {'卡数':>4} {'效率':>8} | {'效率比':>8}")
print("  " + "-" * 68)
for mult in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
    p_win = DEFAULT_MULTIPLIER_SLOTS[mult] / NUM_SLOTS
    n_floor = max(MIN_BET, math.ceil(T / mult))
    n_cap = max(MIN_BET, min(MAX_BET, math.ceil(T * J / mult)))
    cards_floor = min(mult * n_floor // T, J)
    cards_cap = min(mult * n_cap // T, J)
    eff_floor = p_win * cards_floor / n_floor
    eff_cap = p_win * cards_cap / n_cap
    ratio = eff_floor / eff_cap if eff_cap > 0 else float('inf')
    print(f"  {mult:>4}x  | {n_floor:>8} {cards_floor:>4} {eff_floor:>8.5f} | {n_cap:>8} {cards_cap:>4} {eff_cap:>8.5f} | {ratio:>8.3f}")

print()
print("=" * 80)
print("  第二部分：贝叶斯概率估计收敛验证")
print("=" * 80)

# 7. 模拟概率收敛 - 不同局数后的MAE
print("\n7. 贝叶斯估计随局数增加的收敛性 (prior_weight=24, 轻微偏斜分布)：")
hole_probs_slight = normalize_probs([
    0.10, 0.10, 0.10, 0.09, 0.08, 0.07,
    0.07, 0.09, 0.08, 0.08, 0.07, 0.07,
])
checkpoints = [10, 30, 50, 100, 200, 500, 1000, 3000]
N_TRIALS = 50
print(f"  {'局数':>6}  {'平均MAE':>10}  {'最大MAE':>10}  {'收敛图示'}")
print("  " + "-" * 50)
for target_rounds in checkpoints:
    maes = []
    for seed in range(N_TRIALS):
        r = run_simulation(
            initial_marbles=999999, consume_target=999999,
            T=20, J=10, priority="cards", hole_probs=hole_probs_slight,
            seed=seed, verbose=False, confidence_threshold=0.0,
            max_rounds=target_rounds,
        )
        maes.append(r["mae"])
    avg_mae = sum(maes) / len(maes)
    max_mae = max(maes)
    bar = "█" * int(avg_mae * 800)
    print(f"  {target_rounds:>6}  {avg_mae:>10.6f}  {max_mae:>10.6f}  {bar}")

print()
print("=" * 80)
print("  第三部分：V2策略模拟对比")
print("=" * 80)

# 8. 四种分布下 ct=0 vs ct=5 对比 (J=10)
DISTRIBUTIONS = {
    "均匀分布": [1.0] * 12,
    "轻微偏斜": [0.10, 0.10, 0.10, 0.09, 0.08, 0.07, 0.07, 0.09, 0.08, 0.08, 0.07, 0.07],
    "中等偏斜": [0.15, 0.13, 0.11, 0.10, 0.08, 0.07, 0.06, 0.06, 0.05, 0.07, 0.06, 0.06],
    "严重偏斜": [0.25, 0.18, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.04, 0.03, 0.02],
}

N_SEEDS = 50
for J_val in [10, 5]:
    print(f"\n8{'a' if J_val==10 else 'b'}. V2策略对比 (T=20, J={J_val}, 初始=10000, 消耗=3000, N={N_SEEDS})：")
    print(f"  {'分布':>10} | {'策略':>8} | {'平均积分卡':>10} {'平均消耗':>10} {'平均局数':>8} {'胜率':>6} {'卡/珠':>8} | {'提升':>8}")
    print("  " + "-" * 90)
    for dist_name, probs in DISTRIBUTIONS.items():
        results = {}
        for ct in [0, 5]:
            total_cards = 0
            total_consumed = 0
            total_rounds = 0
            total_wins = 0
            total_total_rounds = 0
            for seed in range(N_SEEDS):
                r = run_simulation(
                    initial_marbles=10000, consume_target=3000,
                    T=20, J=J_val, priority="cards", hole_probs=probs,
                    seed=seed, verbose=False, confidence_threshold=float(ct),
                    max_rounds=10000,
                )
                total_cards += r["total_cards_won"]
                total_consumed += r["net_consumed"]
                total_rounds += r["total_rounds"]
                total_wins += r["wins"]
            avg_cards = total_cards / N_SEEDS
            avg_consumed = total_consumed / N_SEEDS
            avg_rounds_val = total_rounds / N_SEEDS
            win_rate = total_wins / total_rounds if total_rounds > 0 else 0
            cpm = total_cards / total_consumed if total_consumed > 0 else float('inf')
            results[ct] = {"avg_cards": avg_cards, "avg_consumed": avg_consumed,
                           "avg_rounds": avg_rounds_val, "win_rate": win_rate, "cpm": cpm}
        
        for ct in [0, 5]:
            r = results[ct]
            label = "原始" if ct == 0 else "V2(ct=5)"
            cpm_s = f"{r['cpm']:.4f}" if r['cpm'] != float('inf') else "盈利∞"
            improvement = ""
            if ct == 5:
                imp_pct = (results[5]["avg_cards"] - results[0]["avg_cards"]) / results[0]["avg_cards"] * 100
                improvement = f"+{imp_pct:.1f}%"
            print(f"  {dist_name:>10} | {label:>8} | {r['avg_cards']:>10.1f} {r['avg_consumed']:>10.1f} {r['avg_rounds']:>8.1f} {r['win_rate']:>6.1%} {cpm_s:>8} | {improvement:>8}")
        print()

# 9. 信心函数曲线
print("\n9. 信心函数 confidence(n) = n/(n+ct) 的值 (ct=5)：")
print(f"  {'局数n':>6}  {'confidence':>12}  {'图示'}")
print("  " + "-" * 45)
for n in [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]:
    ct = 5
    conf = n / (n + ct)
    bar = "█" * int(conf * 40)
    print(f"  {n:>6}  {conf:>12.4f}  {bar}")

# 10. 期望积分卡效率的理论公式验证
print("\n10. 理论 vs 模拟: 单轮期望积分卡 (T=20, J=10, 均匀分布)：")
print(f"  {'倍率':>4}  {'投注':>4}  {'理论E[cards]':>14}  {'说明'}")
print("  " + "-" * 50)
for mult, n_lit in sorted(DEFAULT_MULTIPLIER_SLOTS.items()):
    p_win = n_lit / NUM_SLOTS
    for bet_label, bet in [("n_floor", max(5, math.ceil(20/mult))), ("n_cap", max(5, min(99, math.ceil(200/mult))))]:
        cards_if_win = min(mult * bet // 20, 10)
        e_cards = p_win * cards_if_win
        print(f"  {mult:>4}x  {bet:>4}  {e_cards:>14.4f}  {bet_label}: 中奖得{cards_if_win}卡, P={p_win:.4f}")

print("\n" + "=" * 80)
print("  计算完成")
print("=" * 80)
