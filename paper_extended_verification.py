#!/usr/bin/env python3
"""
论文扩展研究验证脚本

新增研究方向：
  第8章 - 商家操控对抗博弈论分析
  第9章 - 最大投注约束下的策略退化与临界阈值
  第10章 - 实时在线操控检测的统计功效

运行:
    python paper_extended_verification.py
"""

import math
import random
import sys
from typing import Dict, List, Tuple

from pinball_strategy import (
    NUM_SLOTS, MIN_BET, MAX_BET, DEFAULT_MULTIPLIER_SLOTS, PinballStrategy
)
from simulation_test import (
    run_simulation, normalize_probs, MULTIPLIER_PROBS,
    roll_multiplier, roll_lit_slots, roll_landing
)

# ══════════════════════════════════════════════════════════════════════════════
#  常量与分布定义
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_MULT_PROBS = dict(MULTIPLIER_PROBS)
BASELINE_SLOTS = dict(DEFAULT_MULTIPLIER_SLOTS)

DISTRIBUTIONS = {
    "均匀": [1/12]*12,
    "轻微偏斜": normalize_probs([0.10,0.10,0.10,0.09,0.08,0.07,0.07,0.09,0.08,0.08,0.07,0.07]),
    "中等偏斜": normalize_probs([0.15,0.13,0.11,0.09,0.08,0.07,0.06,0.05,0.06,0.07,0.07,0.06]),
    "严重偏斜": normalize_probs([0.25,0.15,0.12,0.10,0.08,0.06,0.05,0.04,0.04,0.04,0.04,0.03]),
}

def calc_weighted_ev(mult_probs, slot_config):
    """计算加权EV"""
    ev = 0
    for m, prob in mult_probs.items():
        n_lit = slot_config[m]
        p_win = n_lit / NUM_SLOTS
        ev += prob * m * p_win
    return ev

def run_batch(dist_name, hole_probs, n_seeds=50, T=20, J=10,
              priority='cards', ct=0, max_bet=MAX_BET,
              initial=10000, consume=3000, max_rounds=10000,
              mult_probs_override=None, slots_override=None):
    """批量运行模拟，返回汇总统计"""
    results = []
    for seed in range(n_seeds):
        r = run_simulation(
            initial_marbles=initial,
            consume_target=consume,
            T=T, J=J, priority=priority,
            hole_probs=hole_probs,
            seed=seed,
            verbose=False,
            confidence_threshold=ct,
            max_rounds=max_rounds,
            max_bet=max_bet,
            mult_probs=mult_probs_override,
        )
        results.append(r)
    
    n = len(results)
    avg_cards = sum(r['total_cards_won'] for r in results) / n
    avg_spent = sum(r['total_marbles_spent'] for r in results) / n
    avg_won = sum(r['total_marbles_won'] for r in results) / n
    avg_consume = sum(r['net_consumed'] for r in results) / n
    avg_rounds = sum(r['total_rounds'] for r in results) / n
    avg_winrate = sum(r['wins']/(r['wins']+r['losses']) if (r['wins']+r['losses'])>0 else 0 for r in results) / n
    avg_efficiency = avg_cards / avg_consume if avg_consume > 0 else float('inf')
    avg_roi = avg_won / avg_spent if avg_spent > 0 else 0
    
    return {
        'avg_cards': avg_cards,
        'avg_consume': avg_consume,
        'avg_rounds': avg_rounds,
        'avg_winrate': avg_winrate,
        'avg_efficiency': avg_efficiency,
        'avg_roi': avg_roi,
        'avg_spent': avg_spent,
        'avg_won': avg_won,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  第8章：商家操控对抗博弈论分析
# ══════════════════════════════════════════════════════════════════════════════

def section_8():
    print("\n" + "=" * 80)
    print("  第8章：商家操控对抗博弈论分析")
    print("=" * 80)
    
    # ── 8.1 操控向量的独立与联合效应 ──
    print("\n── 8.1 三种操控向量的独立影响 ──")
    
    # 向量1：倍率概率操控
    mult_scenarios = {
        "基准":       {2:0.420, 4:0.288, 6:0.127, 8:0.108, 10:0.057},
        "温和(2x↑)":  {2:0.500, 4:0.280, 6:0.110, 8:0.070, 10:0.040},
        "严重(2x↑↑)": {2:0.600, 4:0.250, 6:0.080, 8:0.050, 10:0.020},
        "极端(2x↑↑↑)":{2:0.700, 4:0.200, 6:0.060, 8:0.030, 10:0.010},
    }
    
    print("\n  表A1：倍率概率操控对EV的影响")
    print(f"  {'方案':<16} {'2x%':>6} {'4x%':>6} {'10x%':>6} {'加权EV':>8} {'庄家利润率':>10} {'vs基准':>8}")
    print("  " + "-" * 65)
    base_ev = calc_weighted_ev(mult_scenarios["基准"], BASELINE_SLOTS)
    for name, mp in mult_scenarios.items():
        ev = calc_weighted_ev(mp, BASELINE_SLOTS)
        house = (1 - ev) * 100
        diff = (ev - base_ev) / base_ev * 100
        print(f"  {name:<16} {mp[2]*100:>5.1f}% {mp[4]*100:>5.1f}% {mp[10]*100:>5.1f}% {ev:>8.4f} {house:>9.2f}% {diff:>+7.1f}%")
    
    # 向量2：亮灯格数操控
    slot_scenarios = {
        "标准":         {2:4, 4:3, 6:2, 8:1, 10:1},
        "温和减灯":     {2:3, 4:2, 6:2, 8:1, 10:1},
        "严重减灯":     {2:3, 4:2, 6:1, 8:1, 10:1},
        "极端减灯":     {2:2, 4:2, 6:1, 8:1, 10:1},
    }
    
    print("\n  表A2：亮灯格数操控对EV的影响")
    print(f"  {'方案':<12} {'2x灯':>5} {'4x灯':>5} {'6x灯':>5} {'加权EV':>8} {'庄家利润率':>10} {'vs基准':>8}")
    print("  " + "-" * 60)
    for name, sc in slot_scenarios.items():
        ev = calc_weighted_ev(BASELINE_MULT_PROBS, sc)
        house = (1 - ev) * 100
        diff = (ev - base_ev) / base_ev * 100
        print(f"  {name:<12} {sc[2]:>5} {sc[4]:>5} {sc[6]:>5} {ev:>8.4f} {house:>9.2f}% {diff:>+7.1f}%")
    
    # 向量3：max_bet 限制
    print("\n  表A3：max_bet限制对策略效率的影响（理论分析）")
    print(f"  {'max_bet':>8} {'n_floor可用':>12} {'正EV上限珠':>12} {'理论损害':>10}")
    print("  " + "-" * 48)
    T = 20
    for mb in [5, 8, 10, 15, 20, 30, 50, 99]:
        n_floor_ok = all(max(MIN_BET, math.ceil(T/m)) <= mb for m in [2,4,6,8,10])
        pos_ev_cap = mb  # 正EV时最多只能投这么多
        damage = "策略失效" if not n_floor_ok else f"正EV上限{mb}珠" if mb < 99 else "无限制"
        print(f"  {mb:>8} {'Y' if n_floor_ok else 'N':>12} {pos_ev_cap:>12} {damage:>10}")
    
    # ── 8.2 组合操控的超线性效应 ──
    print("\n── 8.2 组合操控的超线性（交互）效应 ──")
    combos = {
        "基准":              (BASELINE_MULT_PROBS, BASELINE_SLOTS),
        "仅倍率操控(严重)":   (mult_scenarios["严重(2x↑↑)"], BASELINE_SLOTS),
        "仅减灯(严重)":      (BASELINE_MULT_PROBS, slot_scenarios["严重减灯"]),
        "组合(严重+严重)":   (mult_scenarios["严重(2x↑↑)"], slot_scenarios["严重减灯"]),
    }
    
    print(f"\n  表A4：组合操控的交互效应")
    print(f"  {'方案':<22} {'加权EV':>8} {'庄家利润':>10} {'EV下降':>8} {'超线性':>8}")
    print("  " + "-" * 62)
    ev_base = calc_weighted_ev(*combos["基准"])
    ev_mult_only = calc_weighted_ev(*combos["仅倍率操控(严重)"])
    ev_slot_only = calc_weighted_ev(*combos["仅减灯(严重)"])
    ev_combo = calc_weighted_ev(*combos["组合(严重+严重)"])
    
    drop_mult = ev_base - ev_mult_only
    drop_slot = ev_base - ev_slot_only
    drop_combo = ev_base - ev_combo
    expected_additive = drop_mult + drop_slot
    superlinear = drop_combo / expected_additive if expected_additive > 0 else 0
    
    for name, (mp, sc) in combos.items():
        ev = calc_weighted_ev(mp, sc)
        house = (1 - ev) * 100
        drop = (ev_base - ev) * 100
        sl = ""
        if name == "组合(严重+严重)":
            sl = f"{superlinear:.2f}x"
        print(f"  {name:<22} {ev:>8.4f} {house:>9.2f}% {drop:>7.2f}pp {sl:>8}")
    
    print(f"\n  理论分析：单独操控EV下降 {drop_mult*100:.2f}pp + {drop_slot*100:.2f}pp = {expected_additive*100:.2f}pp")
    print(f"            组合操控实际下降 {drop_combo*100:.2f}pp，超线性系数 = {superlinear:.2f}")
    
    # ── 8.3 V2策略在各操控场景下的韧性（蒙特卡洛验证） ──
    print("\n── 8.3 V2策略的抗操控韧性（蒙特卡洛模拟, 30组种子） ──")
    print("  注：使用轻微偏斜分布，T=20, J=10, 初始10000珠，消耗目标3000")
    
    hole_probs = DISTRIBUTIONS["轻微偏斜"]
    manipulation_levels = {
        "无操控": None,
        "温和操控": mult_scenarios["温和(2x↑)"],
        "严重操控": mult_scenarios["严重(2x↑↑)"],
        "极端操控": mult_scenarios["极端(2x↑↑↑)"],
    }
    
    print(f"\n  表A5：V2策略(ct=5) vs 原始策略在操控环境下的对比")
    print(f"  {'操控等级':<12} {'策略':>6} {'平均积分卡':>10} {'净消耗':>8} {'效率':>8} {'V2提升':>8}")
    print("  " + "-" * 60)
    
    for manip_name, manip_mult in manipulation_levels.items():
        for ct_val, strat_name in [(0, "原始"), (5, "V2")]:
            r = run_batch("轻微偏斜", hole_probs, n_seeds=30, ct=ct_val,
                          mult_probs_override=manip_mult)
            v2_lift = ""
            if strat_name == "V2":
                v2_lift = f"+{(r['avg_cards']/orig_cards-1)*100:.0f}%" if orig_cards > 0 else ""
            else:
                orig_cards = r['avg_cards']
            
            consume_str = f"{r['avg_consume']:.0f}"
            eff_str = f"{r['avg_efficiency']:.4f}" if r['avg_consume'] > 0 else "盈利"
            print(f"  {manip_name:<12} {strat_name:>6} {r['avg_cards']:>10.1f} {consume_str:>8} {eff_str:>8} {v2_lift:>8}")
    
    # ── 8.4 纳什均衡分析 ──
    print("\n── 8.4 商家-玩家博弈的策略均衡分析 ──")
    print("  商家操控策略空间: {无操控, 温和, 严重, 极端}")
    print("  玩家策略空间: {原始策略, V2(ct=5)}")
    print("\n  表A6：商家利润矩阵 (每100珠利润，轻微偏斜)")
    print(f"  {'':>16} {'玩家:原始':>12} {'玩家:V2':>12}")
    print("  " + "-" * 42)
    
    for manip_name, manip_mult in manipulation_levels.items():
        losses = []
        for ct_val in [0, 5]:
            r = run_batch("轻微偏斜", hole_probs, n_seeds=30, ct=ct_val,
                          mult_probs_override=manip_mult)
            # 商家利润 = 玩家净消耗 / 玩家总投入 * 100
            merchant_profit = r['avg_consume'] / r['avg_spent'] * 100 if r['avg_spent'] > 0 else 0
            losses.append(merchant_profit)
        print(f"  {manip_name:>16} {losses[0]:>11.1f}% {losses[1]:>11.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
#  第9章：max_bet约束下的策略退化分析
# ══════════════════════════════════════════════════════════════════════════════

def section_9():
    print("\n" + "=" * 80)
    print("  第9章：最大投注约束下的策略退化与临界阈值")
    print("=" * 80)
    
    # ── 9.1 积分卡策略在不同max_bet下的退化曲线 ──
    print("\n── 9.1 积分卡策略的max_bet敏感性（30组种子） ──")
    
    max_bet_values = [5, 8, 10, 15, 20, 30, 50, 99]
    
    for dist_name in ["均匀", "轻微偏斜", "严重偏斜"]:
        hole_probs = DISTRIBUTIONS[dist_name]
        print(f"\n  表B1-{dist_name}：积分卡优先策略 (V2, ct=5, T=20, J=10)")
        print(f"  {'max_bet':>8} {'平均积分卡':>10} {'净消耗':>8} {'效率':>8} {'vs无限制':>10} {'局数':>6}")
        print("  " + "-" * 56)
        
        baseline_cards = None
        for mb in max_bet_values:
            r = run_batch(dist_name, hole_probs, n_seeds=30, ct=5, max_bet=mb)
            if baseline_cards is None or mb == 99:
                baseline_cards_99 = r['avg_cards']
            if mb == 99:
                baseline_cards = r['avg_cards']
            
            vs_base = f"{(r['avg_cards']/baseline_cards_99-1)*100:>+.1f}%" if baseline_cards_99 > 0 else "—"
            eff_str = f"{r['avg_efficiency']:.4f}" if r['avg_consume'] > 0 else "盈利"
            print(f"  {mb:>8} {r['avg_cards']:>10.1f} {r['avg_consume']:>8.0f} {eff_str:>8} {vs_base:>10} {r['avg_rounds']:>6.0f}")
    
    # ── 9.2 珠子策略在不同max_bet下的退化曲线 ──
    print("\n── 9.2 珠子优先策略的max_bet敏感性（30组种子） ──")
    
    for dist_name in ["轻微偏斜", "严重偏斜"]:
        hole_probs = DISTRIBUTIONS[dist_name]
        print(f"\n  表B2-{dist_name}：珠子优先策略 (V2, ct=5, T=20, J=10)")
        print(f"  {'max_bet':>8} {'平均积分卡':>10} {'净消耗':>8} {'ROI':>8} {'vs无限制':>10}")
        print("  " + "-" * 50)
        
        baseline_roi_99 = None
        for mb in max_bet_values:
            r = run_batch(dist_name, hole_probs, n_seeds=30, ct=5, max_bet=mb, priority='marbles')
            if mb == 99:
                baseline_roi_99 = r['avg_roi']
            
            vs_base = f"{(r['avg_roi']/baseline_roi_99-1)*100:>+.1f}%" if baseline_roi_99 and baseline_roi_99 > 0 else "—"
            print(f"  {mb:>8} {r['avg_cards']:>10.1f} {r['avg_consume']:>8.0f} {r['avg_roi']:>8.4f} {vs_base:>10}")
    
    # ── 9.3 临界阈值精确定位 ──
    print("\n── 9.3 临界阈值分析 ──")
    T = 20
    print(f"\n  n_floor 值与 max_bet 限制的关系 (T={T})：")
    print(f"  {'倍率':>6} {'n_floor':>8} {'max_bet≥n_floor?':>20}")
    print("  " + "-" * 38)
    max_nfloor = 0
    for m in sorted(DEFAULT_MULTIPLIER_SLOTS.keys()):
        nf = max(MIN_BET, math.ceil(T / m))
        max_nfloor = max(max_nfloor, nf)
        limits_ok = {mb: "Y" if mb >= nf else "N" for mb in [5, 8, 10, 15]}
        print(f"  {m:>4}x  {nf:>8}    5→{limits_ok[5]} 8→{limits_ok[8]} 10→{limits_ok[10]} 15→{limits_ok[15]}")
    print(f"\n  -> 绝对最低临界值 = {max_nfloor}（max_bet低于此值则n_floor策略无法执行2x倍率）")
    
    # 不同T值的临界max_bet
    print(f"\n  表B3：不同T值的n_floor(2x)临界值")
    print(f"  {'T值':>6} {'n_floor(2x)':>12} {'n_floor(4x)':>12} {'绝对临界':>10}")
    print("  " + "-" * 44)
    for t in [10, 15, 20, 25, 30, 40, 50]:
        nf_2 = max(MIN_BET, math.ceil(t / 2))
        nf_4 = max(MIN_BET, math.ceil(t / 4))
        nf_max = nf_2
        print(f"  {t:>6} {nf_2:>12} {nf_4:>12} {nf_max:>10}")

# ══════════════════════════════════════════════════════════════════════════════
#  第10章：实时在线操控检测的统计功效分析
# ══════════════════════════════════════════════════════════════════════════════

def section_10():
    print("\n" + "=" * 80)
    print("  第10章：实时在线操控检测的统计功效分析")
    print("=" * 80)
    
    # ── 10.1 基于TVD的落点偏差检测功效 ──
    print("\n── 10.1 落点偏差检测：TVD随样本量的收敛 ──")
    
    n_trials = 200
    sample_sizes = [20, 50, 100, 200, 500, 1000]
    
    print(f"\n  表C1：不同落点分布的TVD估计收敛性 ({n_trials}次试验)")
    print(f"  {'分布':<12} {'真实TVD':>8} {'N=20':>10} {'N=50':>10} {'N=100':>10} {'N=200':>10} {'N=500':>10} {'N=1000':>10}")
    print("  " + "-" * 80)
    
    for dist_name, probs in DISTRIBUTIONS.items():
        # True TVD
        uniform = 1.0 / NUM_SLOTS
        true_tvd = sum(abs(p - uniform) for p in probs) / 2
        
        row = f"  {dist_name:<12} {true_tvd:>8.4f}"
        for n_sample in sample_sizes:
            tvd_estimates = []
            for trial in range(n_trials):
                rng = random.Random(trial * 1000 + n_sample)
                # Simulate n_sample landings
                counts = [0] * NUM_SLOTS
                for _ in range(n_sample):
                    r = rng.random()
                    cum = 0
                    for i, p in enumerate(probs):
                        cum += p
                        if r < cum:
                            counts[i] += 1
                            break
                    else:
                        counts[-1] += 1
                # Estimate TVD
                est_probs = [c / n_sample for c in counts]
                est_tvd = sum(abs(ep - uniform) for ep in est_probs) / 2
                tvd_estimates.append(est_tvd)
            
            mean_tvd = sum(tvd_estimates) / len(tvd_estimates)
            row += f" {mean_tvd:>10.4f}"
        print(row)
    
    # ── 10.2 倍率频率检测：样本量与检测功效 ──
    print("\n── 10.2 倍率频率异常检测功效 ──")
    print("  问题：观察N到倍率后，能否检测出2x概率被提升至55%？")
    
    true_2x_probs = [0.420, 0.500, 0.550, 0.600, 0.700]
    sample_sizes_mult = [30, 50, 100, 200, 500]
    n_trials_mult = 1000
    alpha = 0.05  # 显著性水平
    
    print(f"\n  表C2：2x频率异常检测功效 (α={alpha}, 基准=42%, {n_trials_mult}次试验)")
    print(f"  {'真实2x%':>10} {'N=30':>8} {'N=50':>8} {'N=100':>8} {'N=200':>8} {'N=500':>8}")
    print("  " + "-" * 55)
    
    for true_p in true_2x_probs:
        row = f"  {true_p*100:>9.1f}%"
        for n_sample in sample_sizes_mult:
            detections = 0
            for trial in range(n_trials_mult):
                rng = random.Random(trial * 10000 + n_sample + int(true_p*1000))
                # Simulate multiplier draws
                count_2x = sum(1 for _ in range(n_sample) if rng.random() < true_p)
                obs_rate = count_2x / n_sample
                # Simple z-test against H0: p = 0.42
                p0 = 0.42
                se = math.sqrt(p0 * (1-p0) / n_sample)
                z = (obs_rate - p0) / se if se > 0 else 0
                if z > 1.645:  # one-sided α=0.05
                    detections += 1
            power = detections / n_trials_mult
            row += f" {power*100:>7.1f}%"
        print(row)
    
    # ── 10.3 综合评分的操控判别准确率 ──
    print("\n── 10.3 综合操控判别准确率 ──")
    print("  场景：H0=无操控(标准概率) vs H1=操控(2x↑至55%)")
    print("  判别规则：观测2x频率 > 阈值，则判定操控")
    
    print(f"\n  表C3：不同阈值下的判别性能 (N=100, {n_trials_mult}次试验)")
    print(f"  {'阈值':>8} {'真阴性(无操控)':>14} {'真阳性(55%)':>14} {'真阳性(60%)':>14} {'综合准确率':>12}")
    print("  " + "-" * 65)
    
    for threshold in [0.45, 0.47, 0.48, 0.50, 0.52]:
        n_sample = 100
        # False positive rate (no manipulation, p=0.42)
        fp = 0
        for trial in range(n_trials_mult):
            rng = random.Random(trial * 50000)
            count_2x = sum(1 for _ in range(n_sample) if rng.random() < 0.42)
            if count_2x / n_sample > threshold:
                fp += 1
        true_neg = 1 - fp / n_trials_mult
        
        # True positive rate (manipulation, p=0.55)
        tp_55 = 0
        for trial in range(n_trials_mult):
            rng = random.Random(trial * 50000 + 1)
            count_2x = sum(1 for _ in range(n_sample) if rng.random() < 0.55)
            if count_2x / n_sample > threshold:
                tp_55 += 1
        tp_rate_55 = tp_55 / n_trials_mult
        
        # True positive rate (manipulation, p=0.60)
        tp_60 = 0
        for trial in range(n_trials_mult):
            rng = random.Random(trial * 50000 + 2)
            count_2x = sum(1 for _ in range(n_sample) if rng.random() < 0.60)
            if count_2x / n_sample > threshold:
                tp_60 += 1
        tp_rate_60 = tp_60 / n_trials_mult

        accuracy = (true_neg + tp_rate_55) / 2
        print(f"  {threshold*100:>7.0f}% {true_neg*100:>13.1f}% {tp_rate_55*100:>13.1f}% {tp_rate_60*100:>13.1f}% {accuracy*100:>11.1f}%")
    
    # ── 10.4 最小可检测效应量 (MDE) ──
    print("\n── 10.4 最小可检测效应量 (MDE) ──")
    print("  给定样本量N和显著性α=0.05，检测功效≥80%所需的最小2x概率偏离")
    
    print(f"\n  表C4：最小可检测效应量 (功效≥80%, {n_trials_mult}次试验)")
    print(f"  {'N':>6} {'MDE(Δp)':>10} {'最小检测2x%':>14}")
    print("  " + "-" * 34)
    
    for n_sample in [30, 50, 100, 200, 500]:
        # Binary search for MDE
        lo, hi = 0.001, 0.30
        for _ in range(20):
            mid = (lo + hi) / 2
            test_p = 0.42 + mid
            detections = 0
            for trial in range(n_trials_mult):
                rng = random.Random(trial * 100000 + n_sample)
                count_2x = sum(1 for _ in range(n_sample) if rng.random() < test_p)
                obs_rate = count_2x / n_sample
                se = math.sqrt(0.42 * 0.58 / n_sample)
                z = (obs_rate - 0.42) / se if se > 0 else 0
                if z > 1.645:
                    detections += 1
            power = detections / n_trials_mult
            if power >= 0.80:
                hi = mid
            else:
                lo = mid
        mde = (lo + hi) / 2
        print(f"  {n_sample:>6} {mde*100:>9.1f}% {(0.42+mde)*100:>13.1f}%")
    
    # ── 10.5 EV回报率检测 ──
    print("\n── 10.5 EV回报率异常检测 ──")
    print("  基准 ROI ≈ 0.8145，问需要多少珠消耗量才能可靠区分异常机器")
    
    print(f"\n  表C5：ROI异常检测（观测ROI vs 标准0.8145）")
    standard_roi = 0.8145
    
    for true_roi_label, true_mult_probs in [
        ("标准", BASELINE_MULT_PROBS),
        ("温和操控", {2:0.500, 4:0.280, 6:0.110, 8:0.070, 10:0.040}),
        ("严重操控", {2:0.600, 4:0.250, 6:0.080, 8:0.050, 10:0.020}),
    ]:
        true_ev = calc_weighted_ev(true_mult_probs, BASELINE_SLOTS)
        print(f"  {true_roi_label}: 真实EV={true_ev:.4f}（偏差{(true_ev-standard_roi)/standard_roi*100:+.1f}%）")

# ══════════════════════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("  论文扩展研究 —— 三个新方向的验证实验")
    print("  基于 12口弹珠机自适应投注策略研究")
    print("=" * 80)
    
    section_8()
    section_9()
    section_10()
    
    print("\n" + "=" * 80)
    print("  所有验证实验完成")
    print("=" * 80)
