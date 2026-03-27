#!/usr/bin/env python3
"""
弹珠机策略模拟测试

模拟一台具有固定落点概率的12口弹珠机，自动运行策略程序，
在消耗指定数量的珠子后，输出策略拟合出的概率与真实概率的对比。

机器参数（由测试者设定，策略程序不知道）：
- 每个落点有固有的落球概率
- 奖励倍数出现概率: 2x/4x/6x/8x/10x = 42%/28.8%/12.7%/10.8%/5.7%
- 每次哪几个灯亮是随机的

用法:
    python simulation_test.py [--initial 初始珠子] [-Y 消耗量] [--T T值] [--J J值]
                              [--priority cards|marbles] [--seed 随机种子]
"""

import argparse
import random
import sys
from typing import List, Tuple

from pinball_strategy import (
    NUM_SLOTS,
    MIN_BET,
    MAX_BET,
    DEFAULT_MULTIPLIER_SLOTS,
    PinballStrategy,
)

# ── 机器固有参数（策略程序不知道这些） ────────────────────────────────────────

# 奖励倍数及其出现概率
MULTIPLIER_PROBS = {
    2: 0.420,
    4: 0.288,
    6: 0.127,
    8: 0.108,
    10: 0.057,
}

# 默认的12个落点固有概率（不均匀分布，策略程序不知道）
# 可通过命令行参数 --hole-probs 自定义
DEFAULT_HOLE_PROBS = [
    0.10, 0.10, 0.10, 0.09, 0.08, 0.07,
    0.07, 0.09, 0.08, 0.08, 0.07, 0.07,
]


def normalize_probs(probs: List[float]) -> List[float]:
    """归一化概率使其总和为1。"""
    total = sum(probs)
    return [p / total for p in probs]


def roll_multiplier(rng: random.Random, mult_probs: dict | None = None) -> int:
    """根据固有概率随机抽取本轮奖励倍数。"""
    probs = mult_probs if mult_probs is not None else MULTIPLIER_PROBS
    r = rng.random()
    cumulative = 0.0
    for mult, prob in probs.items():
        cumulative += prob
        if r < cumulative:
            return mult
    return list(probs.keys())[-1]  # 兜底


def roll_lit_slots(multiplier: int, rng: random.Random) -> List[int]:
    """根据倍数随机选择哪些灯格亮起（0-indexed）。"""
    n_lit = DEFAULT_MULTIPLIER_SLOTS[multiplier]
    return sorted(rng.sample(range(NUM_SLOTS), n_lit))


def roll_landing(hole_probs: List[float], rng: random.Random) -> int:
    """根据落点固有概率随机决定弹珠落入哪个格（0-indexed）。"""
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(hole_probs):
        cumulative += p
        if r < cumulative:
            return i
    return NUM_SLOTS - 1  # 兜底


def run_simulation(
    initial_marbles: int,
    consume_target: int,
    T: int,
    J: int,
    priority: str,
    hole_probs: List[float],
    seed: int | None = None,
    verbose: bool = True,
    confidence_threshold: float = 0.0,
    max_rounds: int = 10000,
    max_bet: int = MAX_BET,
    mult_probs: dict | None = None,
) -> dict:
    """
    运行自动模拟。

    Parameters
    ----------
    initial_marbles : int
        初始珠子数量。
    consume_target : int
        净消耗目标 Y：当 (初始珠子 - 剩余珠子) >= Y 时停止。
    T, J : int
        机器的积分卡参数。
    priority : str
        策略优先级 ('cards' 或 'marbles')。
    hole_probs : list[float]
        12个落点的固有概率（策略不知道）。
    seed : int | None
        随机种子，用于可复现的测试。
    verbose : bool
        是否输出每一轮的详细信息。
    confidence_threshold : float
        自适应投注的信心阈值。0=原始策略，>0=渐进式投注。
    max_rounds : int
        最大局数上限，防止在正EV策略下游戏无限延长。默认10000。
    max_bet : int
        每局最大投注珠数。默认99。

    Returns
    -------
    dict
        模拟结果汇总。
    """
    rng = random.Random(seed)
    hole_probs = normalize_probs(hole_probs)
    strategy = PinballStrategy(T=T, J=J, priority=priority,
                               confidence_threshold=confidence_threshold,
                               max_bet=max_bet)

    marbles_remaining = initial_marbles
    total_marbles_spent = 0
    total_marbles_won = 0
    total_cards_won = 0
    total_rounds = 0
    wins = 0
    losses = 0

    if verbose:
        print()
        print("=" * 70)
        print(f"  弹珠机策略模拟测试")
        ct_str = f" | 信心阈值: {confidence_threshold}" if confidence_threshold > 0 else " | 原始策略"
        print(f"  初始珠子: {initial_marbles} | 消耗目标: {consume_target} | T={T}, J={J} | 优先级: {priority}{ct_str}")
        print(f"  随机种子: {seed}")
        print("=" * 70)
        print()
        print("  真实落点概率（策略不知道）:")
        for i, p in enumerate(hole_probs):
            bar = "█" * int(p * 80)
            print(f"    槽{i+1:>2}: {p:.4f}  {bar}")
        print()
        print("-" * 70)

    while marbles_remaining >= MIN_BET:
        total_rounds += 1

        # 1. 机器随机确定倍数
        multiplier = roll_multiplier(rng, mult_probs)

        # 2. 随机确定哪些灯亮
        lit_slots = roll_lit_slots(multiplier, rng)

        # 3. 策略给出建议下注数
        recommended_bet = strategy.optimal_bet(multiplier, lit_slots)
        actual_bet = min(recommended_bet, marbles_remaining)
        actual_bet = max(actual_bet, MIN_BET)
        if actual_bet > marbles_remaining:
            break  # 珠子不够了

        # 4. 弹珠发射，根据固有概率决定落点
        landing = roll_landing(hole_probs, rng)

        # 5. 判断是否中奖
        is_win = landing in lit_slots
        marbles_won = multiplier * actual_bet if is_win else 0
        cards_won = min(marbles_won // T, J) if is_win else 0

        # 6. 更新珠子余额
        marbles_remaining -= actual_bet
        marbles_remaining += marbles_won
        total_marbles_spent += actual_bet
        total_marbles_won += marbles_won
        total_cards_won += cards_won
        if is_win:
            wins += 1
        else:
            losses += 1

        # 7. 记录落点到策略（策略由此学习概率分布）
        strategy.record_landing(landing)

        # 8. 输出本轮信息
        net_consumed = initial_marbles - marbles_remaining
        if verbose:
            lit_display = [s + 1 for s in lit_slots]
            win_str = "✓ 中奖" if is_win else "✗ 未中"
            p_win = strategy.win_probability(lit_slots)
            print(
                f"  第{total_rounds:>4}局 | "
                f"{multiplier:>2}x 亮灯{lit_display} | "
                f"下注{actual_bet:>2} | "
                f"落点槽{landing+1:>2} | "
                f"{win_str} | "
                f"赢珠{marbles_won:>4} 积分卡{cards_won:>2} | "
                f"余{marbles_remaining:>5} | "
                f"已消耗{net_consumed:>5} | "
                f"策略估算胜率{p_win:.3f}"
            )

        # 净消耗达到目标Y时停止
        if net_consumed >= consume_target:
            break
        # 达到最大局数上限时停止
        if total_rounds >= max_rounds:
            break

    # ── 输出结果汇总 ──────────────────────────────────────────────────────────
    estimated_probs = strategy.get_landing_probs()

    net_consumed = initial_marbles - marbles_remaining

    if verbose:
        print()
        print("=" * 70)
        print("  模拟结束 — 结果汇总")
        print("=" * 70)
        print(f"  总局数: {total_rounds}")
        print(f"  胜/负:  {wins} / {losses}  (胜率 {wins/total_rounds*100:.1f}%)" if total_rounds > 0 else "")
        print(f"  总投珠: {total_marbles_spent}")
        print(f"  总赢珠: {total_marbles_won}")
        print(f"  净消耗: {net_consumed} 珠 (初始{initial_marbles} → 剩余{marbles_remaining})")
        print(f"  总积分卡: {total_cards_won}")
        print(f"  剩余珠子: {marbles_remaining}")
        print()
        print("  概率对比（真实 vs 策略估计）:")
        print(f"  {'槽位':>4}  {'真实概率':>8}  {'策略估计':>8}  {'误差':>8}  {'对比'}")
        print("  " + "-" * 58)
        total_abs_error = 0.0
        for i in range(NUM_SLOTS):
            real = hole_probs[i]
            est = estimated_probs[i]
            err = est - real
            total_abs_error += abs(err)
            bar_real = "▓" * int(real * 80)
            bar_est = "░" * int(est * 80)
            print(
                f"  槽{i+1:>2}  {real:>8.4f}  {est:>8.4f}  {err:>+8.4f}  "
                f"真实{bar_real} 估计{bar_est}"
            )
        mae = total_abs_error / NUM_SLOTS
        print()
        print(f"  平均绝对误差 (MAE): {mae:.4f}")
        print(f"  策略观测到的总局数: {strategy.total_plays}")
        print()

    return {
        "total_rounds": total_rounds,
        "wins": wins,
        "losses": losses,
        "total_marbles_spent": total_marbles_spent,
        "total_marbles_won": total_marbles_won,
        "net_consumed": net_consumed,
        "total_cards_won": total_cards_won,
        "initial_marbles": initial_marbles,
        "marbles_remaining": marbles_remaining,
        "real_probs": hole_probs,
        "estimated_probs": estimated_probs,
        "mae": sum(abs(estimated_probs[i] - hole_probs[i]) for i in range(NUM_SLOTS)) / NUM_SLOTS,
    }


def main():
    parser = argparse.ArgumentParser(
        description="弹珠机策略模拟测试 — 用固定概率机器检验策略拟合效果"
    )
    parser.add_argument(
        "--initial", type=int, default=10000,
        help="初始珠子数量 (默认10000)"
    )
    parser.add_argument(
        "-Y", "--consume", type=int, default=3000,
        help="净消耗目标：剩余珠子减少Y个后停止 (默认500)"
    )
    parser.add_argument(
        "--T", type=int, default=20,
        help="积分卡分母T (默认20)"
    )
    parser.add_argument(
        "--J", type=int, default=100,
        help="单次最多积分卡数J (默认100)"
    )
    parser.add_argument(
        "--priority", choices=["cards", "marbles"], default="cards",
        help="策略优先级 (默认cards)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子（可复现结果）"
    )
    parser.add_argument(
        "--hole-probs", type=float, nargs=12, default=None,
        metavar="P",
        help="12个落点的固有概率（空格分隔，会自动归一化）"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="安静模式，只输出最终结果"
    )
    parser.add_argument(
        "--confidence-threshold", "-ct", type=float, default=0.0,
        help="自适应投注信心阈值 (0=原始策略, >0=渐进式, 默认0)"
    )

    args = parser.parse_args()

    hole_probs = args.hole_probs if args.hole_probs else DEFAULT_HOLE_PROBS

    if len(hole_probs) != NUM_SLOTS:
        print(f"错误: 需要{NUM_SLOTS}个概率值，但提供了{len(hole_probs)}个", file=sys.stderr)
        sys.exit(1)

    result = run_simulation(
        initial_marbles=args.initial,
        consume_target=args.consume,
        T=args.T,
        J=args.J,
        priority=args.priority,
        hole_probs=hole_probs,
        seed=args.seed,
        verbose=not args.quiet,
        confidence_threshold=args.confidence_threshold,
    )

    if args.quiet:
        print(f"局数={result['total_rounds']} "
              f"胜率={result['wins']/result['total_rounds']*100:.1f}% "
              f"净消耗={result['net_consumed']} "
              f"积分卡={result['total_cards_won']} "
              f"MAE={result['mae']:.4f}")


if __name__ == "__main__":
    main()
