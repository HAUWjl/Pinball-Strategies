#!/usr/bin/env python3
"""
12口弹珠机最佳出卡策略 — 交互式命令行工具
Interactive CLI for the 12-slot pinball machine strategy advisor.

Usage:
    python main.py
"""

import sys
from pinball_strategy import (
    NUM_SLOTS,
    MIN_BET,
    MAX_BET,
    DEFAULT_MULTIPLIER_SLOTS,
    PinballStrategy,
)

# ── ANSI colour helpers ────────────────────────────────────────────────────────
_USE_COLOUR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def cyan(t: str) -> str:
    return _c(t, "36")


def bold(t: str) -> str:
    return _c(t, "1")


# ── Input helpers ──────────────────────────────────────────────────────────────

def _read_int(prompt: str, lo: int, hi: int, default: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"  请输入 {lo}~{hi} 之间的整数。")


def _read_choice(prompt: str, choices: list) -> str:
    while True:
        raw = input(prompt).strip().lower()
        if raw in choices:
            return raw
        print(f"  请输入以下选项之一: {choices}")


# ── Startup banner ─────────────────────────────────────────────────────────────

def _print_banner() -> None:
    print()
    print(bold("=" * 56))
    print(bold("   12口弹珠机最佳出卡策略顾问"))
    print(bold("   Optimal Strategy for 12-Slot Pinball Machine"))
    print(bold("=" * 56))
    print()


# ── EV table ──────────────────────────────────────────────────────────────────

def _print_ev_table(strategy: PinballStrategy) -> None:
    rows = PinballStrategy.expected_value_table(strategy.T, strategy.J)
    print()
    print(bold("基准期望值分析 (均匀落点假设 / uniform landing assumption)"))
    print(
        f"  {'倍数':>4}  {'亮灯格':>5}  {'中奖概率':>8}  "
        f"{'珠子ROI':>8}  {'积分最优投珠':>12}  {'每珠积分':>8}"
    )
    print("  " + "-" * 56)
    for r in rows:
        roi_val = f"{r['marble_ev_ratio']:.4f}"
        roi_str = green(roi_val) if r["marble_ev_ratio"] >= 1 else yellow(roi_val)
        print(
            f"  {r['multiplier']:>4}x  {r['lit_slots']:>5}  {r['p_win']:>8.4f}  "
            f"{roi_str:>17}  {r['card_optimal_bet']:>12}  {r['cards_per_marble']:>8.4f}"
        )
    print()


# ── Landing probability summary ───────────────────────────────────────────────

def _print_probs(strategy: PinballStrategy) -> None:
    probs = strategy.get_landing_probs()
    plays = strategy.total_plays
    print()
    if plays == 0:
        print(cyan("  (尚无历史数据，使用均匀分布 / no history yet, using uniform distribution)"))
    else:
        print(cyan(f"  历史记录: {plays} 局"))
        print(cyan("  槽位落点概率估计:"))
        for i, p in enumerate(probs):
            bar = "█" * int(p * 60)
            print(f"    槽{i+1:>2}: {p:.4f}  {bar}")
    print()


# ── Per-round recommendation ──────────────────────────────────────────────────

def _run_round(strategy: PinballStrategy) -> None:
    valid_multipliers = sorted(DEFAULT_MULTIPLIER_SLOTS.keys())
    mult_str = "/".join(str(m) for m in valid_multipliers)

    print()
    print(bold("── 新一局 ──────────────────────────────────────────────"))

    # 1. Ask for multiplier
    while True:
        raw = input(f"  按按钮后显示的倍数 ({mult_str}): ").strip()
        if raw.isdigit() and int(raw) in DEFAULT_MULTIPLIER_SLOTS:
            multiplier = int(raw)
            break
        print(f"  请输入有效倍数: {mult_str}")

    # 2. Ask for lit slots
    n_lit = DEFAULT_MULTIPLIER_SLOTS[multiplier]
    print(f"  {multiplier}x 倍数共亮 {n_lit} 个灯格 (槽位编号 1~{NUM_SLOTS})")
    lit_slots: list = []
    for i in range(n_lit):
        while True:
            raw = input(f"    第{i + 1}个亮灯槽位: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= NUM_SLOTS:
                s = int(raw) - 1  # convert to 0-indexed
                if s not in lit_slots:
                    lit_slots.append(s)
                    break
            print(f"  请输入 1~{NUM_SLOTS} 之间的不重复整数。")

    # 3. Get recommendation
    rec = strategy.recommend(multiplier, lit_slots)
    print()
    print(bold("  ── 策略建议 ──────────────────────────"))
    print(f"  中奖概率估计: {rec['win_probability']:.4f}")
    print(f"  建议投珠数  : {bold(str(rec['optimal_bet']))} 颗")
    print(f"  期望返珠数  : {rec['expected_marble_return']:.2f}")
    print(f"  期望积分卡  : {rec['expected_score_cards']:.4f}")
    print(f"  珠子ROI     : {rec['marble_roi']:.4f}")
    if strategy.priority == "marbles":
        if rec["marble_roi"] > 1.0:
            print(green("  ✓ 期望正收益！建议多投。"))
        elif rec["marble_roi"] == 1.0:
            print(cyan("  ≈ 收支平衡，投珠数对期望无影响。"))
        else:
            print(yellow(f"  ✗ 期望负收益，建议仅投最低数量 ({MIN_BET} 颗)。"))
    else:
        # card priority: the bet is already optimised for card yield
        if rec["marble_roi"] >= 1.0:
            print(green("  ✓ 珠子期望不亏，积分卡最优投珠数如上。"))
        else:
            print(yellow("  ✗ 珠子期望亏损，但积分卡最优投珠数如上。"))
    print()

    # 4. Ask for result
    result = _read_choice(
        "  本局结果 (w=赢/中奖, l=输/未中奖, s=跳过不记录): ", ["w", "l", "s"]
    )
    if result == "s":
        return

    if result == "w":
        landed_slot = _read_int(
            f"  落入哪个槽位？(1~{NUM_SLOTS}): ", 1, NUM_SLOTS
        ) - 1
    else:
        print(f"  弹珠未落入亮灯格。是否知道落入了哪个槽位？(y=是/n=否): ", end="")
        if input().strip().lower() == "y":
            landed_slot = _read_int(
                f"  落入哪个槽位？(1~{NUM_SLOTS}): ", 1, NUM_SLOTS
            ) - 1
        else:
            landed_slot = None

    if landed_slot is not None:
        strategy.record_landing(landed_slot)
        print(cyan(f"  已记录槽位 {landed_slot + 1}，总记录局数: {strategy.total_plays}"))


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    _print_banner()

    # ── Initialise machine parameters ─────────────────────────────────────────
    print(bold("机器参数设置"))
    T = _read_int("  T 值 (积分卡分母：每T个返珠=1积分卡，通常20~50): ", 1, 9999)
    J = _read_int("  J 值 (单次中奖最多积分卡数, 1~9999): ", 1, 9999)
    priority = _read_choice(
        "  优先目标 (c=积分卡优先 / m=珠子收益优先): ", ["c", "m"]
    )
    priority_key = "cards" if priority == "c" else "marbles"
    prior_weight = _read_int(
        "  先验强度 (越大越稳定，建议12~48，默认24，直接回车=24): ",
        0,
        9999,
        default=24,
    )
    ct = _read_int(
        "  V2自适应信心阈值 (0=关闭/原始策略，建议5，默认5，直接回车=5): ",
        0,
        9999,
        default=5,
    )

    strategy = PinballStrategy(
        T=T, J=J, priority=priority_key,
        prior_weight=prior_weight, confidence_threshold=float(ct),
    )

    print()
    ct_desc = f"V2自适应(ct={ct})" if ct > 0 else "原始策略"
    print(green(f"  已配置: T={T}, J={J}, 优先={priority_key}, 先验强度={prior_weight}, {ct_desc}"))

    # ── Show baseline EV table ─────────────────────────────────────────────────
    _print_ev_table(strategy)

    # ── Main game loop ─────────────────────────────────────────────────────────
    while True:
        print()
        cmd = _read_choice(
            "操作 (n=新一局, p=显示落点概率, e=显示期望值表, q=退出): ",
            ["n", "p", "e", "q"],
        )
        if cmd == "q":
            print(bold("退出策略顾问。祝好运！"))
            break
        elif cmd == "p":
            _print_probs(strategy)
        elif cmd == "e":
            _print_ev_table(strategy)
        elif cmd == "n":
            _run_round(strategy)


if __name__ == "__main__":
    main()
