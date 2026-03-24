"""扩展测试: 更高阈值 + 更多种子"""
from simulation_test import run_simulation, normalize_probs

DISTS = {
    "轻微偏斜": [0.10,0.10,0.10,0.09,0.08,0.07,0.07,0.09,0.08,0.08,0.07,0.07],
    "中等偏斜": [0.15,0.13,0.11,0.10,0.08,0.07,0.06,0.06,0.05,0.07,0.06,0.06],
    "严重偏斜": [0.25,0.18,0.12,0.10,0.08,0.06,0.05,0.04,0.03,0.04,0.03,0.02],
}
THRESHOLDS = [0, 5, 10, 20, 50, 120, 500]
N = 100

for dname, hp in DISTS.items():
    print(f"\n{dname}:")
    print(f"  {'阈值':>6}  {'平均卡':>8}  {'卡/珠':>8}  {'平均局数':>8}")
    for t in THRESHOLDS:
        cards, consumed, rounds = 0, 0, 0
        for s in range(N):
            r = run_simulation(initial_marbles=10000, consume_target=3000, T=20, J=10,
                               priority="cards", hole_probs=hp, seed=s, verbose=False,
                               confidence_threshold=float(t))
            cards += r["total_cards_won"]
            consumed += r["net_consumed"]
            rounds += r["total_rounds"]
        cpm = cards / consumed if consumed > 0 else 0
        label = "原始" if t == 0 else f"ct={t}"
        print(f"  {label:>6}  {cards/N:>8.1f}  {cpm:>8.4f}  {rounds/N:>8.1f}")
