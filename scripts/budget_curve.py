"""Search budget vs strength for Rollout and ISMCTS (opponent: GreedyBelief).

    python scripts/budget_curve.py --games 200
"""
from __future__ import annotations

import os

from common import SERIES, base_parser, label, legend, new_axes, run, save

from fish.arena import RESULT_FIELDS, match_jobs, summarize, write_csv


def main() -> None:
    p = base_parser(__doc__, games=200)
    p.add_argument("--deals", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    p.add_argument("--iterations", nargs="+", type=int, default=[25, 50, 100, 200, 400, 800])
    p.add_argument("--opponent", default="greedy", choices=["greedy", "rule"])
    args = p.parse_args()
    opp = (args.opponent, {})
    configs = [("rollout", "n_deals", b) for b in args.deals] + \
              [("ismcts", "iterations", b) for b in args.iterations]
    jobs = []
    for kind, key, b in configs:
        jobs += match_jobs((kind, {key: b, "name": f"{kind}-{b}"}), opp, args.games, args.seed,
                           match=f"{kind}|{b}")
    jobs.sort(key=lambda j: -j[0][1].get("n_deals", 0) * 12 - j[0][1].get("iterations", 0))
    rows = [r for r, _ in run(jobs, args.workers)]
    write_csv(rows, os.path.join(args.out, "budget_curve_games.csv"), RESULT_FIELDS)
    summary = []
    for kind, key, b in configs:
        s = summarize([r for r in rows if r["match"] == f"{kind}|{b}"])
        summary.append({"agent": kind, "budget": b, **s})
        print(f"{kind:8s} {key}={b:4d}: diff {s['mean_diff']:+5.2f} ± {s['ci95']:.2f}  "
              f"win {s['win_rate']:.2f} loss {s['loss_rate']:.2f}  "
              f"{s['time_mean_x'] * 1e3:6.1f} ms/decision (max {s['time_max_x']:.2f}s)")
    write_csv(summary, os.path.join(args.out, "budget_curve_summary.csv"))

    fig, (a1, a2) = new_axes(2)
    for i, kind in enumerate(("rollout", "ismcts")):
        ss = [s for s in summary if s["agent"] == kind]
        xs = [s["time_mean_x"] * 1e3 for s in ss]
        a1.errorbar(xs, [s["mean_diff"] for s in ss], yerr=[s["ci95"] for s in ss], color=SERIES[i],
                    lw=2, marker="o", ms=5, capsize=3, label=kind)
        a2.plot(xs, [s["win_rate"] for s in ss], color=SERIES[i], lw=2, marker="o", ms=5, label=kind)
        for s, x in zip(ss, xs):
            a2.annotate(str(s["budget"]), (x, s["win_rate"]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color="#52514e")
    for ax in (a1, a2):
        ax.set_xscale("log")
    a1.axhline(0, color="#52514e", lw=0.8)
    a2.axhline(0.5, color="#52514e", lw=0.8)
    label(a1, f"Set differential vs {args.opponent} by search budget\n({args.games} games per point, 95% CI)",
          "mean time per decision (ms, log scale)", "mean set differential")
    label(a2, "Win rate (points labelled with deals / iterations)",
          "mean time per decision (ms, log scale)", f"fraction of games won vs {args.opponent}")
    legend(a1)
    legend(a2)
    save(fig, os.path.join(args.out, "budget_curve.png"))


if __name__ == "__main__":
    main()
