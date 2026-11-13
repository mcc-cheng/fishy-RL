"""Declare-threshold sweep for GreedyBelief against fixed opponents.

    python scripts/threshold_sweep.py --games 400
"""
from __future__ import annotations

import os

from common import SERIES, base_parser, label, legend, new_axes, run, save

from fish.arena import RESULT_FIELDS, match_jobs, summarize, write_csv

OPPONENTS = {"rule": ("rule", {}), "greedy@0.9": ("greedy", {"name": "greedy_ref"})}


def main() -> None:
    p = base_parser(__doc__, games=400)
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    args = p.parse_args()
    jobs = []
    for th in args.thresholds:
        x = ("greedy", {"declare_threshold": th, "name": f"greedy@{th:g}"})
        for oname, opp in OPPONENTS.items():
            jobs += match_jobs(x, opp, args.games, args.seed, match=f"{th:g}|{oname}")
    rows = [r for r, _ in run(jobs, args.workers)]
    write_csv(rows, os.path.join(args.out, "threshold_sweep_games.csv"), RESULT_FIELDS)
    summary = []
    for th in args.thresholds:
        for oname in OPPONENTS:
            s = summarize([r for r in rows if r["match"] == f"{th:g}|{oname}"])
            summary.append({"threshold": th, "opponent": oname, **s})
            print(f"threshold {th:4.2f} vs {oname:10s}: diff {s['mean_diff']:+5.2f} ± {s['ci95']:.2f}  "
                  f"win {s['win_rate']:.2f}  declaration accuracy {s['decl_acc_x']:.3f}  "
                  f"len {s['length']:.0f}")
    write_csv(summary, os.path.join(args.out, "threshold_sweep_summary.csv"))

    fig, (a1, a2) = new_axes(2)
    for i, oname in enumerate(OPPONENTS):
        ss = [s for s in summary if s["opponent"] == oname]
        xs = [s["threshold"] for s in ss]
        a1.errorbar(xs, [s["mean_diff"] for s in ss], yerr=[s["ci95"] for s in ss], color=SERIES[i],
                    lw=2, marker="o", ms=5, capsize=3, label=f"vs {oname}")
        a2.plot(xs, [s["decl_acc_x"] for s in ss], color=SERIES[i], lw=2, marker="o", ms=5,
                label=f"vs {oname}")
    a1.axhline(0, color="#52514e", lw=0.8)
    label(a1, f"GreedyBelief: set differential vs declare threshold\n({args.games} games per point, 95% CI)",
          "declare threshold  P(correct) >", "mean set differential")
    label(a2, "Declaration accuracy vs declare threshold", "declare threshold  P(correct) >",
          "fraction of own declarations correct")
    legend(a1)
    legend(a2)
    save(fig, os.path.join(args.out, "threshold_sweep.png"))


if __name__ == "__main__":
    main()
