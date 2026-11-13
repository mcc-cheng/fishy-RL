"""Round-robin of all agents: mean set differential with 95% CIs.

    python scripts/round_robin.py --games 1000
"""
from __future__ import annotations

import itertools
import os

import numpy as np
from common import DIVERGING, INK, SPECS, base_parser, label, new_axes, run, save

from fish.arena import RESULT_FIELDS, match_jobs, summarize, write_csv


def main() -> None:
    p = base_parser(__doc__, games=1000)
    p.add_argument("--agents", nargs="+", default=list(SPECS), choices=list(SPECS))
    args = p.parse_args()
    names = args.agents
    pairs = list(itertools.combinations(names, 2))
    jobs = []
    for i, (a, b) in enumerate(pairs):
        jobs += match_jobs(SPECS[a], SPECS[b], args.games, args.seed + i, match=f"{a}_vs_{b}")
    # slow match-ups first so the pool stays busy
    res = run(jobs, args.workers)
    rows = [r for r, _ in res]
    write_csv(rows, os.path.join(args.out, "round_robin_games.csv"), RESULT_FIELDS)

    n = len(names)
    diff = np.full((n, n), np.nan)
    ci = np.full((n, n), np.nan)
    summary = []
    for a, b in pairs:
        s = summarize([r for r in rows if r["match"] == f"{a}_vs_{b}"])
        i, j = names.index(a), names.index(b)
        diff[i, j], diff[j, i] = s["mean_diff"], -s["mean_diff"]
        ci[i, j] = ci[j, i] = s["ci95"]
        summary.append({"x": a, "y": b, **s})
    write_csv(summary, os.path.join(args.out, "round_robin_summary.csv"))

    print(f"\nMean set differential (row minus column), {args.games} games per pairing, 95% CI")
    print(" " * 9 + "".join(f"{b:>15s}" for b in names))
    for i, a in enumerate(names):
        cells = ["      -        " if i == j else f"  {diff[i, j]:+5.2f} ± {ci[i, j]:4.2f} "
                 for j in range(n)]
        print(f"{a:>9s}" + "".join(cells))
    print("\nDetails (x vs y):")
    for s in summary:
        print(f"  {s['x']:>8s} vs {s['y']:<8s} diff {s['mean_diff']:+5.2f} ± {s['ci95']:.2f}  "
              f"win {s['win_rate']:.2f}  len {s['length']:5.0f}  decl-acc {s['decl_acc_x']:.2f}/"
              f"{s['decl_acc_y']:.2f}  s/decision {s['time_mean_x']:.3f}/{s['time_mean_y']:.3f} "
              f"(max {s['time_max_x']:.2f}/{s['time_max_y']:.2f})  truncated {s['truncated']:.0f}")

    from matplotlib.colors import LinearSegmentedColormap

    fig, (ax,) = new_axes(1, width=6.2, height=5.0)
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    ax.imshow(diff, cmap=cmap, vmin=-9, vmax=9)
    ax.set_xticks(range(n), names)
    ax.set_yticks(range(n), names)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{diff[i, j]:+.2f}\n±{ci[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color=INK)
    label(ax, f"Mean set differential, row agent minus column agent\n"
              f"({args.games} games per pairing, ± 95% CI)", "opponent", "agent")
    save(fig, os.path.join(args.out, "round_robin.png"))


if __name__ == "__main__":
    main()
