"""Belief calibration: reliability diagram and Brier score vs turn.

GreedyBelief agents play each other; at every turn the acting player's belief
is scored against the true deal for three estimators: the particle filter, the
particle filter with ask-likelihood reweighting, and the particle-free heuristic
used inside rollouts.

    python scripts/calibration.py --games 200
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
from common import INK2, SERIES, base_parser, label, legend, new_axes, save

from fish.agents import heuristic_marginals, make_agent
from fish.arena import play_game, write_csv
from fish.belief import ask_likelihood_weights, belief_scores, marginals, sample_deals

N_BINS = 10
TURN_BUCKET = 10
METHODS = ["particles", "particles+reweight", "heuristic", "uniform-possible"]


def one_game(args: tuple[int, int, int, float, str]) -> dict:
    """Play one game and return calibration bins and per-turn scores."""
    seed, game, n_particles, alpha, policy = args
    rng = np.random.default_rng([seed, game, 7])
    bins = {m: np.zeros((N_BINS, 3)) for m in METHODS}  # count, sum p, sum outcome
    turns: list[tuple[int, str, float, float]] = []

    def hook(t, state, view, agent, dt):
        parts = sample_deals(view.know, n_particles, rng)
        est = {
            "particles": marginals(parts),
            "particles+reweight": marginals(parts, ask_likelihood_weights(parts, view.know, alpha)),
            "heuristic": heuristic_marginals(view.know),
        }
        poss = view.know.possible
        est["uniform-possible"] = poss / np.maximum(poss.sum(0), 1)
        cards = np.array([c for c in range(54) if state.owner[c] >= 0 and not view.hand >> c & 1])
        if cards.size == 0:
            return
        others = [p for p in range(6) if p != view.player]
        truth = (state.owner[cards][None, :] == np.array(others)[:, None]).ravel()
        for m, P in est.items():
            b, ll, _ = belief_scores(P, state, view.hand)
            turns.append((t // TURN_BUCKET, m, b, ll))
            pr = P[np.ix_(others, cards)].ravel()
            idx = np.minimum((pr * N_BINS).astype(int), N_BINS - 1)
            np.add.at(bins[m], (idx, 0), 1)
            np.add.at(bins[m], (idx, 1), pr)
            np.add.at(bins[m], (idx, 2), truth)

    agents = [make_agent(policy) for _ in range(6)]
    play_game(agents, np.random.default_rng([seed, game]), np.random.default_rng([seed, game, 1]),
              on_turn=hook)
    return {"bins": bins, "turns": turns}


def main() -> None:
    p = base_parser(__doc__, games=200)
    p.add_argument("--particles", type=int, default=300)
    p.add_argument("--alpha", type=float, default=1.0, help="ask-likelihood exponent")
    p.add_argument("--policy", default="greedy", help="agent kind that plays the games")
    args = p.parse_args()
    jobs = [(args.seed, g, args.particles, args.alpha, args.policy) for g in range(args.games)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(one_game, jobs, chunksize=1)

    bins = {m: sum(r["bins"][m] for r in res) for m in METHODS}
    bin_rows = []
    for m in METHODS:
        for i in range(N_BINS):
            n, sp, so = bins[m][i]
            if n:
                bin_rows.append({"method": m, "bin": i, "count": int(n), "mean_predicted": sp / n,
                                 "observed_frequency": so / n})
    write_csv(bin_rows, os.path.join(args.out, "calibration_bins.csv"))

    acc: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for r in res:
        for bucket, m, b, ll in r["turns"]:
            acc.setdefault((m, bucket), []).append((b, ll))
    turn_rows = [{"method": m, "turn_from": bucket * TURN_BUCKET, "n": len(v),
                  "brier": float(np.mean([x[0] for x in v])),
                  "logloss": float(np.mean([x[1] for x in v]))}
                 for (m, bucket), v in sorted(acc.items())]
    write_csv(turn_rows, os.path.join(args.out, "brier_by_turn.csv"))

    print(f"{'method':20s} {'Brier':>7s} {'log-loss':>9s} {'ECE':>7s}   (per-turn means over {args.games} games)")
    for m in METHODS:
        v = [x for (mm, _), xs in acc.items() if mm == m for x in xs]
        tot = bins[m][:, 0].sum()
        ece = sum(abs(sp - so) for n, sp, so in bins[m] if n) / tot
        print(f"{m:20s} {np.mean([x[0] for x in v]):7.4f} {np.mean([x[1] for x in v]):9.4f} {ece:7.4f}")

    fig, (a1, a2) = new_axes(2)
    a1.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--", label="perfect calibration")
    for i, m in enumerate(METHODS):
        rows = [r for r in bin_rows if r["method"] == m]
        a1.plot([r["mean_predicted"] for r in rows], [r["observed_frequency"] for r in rows],
                color=SERIES[i], lw=2, marker="o", ms=5, label=m)
        rows = [r for r in turn_rows if r["method"] == m and r["n"] >= max(5, args.games // 10)]
        a2.plot([r["turn_from"] + TURN_BUCKET / 2 for r in rows], [r["brier"] for r in rows],
                color=SERIES[i], lw=2, marker="o", ms=4, label=m)
    label(a1, "Reliability: P(opponent/teammate holds card)", "predicted probability",
          "observed frequency")
    label(a2, "Belief Brier score by turn (lower is better)", f"turn (buckets of {TURN_BUCKET})",
          "mean multi-class Brier score over hidden cards")
    legend(a1, loc="upper left")
    legend(a2)
    save(fig, os.path.join(args.out, "calibration.png"))


if __name__ == "__main__":
    main()
