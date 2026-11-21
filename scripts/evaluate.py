#!/usr/bin/env python
"""Measure how good a player is by playing it against others.

    python scripts/evaluate.py                         # the solver vs rule-based and heuristic
    python scripts/evaluate.py --games 200 --against rule random heuristic
    python scripts/evaluate.py --player heuristic --against rule    # the heuristic itself

Players:

    solver      the trained value network (needs --checkpoint and PyTorch)
    heuristic   hand-written scoring, nothing learned
    rule        asks for cards it knows about, otherwise random
    random      random legal asks

Every deal is played twice with the teams swapped, so card luck mostly
cancels.  For each opponent it reports the player's average set margin (out of
9 sets; positive = better than the opponent), a 95% confidence interval and
the share of games won.

Every finished game is appended to a CSV file, so a run that is stopped can be
continued with ``--resume``.
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from fish.baselines import RandomAgent, RuleBasedAgent  # noqa: E402
from fish.env import duel  # noqa: E402
from fish.heuristic import HeuristicAgent  # noqa: E402

FIELDS = ["player", "opponent", "game", "deal", "margin", "length", "truncated", "seconds"]
PLAYERS = ("solver", "heuristic", "rule", "random")
_cache: dict = {}


def make_player(kind: str, opts: dict):
    """Build a player by name (cached per process)."""
    if kind not in _cache:
        if kind == "random":
            _cache[kind] = RandomAgent()
        elif kind == "rule":
            _cache[kind] = RuleBasedAgent()
        elif kind == "heuristic":
            _cache[kind] = HeuristicAgent(opts["heuristic_particles"], opts["alpha"])
        elif kind == "solver":
            import torch

            from fish.chooser import NetAgent
            from fish.value_net import load

            torch.set_num_threads(1)
            net, info = load(opts["checkpoint"])
            _cache[kind] = NetAgent(
                net,
                n_particles=opts["particles"] or int(info.get("particles", 64)),
                max_asks=opts["max_asks"] or int(info.get("max_asks", 12)),
                alpha=opts["alpha"],
            )
        else:
            raise ValueError(f"unknown player {kind!r}")
    return _cache[kind]


def run_game(job: tuple) -> dict:
    """Play one game (runs inside a worker process)."""
    player, opponent, game, seed, opts = job
    t0 = time.perf_counter()
    margin, length, truncated = duel(make_player(player, opts), make_player(opponent, opts),
                                     game, seed)
    return {"player": player, "opponent": opponent, "game": game, "deal": game // 2,
            "margin": margin, "length": length, "truncated": int(truncated),
            "seconds": round(time.perf_counter() - t0, 3)}


def summarize(rows: list[dict]) -> dict:
    """Average margin, 95% confidence interval (over deals) and win rate."""
    margins = np.array([float(r["margin"]) for r in rows])
    by_deal: dict[int, list[float]] = {}
    for r in rows:
        by_deal.setdefault(int(r["deal"]), []).append(float(r["margin"]))
    per_deal = np.array([np.mean(v) for v in by_deal.values()])
    ci = 1.96 * per_deal.std(ddof=1) / math.sqrt(len(per_deal)) if len(per_deal) > 1 else math.nan
    return {"games": len(rows), "margin": float(margins.mean()), "ci95": float(ci),
            "wins": float((margins > 0).mean()),
            "length": float(np.mean([float(r["length"]) for r in rows])),
            "truncated": int(sum(int(r["truncated"]) for r in rows))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Play one Fish player against others.")
    ap.add_argument("--player", choices=PLAYERS, default="solver", help="the player being measured")
    ap.add_argument("--against", nargs="+", choices=PLAYERS, default=["rule", "heuristic"],
                    help="opponents to play")
    ap.add_argument("--games", type=int, default=100,
                    help="games per opponent (rounded up to an even number)")
    ap.add_argument("--checkpoint", default="checkpoints/net.pt", help="trained network to load")
    ap.add_argument("--out", default="results/evaluate.csv", help="per-game results file")
    ap.add_argument("--resume", action="store_true", help="keep finished games already in --out")
    ap.add_argument("--workers", type=int, default=2, help="games played at the same time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--particles", type=int, default=0,
                    help="deals imagined per decision by the solver (0 = as trained)")
    ap.add_argument("--max-asks", type=int, default=0,
                    help="asks examined per decision by the solver (0 = as trained)")
    ap.add_argument("--heuristic-particles", type=int, default=150,
                    help="deals imagined per decision by the heuristic")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    kinds = {args.player, *args.against}
    if "solver" in kinds and not os.path.exists(args.checkpoint):
        sys.exit(f"no checkpoint at {args.checkpoint}; train one with scripts/train.py "
                 "or use --player heuristic")
    opponents = [o for o in dict.fromkeys(args.against) if o != args.player]
    if not opponents:
        sys.exit("nothing to do: the only opponent is the player itself")
    games = args.games + args.games % 2
    opts = {"checkpoint": args.checkpoint, "particles": args.particles, "max_asks": args.max_asks,
            "heuristic_particles": args.heuristic_particles, "alpha": args.alpha}

    rows: list[dict] = []
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["player"] == args.player]
    done = {(r["opponent"], int(r["game"])) for r in rows}
    jobs = [(args.player, o, g, args.seed, opts)
            for o in opponents for g in range(games) if (o, g) not in done]
    print(f"{args.player} vs {', '.join(opponents)}: {games} games each, "
          f"{len(jobs)} to play, {args.workers} at a time")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fresh = not (args.resume and os.path.exists(args.out))
    t0 = time.perf_counter()
    with open(args.out, "w" if fresh else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()

        def record(row: dict) -> None:
            writer.writerow(row)
            f.flush()
            rows.append(row)
            n = len(rows) - len(done)
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} games  ({time.perf_counter() - t0:.0f}s)", flush=True)

        try:
            if args.workers <= 1 or len(jobs) <= 1:
                for job in jobs:
                    record(run_game(job))
            else:
                with mp.get_context("spawn").Pool(args.workers) as pool:
                    for row in pool.imap_unordered(run_game, jobs):
                        record(row)
        except KeyboardInterrupt:
            print("\ninterrupted; finished games are saved, continue with --resume")

    print(f"\n{args.player} against:")
    for o in opponents:
        mine = [r for r in rows if r["opponent"] == o and int(r["game"]) < games]
        if not mine:
            continue
        s = summarize(mine)
        cut = f", {s['truncated']} cut off" if s["truncated"] else ""
        print(f"  {o:<9} {s['margin']:+.2f} sets  (95% CI ±{s['ci95']:.2f})   "
              f"wins {s['wins']:.0%}   {s['games']} games, {s['length']:.0f} moves on average{cut}")
    print(f"per-game results: {args.out}")


if __name__ == "__main__":
    main()
