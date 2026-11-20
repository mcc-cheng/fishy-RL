#!/usr/bin/env python
"""Train the value network by self-play.

    python scripts/train.py --games 2000
    python scripts/train.py --games 5000 --resume      # continue from the checkpoint

Each round: the current network plays one game against itself, the positions
of that game go into a memory of recent positions, and the network takes a few
learning steps on random positions from that memory.

A checkpoint is written every ``--save-every`` games (and at the end, and when
you press Ctrl-C), so a run can be stopped and continued with ``--resume``.
Progress is printed and appended to a CSV file next to the checkpoint.

Memory use is dominated by ``--buffer`` (about 1.5 MB per thousand positions,
so 60 MB at the default).  Lower ``--particles`` or ``--max-asks`` to make
games faster.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fish.baselines import RuleBasedAgent  # noqa: E402
from fish.chooser import NetAgent  # noqa: E402
from fish.features import FEATURE_SIZE  # noqa: E402
from fish.selfplay import ReplayBuffer, compare, play_training_game, td_targets  # noqa: E402
from fish.value_net import ValueNet, load, save  # noqa: E402

LOG_FIELDS = ["games", "loss", "length", "truncated", "epsilon", "sec_per_game",
              "vs_rule_margin", "vs_rule_wins"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the Fish value network by self-play.")
    ap.add_argument("--games", type=int, default=2000, help="total self-play games to reach")
    ap.add_argument("--out", default="checkpoints/net.pt", help="checkpoint file")
    ap.add_argument("--resume", action="store_true", help="continue from the checkpoint in --out")
    ap.add_argument("--seed", type=int, default=0)
    # network
    ap.add_argument("--hidden", type=int, default=256, help="width of the hidden layers")
    ap.add_argument("--layers", type=int, default=2, help="number of hidden layers")
    # playing
    ap.add_argument("--particles", type=int, default=48, help="deals imagined per decision")
    ap.add_argument("--max-asks", type=int, default=10, help="asks examined per decision")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="how strongly asks are read as holding many cards of a set")
    ap.add_argument("--eps-start", type=float, default=0.25, help="random-move rate at the start")
    ap.add_argument("--eps-end", type=float, default=0.03, help="random-move rate at the end")
    ap.add_argument("--max-actions", type=int, default=2000, help="cut a game off after this many moves")
    # learning
    ap.add_argument("--lam", type=float, default=0.8,
                    help="1 = learn from final results only, lower = lean on the next prediction")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    ap.add_argument("--batch", type=int, default=256, help="positions per learning step")
    ap.add_argument("--updates", type=int, default=8, help="learning steps per game")
    ap.add_argument("--buffer", type=int, default=40000, help="positions kept in memory")
    ap.add_argument("--keep", type=float, default=0.25,
                    help="fraction of each game's positions that is stored")
    # housekeeping
    ap.add_argument("--save-every", type=int, default=50, help="games between checkpoints")
    ap.add_argument("--eval-games", type=int, default=10,
                    help="games against the rule-based player at each checkpoint (0 = none)")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads PyTorch may use")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)

    start = 0
    if args.resume and os.path.exists(args.out):
        net, info = load(args.out)
        start = int(info.get("games", 0))
        print(f"resumed from {args.out} at {start} games")
    else:
        if args.resume:
            print(f"no checkpoint at {args.out}; starting fresh")
        net = ValueNet(args.hidden, args.layers)
    if start >= args.games:
        print(f"already at {start} games; raise --games to train further")
        return

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer)
    rng = np.random.default_rng([args.seed, start])
    play_opts = {"n_particles": args.particles, "alpha": args.alpha, "max_asks": args.max_asks}
    log_path = os.path.splitext(args.out)[0] + "_log.csv"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new_log = not (args.resume and os.path.exists(log_path))
    log_file = open(log_path, "w" if new_log else "a", newline="")
    log = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    if new_log:
        log.writeheader()

    def checkpoint(done: int, stats: dict) -> None:
        margin = wins = float("nan")
        if args.eval_games > 0:
            agent = NetAgent(net, **play_opts)
            margin, wins = compare(agent, RuleBasedAgent(), args.eval_games, seed=args.seed + 12345)
        save(net, args.out, games=done, hidden=args.hidden, layers=args.layers,
             particles=args.particles, max_asks=args.max_asks, alpha=args.alpha)
        row = {"games": done, **stats, "vs_rule_margin": margin, "vs_rule_wins": wins}
        log.writerow(row)
        log_file.flush()
        vs = "" if args.eval_games <= 0 else f"  vs rule-based: {margin:+.2f} sets, {wins:.0%} wins"
        print(f"[{done:>6} games] loss {stats['loss']:.4f}  length {stats['length']:.0f}  "
              f"eps {stats['epsilon']:.2f}  {stats['sec_per_game']:.1f}s/game{vs}", flush=True)

    losses: list[float] = []
    lengths: list[int] = []
    cut = 0
    eps = args.eps_start
    t0 = time.perf_counter()
    since = 0
    done = start
    stats = {"loss": float("nan"), "length": float("nan"), "truncated": 0,
             "epsilon": eps, "sec_per_game": float("nan")}
    try:
        for game in range(start, args.games):
            frac = game / max(1, args.games - 1)
            eps = args.eps_start + (args.eps_end - args.eps_start) * frac
            frames, margin, length, truncated = play_training_game(
                net, rng, eps, args.max_actions, **play_opts)
            targets = td_targets(net, frames, margin, args.lam)
            x = frames.reshape(-1, FEATURE_SIZE)
            y = targets.reshape(-1)
            pick = rng.random(len(x)) < args.keep
            pick[-6:] = True  # always keep the final position
            buffer.add(x[pick], y[pick])
            lengths.append(length)
            cut += int(truncated)

            if buffer.size >= args.batch:
                net.train()
                for _ in range(args.updates):
                    bx, by = buffer.sample(args.batch, rng)
                    loss = torch.mean((net(torch.as_tensor(bx)) - torch.as_tensor(by)) ** 2)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(float(loss))

            done = game + 1
            since += 1
            if done % args.save_every == 0 or done == args.games:
                stats = {
                    "loss": float(np.mean(losses)) if losses else float("nan"),
                    "length": float(np.mean(lengths)),
                    "truncated": cut,
                    "epsilon": eps,
                    "sec_per_game": (time.perf_counter() - t0) / since,
                }
                checkpoint(done, stats)
                losses, lengths, cut, since = [], [], 0, 0
                t0 = time.perf_counter()
    except KeyboardInterrupt:
        print("\ninterrupted; saving")
        save(net, args.out, games=done, hidden=args.hidden, layers=args.layers,
             particles=args.particles, max_asks=args.max_asks, alpha=args.alpha)
    finally:
        log_file.close()
    print(f"checkpoint: {args.out}   log: {log_path}")


if __name__ == "__main__":
    main()
