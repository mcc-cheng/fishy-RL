#!/usr/bin/env python
"""Train the solver's value network by self-play.

    python scripts/train.py --games 2000
    python scripts/train.py --games 5000 --resume      # continue from the checkpoint

Each round: one game is played, the position after every move goes into a
memory of recent positions, and the network takes a few learning steps on
random positions from that memory.  For every set still in play in a position,
the network learns to predict who ended up with it: +1 the player's team, -1
the opponents.

The printed loss is the average squared error of those predictions.  During
the warm start, 1.0 is what always answering "no idea" would score.  After it,
the targets lean on the network's own later predictions, so the loss is much
lower and no longer comparable: it drops as the warm-start positions leave the
memory, not because the network suddenly got better.  The games against the
two baselines are the real measure of progress.

Who plays the games:

  1. Warm start (the first ``--warmup`` games): all six seats are the
     hand-written heuristic, and the network learns how those games turn out.
  2. After that, most games are the network against itself.  In a share of
     them (``--heuristic-share``) the network's team plays a team of three
     heuristic players, and in another share (``--past-share``) a team of an
     older saved copy of itself.  From those games the network learns only
     from its own team's seats.

For comparison runs: ``--warmup 0 --heuristic-share 0`` is pure self-play and
``--heuristic-share 1`` always plays the heuristic.

A checkpoint is written every ``--save-every`` games (and at the end, and when
you press Ctrl-C), so a run can be stopped and continued with ``--resume``.
Progress is printed and appended to a CSV file next to the checkpoint.

The memory of positions is small (about 0.4 MB per thousand, so 17 MB at the
default ``--buffer``).  Lower ``--particles`` or ``--max-asks`` to make games
faster.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fish.baselines import RuleBasedAgent  # noqa: E402
from fish.chooser import NetAgent  # noqa: E402
from fish.heuristic import HeuristicAgent  # noqa: E402
from fish.cards import N_PLAYERS  # noqa: E402
from fish.selfplay import ReplayBuffer, compare, play_training_game, td_targets  # noqa: E402
from fish.value_net import ValueNet, load, save  # noqa: E402

LOG_FIELDS = ["games", "loss", "length", "truncated", "epsilon", "sec_per_game",
              "vs_rule_margin", "vs_rule_wins", "vs_heuristic_margin", "vs_heuristic_wins"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the Fish value network by self-play.")
    ap.add_argument("--games", type=int, default=2000, help="total self-play games to reach")
    ap.add_argument("--out", default="checkpoints/net.pt", help="checkpoint file")
    ap.add_argument("--resume", action="store_true", help="continue from the checkpoint in --out")
    ap.add_argument("--seed", type=int, default=0)
    # network
    ap.add_argument("--hidden", type=int, default=64, help="width of the hidden layers")
    ap.add_argument("--layers", type=int, default=2, help="number of hidden layers")
    # playing
    ap.add_argument("--particles", type=int, default=48, help="deals imagined per decision")
    ap.add_argument("--max-asks", type=int, default=10, help="asks examined per decision")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="how strongly asks are read as holding many cards of a set")
    ap.add_argument("--eps-start", type=float, default=0.10, help="random-move rate at the start")
    ap.add_argument("--eps-end", type=float, default=0.02, help="random-move rate at the end")
    ap.add_argument("--max-actions", type=int, default=300,
                    help="cut a training game off after this many moves (a normal game takes about 100)")
    # who plays
    ap.add_argument("--warmup", type=int, default=200,
                    help="games at the start played entirely by the heuristic")
    ap.add_argument("--heuristic-share", type=float, default=0.3,
                    help="after the warm start, fraction of games against a heuristic team")
    ap.add_argument("--past-share", type=float, default=0.0,
                    help="after the warm start, fraction of games against an older copy of the network")
    ap.add_argument("--past-pool", type=int, default=5, help="older copies kept to play against")
    ap.add_argument("--heuristic-particles", type=int, default=100,
                    help="deals imagined per decision by the heuristic")
    # learning
    ap.add_argument("--lam", type=float, default=0.9,
                    help="1 = learn from real outcomes only, lower = lean on the next prediction")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    ap.add_argument("--batch", type=int, default=256, help="positions per learning step")
    ap.add_argument("--updates", type=int, default=16, help="learning steps per game")
    ap.add_argument("--buffer", type=int, default=40000, help="positions kept in memory")
    ap.add_argument("--keep", type=float, default=1.0,
                    help="fraction of each game's positions that is stored")
    # housekeeping
    ap.add_argument("--save-every", type=int, default=100, help="games between checkpoints")
    ap.add_argument("--eval-games", type=int, default=20,
                    help="games against the rule-based player and against the heuristic "
                         "at each checkpoint (0 = none)")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads PyTorch may use")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)

    start = 0
    if args.resume and os.path.exists(args.out):
        try:
            net, info = load(args.out)
        except ValueError as err:  # written by an older version of the code
            raise SystemExit(f"cannot resume from {args.out}: {err}. "
                             "Move it aside and train without --resume.")
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
    if args.heuristic_share + args.past_share > 1.0:
        raise SystemExit("--heuristic-share and --past-share must not add up to more than 1")
    heuristic = HeuristicAgent(args.heuristic_particles, args.alpha)
    past: list[NetAgent] = []  # older copies of the network, newest last
    all_seats = tuple(range(N_PLAYERS))
    print(f"warm start: {args.warmup} heuristic games; then {args.heuristic_share:.0%} of games "
          f"against the heuristic, {args.past_share:.0%} against older copies, the rest self-play")
    log_path = os.path.splitext(args.out)[0] + "_log.csv"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new_log = not (start > 0 and os.path.exists(log_path))
    log_file = open(log_path, "w" if new_log else "a", newline="")
    log = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    if new_log:
        log.writeheader()

    def checkpoint(done: int, stats: dict) -> None:
        margin = wins = h_margin = h_wins = float("nan")
        if args.eval_games > 0:
            agent = NetAgent(net, **play_opts)
            margin, wins = compare(agent, RuleBasedAgent(), args.eval_games,
                                   seed=args.seed + 12345, max_actions=1000)
            h_margin, h_wins = compare(agent, heuristic, args.eval_games,
                                       seed=args.seed + 12345, max_actions=1000)
        save(net, args.out, games=done, hidden=args.hidden, layers=args.layers,
             particles=args.particles, max_asks=args.max_asks, alpha=args.alpha)
        row = {"games": done, **stats, "vs_rule_margin": margin, "vs_rule_wins": wins,
               "vs_heuristic_margin": h_margin, "vs_heuristic_wins": h_wins}
        log.writerow(row)
        log_file.flush()
        vs = "" if args.eval_games <= 0 else (
            f"  vs rule: {margin:+.2f} sets, {wins:.0%} wins"
            f"  vs heuristic: {h_margin:+.2f} sets, {h_wins:.0%} wins")
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
            frac = max(0, game - args.warmup) / max(1, args.games - 1 - args.warmup)
            eps = args.eps_start + (args.eps_end - args.eps_start) * frac

            # who plays this game, and which seats the network learns from
            others, other_seats, learn_seats, lam = None, (), all_seats, args.lam
            if game < args.warmup:
                # heuristic in every seat; learn the real outcomes of its games
                others, other_seats, lam = heuristic, all_seats, 1.0
            else:
                r = rng.random()
                if r < args.heuristic_share:
                    others = heuristic
                elif r < args.heuristic_share + args.past_share and past:
                    others = past[int(rng.integers(len(past)))]
                if others is not None:
                    team = int(rng.integers(2))  # the side the other players take
                    other_seats = tuple(s for s in all_seats if s % 2 == team)
                    learn_seats = tuple(s for s in all_seats if s % 2 != team)

            played = play_training_game(
                net, rng, eps, args.max_actions, others, other_seats, learn_seats, **play_opts)
            targets = td_targets(net, played, lam)
            pick = rng.random(len(played.frames)) < args.keep
            buffer.add(played.frames[pick], targets[pick], played.live[pick])
            lengths.append(played.length)
            cut += int(played.truncated)

            if buffer.size >= args.batch:
                net.train()
                for _ in range(args.updates):
                    bx, by, live = buffer.sample(args.batch, rng)
                    counts = torch.as_tensor(live)  # only sets still in play have a target
                    error = (net(torch.as_tensor(bx)) - torch.as_tensor(by)) ** 2
                    loss = (error * counts).sum() / counts.sum().clamp(min=1.0)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())

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
                if args.past_share > 0 and done >= args.warmup:
                    past.append(NetAgent(copy.deepcopy(net).eval(), name="past", **play_opts))
                    del past[:-max(1, args.past_pool)]
                losses, lengths, cut, since = [], [], 0, 0
                t0 = time.perf_counter()
            elif done % 25 == 0:  # a sign of life between checkpoints
                print(f"  ... {done} games ({(time.perf_counter() - t0) / since:.1f}s/game)",
                      flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted; saving")
        save(net, args.out, games=done, hidden=args.hidden, layers=args.layers,
             particles=args.particles, max_asks=args.max_asks, alpha=args.alpha)
    finally:
        log_file.close()
    print(f"checkpoint: {args.out}   log: {log_path}")


if __name__ == "__main__":
    main()
