#!/usr/bin/env python
"""Interactive Fish solver.

Run it, type in your cards, then type in every move as it happens at the
table.  Whenever it is your turn it prints the best moves.

    python scripts/solve.py
    python scripts/solve.py --seat 2 --particles 500
    python scripts/solve.py --heuristic      # hand-written scoring, no network

Moves are ranked by the trained network in ``--checkpoint`` (made by
``scripts/train.py``).  If there is none, the hand-written heuristic is used.

Seats are numbered 0-5 clockwise.  Your teammates are the seats 2 and 4 places
after you; everyone else is an opponent.

Commands (the player on turn is always the one acting):

    a <seat> <card> <y/n>     the player on turn asked <seat> for <card>;
                              y = they got it, n = they did not
                              (leave out y/n when you are the one being asked)
    d <set> <y/n> [counts]    the player on turn declared <set>; y = correct.
                              counts = how many cards of the set each seat
                              held, six numbers for seats 0-5
    best                      show the best moves again (your turn only)
    show                      your hand, card counts and the score
    undo                      take back the last move entered
    help                      this text
    quit

Cards: 2S 3S ... 7S, 9S TS JS QS KS AS (suits S H D C), 8S 8H 8D 8C, JK1 JK2.
Sets:  low-S high-S low-H high-H low-D high-D low-C high-C eights (or 0-8).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fish.cards import N_PLAYERS, SET_NAMES  # noqa: E402
from fish.engine import Declare  # noqa: E402
from fish.solver import BadInput, Solver, describe  # noqa: E402

YES = {"y", "yes", "1", "hit", "got", "right", "correct"}
NO = {"n", "no", "0", "miss", "wrong"}


def yes_no(tok: str) -> bool:
    t = tok.lower()
    if t in YES:
        return True
    if t in NO:
        return False
    raise BadInput(f"expected y or n, got {tok!r}")


def seat_of(tok: str) -> int:
    if tok.isdigit() and int(tok) < N_PLAYERS:
        return int(tok)
    raise BadInput(f"expected a seat 0-5, got {tok!r}")


def set_of(tok: str) -> int:
    if tok.isdigit() and int(tok) < len(SET_NAMES):
        return int(tok)
    names = [n.lower() for n in SET_NAMES]
    if tok.lower() in names:
        return names.index(tok.lower())
    raise BadInput(f"unknown set {tok!r}; sets are {' '.join(SET_NAMES)}")


def counts_of(toks: list[str]) -> list[int]:
    if len(toks) != N_PLAYERS or not all(t.isdigit() for t in toks):
        raise BadInput("expected six numbers, one per seat 0-5")
    return [int(t) for t in toks]


def ask_line(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        print()
        raise SystemExit(0)


def show_advice(solver: Solver, top: int, last: dict[int, tuple[int, ...]]) -> None:
    moves = solver.best_moves()
    last.clear()
    for m in moves:
        if isinstance(m.action, Declare):
            last.setdefault(m.action.set_id, m.action.assignment)
    print("\nYour turn. Best moves:")
    for i, m in enumerate(moves[:top], 1):
        if m.certain:
            note = "certain"
        elif isinstance(m.action, Declare):
            note = f"{m.chance:.0%} to be right"
        else:
            note = f"{m.chance:.0%} to get it"
        print(f"  {i}. {describe(m)}   [{note}, value {m.value:+.2f}]")
    print()


def load_net(args: argparse.Namespace):
    """The trained network, or None to use the heuristic."""
    if args.heuristic:
        print("Using the hand-written heuristic.")
        return None
    if not os.path.exists(args.checkpoint):
        print(f"No trained network at {args.checkpoint} (train one with scripts/train.py); "
              "using the hand-written heuristic.")
        return None
    try:
        from fish.value_net import load
    except ImportError:
        print("PyTorch is not installed; using the hand-written heuristic.")
        return None
    try:
        net, info = load(args.checkpoint)
    except ValueError as err:  # trained with an older version of the code
        print(f"Cannot use {args.checkpoint} ({err}); retrain it with scripts/train.py. "
              "Using the hand-written heuristic.")
        return None
    print(f"Using the trained network in {args.checkpoint} ({info.get('games', '?')} training games).")
    return net


def start(args: argparse.Namespace, net) -> Solver:
    print("Seats are 0-5 clockwise. You are seat "
          f"{args.seat}; your teammates are seats {(args.seat + 2) % 6} and {(args.seat + 4) % 6}.")
    while True:
        try:
            if args.midgame:
                declared = [set_of(t) for t in ask_line("Sets already declared (blank if none): ").split()]
                sizes = counts_of(ask_line("Cards held by seats 0-5 (six numbers): ").split())
                score = ask_line("Sets won so far, even seats then odd seats (e.g. 2 1): ").split()
                if len(score) != 2 or not all(t.isdigit() for t in score):
                    raise BadInput("expected two numbers")
                extra = {"declared": declared, "hand_sizes": sizes,
                         "score": (int(score[0]), int(score[1]))}
            else:
                extra = {}
            hand = ask_line("Your cards (e.g. 2S 3S QH 8D JK1 ...): ").replace(",", " ").split()
            first = seat_of(ask_line("Which seat moves first? "))
            return Solver(args.seat, hand, first, n_particles=args.particles,
                          alpha=args.alpha, seed=args.seed, net=net, **extra)
        except BadInput as err:
            print(f"  ! {err}. Let's try again.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive Fish solver.")
    ap.add_argument("--seat", type=int, default=0, help="your seat, 0-5 (default 0)")
    ap.add_argument("--particles", type=int, default=300,
                    help="deals imagined per decision (default 300)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="how strongly asks are read as holding many cards of a set; 0 = off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=5, help="how many moves to show (default 5)")
    ap.add_argument("--midgame", action="store_true",
                    help="start from the middle of a game (asks for card counts and score)")
    ap.add_argument("--checkpoint", default="checkpoints/net.pt", help="trained network to use")
    ap.add_argument("--heuristic", action="store_true",
                    help="use the hand-written heuristic instead of the network")
    args = ap.parse_args()

    solver = start(args, load_net(args))
    print("\n" + solver.summary())
    print("Type 'help' for the commands.\n")
    last: dict[int, tuple[int, ...]] = {}  # declarations suggested in the last advice
    advised = False
    while not solver.done:
        if solver.my_turn and not advised:
            show_advice(solver, args.top, last)
            advised = True
        who = "you" if solver.my_turn else f"seat {solver.turn}"
        toks = ask_line(f"[{who} to move] > ").replace(",", " ").split()
        if not toks:
            continue
        cmd, rest = toks[0].lower(), toks[1:]
        try:
            if cmd in ("q", "quit", "exit"):
                return
            elif cmd in ("h", "help", "?"):
                print(__doc__)
            elif cmd == "show":
                print(solver.summary())
            elif cmd == "best":
                show_advice(solver, args.top, last)
            elif cmd == "undo":
                solver.undo()
                advised = False
                print("  undone.")
            elif cmd in ("a", "ask"):
                if len(rest) not in (2, 3):
                    raise BadInput("use: a <seat> <card> <y/n>")
                obs = solver.ask(seat_of(rest[0]), rest[1],
                                 yes_no(rest[2]) if len(rest) == 3 else None)
                print(f"  seat {obs.asker} asked seat {obs.target}: "
                      f"{'got the card' if obs.success else 'no luck'}.")
                advised = False
            elif cmd in ("d", "declare"):
                if len(rest) < 2:
                    raise BadInput("use: d <set> <y/n> [six counts]")
                h, ok = set_of(rest[0]), yes_no(rest[1])
                if len(rest) > 2:
                    obs = solver.declare(h, ok, lost=counts_of(rest[2:]))
                elif ok and solver.my_turn and h in last:
                    obs = solver.declare(h, ok, assignment=last[h])  # the suggested layout was right
                else:
                    counts = counts_of(ask_line(
                        "  cards of that set held by seats 0-5 (six numbers): ").split())
                    obs = solver.declare(h, ok, lost=counts)
                print(f"  seat {obs.declarer} declared {SET_NAMES[h]}: "
                      f"{'correct' if obs.correct else 'wrong'}.")
                advised = False
            else:
                raise BadInput(f"unknown command {cmd!r}; type 'help'")
        except BadInput as err:
            print(f"  ! {err}")
    print("\nGame over.\n" + solver.summary())


if __name__ == "__main__":
    main()
