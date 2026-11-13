"""Arena: play seeded games between agent configurations and log metrics.

A *spec* is ``(kind, kwargs)`` for :func:`fish.agents.make_agent`, so matches
can be shipped to worker processes.  Games come in pairs: deal ``d`` is played
once with config X on team A and once with X on team B, with a random first
player per deal, which removes most deal luck from the set differential.
"""
from __future__ import annotations

import csv
import math
import multiprocessing as mp
import os
import time
from typing import Any, Callable, Sequence

import numpy as np

from .agents import Agent, make_agent
from .belief import belief_scores
from .cards import N_PLAYERS
from .engine import Declare, State, is_terminal, new_game, step
from .knowledge import Knowledge, View

Spec = tuple[str, dict[str, Any]]
TurnHook = Callable[[int, State, View, Agent, float], None]

RESULT_FIELDS = [
    "match", "game", "deal", "seed", "x", "y", "x_team", "sets_x", "sets_y", "diff",
    "length", "truncated", "decl_x", "decl_correct_x", "decl_y", "decl_correct_y",
    "brier_x", "logloss_x", "brier_y", "logloss_y",
    "time_mean_x", "time_max_x", "time_mean_y", "time_max_y",
]
TURN_FIELDS = ["match", "game", "turn", "player", "config", "live_sets", "action",
               "brier", "logloss", "seconds"]


def spec_name(spec: Spec) -> str:
    """Display name of a spec (its ``name`` kwarg or its kind)."""
    return str(spec[1].get("name", spec[0]))


def play_game(
    agents: Sequence[Agent],
    deal_rng: np.random.Generator,
    agent_rng: np.random.Generator,
    max_actions: int = 3000,
    check: bool = False,
    on_turn: TurnHook | None = None,
    turn_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Play one game; ``agents[p]`` controls seat ``p``.

    Agents receive only their :class:`View`.  With ``check=True`` every view is
    verified against the true state (slow; for tests).  Returns per-seat
    statistics; a game exceeding ``max_actions`` is truncated (remaining sets
    are not scored).
    """
    state = new_game(deal_rng)
    public = Knowledge(state.turn)
    n = 0
    decl = [0] * N_PLAYERS
    decl_ok = [0] * N_PLAYERS
    times: list[list[float]] = [[] for _ in range(N_PLAYERS)]
    briers: list[list[float]] = [[] for _ in range(N_PLAYERS)]
    losses: list[list[float]] = [[] for _ in range(N_PLAYERS)]
    while not is_terminal(state) and n < max_actions:
        p = state.turn
        agent = agents[p]
        view = public.view_for(p, state)
        if check:
            bad = public.violations(state) + view.know.violations(state)
            if bad:
                raise AssertionError(f"knowledge inconsistent with state: {bad}")
        if hasattr(agent, "last_marginals"):
            agent.last_marginals = None
        t0 = time.perf_counter()
        action = agent.act(view, agent_rng)
        dt = time.perf_counter() - t0
        times[p].append(dt)
        brier = logloss = math.nan
        P = getattr(agent, "last_marginals", None)
        if P is not None:
            brier, logloss, cnt = belief_scores(P, state, view.hand)
            if cnt:
                briers[p].append(brier)
                losses[p].append(logloss)
        if on_turn is not None:
            on_turn(n, state, view, agent, dt)
        if turn_rows is not None:
            turn_rows.append({
                "turn": n, "player": p, "config": agent.name,
                "live_sets": 9 - bin(state.declared).count("1"),
                "action": "declare" if isinstance(action, Declare) else "ask",
                "brier": brier, "logloss": logloss, "seconds": dt,
            })
        state, obs = step(state, action)
        public.update(obs)
        if isinstance(action, Declare):
            decl[p] += 1
            decl_ok[p] += obs.correct
        n += 1
    return {
        "score": state.score, "length": n, "truncated": not is_terminal(state),
        "decl": decl, "decl_ok": decl_ok, "times": times, "briers": briers, "losses": losses,
    }


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else math.nan


def play_match_game(x: Spec, y: Spec, game: int, seed: int, match: str = "",
                    max_actions: int = 3000, check: bool = False,
                    turn_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Game ``game`` of the X-vs-Y match: deal ``game // 2``, X on team ``game % 2``."""
    deal = game // 2
    x_team = game % 2
    seats = [make_agent(x[0], **x[1]) if p % 2 == x_team else make_agent(y[0], **y[1])
             for p in range(N_PLAYERS)]
    rows: list[dict[str, Any]] | None = [] if turn_rows is not None else None
    res = play_game(seats, np.random.default_rng([seed, deal]),
                    np.random.default_rng([seed, game, 1]), max_actions, check, None, rows)
    if rows is not None and turn_rows is not None:
        for r in rows:
            r.update(match=match, game=game)
        turn_rows.extend(rows)
    xs = [p for p in range(N_PLAYERS) if p % 2 == x_team]
    ys = [p for p in range(N_PLAYERS) if p % 2 != x_team]

    def cat(key: str, seats_: list[int]) -> list[float]:
        return [v for p in seats_ for v in res[key][p]]

    tx, ty = cat("times", xs), cat("times", ys)
    sx, sy = res["score"][x_team], res["score"][1 - x_team]
    return {
        "match": match, "game": game, "deal": deal, "seed": seed,
        "x": spec_name(x), "y": spec_name(y), "x_team": x_team,
        "sets_x": sx, "sets_y": sy, "diff": sx - sy,
        "length": res["length"], "truncated": int(res["truncated"]),
        "decl_x": sum(res["decl"][p] for p in xs),
        "decl_correct_x": sum(res["decl_ok"][p] for p in xs),
        "decl_y": sum(res["decl"][p] for p in ys),
        "decl_correct_y": sum(res["decl_ok"][p] for p in ys),
        "brier_x": _mean(cat("briers", xs)), "logloss_x": _mean(cat("losses", xs)),
        "brier_y": _mean(cat("briers", ys)), "logloss_y": _mean(cat("losses", ys)),
        "time_mean_x": _mean(tx), "time_max_x": max(tx, default=math.nan),
        "time_mean_y": _mean(ty), "time_max_y": max(ty, default=math.nan),
    }


def _worker(args: tuple[Spec, Spec, int, int, str, int, bool]) -> tuple[dict, list[dict]]:
    x, y, game, seed, match, max_actions, log_turns = args
    turn_rows: list[dict[str, Any]] | None = [] if log_turns else None
    row = play_match_game(x, y, game, seed, match, max_actions, False, turn_rows)
    return row, turn_rows or []


def run_jobs(jobs: list[tuple], workers: int | None = None,
             progress: Callable[[int, int], None] | None = None) -> list[tuple[dict, list[dict]]]:
    """Run match-game jobs (see :func:`match_jobs`), in parallel when ``workers > 1``."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    out = []
    if workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs):
            out.append(_worker(job))
            if progress:
                progress(i + 1, len(jobs))
        return out
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for i, res in enumerate(pool.imap(_worker, jobs, chunksize=1)):
            out.append(res)
            if progress:
                progress(i + 1, len(jobs))
    return out


def match_jobs(x: Spec, y: Spec, n_games: int, seed: int, match: str | None = None,
               max_actions: int = 3000, log_turns: bool = False) -> list[tuple]:
    """Job list for an ``n_games`` match between two specs."""
    name = match or f"{spec_name(x)}_vs_{spec_name(y)}"
    return [(x, y, g, seed, name, max_actions, log_turns) for g in range(n_games)]


def play_match(x: Spec, y: Spec, n_games: int, seed: int = 0, workers: int | None = None,
               match: str | None = None, max_actions: int = 3000, log_turns: bool = False,
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Play ``n_games`` between X and Y; returns ``(game_rows, turn_rows)``.

    Results depend only on ``seed`` (not on the number of workers).
    """
    res = run_jobs(match_jobs(x, y, n_games, seed, match, max_actions, log_turns), workers)
    return [r for r, _ in res], [t for _, ts in res for t in ts]


def write_csv(rows: list[dict[str, Any]], path: str, fields: list[str] | None = None) -> None:
    """Write dict rows to ``path`` (creating directories)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: str) -> list[dict[str, str]]:
    """Read a CSV written by :func:`write_csv`."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Match summary from X's perspective.

    The 95% CI of the mean set differential treats the two games of a deal
    (teams swapped) as one sample, since they are correlated.
    """
    diff = np.array([float(r["diff"]) for r in rows])
    deals: dict[Any, list[float]] = {}
    for r in rows:
        deals.setdefault(r["deal"], []).append(float(r["diff"]))
    per_deal = np.array([np.mean(v) for v in deals.values()])
    half = 1.96 * per_deal.std(ddof=1) / math.sqrt(len(per_deal)) if len(per_deal) > 1 else math.nan

    def tot(key: str) -> float:
        return float(sum(float(r[key]) for r in rows))

    def avg(key: str) -> float:
        v = [float(r[key]) for r in rows if not math.isnan(float(r[key]))]
        return float(np.mean(v)) if v else math.nan

    return {
        "games": len(rows), "mean_diff": float(diff.mean()), "ci95": float(half),
        "win_rate": float((diff > 0).mean()), "loss_rate": float((diff < 0).mean()),
        "length": avg("length"), "truncated": tot("truncated"),
        "decl_acc_x": tot("decl_correct_x") / max(tot("decl_x"), 1),
        "decl_acc_y": tot("decl_correct_y") / max(tot("decl_y"), 1),
        "brier_x": avg("brier_x"), "brier_y": avg("brier_y"),
        "time_mean_x": avg("time_mean_x"), "time_mean_y": avg("time_mean_y"),
        "time_max_x": max((float(r["time_max_x"]) for r in rows), default=math.nan),
        "time_max_y": max((float(r["time_max_y"]) for r in rows), default=math.nan),
    }
