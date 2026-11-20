"""Baseline agents and the helpers every agent shares.

Every agent implements ``act(view, rng) -> Action`` and only ever touches its
:class:`~fish.knowledge.View` (own hand + public knowledge), never a ``State``.

* :class:`RandomAgent` asks a uniformly random legal question.
* :class:`RuleBasedAgent` asks for a card it knows an opponent holds, if any.

Both declare a set as soon as their knowledge proves the declaration, and make
a best guess when a declaration is forced.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .cards import N_PLAYERS, N_SETS, OPPONENTS, SET_MASKS, SET_SIZE, TEAMMATES
from .engine import Action, Ask, Declare, enumerate_asks
from .knowledge import Knowledge, View


class Agent(Protocol):
    """Interface shared by all agents."""

    name: str

    def act(self, view: View, rng: np.random.Generator) -> Action:
        """Choose an action for ``view.player`` (who is on turn)."""
        ...


# ============================================================ shared helpers
def legal_asks(view: View) -> list[Ask]:
    """Legal asks derivable from the view (own hand + public hand sizes)."""
    return enumerate_asks(view.player, view.hand, view.know.hand_size)


def certain_declaration(k: Knowledge, player: int) -> Declare | None:
    """A declaration that is provably correct under ``k`` (lowest set id), if any."""
    mates = TEAMMATES[player]
    team_known = k.known[mates[0]] | k.known[mates[1]] | k.known[mates[2]]
    for h in range(N_SETS):
        sm = SET_MASKS[h]
        if not k.declared >> h & 1 and team_known & sm == sm:
            return Declare(h, tuple(k.known_owner(c) for c in range(SET_SIZE * h, SET_SIZE * h + 6)))
    return None


def _densities(k: Knowledge) -> tuple[list[int], list[list[float]]]:
    """Per player: mask of unknown-but-possible cards and, per set, a rough
    probability that such a card is actually held.

    Base density is ``free slots / possible unknown cards``; in a set where the
    player provably holds more cards than are located, it is raised to
    ``missing / possible cards in the set``.
    """
    unk: list[int] = []
    dens: list[list[float]] = []
    for p in range(N_PLAYERS):
        kn = k.known[p]
        u = k.poss[p] & ~kn
        unk.append(u)
        free = k.hand_size[p] - kn.bit_count()
        if not u or free <= 0:
            dens.append([0.0] * N_SETS)
            continue
        d = free / u.bit_count()
        row = [d] * N_SETS
        mins = k.min_in_set[p]
        for h in range(N_SETS):
            mn = mins[h]
            if mn:
                sm = SET_MASKS[h]
                need = mn - (kn & sm).bit_count()
                if need > 0:
                    x = need / (u & sm).bit_count()
                    if x > d:
                        row[h] = x
        dens.append(row)
    return unk, dens


def _guess_set(k: Knowledge, player: int, h: int, unk: list[int],
               dens: list[list[float]], u: float = 0.0) -> tuple[tuple[int, ...], float]:
    """Best per-card assignment of set ``h`` to the team and a rough probability
    that it is correct (cards treated as independent).

    Unlocated cards go to the likeliest teammate that still has a free hand
    slot; ``u`` in [0, 1) picks which card is considered first, so that equally
    plausible guesses are not always resolved the same way.
    """
    mates = TEAMMATES[player]
    opps = OPPONENTS[player]
    known = k.known
    prob = 1.0
    sigma = [player] * SET_SIZE
    free = {q: k.hand_size[q] - known[q].bit_count() for q in mates}
    start = int(u * SET_SIZE)
    for step_i in range(SET_SIZE):
        i = (start + step_i) % SET_SIZE
        bit = 1 << (SET_SIZE * h + i)
        best_q, best_w, tot = player, 0.0, 0.0
        done = False
        for q in mates:
            if known[q] & bit:
                sigma[i] = q
                done = True
                break
            if unk[q] & bit:
                w = dens[q][h]
                tot += w
                if w > best_w and free[q] > 0:
                    best_q, best_w = q, w
        if done:
            continue
        free[best_q] -= 1
        for o in opps:
            if known[o] & bit:
                tot = 0.0
                best_w = 0.0
                break
            if unk[o] & bit:
                tot += dens[o][h]
        sigma[i] = best_q
        prob *= best_w / tot if tot > 0 else 0.0
    return tuple(sigma), prob


def guess_declaration(k: Knowledge, player: int, u: float = 0.0) -> Declare:
    """Best-effort declaration when one is forced and nothing better is available."""
    unk, dens = _densities(k)
    best: tuple[float, int, tuple[int, ...]] | None = None
    for h in range(N_SETS):
        if k.declared >> h & 1:
            continue
        sigma, pr = _guess_set(k, player, h, unk, dens, u)
        if best is None or pr > best[0]:
            best = (pr, h, sigma)
    assert best is not None
    return Declare(best[1], best[2])


# ================================================================ baselines
class RandomAgent:
    """Uniformly random legal ask; declares only sets that are fully known."""

    def __init__(self, name: str = "random") -> None:
        self.name = name

    def act(self, view: View, rng: np.random.Generator) -> Action:
        cert = certain_declaration(view.know, view.player)
        if cert is not None:
            return cert
        asks = legal_asks(view)
        if not asks:
            return guess_declaration(view.know, view.player, float(rng.random()))
        return asks[int(rng.integers(len(asks)))]


class RuleBasedAgent:
    """Asks for a card known to be with an opponent if any, otherwise random."""

    def __init__(self, name: str = "rule") -> None:
        self.name = name

    def act(self, view: View, rng: np.random.Generator) -> Action:
        k = view.know
        cert = certain_declaration(k, view.player)
        if cert is not None:
            return cert
        asks = legal_asks(view)
        if not asks:
            return guess_declaration(k, view.player, float(rng.random()))
        sure = [a for a in asks if k.known[a.target] >> a.card & 1]
        pool = sure or asks
        return pool[int(rng.integers(len(pool)))]


# ================================================================== registry
AGENT_TYPES: dict[str, type] = {
    "random": RandomAgent,
    "rule": RuleBasedAgent,
}


def make_agent(kind: str, **kwargs: Any) -> Agent:
    """Instantiate an agent by registry name (specs stay picklable for multiprocessing)."""
    return AGENT_TYPES[kind](**kwargs)
