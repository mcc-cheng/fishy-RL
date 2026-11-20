"""Pick a move with the value network.

For every candidate move the chooser works out the positions it can lead to
(an ask either succeeds or fails; a declaration is either right or wrong),
scores each of those positions with the network, and weights them by how
likely they are.  The move with the best expected value wins.

The probabilities come from the particle belief (:mod:`fish.belief`).  A
declaration that is certain to be right is always played straight away.

Needs PyTorch.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .baselines import certain_declaration, guess_declaration
from .belief import ask_likelihood_weights, best_declaration, marginals, sample_deals
from .cards import N_PLAYERS, N_SETS, OPPONENTS, SET_MASKS, SET_SIZE
from .engine import Action, Ask, AskObs, Declare, DeclareObs, enumerate_asks, resolve_turn
from .features import encode
from .knowledge import Contradiction, Knowledge, View
from .value_net import MARGIN_SCALE, ValueNet, predict


class Scored(NamedTuple):
    """A candidate move: its expected value for the player's team (-1 to +1)
    and the chance that the ask succeeds / the declaration is right."""

    action: Action
    value: float
    chance: float


def after_ask(know: Knowledge, me: int, ask: Ask, success: bool) -> Knowledge | None:
    """What ``me`` would know after this ask with this result (None if impossible)."""
    k = know.copy(False)
    try:
        k.update(AskObs(me, ask.target, ask.card, success, me if success else ask.target))
    except Contradiction:
        return None
    return k


def after_declare(know: Knowledge, me: int, set_id: int, sigma: tuple[int, ...],
                  correct: bool, removed: list[int]) -> Knowledge | None:
    """What ``me`` would know after this declaration; ``removed[p]`` is how
    many cards of the set seat ``p`` turns out to hold."""
    k = know.copy(False)
    sizes = [know.hand_size[p] - removed[p] for p in range(N_PLAYERS)]
    if min(sizes) < 0:
        return None
    try:
        k.update(DeclareObs(me, set_id, sigma, correct, tuple(removed), resolve_turn(sizes, me)))
    except Contradiction:
        return None
    return k


def _position_values(net: ValueNet, knows: list[Knowledge], me: int) -> np.ndarray:
    """Value of each position for ``me``'s team; finished games are scored exactly."""
    out = np.zeros(len(knows), dtype=np.float32)
    team = me & 1
    open_idx = []
    for i, k in enumerate(knows):
        if k.is_terminal():
            out[i] = (k.score[team] - k.score[1 - team]) / MARGIN_SCALE
        else:
            open_idx.append(i)
    if open_idx:
        feats = np.stack([encode(knows[i], me) for i in open_idx])
        out[open_idx] = predict(net, feats)
    return out


def score_moves(
    net: ValueNet,
    know: Knowledge,
    me: int,
    hand: int,
    rng: np.random.Generator,
    n_particles: int = 64,
    alpha: float = 1.0,
    max_asks: int = 12,
) -> list[Scored]:
    """Candidate moves for ``me`` (who is on turn), best first.

    ``know`` is the player's private knowledge and ``hand`` their cards.
    Candidates are the ``max_asks`` most promising asks and, for every set the
    team might hold completely, its most likely declaration.
    """
    team = me & 1
    # each candidate: (action, chance, [(probability, resulting knowledge), ...])
    cands: list[tuple[Action, float, list[tuple[float, Knowledge]]]] = []

    sure = certain_declaration(know, me)
    if sure is not None:
        removed = [sure.assignment.count(p) for p in range(N_PLAYERS)]
        k = after_declare(know, me, sure.set_id, sure.assignment, True, removed)
        if k is not None:
            cands.append((sure, 1.0, [(1.0, k)]))
    else:
        parts = sample_deals(know, n_particles, rng)
        w = ask_likelihood_weights(parts, know, alpha)
        P = marginals(parts, w)

        asks = enumerate_asks(me, hand, know.hand_size)
        if asks:
            probs = np.array([P[a.target, a.card] for a in asks])
            order = np.argsort(-(probs + 1e-9 * rng.random(len(asks))))  # random tie-break
            keep = [int(i) for i in order[:max_asks] if probs[i] > 0]
            if not keep:  # nothing can work: an ask only passes the turn
                keep = [int(i) for i in order[:3]]
            for i in keep:
                a, p = asks[i], float(min(max(probs[i], 0.0), 1.0))
                outs = []
                if p > 0:
                    k = after_ask(know, me, a, True)
                    if k is not None:
                        outs.append((p, k))
                if p < 1:
                    k = after_ask(know, me, a, False)
                    if k is not None:
                        outs.append((1.0 - p, k))
                if outs:
                    cands.append((a, p, outs))

        opps = OPPONENTS[me]
        opp_known = know.known[opps[0]] | know.known[opps[1]] | know.known[opps[2]]
        for h in range(N_SETS):
            if know.declared >> h & 1 or opp_known & SET_MASKS[h]:
                continue
            sigma, q = best_declaration(parts, team, h, w)
            if sigma is None:
                continue
            q = float(min(max(q, 0.0), 1.0))
            outs = []
            k = after_declare(know, me, h, sigma, True,
                              [sigma.count(p) for p in range(N_PLAYERS)])
            if k is not None:
                outs.append((q, k))
            if q < 1.0 - 1e-9:
                # a likely layout in which this declaration is wrong
                cols = parts[:, SET_SIZE * h: SET_SIZE * (h + 1)]
                wrong = (cols != np.array(sigma, dtype=cols.dtype)).any(1)
                if wrong.any():
                    idx = int(np.argmax(np.where(wrong, w, -1.0)))
                    removed = np.bincount(cols[idx].astype(np.int64), minlength=N_PLAYERS).tolist()
                    k = after_declare(know, me, h, sigma, False, removed)
                    if k is not None:
                        outs.append((1.0 - q, k))
            if outs:
                cands.append((Declare(h, sigma), q, outs))

    if not cands:  # forced to declare something hopeless
        return [Scored(guess_declaration(know, me, float(rng.random())), -1.0, 0.0)]

    flat = [k for _, _, outs in cands for _, k in outs]
    vals = _position_values(net, flat, me)
    scored = []
    pos = 0
    for action, chance, outs in cands:
        total = sum(p for p, _ in outs)
        value = sum(p * float(vals[pos + i]) for i, (p, _) in enumerate(outs)) / total
        pos += len(outs)
        scored.append(Scored(action, value, chance))
    scored.sort(key=lambda s: -s.value)
    return scored


def choose(
    net: ValueNet,
    know: Knowledge,
    me: int,
    hand: int,
    rng: np.random.Generator,
    epsilon: float = 0.0,
    **options,
) -> tuple[Action, list[Scored]]:
    """The move to play and the full ranking.

    With probability ``epsilon`` a random candidate ask is played instead of
    the best move (exploration during training).  ``options`` are passed to
    :func:`score_moves`.
    """
    moves = score_moves(net, know, me, hand, rng, **options)
    if epsilon > 0 and rng.random() < epsilon:
        asks = [m for m in moves if isinstance(m.action, Ask)]
        if asks:
            return asks[int(rng.integers(len(asks)))].action, moves
    return moves[0].action, moves


class NetAgent:
    """A player that follows the value network (``act(view, rng)`` like the baselines)."""

    def __init__(self, net: ValueNet, name: str = "net", **options) -> None:
        self.net = net
        self.name = name
        self.options = options

    def act(self, view: View, rng: np.random.Generator) -> Action:
        return choose(self.net, view.know, view.player, view.hand, rng, **self.options)[0]
