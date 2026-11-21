"""Pick a move with the value network.

For every candidate move the chooser works out the positions it can lead to
(an ask either succeeds or fails; a declaration is either right or wrong),
scores each of those positions with the network, and weights them by how
likely they are.  The move with the best expected value wins.

The probabilities come from the particle belief (:mod:`fish.belief`): a set
of imagined deals that fit everything the player knows.  The same deals also
describe each position to the network.  For each outcome, only the deals in
which that outcome happens are kept.

A position is worth the sets already scored plus, for every set still in
play, the network's prediction of who will end up with it.

A declaration that is certain to be right is always played straight away.

Needs PyTorch.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from .baselines import certain_declaration, guess_declaration
from .belief import ask_likelihood_weights, best_declaration, marginals, sample_deals
from .cards import N_PLAYERS, N_SETS, OPPONENTS, SET_MASKS, SET_SIZE
from .engine import Action, Ask, AskObs, Declare, DeclareObs, PublicObs, enumerate_asks, resolve_turn
from .features import set_features
from .knowledge import Contradiction, Knowledge, View
from .value_net import MARGIN_SCALE, ValueNet, predict

# A position: what the player would know, one row of numbers per set, and
# which sets are still in play.  The last two are None when the game is over.
Position = tuple[Knowledge, Optional[np.ndarray], Optional[np.ndarray]]


class Scored(NamedTuple):
    """A candidate move: its expected value for the player's team (the
    expected final set margin divided by 9, so -1 to +1) and the chance that
    the ask succeeds / the declaration is right."""

    action: Action
    value: float
    chance: float


def card_chances(know: Knowledge, rng: np.random.Generator, n_particles: int = 64,
                 alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Imagined deals that fit ``know``, their weights, and the resulting
    card probabilities ``P[seat, card]``."""
    parts = sample_deals(know, n_particles, rng)
    w = ask_likelihood_weights(parts, know, alpha)
    return parts, w, marginals(parts, w)


def _declare_obs(know: Knowledge, me: int, set_id: int, sigma: tuple[int, ...],
                 correct: bool, removed: list[int]) -> DeclareObs | None:
    """What everyone would see if ``me`` declared; ``removed[p]`` is how many
    cards of the set seat ``p`` turns out to hold (None if that cannot be)."""
    sizes = [know.hand_size[p] - removed[p] for p in range(N_PLAYERS)]
    if min(sizes) < 0:
        return None
    return DeclareObs(me, set_id, tuple(sigma), correct, tuple(removed), resolve_turn(sizes, me))


def _outcome(know: Knowledge, public: Knowledge, me: int, obs: PublicObs,
             parts: np.ndarray, w: np.ndarray, keep: np.ndarray) -> Position | None:
    """The position after ``obs``, as ``me`` would see it.

    ``keep`` marks the imagined deals in which this outcome happens; the
    position is described from those.  None if the outcome cannot happen.
    """
    k, pub = know.copy(False), public.copy(False)
    try:
        k.update(obs)
        pub.update(obs)
    except Contradiction:
        return None
    if k.is_terminal():
        return k, None, None
    ww = w[keep]
    if ww.sum() <= 0:
        return None
    sub = parts[keep]  # a copy, so it can be changed
    if isinstance(obs, AskObs):
        if obs.success:
            sub[:, obs.card] = obs.asker  # the card changes hands
    else:
        sub[:, SET_SIZE * obs.set_id: SET_SIZE * (obs.set_id + 1)] = -1  # the set leaves the game
    rows, live = set_features(k, pub, me, sub, ww)
    return k, rows, live


def _position_values(net: ValueNet, positions: list[Position], me: int) -> np.ndarray:
    """Expected final margin (divided by 9) of each position for ``me``'s team:
    the sets already scored, which are known exactly, plus the network's
    prediction for every set still in play."""
    out = np.zeros(len(positions), dtype=np.float32)
    team = me & 1
    open_idx = []
    for i, (k, rows, _) in enumerate(positions):
        out[i] = k.score[team] - k.score[1 - team]
        if rows is not None:
            open_idx.append(i)
    if open_idx:
        rows = np.stack([positions[i][1] for i in open_idx])   # (position, set, feature)
        live = np.stack([positions[i][2] for i in open_idx])   # (position, set)
        out[open_idx] += (predict(net, rows) * live).sum(1)
    return out / MARGIN_SCALE


def score_moves(
    net: ValueNet,
    know: Knowledge,
    public: Knowledge,
    me: int,
    hand: int,
    rng: np.random.Generator,
    n_particles: int = 64,
    alpha: float = 1.0,
    max_asks: int = 12,
) -> list[Scored]:
    """Candidate moves for ``me`` (who is on turn), best first.

    ``know`` is the player's private knowledge, ``public`` what everyone
    knows and ``hand`` the player's cards.  Candidates are the ``max_asks``
    most promising asks and, for every set the team might hold completely,
    its most likely declaration.
    """
    team = me & 1
    parts, w, P = card_chances(know, rng, n_particles, alpha)
    every = np.ones(len(parts), dtype=bool)
    # each candidate: (action, chance, [(probability, position), ...])
    cands: list[tuple[Action, float, list[tuple[float, Position]]]] = []

    sure = certain_declaration(know, me)
    if sure is not None:
        obs = _declare_obs(know, me, sure.set_id, sure.assignment, True,
                           [sure.assignment.count(p) for p in range(N_PLAYERS)])
        pos = None if obs is None else _outcome(know, public, me, obs, parts, w, every)
        if pos is not None:
            cands.append((sure, 1.0, [(1.0, pos)]))
    else:
        asks = enumerate_asks(me, hand, know.hand_size)
        if asks:
            probs = np.array([P[a.target, a.card] for a in asks])
            order = np.argsort(-(probs + 1e-9 * rng.random(len(asks))))  # random tie-break
            keep = [int(i) for i in order[:max_asks] if probs[i] > 0]
            if not keep:  # nothing can work: an ask only passes the turn
                keep = [int(i) for i in order[:3]]
            for i in keep:
                a, p = asks[i], float(min(max(probs[i], 0.0), 1.0))
                held = parts[:, a.card] == a.target  # deals in which the ask works
                outs = []
                if p > 0:
                    pos = _outcome(know, public, me, AskObs(me, a.target, a.card, True, me),
                                   parts, w, held)
                    if pos is not None:
                        outs.append((p, pos))
                if p < 1:
                    pos = _outcome(know, public, me, AskObs(me, a.target, a.card, False, a.target),
                                   parts, w, ~held)
                    if pos is not None:
                        outs.append((1.0 - p, pos))
                if outs:
                    cands.append((a, p, outs))

        opps = OPPONENTS[me]
        opp_known = know.known[opps[0]] | know.known[opps[1]] | know.known[opps[2]]
        # A declaration is only a candidate here when it is not proven, so its
        # chance is kept below 1 even if every imagined deal agrees with it.
        # The fewer deals really count (a few may carry most of the weight),
        # the lower the cap.
        deals = 1.0 / float(np.sum(w ** 2))
        cap = deals / (deals + 1.0)
        for h in range(N_SETS):
            if know.declared >> h & 1 or opp_known & SET_MASKS[h]:
                continue
            sigma, q = best_declaration(parts, team, h, w)
            if sigma is None:
                continue
            q = float(min(max(q, 0.0), 1.0)) * cap
            cols = parts[:, SET_SIZE * h: SET_SIZE * (h + 1)]
            wrong = (cols != np.array(sigma, dtype=cols.dtype)).any(1)
            counts = [sigma.count(p) for p in range(N_PLAYERS)]
            outs = []
            obs = _declare_obs(know, me, h, sigma, True, counts)
            pos = None if obs is None else _outcome(know, public, me, obs, parts, w, ~wrong)
            if pos is not None:
                outs.append((q, pos))
            if wrong.any():
                # a likely layout in which this declaration is wrong
                idx = int(np.argmax(np.where(wrong, w, -1.0)))
                removed = np.bincount(cols[idx].astype(np.int64), minlength=N_PLAYERS).tolist()
                match = (cols == cols[idx]).all(1)
            else:
                # no imagined deal contradicts it: judge the rest of the
                # position as if the cards were where we think
                removed, match = counts, ~wrong
            obs = _declare_obs(know, me, h, sigma, False, removed)
            pos = None if obs is None else _outcome(know, public, me, obs, parts, w, match)
            if pos is not None:
                outs.append((1.0 - q, pos))
            if outs:
                cands.append((Declare(h, sigma), q, outs))

    if not cands:  # forced to declare something hopeless
        return [Scored(guess_declaration(know, me, float(rng.random())), -1.0, 0.0)]

    vals = _position_values(net, [pos for _, _, outs in cands for _, pos in outs], me)
    scored = []
    at = 0
    for action, chance, outs in cands:
        total = sum(p for p, _ in outs)
        value = sum(p * float(vals[at + i]) for i, (p, _) in enumerate(outs)) / total
        at += len(outs)
        scored.append(Scored(action, value, chance))
    scored.sort(key=lambda s: -s.value)
    return scored


def choose(
    net: ValueNet,
    know: Knowledge,
    public: Knowledge,
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
    moves = score_moves(net, know, public, me, hand, rng, **options)
    if epsilon > 0 and rng.random() < epsilon:
        asks = [m for m in moves if isinstance(m.action, Ask)]
        if asks:
            return asks[int(rng.integers(len(asks)))].action, moves
    return moves[0].action, moves


class NetAgent:
    """The solver as a computer player: follows the value network
    (``act(view, rng)`` like the baselines)."""

    def __init__(self, net: ValueNet, name: str = "solver", **options) -> None:
        self.net = net
        self.name = name
        self.options = options

    def act(self, view: View, rng: np.random.Generator) -> Action:
        return choose(self.net, view.know, view.public, view.player, view.hand, rng,
                      **self.options)[0]
