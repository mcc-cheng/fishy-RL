"""The hand-written heuristic: a baseline that scores moves with fixed formulas.

Nothing here is learned.  It is the opponent the solver trains against and is
measured against (see ``scripts/train.py`` and ``scripts/evaluate.py``), and
the fallback when no trained network is available.

How it chooses a move
---------------------
1. If a declaration is certain to be right, play it.
2. Otherwise imagine a few hundred complete deals that fit everything seen so
   far, and use them as probabilities for where each card is.
3. Score every legal move and play the highest:

   * an ask is worth ``chance * GAIN - (1 - chance) * threat``.  ``GAIN`` is
     what one card is worth; ``threat`` is how much the asked opponent is
     expected to take back if the ask fails and they get the turn;
   * a declaration is worth ``(2 * chance - 1) - (what the set is worth now)``,
     i.e. the expected result of declaring compared with holding on.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .baselines import guess_declaration
from .belief import ask_likelihood_weights, best_declaration, marginals, sample_deals
from .cards import N_SETS, OPPONENTS, SET_MASKS, SET_SIZE, TEAMMATES, card_name, set_name
from .engine import Action, Ask, Declare, enumerate_asks
from .knowledge import Knowledge, View

# A set is scored from -1 (opponents hold all six cards) to +1 (we hold all
# six), so moving one card to our side is worth 2/6.
GAIN = 2.0 / SET_SIZE


class Move(NamedTuple):
    """A candidate move with its score.

    ``chance`` is the probability that the ask succeeds or that the declaration
    is correct.  ``certain`` marks a declaration that is proven right.
    """

    action: Action
    value: float
    chance: float
    certain: bool


def describe(move: Move | Action) -> str:
    """A move in words, e.g. ``'ask seat 3 for QH'``."""
    action = move.action if isinstance(move, Move) else move
    if isinstance(action, Ask):
        return f"ask seat {action.target} for {card_name(action.card)}"
    base = SET_SIZE * action.set_id
    who = ", ".join(f"{card_name(base + i)}: seat {q}" for i, q in enumerate(action.assignment))
    return f"declare {set_name(action.set_id)} ({who})"


def rank_moves(
    know: Knowledge,
    public: Knowledge,
    me: int,
    hand: int,
    rng: np.random.Generator,
    n_particles: int = 300,
    alpha: float = 1.0,
) -> list[Move]:
    """Every sensible move for ``me`` (who is on turn), best first.

    ``know`` is what ``me`` knows (public knowledge plus own hand), ``public``
    what everyone knows, ``hand`` the player's cards.
    """
    k = know
    team = me & 1
    mates, opps = TEAMMATES[me], OPPONENTS[me]
    live = [h for h in range(N_SETS) if not k.declared >> h & 1]
    team_known = k.known[mates[0]] | k.known[mates[1]] | k.known[mates[2]]
    opp_known = k.known[opps[0]] | k.known[opps[1]] | k.known[opps[2]]

    moves: list[Move] = []
    sure: set[int] = set()
    for h in live:
        if team_known & SET_MASKS[h] == SET_MASKS[h]:
            sigma = tuple(k.known_owner(c) for c in range(SET_SIZE * h, SET_SIZE * (h + 1)))
            moves.append(Move(Declare(h, sigma), 0.0, 1.0, True))
            sure.add(h)

    # where the cards probably are, from the player's point of view
    parts = sample_deals(k, n_particles, rng)
    w = ask_likelihood_weights(parts, k, alpha)
    P = marginals(parts, w)
    ours = (parts >= 0) & ((parts & 1) == team)
    share = w @ ours.reshape(-1, N_SETS, SET_SIZE).sum(2)   # expected cards we hold, per set
    worth = 2.0 * share / SET_SIZE - 1.0                    # each set, from -1 to +1

    asks = enumerate_asks(me, hand, k.hand_size)
    if asks:
        threat = _threats(public, me, parts, w, rng, n_particles, alpha)
        for a in asks:
            p = float(P[a.target, a.card])
            moves.append(Move(a, p * GAIN - (1.0 - p) * threat[a.target], p, False))

    for h in live:
        if h in sure or opp_known & SET_MASKS[h]:
            continue
        sigma, p = best_declaration(parts, team, h, w)
        if sigma is not None:
            moves.append(Move(Declare(h, sigma), (2.0 * p - 1.0) - float(worth[h]), p, False))

    if not moves:  # forced to declare something that cannot be right
        guess = guess_declaration(k, me, float(rng.random()))
        moves.append(Move(guess, -1.0 - float(worth[guess.set_id]), 0.0, False))
    moves.sort(key=lambda m: (not m.certain, -m.value, -m.chance))
    return moves


def _threats(public: Knowledge, me: int, parts: np.ndarray, w: np.ndarray,
             rng: np.random.Generator, n_particles: int, alpha: float) -> dict[int, float]:
    """For each opponent: what they are expected to gain from one ask if
    they get the turn.

    An opponent can take a card from us when (a) they hold another card of
    its set, which is judged from the player's own beliefs, and (b) they know
    where it is, which is judged from what is *public* (they cannot see the
    player's hand).
    """
    pub = sample_deals(public, n_particles, rng)
    Q = marginals(pub, ask_likelihood_weights(pub, public, alpha))
    exposed = Q[list(TEAMMATES[me])].max(0).reshape(N_SETS, SET_SIZE).max(1)  # per set
    out: dict[int, float] = {}
    for j in OPPONENTS[me]:
        in_set = w @ (parts == j).reshape(-1, N_SETS, SET_SIZE).any(2)
        out[j] = GAIN * float((exposed * in_set).max())
    return out


class HeuristicAgent:
    """The heuristic as a computer player (``act(view, rng)``)."""

    def __init__(self, n_particles: int = 300, alpha: float = 1.0, name: str = "heuristic") -> None:
        self.n_particles = n_particles
        self.alpha = alpha
        self.name = name

    def act(self, view: View, rng: np.random.Generator) -> Action:
        return rank_moves(view.know, view.public, view.player, view.hand, rng,
                          self.n_particles, self.alpha)[0].action
