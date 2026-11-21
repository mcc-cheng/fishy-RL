"""Turn a position into the numbers the value network reads.

The network does not look at the 54 cards one by one.  Every set that is still
in play gets one short row of numbers describing that set as the player sees
it, and the network scores each row on its own: how likely is this set to end
up with my team?  All sets share the same network, so whatever is learned
about one set applies to every other, and to every card in it.

Everything is written from the player's own side ("ours" is the player's
team, "theirs" the opponents), so one network can play every seat.

One row, for one set (all numbers between 0 and 1):

==================  =========================================================
mine                cards of the set I hold / 6
team                cards of the set my team is expected to hold / 6
all_ours            chance that my team holds all six
all_theirs          chance that the opponents hold all six
known_ours          cards I know for certain are with a player of my team / 6
known_theirs        cards I know for certain are with an opponent / 6
public_ours         cards everyone knows are with a player of my team / 6
public_theirs       cards everyone knows are with an opponent / 6
public_min_ours     cards my team must hold, as far as everyone knows / 6
public_min_theirs   the same for the opponents
turn_holds          chance that the player to move holds a card of the set
turn_share          cards of the set the player to move is expected to hold / 6
best_ask            if I am to move: my best chance to get a card of the set
------------------  ---------------------------------------------------------
turn_me             1 if I am to move
turn_mate           1 if a teammate is to move
turn_opponent       1 if an opponent is to move
sets_left           sets still in play / 9
my_cards            cards I hold / 18
turn_cards          cards the player to move holds / 18
team_cards          share of all cards in play that my team holds
best_ask_any        if I am to move: my best chance to get any card
==================  =========================================================

The last eight describe the whole position and are repeated in every row.
"""
from __future__ import annotations

import numpy as np

from .cards import HAND_SIZE, N_PLAYERS, N_SETS, OPPONENTS, SET_MASKS, SET_SIZE, TEAMMATES
from .knowledge import Knowledge

#: the numbers in one row, in order
FEATURE_NAMES: tuple[str, ...] = (
    "mine", "team", "all_ours", "all_theirs", "known_ours", "known_theirs",
    "public_ours", "public_theirs", "public_min_ours", "public_min_theirs",
    "turn_holds", "turn_share", "best_ask",
    "turn_me", "turn_mate", "turn_opponent", "sets_left", "my_cards", "turn_cards",
    "team_cards", "best_ask_any",
)
N_FEATURES: int = len(FEATURE_NAMES)

_SEATS = np.arange(N_PLAYERS, dtype=np.int8)
_MANY_CARDS = 2 * HAND_SIZE


def _per_set(mask: int) -> np.ndarray:
    """How many cards of each set are in a card bitmask."""
    return np.array([(mask & SET_MASKS[h]).bit_count() for h in range(N_SETS)], dtype=np.float64)


def _union(masks: list[int], seats: tuple[int, ...]) -> int:
    out = 0
    for s in seats:
        out |= masks[s]
    return out


def set_features(
    know: Knowledge,
    public: Knowledge,
    me: int,
    parts: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One row of numbers per set, for the position as ``me`` sees it.

    ``know`` is what ``me`` knows (public knowledge plus own hand), ``public``
    what everyone knows.  ``parts`` are imagined deals that fit ``know``
    (``int8[n, 54]`` owner arrays, see :func:`fish.belief.sample_deals`) and
    ``w`` their weights.

    Returns ``(rows, live)``: ``rows[set]`` is the row described in the module
    docstring (shape ``(9, N_FEATURES)``) and ``live[set]`` is True for sets
    still in play.  Rows of declared sets are all zero.
    """
    mates, opps = TEAMMATES[me], OPPONENTS[me]
    n = parts.shape[0]
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    holds = parts[:, None, :] == _SEATS[None, :, None]                 # (deal, seat, card)
    count = holds.reshape(n, N_PLAYERS, N_SETS, SET_SIZE).sum(3)       # (deal, seat, set)
    ours = count[:, list(mates)].sum(1)                                # (deal, set)
    turn = know.turn
    sizes = know.hand_size
    live = np.array([not (know.declared >> h & 1) for h in range(N_SETS)])

    mine = _per_set(know.known[me])  # a player's own cards are always known to them
    best = np.zeros(N_SETS)
    if turn == me:
        askable = [s for s in opps if sizes[s] > 0]
        if askable:
            chance = np.tensordot(w, holds[:, askable], axes=(0, 0))   # (opponent, card)
            best = chance.max(0).reshape(N_SETS, SET_SIZE).max(1) * (mine > 0)

    cols = {
        "mine": mine / SET_SIZE,
        "team": (w @ ours) / SET_SIZE,
        "all_ours": w @ (ours == SET_SIZE),
        "all_theirs": w @ (ours == 0),
        "known_ours": _per_set(_union(know.known, mates)) / SET_SIZE,
        "known_theirs": _per_set(_union(know.known, opps)) / SET_SIZE,
        "public_ours": _per_set(_union(public.known, mates)) / SET_SIZE,
        "public_theirs": _per_set(_union(public.known, opps)) / SET_SIZE,
        "public_min_ours": np.minimum(
            np.sum([public.min_in_set[s] for s in mates], axis=0), SET_SIZE) / SET_SIZE,
        "public_min_theirs": np.minimum(
            np.sum([public.min_in_set[s] for s in opps], axis=0), SET_SIZE) / SET_SIZE,
        "turn_holds": w @ (count[:, turn] > 0),
        "turn_share": (w @ count[:, turn]) / SET_SIZE,
        "best_ask": best,
        "turn_me": float(turn == me),
        "turn_mate": float(turn != me and turn in mates),
        "turn_opponent": float(turn in opps),
        "sets_left": live.sum() / N_SETS,
        "my_cards": min(sizes[me], _MANY_CARDS) / _MANY_CARDS,
        "turn_cards": min(sizes[turn], _MANY_CARDS) / _MANY_CARDS,
        "team_cards": sum(sizes[s] for s in mates) / max(1, sum(sizes)),
        "best_ask_any": float(best.max()),
    }
    rows = np.stack(
        [np.broadcast_to(np.asarray(cols[name], dtype=np.float32), (N_SETS,))
         for name in FEATURE_NAMES], axis=1)
    rows = np.clip(rows, 0.0, 1.0)
    rows[~live] = 0.0
    return rows, live
