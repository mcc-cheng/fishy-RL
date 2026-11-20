"""Turn what one player knows into a fixed-length list of numbers.

This is the input of the value network.  Everything is written from the
player's own seat: position 0 is the player, positions 2 and 4 are their
teammates and positions 1, 3, 5 the opponents, going clockwise.  That way one
network can play every seat.

Layout of the vector (``FEATURE_SIZE`` numbers, all between 0 and 1):

=====================  =====  ==============================================
block                  size   meaning
=====================  =====  ==============================================
known[position, card]  6x54   1 if that player certainly holds the card
poss[position, card]   6x54   1 if that player could hold the card
hand size[position]    6      cards held / 9
set minimum[pos, set]  6x9    cards of the set the player must hold / 6
declared[set]          9      1 if the set is gone
score                  2      sets won by us / 9, by them / 9
turn[position]         6      1 at the player whose move it is
=====================  =====  ==============================================
"""
from __future__ import annotations

import numpy as np

from .cards import HAND_SIZE, N_CARDS, N_PLAYERS, N_SETS, SET_SIZE
from .knowledge import Knowledge, masks_to_matrix

_BLOCKS = (
    ("known", N_PLAYERS * N_CARDS),
    ("poss", N_PLAYERS * N_CARDS),
    ("hand_size", N_PLAYERS),
    ("set_min", N_PLAYERS * N_SETS),
    ("declared", N_SETS),
    ("score", 2),
    ("turn", N_PLAYERS),
)

#: where each block sits in the vector, e.g. ``SLICES["score"]``
SLICES: dict[str, slice] = {}
_start = 0
for _name, _size in _BLOCKS:
    SLICES[_name] = slice(_start, _start + _size)
    _start += _size
FEATURE_SIZE: int = _start


def seat_order(seat: int) -> list[int]:
    """Seats clockwise starting from ``seat``: ``seat_order(seat)[i]`` sits at position ``i``."""
    return [(seat + i) % N_PLAYERS for i in range(N_PLAYERS)]


def encode(know: Knowledge, seat: int) -> np.ndarray:
    """Feature vector of ``know`` (public knowledge plus ``seat``'s own hand).

    ``know`` should be the player's private knowledge, i.e. ``view.know`` or
    ``Solver.know``; it works for any position, including ones where it is not
    ``seat``'s turn.
    """
    order = seat_order(seat)
    x = np.zeros(FEATURE_SIZE, dtype=np.float32)
    x[SLICES["known"]] = masks_to_matrix([know.known[p] for p in order]).ravel()
    x[SLICES["poss"]] = masks_to_matrix([know.poss[p] for p in order]).ravel()
    x[SLICES["hand_size"]] = [know.hand_size[p] / HAND_SIZE for p in order]
    x[SLICES["set_min"]] = np.array(
        [know.min_in_set[p] for p in order], dtype=np.float32).ravel() / SET_SIZE
    x[SLICES["declared"]] = [know.declared >> h & 1 for h in range(N_SETS)]
    team = seat & 1
    x[SLICES["score"]] = [know.score[team] / N_SETS, know.score[1 - team] / N_SETS]
    x[SLICES["turn"].start + (know.turn - seat) % N_PLAYERS] = 1.0
    return x


def encode_batch(knows: list[Knowledge], seat: int) -> np.ndarray:
    """Stack of feature vectors, one row per knowledge store."""
    if not knows:
        return np.zeros((0, FEATURE_SIZE), dtype=np.float32)
    return np.stack([encode(k, seat) for k in knows])
