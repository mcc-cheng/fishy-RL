"""Card ids, names, set mapping and pretty printing.

Cards are ints 0..53 and ``set_id = card // 6``:

* sets 0..7: for suit ``s`` in (S, H, D, C): set ``2s`` is the low half
  (2-7), set ``2s + 1`` the high half (9-A);
* set 8: the four 8s (S, H, D, C) followed by the two jokers.

Hands are bitmasks (Python ints) with bit ``c`` set when card ``c`` is held.
"""
from __future__ import annotations

from typing import Iterable

N_CARDS = 54
N_SETS = 9
SET_SIZE = 6
N_PLAYERS = 6
N_TEAMS = 2
HAND_SIZE = N_CARDS // N_PLAYERS

SUITS = "SHDC"
LOW_RANKS = "234567"
HIGH_RANKS = "9TJQKA"

ALL_CARDS_MASK = (1 << N_CARDS) - 1
ALL_SETS_MASK = (1 << N_SETS) - 1
SET_MASKS: tuple[int, ...] = tuple(0b111111 << (SET_SIZE * h) for h in range(N_SETS))
SET_CARDS: tuple[tuple[int, ...], ...] = tuple(
    tuple(range(SET_SIZE * h, SET_SIZE * (h + 1))) for h in range(N_SETS)
)
TEAM_MEMBERS: tuple[tuple[int, ...], ...] = ((0, 2, 4), (1, 3, 5))
OPPONENTS: tuple[tuple[int, ...], ...] = tuple(
    TEAM_MEMBERS[1 - (p % 2)] for p in range(N_PLAYERS)
)
TEAMMATES: tuple[tuple[int, ...], ...] = tuple(
    TEAM_MEMBERS[p % 2] for p in range(N_PLAYERS)
)


def set_of(card: int) -> int:
    """Set id (0..8) of a card."""
    return card // SET_SIZE


def team_of(player: int) -> int:
    """Team id (0 = A = even seats, 1 = B = odd seats)."""
    return player & 1


def _build_names() -> tuple[str, ...]:
    names: list[str] = []
    for s in SUITS:
        names += [r + s for r in LOW_RANKS]
        names += [r + s for r in HIGH_RANKS]
    names += ["8" + s for s in SUITS] + ["JK1", "JK2"]
    return tuple(names)


CARD_NAMES: tuple[str, ...] = _build_names()
SET_NAMES: tuple[str, ...] = tuple(
    [f"{half}-{s}" for s in SUITS for half in ("low", "high")] + ["eights"]
)
_NAME_TO_CARD = {n: i for i, n in enumerate(CARD_NAMES)}


def card_name(card: int) -> str:
    """Short name, e.g. ``'QS'`` or ``'JK1'``."""
    return CARD_NAMES[card]


def parse_card(name: str) -> int:
    """Inverse of :func:`card_name`."""
    return _NAME_TO_CARD[name.upper()]


def set_name(set_id: int) -> str:
    """Human readable set name, e.g. ``'high-H'``."""
    return SET_NAMES[set_id]


def cards_to_mask(cards: Iterable[int]) -> int:
    """Bitmask with one bit per card."""
    m = 0
    for c in cards:
        m |= 1 << c
    return m


def mask_to_cards(mask: int) -> list[int]:
    """Sorted card ids present in a bitmask."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def format_hand(mask: int) -> str:
    """Pretty-print a hand grouped by set."""
    parts = []
    for h in range(N_SETS):
        sub = mask & SET_MASKS[h]
        if sub:
            parts.append(f"{SET_NAMES[h]}[{' '.join(CARD_NAMES[c] for c in mask_to_cards(sub))}]")
    return " ".join(parts) if parts else "(empty)"
