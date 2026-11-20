"""Game engine: state, legal actions, transitions and public observations.

The engine knows the full (hidden) state.  Agents never see a ``State``; they
see a view built from the stream of ``PublicObs`` (see ``knowledge.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence, Union

import numpy as np

from .cards import (
    ALL_SETS_MASK,
    HAND_SIZE,
    N_CARDS,
    N_PLAYERS,
    N_SETS,
    OPPONENTS,
    SET_MASKS,
    SET_SIZE,
)


class IllegalAction(ValueError):
    """Raised by :func:`step` / :func:`validate_action` for an illegal action."""


class Ask(NamedTuple):
    """Ask opponent ``target`` for ``card``."""

    target: int
    card: int


class Declare(NamedTuple):
    """Declare set ``set_id``; ``assignment[i]`` is the claimed holder of card ``6*set_id + i``."""

    set_id: int
    assignment: tuple[int, ...]


Action = Union[Ask, Declare]


class AskObs(NamedTuple):
    """Public result of an ask."""

    asker: int
    target: int
    card: int
    success: bool
    next_turn: int


class DeclareObs(NamedTuple):
    """Public result of a declaration.

    ``removed[p]`` is the number of cards of the set that left player ``p``'s
    hand (hand sizes are public, so this is public too).
    """

    declarer: int
    set_id: int
    assignment: tuple[int, ...]
    correct: bool
    removed: tuple[int, ...]
    next_turn: int


PublicObs = Union[AskObs, DeclareObs]


@dataclass(frozen=True, slots=True)
class State:
    """Full game state.

    ``owner[c]`` is the player holding card ``c`` or -1 once its set has been
    declared.  ``hands`` is the same information as six bitmasks and is kept in
    sync by :func:`step`.  ``declared`` is a bitmask over set ids and
    ``score[t]`` the number of sets won by team ``t``.
    """

    owner: np.ndarray
    hands: tuple[int, ...]
    turn: int
    score: tuple[int, int]
    declared: int

    @staticmethod
    def from_owner(
        owner: Sequence[int] | np.ndarray,
        turn: int,
        score: tuple[int, int] = (0, 0),
        declared: int | None = None,
    ) -> "State":
        """Build a state from an owner array (the turn is resolved past empty hands)."""
        arr = np.array(owner, dtype=np.int8)
        if arr.shape != (N_CARDS,):
            raise ValueError("owner must have length 54")
        hands = [0] * N_PLAYERS
        dec = 0
        for c, p in enumerate(arr.tolist()):
            if p >= 0:
                hands[p] |= 1 << c
            else:
                dec |= 1 << (c // SET_SIZE)
        for h in range(N_SETS):
            gone = sum(1 for c in range(6 * h, 6 * h + 6) if arr[c] < 0)
            if gone not in (0, SET_SIZE):
                raise ValueError(f"set {h} is partially declared")
        if declared is not None and declared != dec:
            raise ValueError("declared mask inconsistent with owner array")
        arr.setflags(write=False)
        hands_t = tuple(hands)
        return State(arr, hands_t, resolve_turn(hands_t, turn), (score[0], score[1]), dec)

    def hand_size(self, player: int) -> int:
        """Number of cards held by ``player``."""
        return self.hands[player].bit_count()

    def hand_sizes(self) -> tuple[int, ...]:
        """Hand sizes of all six players."""
        return tuple(h.bit_count() for h in self.hands)

    def declared_sets(self) -> list[int]:
        """Ids of the sets already declared."""
        return [h for h in range(N_SETS) if self.declared >> h & 1]


def new_game(rng: np.random.Generator, first_player: int | None = None) -> State:
    """Shuffle and deal 9 cards to each player; random first player unless given."""
    perm = rng.permutation(N_CARDS)
    owner = np.empty(N_CARDS, dtype=np.int8)
    owner[perm] = np.repeat(np.arange(N_PLAYERS, dtype=np.int8), HAND_SIZE)
    turn = int(rng.integers(N_PLAYERS)) if first_player is None else first_player
    return State.from_owner(owner, turn)


def is_terminal(state: State) -> bool:
    """True once all nine sets have been declared."""
    return state.declared == ALL_SETS_MASK


def resolve_turn(hands: Sequence[int], turn: int) -> int:
    """Seat that actually moves when the turn nominally belongs to ``turn``.

    A player without cards passes to the next teammate clockwise that has
    cards; if the whole team is out, to the next opponent clockwise with cards.
    """
    if hands[turn]:
        return turn
    for d in (2, 4, 1, 3, 5):
        q = (turn + d) % N_PLAYERS
        if hands[q]:
            return q
    return turn


def enumerate_asks(player: int, hand: int, hand_sizes: Sequence[int]) -> list[Ask]:
    """All legal asks for ``player`` holding ``hand`` given public hand sizes."""
    targets = [q for q in OPPONENTS[player] if hand_sizes[q] > 0]
    out: list[Ask] = []
    if not targets or not hand:
        return out
    for h in range(N_SETS):
        sm = SET_MASKS[h]
        if hand & sm:
            missing = sm & ~hand
            base = SET_SIZE * h
            for i in range(SET_SIZE):
                if missing >> (base + i) & 1:
                    for t in targets:
                        out.append(Ask(t, base + i))
    return out


def legal_actions(state: State, player: int) -> list[Action]:
    """Legal asks for ``player`` (empty unless it is their turn).

    Declarations are legal for the turn player for every undeclared set and
    every assignment to own-team seats; they are too many to enumerate, so
    agents construct them and :func:`validate_action` / :func:`step` check
    them.  See :func:`declarable_sets`.
    """
    if is_terminal(state) or player != state.turn:
        return []
    return list(enumerate_asks(player, state.hands[player], state.hand_sizes()))


def declarable_sets(state: State, player: int) -> list[int]:
    """Sets ``player`` may declare right now (all undeclared sets on their turn)."""
    if is_terminal(state) or player != state.turn:
        return []
    return [h for h in range(N_SETS) if not state.declared >> h & 1]


def validate_action(state: State, action: Action) -> None:
    """Raise :class:`IllegalAction` unless ``action`` is legal for the turn player."""
    if is_terminal(state):
        raise IllegalAction("game is over")
    p = state.turn
    if isinstance(action, Ask):
        t, c = action
        if not (isinstance(c, (int, np.integer)) and 0 <= c < N_CARDS):
            raise IllegalAction(f"bad card {c!r}")
        if not (isinstance(t, (int, np.integer)) and 0 <= t < N_PLAYERS):
            raise IllegalAction(f"bad target {t!r}")
        if (t - p) % 2 == 0:
            raise IllegalAction("target must be an opponent")
        if not state.hands[t]:
            raise IllegalAction("target has no cards")
        hand = state.hands[p]
        if hand >> c & 1:
            raise IllegalAction("cannot ask for a card you hold")
        if not hand & SET_MASKS[c // SET_SIZE]:
            raise IllegalAction("must hold a card of the same set (or the set is declared)")
    elif isinstance(action, Declare):
        h, sigma = action
        if not (isinstance(h, (int, np.integer)) and 0 <= h < N_SETS):
            raise IllegalAction(f"bad set {h!r}")
        if state.declared >> h & 1:
            raise IllegalAction("set already declared")
        if len(sigma) != SET_SIZE:
            raise IllegalAction("assignment must name 6 holders")
        for q in sigma:
            if not (isinstance(q, (int, np.integer)) and 0 <= q < N_PLAYERS and (q - p) % 2 == 0):
                raise IllegalAction("cards may only be assigned to the declarer's own team")
    else:
        raise IllegalAction(f"unknown action {action!r}")


def step(state: State, action: Action, check: bool = True) -> tuple[State, PublicObs]:
    """Apply ``action`` for the turn player; return the new state and what everyone sees."""
    if check:
        validate_action(state, action)
    p = state.turn
    if isinstance(action, Ask):
        t, c = action
        bit = 1 << c
        if state.hands[t] & bit:
            hands = list(state.hands)
            hands[t] ^= bit
            hands[p] |= bit
            owner = state.owner.copy()
            owner[c] = p
            owner.setflags(write=False)
            new = State(owner, tuple(hands), p, state.score, state.declared)
            return new, AskObs(p, t, c, True, p)
        new = State(state.owner, state.hands, t, state.score, state.declared)
        return new, AskObs(p, t, c, False, t)
    h, sigma = action
    sigma = tuple(int(q) for q in sigma)
    base = SET_SIZE * h
    correct = state.owner[base : base + SET_SIZE].tolist() == list(sigma)
    sm = SET_MASKS[h]
    removed = tuple((hd & sm).bit_count() for hd in state.hands)
    hands_t = tuple(hd & ~sm for hd in state.hands)
    owner = state.owner.copy()
    owner[base : base + SET_SIZE] = -1
    owner.setflags(write=False)
    winner = (p & 1) if correct else 1 - (p & 1)
    score = (state.score[0] + (winner == 0), state.score[1] + (winner == 1))
    turn = resolve_turn(hands_t, p)
    new = State(owner, hands_t, turn, score, state.declared | (1 << h))
    return new, DeclareObs(p, h, sigma, correct, removed, turn)


def returns(state: State) -> tuple[int, int]:
    """Reward per team: own sets minus opponent sets."""
    a, b = state.score
    return (a - b, b - a)
