"""Public constraint store and propagation.

A :class:`Knowledge` holds everything that can be *proved* about the hidden
deal from public observations (and optionally one player's own hand):

* ``poss[p]``        bitmask of cards player ``p`` could hold
                     (``possible`` gives the same as a bool matrix),
* ``hand_size[p]``   exact, public,
* ``min_in_set[p][h]`` lower bound on the number of set-``h`` cards ``p`` holds,
* ``known[p]``       cards whose only possible holder is ``p`` (derived).

After every update :meth:`Knowledge.propagate` runs the inference rules to a
fixed point.  All rules are sound: the true deal always satisfies the store.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .cards import (
    ALL_CARDS_MASK,
    HAND_SIZE,
    N_CARDS,
    N_PLAYERS,
    N_SETS,
    SET_MASKS,
    SET_SIZE,
)
from .engine import AskObs, DeclareObs, PublicObs, State


class Contradiction(RuntimeError):
    """The constraint store has no consistent deal (indicates a bug or bad input)."""


class Knowledge:
    """Hard constraints on the current deal; see the module docstring."""

    __slots__ = (
        "poss", "known", "hand_size", "min_in_set", "declared", "score", "turn",
        "n_actions", "ask_log",
    )

    def __init__(self, first_player: int = 0) -> None:
        self.poss: list[int] = [ALL_CARDS_MASK] * N_PLAYERS
        self.known: list[int] = [0] * N_PLAYERS
        self.hand_size: list[int] = [HAND_SIZE] * N_PLAYERS
        self.min_in_set: list[list[int]] = [[0] * N_SETS for _ in range(N_PLAYERS)]
        self.declared: int = 0
        self.score: list[int] = [0, 0]
        self.turn: int = first_player
        self.n_actions: int = 0
        # [asker, set, offset]: the asker held (count_now + offset) cards of the
        # set when asking; used only by optional likelihood reweighting.
        # None disables logging (used for throw-away simulation copies).
        self.ask_log: list[list[int]] | None = []

    # ------------------------------------------------------------------ basics
    def copy(self, with_log: bool = True) -> "Knowledge":
        """Independent copy (``with_log=False`` drops the ask log, for simulations)."""
        k = Knowledge.__new__(Knowledge)
        k.poss = self.poss[:]
        k.known = self.known[:]
        k.hand_size = self.hand_size[:]
        k.min_in_set = [row[:] for row in self.min_in_set]
        k.declared = self.declared
        k.score = self.score[:]
        k.turn = self.turn
        k.n_actions = self.n_actions
        k.ask_log = (
            [e[:] for e in self.ask_log] if with_log and self.ask_log is not None else None
        )
        return k

    @property
    def possible(self) -> np.ndarray:
        """``possible[p, c]`` as a (6, 54) bool array."""
        return masks_to_matrix(self.poss)

    @property
    def live_mask(self) -> int:
        """Bitmask of cards still in play."""
        m = 0
        for h in range(N_SETS):
            if not self.declared >> h & 1:
                m |= SET_MASKS[h]
        return m

    def known_owner(self, card: int) -> int:
        """Player known to hold ``card`` or -1."""
        bit = 1 << card
        for p in range(N_PLAYERS):
            if self.known[p] & bit:
                return p
        return -1

    def is_terminal(self) -> bool:
        """All sets declared."""
        return self.declared == (1 << N_SETS) - 1

    # ----------------------------------------------------------------- updates
    def update(self, obs: PublicObs) -> None:
        """Incorporate one public observation and propagate."""
        if isinstance(obs, AskObs):
            a, j, c = obs.asker, obs.target, obs.card
            h = c // SET_SIZE
            bit = 1 << c
            mins = self.min_in_set
            # The asker held >= 1 card of the set and did not hold c.
            if mins[a][h] < 1:
                mins[a][h] = 1
            log = self.ask_log
            if obs.success:
                poss = self.poss
                for p in range(N_PLAYERS):
                    poss[p] &= ~bit
                poss[a] |= bit
                mins[a][h] += 1
                if mins[j][h] > 0:
                    mins[j][h] -= 1
                self.hand_size[a] += 1
                self.hand_size[j] -= 1
                if log is not None:
                    for e in log:
                        if e[1] == h:
                            if e[0] == a:
                                e[2] -= 1
                            elif e[0] == j:
                                e[2] += 1
                    # after the transfer the asker holds one more than when asking
                    log.append([a, h, -1])
            else:
                self.poss[a] &= ~bit
                self.poss[j] &= ~bit
                if log is not None:
                    log.append([a, h, 0])
        else:
            h = obs.set_id
            keep = ~SET_MASKS[h]
            for p in range(N_PLAYERS):
                self.poss[p] &= keep
                self.min_in_set[p][h] = 0
                self.hand_size[p] -= obs.removed[p]
            self.declared |= 1 << h
            team = obs.declarer & 1
            self.score[team if obs.correct else 1 - team] += 1
            if self.ask_log is not None:
                self.ask_log = [e for e in self.ask_log if e[1] != h]
        self.turn = obs.next_turn
        self.n_actions += 1
        self.propagate()

    def set_hand(self, player: int, hand: int) -> None:
        """Fix ``player``'s exact hand (private information) and propagate."""
        if hand.bit_count() != self.hand_size[player]:
            raise Contradiction("hand does not match public hand size")
        if hand & ~self.poss[player]:
            raise Contradiction("hand contains a card the player cannot hold")
        not_hand = ~hand
        for q in range(N_PLAYERS):
            self.poss[q] &= not_hand
        self.poss[player] = hand
        self.propagate()

    def view_for(self, player: int, true_state: State) -> "View":
        """``player``'s view: this public knowledge plus their own hand."""
        hand = true_state.hands[player]
        k = self.copy()
        k.set_hand(player, hand)
        return View(player, hand, self, k)

    # ------------------------------------------------------------- propagation
    def propagate(self) -> None:
        """Run all inference rules to a fixed point (raises on contradiction)."""
        poss = self.poss
        known = self.known
        hs = self.hand_size
        live = self.live_mask
        live_sets = [h for h in range(N_SETS) if not self.declared >> h & 1]
        while True:
            changed = False
            # Rule 1: a card with exactly one possible holder is known.
            once = 0
            multi = 0
            for m in poss:
                multi |= once & m
                once |= m
            if once != live:
                raise Contradiction("a live card has no possible holder")
            single = once & ~multi
            for p in range(N_PLAYERS):
                pm = poss[p]
                kn = pm & single
                known[p] = kn
                kc = kn.bit_count()
                size = hs[p]
                pc = pm.bit_count()
                if kc > size or pc < size:
                    raise Contradiction(f"player {p}: hand size cannot be met")
                # Rule 2: all cards known -> holds nothing else.
                if kc == size:
                    if pm != kn:
                        poss[p] = kn
                        changed = True
                    continue
                # Rule 4: exactly hand_size possible cards -> all of them are held.
                if pc == size:
                    rm = ~pm
                    for q in range(N_PLAYERS):
                        if q != p:
                            poss[q] &= rm
                    known[p] = pm
                    changed = True
                    continue
                mins = self.min_in_set[p]
                total_min = 0
                for h in live_sets:
                    sm = SET_MASKS[h]
                    k_h = (kn & sm).bit_count()
                    mn = mins[h]
                    if k_h > mn:
                        mn = mins[h] = k_h
                    total_min += mn
                    if mn > k_h:
                        ps = pm & sm
                        n = ps.bit_count()
                        if n < mn:
                            raise Contradiction(f"player {p}: min_in_set[{h}] cannot be met")
                        # Rule 3: min equals number of possible cards -> all held.
                        if n == mn:
                            rm = ~ps
                            for q in range(N_PLAYERS):
                                if q != p:
                                    poss[q] &= rm
                            changed = True
                if total_min > size:
                    raise Contradiction(f"player {p}: set minimums exceed hand size")
                # Rule 5 (extension): minimums fill the whole hand -> the player
                # holds exactly min cards per set, so sets whose known cards
                # already reach the minimum contain nothing else.
                if total_min == size:
                    for h in live_sets:
                        sm = SET_MASKS[h]
                        extra = pm & sm & ~kn
                        if extra and (kn & sm).bit_count() == mins[h]:
                            poss[p] &= ~extra
                            pm = poss[p]
                            changed = True
            if not changed:
                return

    # ------------------------------------------------------------- consistency
    def violations(self, state: State) -> list[str]:
        """Ways in which ``state`` contradicts this store (empty = consistent)."""
        out: list[str] = []
        if state.declared != self.declared:
            out.append("declared sets differ")
        if tuple(self.score) != tuple(state.score):
            out.append("score differs")
        if state.turn != self.turn and not self.is_terminal():
            out.append("turn differs")
        for p in range(N_PLAYERS):
            hand = state.hands[p]
            if hand.bit_count() != self.hand_size[p]:
                out.append(f"hand_size[{p}]")
            if hand & ~self.poss[p]:
                out.append(f"player {p} holds a card marked impossible")
            if self.known[p] & ~hand:
                out.append(f"player {p} lacks a card marked known")
            for h in range(N_SETS):
                if (hand & SET_MASKS[h]).bit_count() < self.min_in_set[p][h]:
                    out.append(f"min_in_set[{p}][{h}]")
        return out

    def consistent_with(self, state: State) -> bool:
        """True when ``state`` satisfies every constraint in the store."""
        return not self.violations(state)


class View(NamedTuple):
    """What one player sees: own hand, public knowledge and the private store
    (public knowledge + own hand, propagated)."""

    player: int
    hand: int
    public: Knowledge
    know: Knowledge


def masks_to_matrix(masks: list[int]) -> np.ndarray:
    """Six card bitmasks -> (6, 54) bool array."""
    raw = b"".join(m.to_bytes(8, "little") for m in masks)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8).reshape(len(masks), 8), axis=1,
                         bitorder="little")
    return bits[:, :N_CARDS].astype(bool)
