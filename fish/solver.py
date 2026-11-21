"""The solver: tell it your cards and every move at the table, ask it what to play.

    net, _ = load("checkpoints/net.pt")             # the trained network
    solver = Solver(seat=0, hand=["2S", "3S", "QH", ...], first_player=3, net=net)
    solver.ask(target=0, card="4S")                 # seat 3 asks you; you don't have it
    for move in solver.best_moves(top=3):           # now it is your turn
        print(describe(move), move.chance)
    solver.ask(target=3, card="5S", success=True)   # you ask seat 3 and get the card

Seats are numbered 0-5 clockwise.  Even seats are one team, odd seats the
other, so your teammates are the two seats that differ from yours by 2 and 4.

The solver has to hear about *every* move at the table, not only your own:
what the other players ask for is where its information comes from.

Moves are ranked by a value network trained by self-play
(:mod:`fish.chooser`).  Without a network the hand-written heuristic
(:mod:`fish.heuristic`) is used instead.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np

from .baselines import certain_declaration
from .belief import ask_likelihood_weights, marginals, sample_deals
from .cards import (
    HAND_SIZE,
    N_CARDS,
    N_PLAYERS,
    N_SETS,
    SET_MASKS,
    SET_SIZE,
    card_name,
    format_hand,
    parse_card,
    set_name,
)
from .engine import AskObs, DeclareObs, PublicObs, resolve_turn
from .heuristic import Move, describe, rank_moves
from .knowledge import Contradiction, Knowledge

__all__ = ["BadInput", "Move", "Solver", "describe"]

Card = Union[int, str]


class BadInput(ValueError):
    """The move that was entered is not possible in the current position."""


def _card(card: Card) -> int:
    if isinstance(card, str):
        try:
            return parse_card(card)
        except KeyError:
            raise BadInput(f"unknown card {card!r}") from None
    c = int(card)
    if not 0 <= c < N_CARDS:
        raise BadInput(f"unknown card {card!r}")
    return c


class Solver:
    """Follows one game of Fish from one seat and recommends moves.

    Parameters
    ----------
    seat:
        Your seat, 0-5.
    hand:
        Your cards (names like ``"QH"`` or card ids).
    first_player:
        Seat that moves next.
    hand_sizes, declared, score:
        Only needed when you start in the middle of a game: how many cards each
        seat holds, which sets are already gone and the sets won by the even
        and the odd team.  Left out, a fresh game with nine cards each is assumed.
    n_particles:
        How many deals are imagined per decision.  More is slower and steadier.
    alpha:
        How strongly asks are read as a sign of holding many cards of that
        set.  0 switches this off.
    seed:
        Makes the recommendations repeatable.
    net:
        A trained value network (from :func:`fish.value_net.load`).  With it,
        moves are ranked by the network; without it, by the hand-written
        heuristic in :mod:`fish.heuristic`.
    """

    def __init__(
        self,
        seat: int,
        hand: Iterable[Card],
        first_player: int,
        hand_sizes: Sequence[int] | None = None,
        declared: Iterable[int] = (),
        score: tuple[int, int] = (0, 0),
        n_particles: int = 300,
        alpha: float = 1.0,
        seed: int = 0,
        net=None,
    ) -> None:
        if not 0 <= seat < N_PLAYERS:
            raise BadInput("seat must be 0-5")
        if not 0 <= first_player < N_PLAYERS:
            raise BadInput("first player must be a seat 0-5")
        cards = [_card(c) for c in hand]
        if len(set(cards)) != len(cards):
            raise BadInput("a card was entered twice")
        mask = 0
        for c in cards:
            mask |= 1 << c
        sizes = [HAND_SIZE] * N_PLAYERS if hand_sizes is None else [int(n) for n in hand_sizes]
        gone = 0
        for h in declared:
            gone |= 1 << int(h)
        n_live = N_SETS - bin(gone).count("1")
        if len(sizes) != N_PLAYERS or min(sizes) < 0 or sum(sizes) != SET_SIZE * n_live:
            raise BadInput(f"hand sizes must add up to {SET_SIZE * n_live} cards")
        if len(cards) != sizes[seat]:
            raise BadInput(f"you should have {sizes[seat]} cards, got {len(cards)}")
        for h in range(N_SETS):
            if gone >> h & 1 and mask & SET_MASKS[h]:
                raise BadInput(f"you hold a card of {set_name(h)}, which is already declared")

        self.seat = seat
        self.n_particles = n_particles
        self.alpha = alpha
        self.net = net
        self.rng = np.random.default_rng(seed)
        self.hand = mask

        public = Knowledge(resolve_turn(sizes, first_player))
        public.declared = gone
        public.hand_size = sizes
        live = public.live_mask
        public.poss = [live if n else 0 for n in sizes]
        public.score = [int(score[0]), int(score[1])]
        public.propagate()
        know = public.copy()
        know.set_hand(seat, mask)
        self.public = public  # what everyone at the table knows
        self.know = know      # the same plus your own hand
        self._undo: list[tuple[Knowledge, Knowledge, int]] = []

    # ------------------------------------------------------------- game state
    @property
    def turn(self) -> int:
        """Seat whose move it is."""
        return self.know.turn

    @property
    def my_turn(self) -> bool:
        return not self.done and self.turn == self.seat

    @property
    def done(self) -> bool:
        """True once every set has been declared."""
        return self.know.is_terminal()

    @property
    def score(self) -> tuple[int, int]:
        """Sets won by the even-seat team and the odd-seat team."""
        return (self.know.score[0], self.know.score[1])

    @property
    def hand_sizes(self) -> list[int]:
        return list(self.know.hand_size)

    def live_sets(self) -> list[int]:
        """Ids of the sets that have not been declared."""
        return [h for h in range(N_SETS) if not self.know.declared >> h & 1]

    def summary(self) -> str:
        """Your hand, everyone's card counts, the score and whose turn it is."""
        sizes = ", ".join(f"seat {p}: {n}" for p, n in enumerate(self.hand_sizes))
        mine, theirs = self.score[self.seat & 1], self.score[1 - (self.seat & 1)]
        state = "game over" if self.done else f"seat {self.turn} to move"
        return (f"your hand: {format_hand(self.hand)}\n"
                f"cards held: {sizes}\n"
                f"score: you {mine}, them {theirs}   ({state})")

    # ------------------------------------------------------- entering moves
    def ask(self, target: int, card: Card, success: bool | None = None) -> AskObs:
        """Record that the player on turn asked ``target`` for ``card``.

        ``success`` says whether the card was handed over.  It can be left out
        when you are the one being asked, since your hand settles it.
        """
        if self.done:
            raise BadInput("the game is over")
        k = self.know
        asker = k.turn
        c = _card(card)
        if not (isinstance(target, (int, np.integer)) and 0 <= target < N_PLAYERS):
            raise BadInput("target must be a seat 0-5")
        if (target - asker) % 2 == 0:
            raise BadInput(f"seat {asker} is on turn and can only ask seats on the other team")
        if k.hand_size[target] == 0:
            raise BadInput(f"seat {target} has no cards")
        h = c // SET_SIZE
        if k.declared >> h & 1:
            raise BadInput(f"{set_name(h)} has already been declared")
        bit = 1 << c
        me = self.seat
        if asker == me:
            if self.hand & bit:
                raise BadInput("you cannot ask for a card you hold")
            if not self.hand & SET_MASKS[h]:
                raise BadInput(f"you hold no card of {set_name(h)}, so you cannot ask for one")
        if target == me:
            truth = bool(self.hand & bit)
            if success is not None and bool(success) != truth:
                raise BadInput(f"you {'do' if truth else 'do not'} hold {card_name(c)}")
            success = truth
        elif success is None:
            raise BadInput("say whether the ask succeeded")
        success = bool(success)
        hand = self.hand
        if success and asker == me:
            hand |= bit
        if success and target == me:
            hand &= ~bit
        obs = AskObs(asker, int(target), c, success, asker if success else int(target))
        self._apply(obs, hand)
        return obs

    def declare(
        self,
        set_id: int,
        correct: bool,
        lost: Sequence[int] | None = None,
        assignment: Sequence[int] | None = None,
    ) -> DeclareObs:
        """Record that the player on turn declared ``set_id``.

        ``lost[p]`` is how many cards of the set seat ``p`` held (they all
        leave the game).  For a correct declaration you can pass the
        ``assignment`` that was announced instead (holder of each of the six
        cards, in card order).
        """
        if self.done:
            raise BadInput("the game is over")
        k = self.know
        declarer = k.turn
        if not (isinstance(set_id, (int, np.integer)) and 0 <= set_id < N_SETS):
            raise BadInput("set must be 0-8")
        if k.declared >> set_id & 1:
            raise BadInput(f"{set_name(set_id)} has already been declared")
        correct = bool(correct)
        sigma: tuple[int, ...] = ()
        if assignment is not None:
            sigma = tuple(int(q) for q in assignment)
            if len(sigma) != SET_SIZE or any(
                    not 0 <= q < N_PLAYERS or (q - declarer) % 2 for q in sigma):
                raise BadInput("an assignment names six seats on the declarer's team")
        if lost is None:
            if not (correct and sigma):
                raise BadInput("say how many cards of the set each seat held")
            lost = [sigma.count(p) for p in range(N_PLAYERS)]
        counts = [int(n) for n in lost]
        if len(counts) != N_PLAYERS or min(counts) < 0 or sum(counts) != SET_SIZE:
            raise BadInput("the six counts must add up to 6")
        if correct and sigma and counts != [sigma.count(p) for p in range(N_PLAYERS)]:
            raise BadInput("the counts do not match the announced assignment")
        for p in range(N_PLAYERS):
            if counts[p] > k.hand_size[p]:
                raise BadInput(f"seat {p} only holds {k.hand_size[p]} cards")
            if correct and counts[p] and (p - declarer) % 2:
                raise BadInput("a correct declaration has every card on the declarer's team")
        sm = SET_MASKS[set_id]
        mine = (self.hand & sm).bit_count()
        if counts[self.seat] != mine:
            raise BadInput(f"you hold {mine} cards of {set_name(set_id)}, not {counts[self.seat]}")
        sizes = [k.hand_size[p] - counts[p] for p in range(N_PLAYERS)]
        obs = DeclareObs(declarer, int(set_id), sigma, correct, tuple(counts),
                         resolve_turn(sizes, declarer))
        self._apply(obs, self.hand & ~sm)
        return obs

    def _apply(self, obs: PublicObs, hand: int) -> None:
        public, know = self.public.copy(), self.know.copy()
        try:
            public.update(obs)
            know.update(obs)
        except Contradiction as err:
            raise BadInput(f"that does not fit the moves entered so far ({err})") from None
        self._undo.append((self.public, self.know, self.hand))
        self.public, self.know, self.hand = public, know, hand

    def undo(self) -> None:
        """Take back the last move that was entered."""
        if not self._undo:
            raise BadInput("nothing to undo")
        self.public, self.know, self.hand = self._undo.pop()

    # ------------------------------------------------------------- deciding
    def chances(self) -> np.ndarray:
        """``P[seat, card]``: how likely each seat is to hold each card, as you see it."""
        parts = sample_deals(self.know, self.n_particles, self.rng)
        return marginals(parts, ask_likelihood_weights(parts, self.know, self.alpha))

    def best_moves(self, top: int | None = None) -> list[Move]:
        """Your legal moves, best first (only on your turn).

        Declarations that are known to be wrong are left out.  With a
        network, ``value`` is the expected final set margin divided by 9; with
        the heuristic it is that module's own score.
        """
        if self.done:
            raise BadInput("the game is over")
        if self.turn != self.seat:
            raise BadInput(f"it is seat {self.turn}'s turn, not yours")
        if self.net is None:
            moves = rank_moves(self.know, self.public, self.seat, self.hand, self.rng,
                               self.n_particles, self.alpha)
        else:
            from .chooser import score_moves  # needs PyTorch, so only loaded when used

            sure = certain_declaration(self.know, self.seat)
            scored = score_moves(self.net, self.know, self.public, self.seat, self.hand, self.rng,
                                 n_particles=self.n_particles, alpha=self.alpha,
                                 max_asks=N_CARDS * N_PLAYERS)  # look at every ask
            moves = [Move(s.action, s.value, s.chance, s.action == sure) for s in scored]
        return moves if top is None else moves[:top]

    def best_move(self) -> Move:
        """The recommended move."""
        return self.best_moves(1)[0]
