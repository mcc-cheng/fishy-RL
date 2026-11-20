"""The Fish environment: one object that runs a game.

The rules themselves live in :mod:`fish.engine` and the bookkeeping of what is
publicly known in :mod:`fish.knowledge`.  :class:`FishEnv` ties the two
together so that playing a game is a simple loop::

    env = FishEnv()
    view = env.reset(seed=0)            # random deal
    while not env.done:
        action = agent.act(view, rng)   # the agent only ever sees its view
        view, obs = env.step(action)
    print(env.score)

A game can also start from a deal you choose: ``env.reset(deal=owner)`` where
``owner[c]`` is the seat holding card ``c`` (-1 for cards of declared sets).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .cards import N_PLAYERS, N_SETS, format_hand, set_name
from .engine import (
    Action,
    Ask,
    PublicObs,
    State,
    enumerate_asks,
    is_terminal,
    new_game,
    returns,
    step,
)
from .knowledge import Knowledge, View


def public_knowledge(state: State) -> Knowledge:
    """What everyone knows at the start of a game from ``state``: hand sizes,
    which sets are gone, the score and whose turn it is.  Nothing about who
    holds which card."""
    k = Knowledge(state.turn)
    k.declared = state.declared
    live = k.live_mask
    k.hand_size = list(state.hand_sizes())
    k.poss = [live if n else 0 for n in k.hand_size]
    k.score = list(state.score)
    k.propagate()
    return k


class FishEnv:
    """A game of Fish in progress.

    The full deal is hidden inside the environment.  Players get a
    :class:`~fish.knowledge.View` (their own hand plus what is public) and
    hand back an action.
    """

    def __init__(self) -> None:
        self._state: State | None = None
        self._public: Knowledge | None = None
        self.history: list[PublicObs] = []

    # ------------------------------------------------------------------ setup
    def reset(
        self,
        seed: int | Sequence[int] | np.random.Generator | None = None,
        deal: Sequence[int] | np.ndarray | None = None,
        first_player: int | None = None,
        score: tuple[int, int] = (0, 0),
    ) -> View | None:
        """Start a new game and return the view of the player on turn.

        * ``seed``: shuffle and deal 9 cards each (same seed, same deal).
        * ``deal``: start from this exact deal instead; ``deal[c]`` is the seat
          holding card ``c``, or -1 if the card's set is already declared.
          ``score`` gives the sets already won by each team in that case.
        * ``first_player``: seat that moves first (random from the seed, or
          seat 0 for a chosen deal, when omitted).
        """
        if deal is not None:
            self._state = State.from_owner(deal, first_player or 0, score)
        else:
            rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
            self._state = new_game(rng, first_player)
        self._public = public_knowledge(self._state)
        self.history = []
        return self.view()

    # ------------------------------------------------------------- game state
    @property
    def state(self) -> State:
        """The full hidden state (for the referee, logging and tests; not for agents)."""
        if self._state is None:
            raise RuntimeError("call reset() first")
        return self._state

    @property
    def public(self) -> Knowledge:
        """Everything that is publicly known so far."""
        if self._public is None:
            raise RuntimeError("call reset() first")
        return self._public

    @property
    def turn(self) -> int:
        """Seat whose move it is."""
        return self.state.turn

    @property
    def done(self) -> bool:
        """True once all nine sets have been declared."""
        return is_terminal(self.state)

    @property
    def score(self) -> tuple[int, int]:
        """Sets won so far by team A (seats 0, 2, 4) and team B (seats 1, 3, 5)."""
        return self.state.score

    def returns(self) -> tuple[int, int]:
        """Reward per team: sets won minus sets lost."""
        return returns(self.state)

    # ---------------------------------------------------------------- playing
    def view(self, player: int | None = None) -> View | None:
        """What ``player`` can see (default: the player on turn); None when the game is over."""
        if player is None:
            if self.done:
                return None
            player = self.turn
        return self.public.view_for(player, self.state)

    def legal_asks(self) -> list[Ask]:
        """Every ask the player on turn may make.

        Declaring is also always legal: any undeclared set, each card assigned
        to a player on the declarer's own team.
        """
        if self.done:
            return []
        s = self.state
        return enumerate_asks(s.turn, s.hands[s.turn], s.hand_sizes())

    def declarable_sets(self) -> list[int]:
        """Sets the player on turn may declare (all that are not declared yet)."""
        if self.done:
            return []
        return [h for h in range(N_SETS) if not self.state.declared >> h & 1]

    def step(self, action: Action) -> tuple[View | None, PublicObs]:
        """Play ``action`` for the player on turn.

        Returns the view of whoever moves next (None if the game just ended)
        and the public observation of what happened.  Raises
        :class:`~fish.engine.IllegalAction` for an illegal move.
        """
        self._state, obs = step(self.state, action)
        self.public.update(obs)
        self.history.append(obs)
        return self.view(), obs

    # ---------------------------------------------------------------- display
    def render(self) -> str:
        """All hands, the score and whose turn it is, as text (shows hidden cards)."""
        s = self.state
        lines = []
        for p in range(N_PLAYERS):
            mark = "->" if p == s.turn and not self.done else "  "
            team = "A" if p % 2 == 0 else "B"
            lines.append(f"{mark} seat {p} ({team}): {format_hand(s.hands[p]) or '-'}")
        left = [set_name(h) for h in self.declarable_sets()] if not self.done else []
        lines.append(f"score A {s.score[0]} - B {s.score[1]}"
                     + (f"   sets left: {', '.join(left)}" if left else "   game over"))
        return "\n".join(lines)
