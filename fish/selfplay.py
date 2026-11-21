"""Self-play: generate training games, turn them into targets, store them.

Every time a player moves, the position that move leads to is recorded as that
player sees it: one row of numbers per set still in play (see
:mod:`fish.features`).  These are exactly the kind of positions the network is
asked to score when a move is chosen.

The network predicts, for one set at a time, who will end up with it.  So the
training target is per set as well: +1 if the player's team won that set later
in the game, -1 if the opponents did, and 0 if the game was cut off before it
was declared.  With ``lam`` below 1 the target is blended with the network's
own prediction at the next recorded position (the TD(lambda) return):
``lam = 1`` uses only real outcomes, smaller values lean more on the next
prediction.

Needs PyTorch.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .cards import N_PLAYERS, N_SETS
from .chooser import card_chances, choose
from .engine import is_terminal, new_game, step
from .env import duel, public_knowledge
from .features import N_FEATURES, set_features
from .knowledge import Knowledge, View
from .value_net import ValueNet, predict


class Game(NamedTuple):
    """One training game.

    * ``frames[i]``: the ``i``-th recorded position, one row per set, from the
      seat of the player who just moved (shape ``(positions, 9, N_FEATURES)``);
    * ``live[i, set]``: True if the set was still in play in that position;
    * ``when[i]``: which move of the game led to it (0 = the first move);
    * ``sides[i]``: +1 if that player sits on an even seat, -1 if odd;
    * ``rewards[t, set]``: +1 if the even seats won the set on move ``t``,
      -1 if the odd seats did, otherwise 0 (shape ``(length, 9)``);
    * ``length``: number of moves played;
    * ``truncated``: True if the game was cut off before all sets were declared.
    """

    frames: np.ndarray
    live: np.ndarray
    when: np.ndarray
    sides: np.ndarray
    rewards: np.ndarray
    length: int
    truncated: bool


def play_training_game(
    net: ValueNet,
    rng: np.random.Generator,
    epsilon: float = 0.0,
    max_actions: int = 300,
    others=None,
    other_seats: tuple[int, ...] = (),
    learn_seats: tuple[int, ...] | None = None,
    **options,
) -> Game:
    """Play one game.  ``net`` plays every seat except those in
    ``other_seats``, which are played by ``others`` (any player with
    ``act(view, rng)``, e.g. the heuristic or an older network).

    Positions are recorded after the moves of the seats in ``learn_seats``
    (default: the seats ``net`` plays).  ``options`` go to
    :func:`fish.chooser.score_moves`.
    """
    if other_seats and others is None:
        raise ValueError("other_seats given without a player for them")
    other = set(other_seats)
    learn = set(range(N_PLAYERS)) - other if learn_seats is None else set(learn_seats)
    n_particles = options.get("n_particles", 64)
    alpha = options.get("alpha", 1.0)
    state = new_game(rng)
    public = public_knowledge(state)
    private = []
    for s in range(N_PLAYERS):
        k = Knowledge(state.turn)
        k.set_hand(s, state.hands[s])
        private.append(k)
    frames, lives, when, sides, rewards = [], [], [], [], []
    n = 0
    while not is_terminal(state) and n < max_actions:
        p = state.turn
        if p in other:
            action = others.act(View(p, state.hands[p], public, private[p]), rng)
        else:
            action, _ = choose(net, private[p], public, p, state.hands[p], rng, epsilon, **options)
        declared, margin = state.declared, state.score[0] - state.score[1]
        state, obs = step(state, action)
        public.update(obs)
        for k in private:
            k.update(obs)
        reward = np.zeros(N_SETS, dtype=np.float32)
        gone = state.declared & ~declared
        if gone:  # a set was declared: who got it?
            reward[gone.bit_length() - 1] = (state.score[0] - state.score[1]) - margin
        rewards.append(reward)
        if p in learn and not is_terminal(state):
            parts, w, _ = card_chances(private[p], rng, n_particles, alpha)
            rows, live = set_features(private[p], public, p, parts, w)
            frames.append(rows)
            lives.append(live)
            when.append(n)
            sides.append(1.0 if p % 2 == 0 else -1.0)
        n += 1
    return Game(
        np.stack(frames) if frames else np.zeros((0, N_SETS, N_FEATURES), dtype=np.float32),
        np.stack(lives) if lives else np.zeros((0, N_SETS), dtype=bool),
        np.array(when, dtype=np.int64),
        np.array(sides, dtype=np.float32),
        np.stack(rewards) if rewards else np.zeros((0, N_SETS), dtype=np.float32),
        n,
        not is_terminal(state),
    )


def td_targets(net: ValueNet, game: Game, lam: float = 0.9) -> np.ndarray:
    """Training target for every set of every recorded position, shape
    ``(positions, 9)``.

    The target for a set is how it ends for the player the position was
    recorded for: +1 won, -1 lost, 0 never declared.  Only the entries where
    ``game.live`` is True mean anything.
    """
    count = len(game.frames)
    targets = np.zeros((count, N_SETS), dtype=np.float32)
    if count == 0:
        return targets
    values = predict(net, game.frames) * game.sides[:, None]  # all seen from the even seats' side
    ahead = np.zeros(N_SETS, dtype=np.float32)  # how each set ends after move t, for the even seats
    i = count - 1
    for t in range(len(game.rewards) - 1, -1, -1):
        if i >= 0 and game.when[i] == t:
            targets[i] = game.sides[i] * ahead
            ahead = np.where(game.live[i], (1.0 - lam) * values[i] + lam * ahead, ahead)
            i -= 1
        ahead = ahead + game.rewards[t]
    return targets


class ReplayBuffer:
    """Fixed-size store of recorded positions with their targets; the oldest
    are overwritten.  Features are kept as 16-bit floats to save memory (about
    0.4 MB per thousand positions)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.x = np.zeros((capacity, N_SETS, N_FEATURES), dtype=np.float16)
        self.y = np.zeros((capacity, N_SETS), dtype=np.float32)
        self.m = np.zeros((capacity, N_SETS), dtype=bool)
        self.size = 0
        self._next = 0

    def add(self, x: np.ndarray, y: np.ndarray, live: np.ndarray) -> None:
        """Store positions ``x`` with targets ``y``; ``live`` marks the sets
        whose target counts."""
        for i in range(0, len(x), self.capacity):  # in case one batch exceeds the capacity
            xs = x[i:i + self.capacity]
            idx = (self._next + np.arange(len(xs))) % self.capacity
            self.x[idx] = xs
            self.y[idx] = y[i:i + self.capacity]
            self.m[idx] = live[i:i + self.capacity]
            self._next = int((self._next + len(xs)) % self.capacity)
            self.size = min(self.capacity, self.size + len(xs))

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``n`` random positions: features, targets and the 0/1 mask of sets that count."""
        idx = rng.integers(0, self.size, size=n)
        return self.x[idx].astype(np.float32), self.y[idx], self.m[idx].astype(np.float32)


def compare(a, b, games: int, seed: int = 0, max_actions: int = 3000) -> tuple[float, float]:
    """Play ``a``'s team against ``b``'s team over paired deals (see
    :func:`fish.env.duel`).  Returns ``a``'s average set margin and the
    fraction of games it won."""
    m = np.array([duel(a, b, g, seed, max_actions)[0] for g in range(games)], dtype=float)
    return float(m.mean()), float((m > 0).mean())
