"""Self-play: generate training games, turn them into targets, store them.

One game gives, for every moment of the game and every seat, the feature
vector that seat would see.  The training target for each of them is a blend
of the network's own prediction one move later and the real final result
(the TD(lambda) return): ``lam = 1`` uses only the final result, smaller
values lean more on the next prediction.

Needs PyTorch.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .cards import N_PLAYERS
from .chooser import choose
from .engine import Action, is_terminal, new_game, step
from .env import FishEnv
from .features import FEATURE_SIZE, encode
from .knowledge import Knowledge, View
from .value_net import MARGIN_SCALE, ValueNet, predict


def play_training_game(
    net: ValueNet,
    rng: np.random.Generator,
    epsilon: float = 0.0,
    max_actions: int = 2000,
    **options,
) -> tuple[np.ndarray, int, int, bool]:
    """All six seats play one game with ``net``.

    Returns ``(frames, margin, length, truncated)``: ``frames[t, s]`` is seat
    ``s``'s feature vector after ``t`` moves (shape ``(length + 1, 6,
    FEATURE_SIZE)``), ``margin`` is sets won by the even seats minus sets won
    by the odd seats.  ``options`` go to :func:`fish.chooser.score_moves`.
    """
    state = new_game(rng)
    private = []
    for s in range(N_PLAYERS):
        k = Knowledge(state.turn)
        k.set_hand(s, state.hands[s])
        private.append(k)
    frames = [np.stack([encode(private[s], s) for s in range(N_PLAYERS)])]
    n = 0
    while not is_terminal(state) and n < max_actions:
        p = state.turn
        action, _ = choose(net, private[p], p, state.hands[p], rng, epsilon, **options)
        state, obs = step(state, action)
        for k in private:
            k.update(obs)
        frames.append(np.stack([encode(private[s], s) for s in range(N_PLAYERS)]))
        n += 1
    margin = state.score[0] - state.score[1]
    return np.stack(frames), margin, n, not is_terminal(state)


def td_targets(net: ValueNet, frames: np.ndarray, margin: int, lam: float = 0.8) -> np.ndarray:
    """Training target for every frame of one game, shape ``(length + 1, 6)``."""
    steps = frames.shape[0]
    values = predict(net, frames.reshape(-1, FEATURE_SIZE)).reshape(steps, N_PLAYERS)
    final = np.array([(margin if s % 2 == 0 else -margin) / MARGIN_SCALE
                      for s in range(N_PLAYERS)], dtype=np.float32)
    targets = np.empty((steps, N_PLAYERS), dtype=np.float32)
    targets[-1] = final
    values[-1] = final
    for t in range(steps - 2, -1, -1):
        targets[t] = (1.0 - lam) * values[t + 1] + lam * targets[t + 1]
    return targets


class ReplayBuffer:
    """Fixed-size store of (features, target) pairs; the oldest are overwritten.

    Features are kept as 16-bit floats to save memory (about 1.5 MB per
    thousand entries).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.x = np.zeros((capacity, FEATURE_SIZE), dtype=np.float16)
        self.y = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self._next = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        for i in range(0, len(x), self.capacity):  # in case one batch exceeds the capacity
            xs, ys = x[i:i + self.capacity], y[i:i + self.capacity]
            idx = (self._next + np.arange(len(xs))) % self.capacity
            self.x[idx] = xs
            self.y[idx] = ys
            self._next = int((self._next + len(xs)) % self.capacity)
            self.size = min(self.capacity, self.size + len(xs))

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, self.size, size=n)
        return self.x[idx].astype(np.float32), self.y[idx]


class _Player(Protocol):
    def act(self, view: View, rng: np.random.Generator) -> Action: ...


def compare(a: _Player, b: _Player, games: int, seed: int = 0,
            max_actions: int = 3000) -> tuple[float, float]:
    """Play ``a``'s team against ``b``'s team.

    Deals are played twice with the teams swapped, so card luck mostly cancels.
    Returns ``a``'s average set margin and the fraction of games it won.
    """
    env = FishEnv()
    margins = []
    for g in range(games):
        a_team = g % 2
        view = env.reset(seed=[seed, g // 2])
        rng = np.random.default_rng([seed, g, 1])
        n = 0
        while not env.done and n < max_actions:
            player = a if view.player % 2 == a_team else b
            view, _ = env.step(player.act(view, rng))
            n += 1
        margins.append(env.score[a_team] - env.score[1 - a_team])
    m = np.array(margins, dtype=float)
    return float(m.mean()), float((m > 0).mean())
