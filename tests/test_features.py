"""Feature encoding of a player's knowledge."""
from __future__ import annotations

import numpy as np

from fish.baselines import RuleBasedAgent
from fish.cards import N_CARDS, N_PLAYERS, N_SETS
from fish.env import FishEnv
from fish.features import FEATURE_SIZE, SLICES, encode, encode_batch, seat_order


def test_size_and_layout():
    assert FEATURE_SIZE == 2 * 6 * 54 + 6 + 6 * 9 + 9 + 2 + 6
    assert SLICES["known"].start == 0 and SLICES["turn"].stop == FEATURE_SIZE
    assert seat_order(4) == [4, 5, 0, 1, 2, 3]
    assert encode_batch([], 0).shape == (0, FEATURE_SIZE)


def test_fresh_game_from_each_seat():
    env = FishEnv()
    env.reset(seed=0, first_player=3)
    for seat in range(N_PLAYERS):
        view = env.view(seat)
        x = encode(view.know, seat)
        assert x.shape == (FEATURE_SIZE,) and x.dtype == np.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
        known = x[SLICES["known"]].reshape(N_PLAYERS, N_CARDS)
        poss = x[SLICES["poss"]].reshape(N_PLAYERS, N_CARDS)
        # position 0 is the player: exactly their nine cards are known
        mine = np.array([view.hand >> c & 1 for c in range(N_CARDS)], dtype=np.float32)
        assert (known[0] == mine).all() and (poss[0] == mine).all()
        assert known[1:].sum() == 0
        # everyone else could hold any card that is not the player's
        assert (poss[1:] == 1 - mine).all()
        assert (x[SLICES["hand_size"]] == 1.0).all()
        assert x[SLICES["declared"]].sum() == 0 and x[SLICES["score"]].sum() == 0
        turn = x[SLICES["turn"]]
        assert turn.sum() == 1 and turn[(3 - seat) % N_PLAYERS] == 1


def test_teammates_see_the_same_score_and_opponents_the_reverse():
    env = FishEnv()
    rng = np.random.default_rng(0)
    agent = RuleBasedAgent()
    view = env.reset(seed=1)
    while not env.done and sum(env.score) < 3:
        view, _ = env.step(agent.act(view, rng))
    a, b = env.score
    for seat in range(N_PLAYERS):
        x = encode(env.view(seat).know, seat)
        want = [a / N_SETS, b / N_SETS] if seat % 2 == 0 else [b / N_SETS, a / N_SETS]
        assert np.allclose(x[SLICES["score"]], want)
        assert x[SLICES["declared"]].sum() == a + b
        # cards of declared sets are possible for nobody
        poss = x[SLICES["poss"]].reshape(N_PLAYERS, N_SETS, 6)
        gone = x[SLICES["declared"]].astype(bool)
        assert poss[:, gone].sum() == 0
        # a known card is always a possible card
        assert (x[SLICES["known"]] <= x[SLICES["poss"]]).all()
    rows = encode_batch([env.view(0).know, env.view(2).know], 0)
    assert rows.shape == (2, FEATURE_SIZE)
