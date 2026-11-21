"""The numbers the value network reads: one row per set."""
from __future__ import annotations

import numpy as np

from fish.baselines import RuleBasedAgent
from fish.belief import sample_deals
from fish.cards import N_PLAYERS, N_SETS, SET_MASKS, SET_SIZE
from fish.env import FishEnv
from fish.features import FEATURE_NAMES, N_FEATURES, set_features

COL = {name: i for i, name in enumerate(FEATURE_NAMES)}


def rows_for(env: FishEnv, seat: int, n: int = 30):
    """The seat's view of the current position and its feature rows."""
    view = env.view(seat)
    parts = sample_deals(view.know, n, np.random.default_rng(seat))
    rows, live = set_features(view.know, view.public, seat, parts, np.full(n, 1.0 / n))
    return view, rows, live


def cards_per_set(mask: int) -> np.ndarray:
    return np.array([(mask & SET_MASKS[h]).bit_count() for h in range(N_SETS)])


def test_layout():
    assert N_FEATURES == len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_fresh_game_from_each_seat():
    env = FishEnv()
    env.reset(seed=0, first_player=3)
    for seat in range(N_PLAYERS):
        view, rows, live = rows_for(env, seat)
        assert rows.shape == (N_SETS, N_FEATURES) and rows.dtype == np.float32
        assert live.all() and rows.min() >= 0.0 and rows.max() <= 1.0
        mine = cards_per_set(view.hand) / SET_SIZE
        assert np.allclose(rows[:, COL["mine"]], mine)
        # nothing has happened yet: I know my own cards and nothing is public
        assert np.allclose(rows[:, COL["known_ours"]], mine)
        for name in ("known_theirs", "public_ours", "public_theirs",
                     "public_min_ours", "public_min_theirs"):
            assert (rows[:, COL[name]] == 0).all(), name
        # my team holds 27 cards in every imagined deal, mine among them
        team = rows[:, COL["team"]]
        assert (team >= mine - 1e-6).all() and np.isclose(team.sum(), 27 / SET_SIZE)
        assert (rows[:, COL["all_ours"]] + rows[:, COL["all_theirs"]] <= 1 + 1e-6).all()
        # whose turn it is, seen from this seat (seat 3 moves first)
        turn = [float(rows[0, COL[name]]) for name in ("turn_me", "turn_mate", "turn_opponent")]
        assert turn == [float(seat == 3), float(seat in (1, 5)), float(seat % 2 == 0)]
        assert rows[0, COL["sets_left"]] == 1.0
        assert rows[0, COL["my_cards"]] == 0.5 and rows[0, COL["team_cards"]] == 0.5
        best = rows[:, COL["best_ask"]]
        if seat == 3:
            # I can only ask in sets I hold a card of
            assert (best[mine == 0] == 0).all() and best.max() > 0
            assert np.allclose(rows[:, COL["turn_holds"]], mine > 0)
            assert np.isclose(rows[0, COL["best_ask_any"]], best.max())
        else:
            assert (best == 0).all() and rows[0, COL["best_ask_any"]] == 0


def test_a_card_seen_changing_hands_becomes_public():
    env = FishEnv()
    env.reset(seed=3, first_player=0)
    hands = env.state.hands
    ask = next(a for a in env.legal_asks() if hands[a.target] >> a.card & 1)
    env.step(ask)
    h = ask.card // SET_SIZE
    assert env.turn == 0  # the ask worked, so seat 0 goes again
    _, rows, _ = rows_for(env, 0)
    assert np.isclose(rows[h, COL["public_ours"]], 1 / SET_SIZE)
    assert rows[h, COL["public_min_ours"]] >= 2 / SET_SIZE - 1e-6  # the card asked from, and the new one
    assert rows[0, COL["turn_me"]] == 1.0
    # the opponents see the same card on the other side
    _, rows, _ = rows_for(env, 1)
    assert np.isclose(rows[h, COL["public_theirs"]], 1 / SET_SIZE)
    assert rows[h, COL["known_theirs"]] >= 1 / SET_SIZE - 1e-6
    assert rows[0, COL["turn_opponent"]] == 1.0


def test_declared_sets_are_blank():
    env = FishEnv()
    rng = np.random.default_rng(0)
    agent = RuleBasedAgent()
    view = env.reset(seed=1)
    while not env.done and sum(env.score) < 3:
        view, _ = env.step(agent.act(view, rng))
    gone = np.array([bool(env.state.declared >> h & 1) for h in range(N_SETS)])
    assert gone.sum() == 3
    for seat in range(N_PLAYERS):
        _, rows, live = rows_for(env, seat)
        assert (live == ~gone).all()
        assert (rows[gone] == 0).all()
        assert np.allclose(rows[live, COL["sets_left"]], 6 / N_SETS)
        assert rows.min() >= 0.0 and rows.max() <= 1.0
        # what everyone knows, I know too
        assert (rows[:, COL["public_ours"]] <= rows[:, COL["known_ours"]] + 1e-6).all()
        assert (rows[:, COL["public_theirs"]] <= rows[:, COL["known_theirs"]] + 1e-6).all()
