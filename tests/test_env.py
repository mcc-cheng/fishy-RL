"""The environment wrapper: reset, views, stepping, chosen deals."""
from __future__ import annotations

import numpy as np
import pytest

from fish.baselines import RuleBasedAgent
from fish.cards import N_CARDS, N_PLAYERS, parse_card
from fish.engine import Ask, AskObs, Declare, IllegalAction
from fish.env import FishEnv, public_knowledge

C = parse_card


def test_reset_deals_nine_cards_each_and_is_seeded():
    env = FishEnv()
    view = env.reset(seed=3)
    assert env.state.hand_sizes() == (9,) * N_PLAYERS
    assert not env.done and env.score == (0, 0) and env.history == []
    assert view.player == env.turn and view.hand == env.state.hands[env.turn]
    first = env.state.owner.copy()
    env.reset(seed=3)
    assert (env.state.owner == first).all()
    env.reset(seed=4)
    assert not (env.state.owner == first).all()
    assert FishEnv().reset(seed=3, first_player=5).player == 5


def test_must_reset_first():
    with pytest.raises(RuntimeError):
        FishEnv().state


def test_views_show_only_own_hand():
    env = FishEnv()
    env.reset(seed=0)
    for p in range(N_PLAYERS):
        v = env.view(p)
        assert v.player == p and v.know.known[p] == v.hand
        # nothing is known about anyone else's cards at the start
        assert all(v.know.known[q] == 0 for q in range(N_PLAYERS) if q != p)
    assert all(k == 0 for k in env.public.known)


def test_step_ask_and_declare_from_a_chosen_deal():
    # one set left: seat 0 holds five low spades, opponent seat 1 holds 7S
    deal = np.full(N_CARDS, -1, dtype=np.int8)
    deal[:6] = [0, 0, 0, 0, 0, 1]
    env = FishEnv()
    view = env.reset(deal=deal, first_player=0, score=(4, 4))
    assert view.player == 0 and env.score == (4, 4)
    assert env.declarable_sets() == [0]
    assert env.legal_asks() == [Ask(1, C("7S"))]
    with pytest.raises(IllegalAction):
        env.step(Ask(2, C("7S")))  # a teammate
    view, obs = env.step(Ask(1, C("7S")))
    assert isinstance(obs, AskObs) and obs.success and view.player == 0
    assert env.public.known_owner(C("7S")) == 0  # everyone saw the card move
    assert env.legal_asks() == []  # the opponents are out of cards
    view, obs = env.step(Declare(0, (0,) * 6))
    assert obs.correct and view is None
    assert env.done and env.score == (5, 4) and env.returns() == (1, -1)
    assert len(env.history) == 2 and env.view() is None
    assert "game over" in env.render()


def test_public_knowledge_matches_state_mid_game():
    deal = np.full(N_CARDS, -1, dtype=np.int8)
    deal[:6] = [0, 0, 2, 4, 1, 3]
    env = FishEnv()
    env.reset(deal=deal, first_player=5)  # seat 5 has no cards: turn moves on
    assert env.turn in (1, 3)
    k = public_knowledge(env.state)
    assert k.hand_size == [2, 1, 1, 1, 1, 0] and k.poss[5] == 0
    assert k.consistent_with(env.state)


def test_full_game_keeps_knowledge_consistent():
    env = FishEnv()
    rng = np.random.default_rng(1)
    agents = [RuleBasedAgent() for _ in range(N_PLAYERS)]
    view = env.reset(seed=1)
    n = 0
    while not env.done:
        assert env.public.consistent_with(env.state)
        assert view.know.consistent_with(env.state)
        view, _ = env.step(agents[view.player].act(view, rng))
        n += 1
        assert n < 3000
    assert sum(env.score) == 9 and len(env.history) == n
    assert "seat 0" in env.render()
