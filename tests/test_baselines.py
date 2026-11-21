"""Baseline agents and shared agent helpers."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from fish.baselines import (
    RandomAgent,
    RuleBasedAgent,
    certain_declaration,
    guess_declaration,
    legal_asks,
)
from fish.cards import N_CARDS, parse_card
from fish.engine import Ask, Declare, State, is_terminal, legal_actions, new_game, step, validate_action
from fish.env import FishEnv, duel
from fish.knowledge import Knowledge, View
from tests.test_engine import _state, random_policy
from tests.test_knowledge import tiny

C = parse_card
ALL = [RandomAgent, RuleBasedAgent]


def known_card_scenario():
    """Player 3 publicly takes 5S from player 2, then fails; player 0 (holding
    low spades) is on turn and knows 5S is with opponent 3."""
    s = _state({0: ["2S", "3S", "9H"], 2: ["5S", "9D"], 3: ["4S", "9C"]}, turn=3)
    pub = Knowledge(3)
    pub.hand_size = list(s.hand_sizes())
    for a in (Ask(2, C("5S")), Ask(0, C("6S"))):
        s, obs = step(s, a)
        pub.update(obs)
    assert s.turn == 0 and pub.known_owner(C("5S")) == 3
    return s, pub


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.__name__)
def test_agents_play_full_legal_games(cls):
    """Self-play; the environment rejects illegal moves, and every view must fit the truth."""
    for seed in range(2):
        agent, rng = cls(), np.random.default_rng(seed + 100)
        env = FishEnv()
        view = env.reset(seed=seed)
        while not env.done:
            assert view.know.consistent_with(env.state)
            view, _ = env.step(agent.act(view, rng))
        assert sum(env.score) == 9


def test_agents_only_receive_views():
    for cls in ALL:
        params = list(inspect.signature(cls.act).parameters)
        assert params == ["self", "view", "rng"]
    assert View._fields == ("player", "hand", "public", "know")
    # nothing reachable from a view is a State
    state = new_game(np.random.default_rng(0))
    v = Knowledge(state.turn).view_for(state.turn, state)
    for k in (v.public, v.know):
        for slot in Knowledge.__slots__:
            assert not isinstance(getattr(k, slot), State)


def test_legal_asks_match_engine():
    rng = np.random.default_rng(0)
    state = new_game(rng)
    pub = Knowledge(state.turn)
    for _ in range(60):
        v = pub.view_for(state.turn, state)
        assert sorted(legal_asks(v)) == sorted(legal_actions(state, state.turn))
        state, obs = step(state, random_policy(state, rng, 0.02))
        pub.update(obs)
        if is_terminal(state):
            break


def test_certain_declaration_is_taken_by_every_agent():
    low_s = ["2S", "3S", "4S", "5S", "6S", "7S"]
    s = _state({0: low_s[:5] + ["9H"], 3: [low_s[5]]}, turn=0)
    pub = Knowledge(0)
    pub.hand_size = list(s.hand_sizes())
    assert certain_declaration(pub.view_for(0, s).know, 0) is None
    s, obs = step(s, Ask(3, C("7S")))
    pub.update(obs)
    v = pub.view_for(0, s)
    want = Declare(0, (0,) * 6)
    assert certain_declaration(v.know, 0) == want
    for cls in ALL:
        assert cls().act(v, np.random.default_rng(0)) == want
    # a teammate who sees the public transfer does not know the other four cards
    assert certain_declaration(pub.view_for(2, s).know, 2) is None


def test_rule_based_asks_for_known_card():
    s, pub = known_card_scenario()
    v = pub.view_for(0, s)
    want = Ask(3, C("5S"))
    for seed in range(5):
        assert RuleBasedAgent().act(v, np.random.default_rng(seed)) == want
    picks = {RandomAgent().act(v, np.random.default_rng(i)) for i in range(40)}
    assert len(picks) > 5  # the random agent does not focus on it


def test_forced_declaration_when_opponents_are_out():
    owner = np.full(N_CARDS, -1, dtype=np.int8)
    owner[:6] = [0, 0, 0, 2, 2, 4]
    s = State.from_owner(owner, turn=0, score=(4, 4))
    pub = tiny([3, 0, 2, 0, 1, 0], [0])
    pub.score = [4, 4]
    v = pub.view_for(0, s)
    assert legal_asks(v) == []
    for cls in ALL:
        a = cls().act(v, np.random.default_rng(1))
        assert isinstance(a, Declare) and a.set_id == 0 and a.assignment[:3] == (0, 0, 0)
        assert all(q in (2, 4) for q in a.assignment[3:])
        validate_action(s, a)
    g = guess_declaration(v.know, 0)
    assert g.set_id == 0 and g.assignment[:3] == (0, 0, 0)


def test_forced_guess_respects_hand_sizes():
    """A guess never gives a player more cards than they hold."""
    owner = np.full(N_CARDS, -1, dtype=np.int8)
    owner[:6] = [0, 0, 0, 0, 2, 4]
    s = State.from_owner(owner, turn=2, score=(4, 4))
    pub = tiny([4, 0, 1, 0, 1, 0], [0])
    g = guess_declaration(pub.view_for(2, s).know, 2)
    assert g.assignment[4] == 2
    assert sorted(g.assignment[:4] + g.assignment[5:]) == [0, 0, 0, 0, 4]


def test_agents_are_seeded():
    rule, rand = RuleBasedAgent(), RandomAgent()
    assert duel(rule, rand, 0, seed=3) == duel(rule, rand, 0, seed=3)


def test_rule_based_beats_random():
    rule, rand = RuleBasedAgent(), RandomAgent()
    margins = np.array([duel(rule, rand, g, seed=11)[0] for g in range(40)], dtype=float)
    per_deal = margins.reshape(-1, 2).mean(1)  # each deal is played from both sides
    ci95 = 1.96 * per_deal.std(ddof=1) / np.sqrt(len(per_deal))
    assert per_deal.mean() - ci95 > 0, (per_deal.mean(), ci95)
