"""The move chooser and the self-play pieces built on it."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fish.baselines import RuleBasedAgent  # noqa: E402
from fish.chooser import NetAgent, choose, score_moves  # noqa: E402
from fish.cards import N_CARDS, N_PLAYERS  # noqa: E402
from fish.engine import Ask, Declare, validate_action  # noqa: E402
from fish.env import FishEnv  # noqa: E402
from fish.features import FEATURE_SIZE  # noqa: E402
from fish.selfplay import ReplayBuffer, compare, play_training_game, td_targets  # noqa: E402
from fish.value_net import MARGIN_SCALE, ValueNet  # noqa: E402

FAST = {"n_particles": 24, "max_asks": 4}


def small_net(seed: int = 0) -> ValueNet:
    torch.manual_seed(seed)
    return ValueNet(hidden=16, layers=1)


def test_moves_are_legal_sorted_and_bounded():
    net = small_net()
    env = FishEnv()
    view = env.reset(seed=0)
    rng = np.random.default_rng(0)
    moves = score_moves(net, view.know, view.player, view.hand, rng, **FAST)
    assert 1 <= sum(isinstance(m.action, Ask) for m in moves) <= 4
    values = [m.value for m in moves]
    assert values == sorted(values, reverse=True)
    for m in moves:
        validate_action(env.state, m.action)
        assert -1.0 <= m.value <= 1.0 and 0.0 <= m.chance <= 1.0
    action, ranking = choose(net, view.know, view.player, view.hand, np.random.default_rng(0), **FAST)
    assert action == ranking[0].action
    # with epsilon = 1 a candidate ask is played
    action, ranking = choose(net, view.know, view.player, view.hand, rng, 1.0, **FAST)
    assert isinstance(action, Ask) and action in [m.action for m in ranking]


def test_certain_declaration_is_the_only_candidate_and_scored_exactly():
    # last set, all six cards in our hand: declaring wins the game 5-4
    deal = np.full(N_CARDS, -1, dtype=np.int8)
    deal[:6] = 0
    env = FishEnv()
    view = env.reset(deal=deal, first_player=0, score=(4, 4))
    moves = score_moves(small_net(), view.know, 0, view.hand, np.random.default_rng(0), **FAST)
    assert len(moves) == 1 and moves[0].action == Declare(0, (0,) * 6)
    assert np.isclose(moves[0].value, 1 / MARGIN_SCALE) and moves[0].chance == 1.0


def test_forced_guess_when_nobody_can_be_asked():
    deal = np.full(N_CARDS, -1, dtype=np.int8)
    deal[:6] = [0, 0, 0, 2, 2, 4]
    env = FishEnv()
    view = env.reset(deal=deal, first_player=0, score=(4, 4))
    moves = score_moves(small_net(), view.know, 0, view.hand, np.random.default_rng(0), **FAST)
    assert all(isinstance(m.action, Declare) for m in moves)
    validate_action(env.state, moves[0].action)
    # right wins 5-4, wrong loses 4-5: the value is between the two
    assert -1 / MARGIN_SCALE <= moves[0].value <= 1 / MARGIN_SCALE
    assert 0.0 < moves[0].chance < 1.0


def test_training_game_and_targets():
    net = small_net()
    frames, margin, length, truncated = play_training_game(
        net, np.random.default_rng(1), epsilon=0.3, max_actions=40, **FAST)
    assert frames.shape == (length + 1, N_PLAYERS, FEATURE_SIZE) and frames.dtype == np.float32
    assert length <= 40 and (truncated or abs(margin) % 2 == 1)
    final = margin / MARGIN_SCALE
    only_result = td_targets(net, frames, margin, lam=1.0)
    assert only_result.shape == (length + 1, N_PLAYERS)
    assert np.allclose(only_result[:, 0::2], final) and np.allclose(only_result[:, 1::2], -final)
    blended = td_targets(net, frames, margin, lam=0.5)
    assert np.allclose(blended[-1], only_result[-1])
    assert np.abs(blended).max() <= 1.0


def test_replay_buffer_wraps_around():
    buf = ReplayBuffer(10)
    x = np.arange(4 * FEATURE_SIZE, dtype=np.float32).reshape(4, FEATURE_SIZE) % 2
    for i in range(4):
        buf.add(x, np.full(4, i, dtype=np.float32))
    assert buf.size == 10
    assert sorted(set(buf.y.tolist())) == [1.0, 2.0, 3.0]  # the oldest entries are gone
    bx, by = buf.sample(32, np.random.default_rng(0))
    assert bx.shape == (32, FEATURE_SIZE) and bx.dtype == np.float32 and by.shape == (32,)


def test_net_agent_plays_whole_games_against_the_rule_player():
    agent = NetAgent(small_net(), **FAST)
    margin, wins = compare(agent, RuleBasedAgent(), games=2, seed=3)
    assert -9 <= margin <= 9 and 0.0 <= wins <= 1.0
