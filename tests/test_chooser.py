"""The move chooser and the self-play pieces built on it."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fish.baselines import RuleBasedAgent  # noqa: E402
from fish.chooser import NetAgent, choose, score_moves  # noqa: E402
from fish.cards import N_CARDS, N_PLAYERS, N_SETS, mask_to_cards  # noqa: E402
from fish.engine import Ask, Declare, validate_action  # noqa: E402
from fish.env import FishEnv  # noqa: E402
from fish.features import N_FEATURES  # noqa: E402
from fish.selfplay import Game, ReplayBuffer, compare, play_training_game, td_targets  # noqa: E402
from fish.solver import Solver  # noqa: E402
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
    moves = score_moves(net, view.know, view.public, view.player, view.hand, rng, **FAST)
    assert 1 <= sum(isinstance(m.action, Ask) for m in moves) <= 4
    values = [m.value for m in moves]
    assert values == sorted(values, reverse=True)
    for m in moves:
        validate_action(env.state, m.action)
        assert -1.0 <= m.value <= 1.0 and 0.0 <= m.chance <= 1.0
    action, ranking = choose(net, view.know, view.public, view.player, view.hand,
                             np.random.default_rng(0), **FAST)
    assert action == ranking[0].action
    # with epsilon = 1 a candidate ask is played
    action, ranking = choose(net, view.know, view.public, view.player, view.hand, rng, 1.0, **FAST)
    assert isinstance(action, Ask) and action in [m.action for m in ranking]


def test_certain_declaration_is_the_only_candidate_and_scored_exactly():
    # last set, all six cards in our hand: declaring wins the game 5-4
    deal = np.full(N_CARDS, -1, dtype=np.int8)
    deal[:6] = 0
    env = FishEnv()
    view = env.reset(deal=deal, first_player=0, score=(4, 4))
    moves = score_moves(small_net(), view.know, view.public, 0, view.hand,
                        np.random.default_rng(0), **FAST)
    assert len(moves) == 1 and moves[0].action == Declare(0, (0,) * 6)
    assert np.isclose(moves[0].value, 1 / MARGIN_SCALE) and moves[0].chance == 1.0


def test_forced_guess_when_nobody_can_be_asked():
    deal = np.full(N_CARDS, -1, dtype=np.int8)
    deal[:6] = [0, 0, 0, 2, 2, 4]
    env = FishEnv()
    view = env.reset(deal=deal, first_player=0, score=(4, 4))
    moves = score_moves(small_net(), view.know, view.public, 0, view.hand,
                        np.random.default_rng(0), **FAST)
    assert all(isinstance(m.action, Declare) for m in moves)
    validate_action(env.state, moves[0].action)
    # right wins 5-4, wrong loses 4-5: the value is between the two
    assert -1 / MARGIN_SCALE <= moves[0].value <= 1 / MARGIN_SCALE
    assert 0.0 < moves[0].chance < 1.0


def test_training_game_and_targets():
    net = small_net()
    g = play_training_game(net, np.random.default_rng(1), epsilon=0.3, max_actions=40, **FAST)
    # one position per move, from the seat of the player who moved; a move
    # that ends the game leaves no position behind
    count = g.length - int(not g.truncated)
    assert g.length <= 40 and count > 0
    assert g.frames.shape == (count, N_SETS, N_FEATURES) and g.frames.dtype == np.float32
    assert g.live.shape == (count, N_SETS) and g.rewards.shape == (g.length, N_SETS)
    assert g.when.tolist() == list(range(count)) and set(g.sides.tolist()) <= {1.0, -1.0}
    assert g.frames.min() >= 0.0 and g.frames.max() <= 1.0
    assert (np.abs(g.rewards).sum(1) <= 1).all()  # at most one set is decided per move
    # a set is in play exactly until the move that declares it
    gone = np.cumsum(np.abs(g.rewards), axis=0) > 0
    assert (g.live == ~gone[g.when]).all()
    # lam = 1: the target of a set is how it ends later on, from the mover's side
    to_come = np.stack([g.rewards[t + 1:].sum(0) for t in g.when]) * g.sides[:, None]
    assert np.allclose(td_targets(net, g, lam=1.0), to_come)
    blended = td_targets(net, g, lam=0.5)
    assert blended.shape == (count, N_SETS) and np.abs(blended).max() <= 1.0 + 1e-6


def test_targets_follow_each_set_to_its_declaration():
    """Hand-made game of four moves by seats 0, 1, 0, 1 in which the even
    seats win set 2 on the third move."""
    net = small_net()
    frames = np.zeros((4, N_SETS, N_FEATURES), dtype=np.float32)
    live = np.ones((4, N_SETS), dtype=bool)
    live[2:, 2] = False
    rewards = np.zeros((4, N_SETS), dtype=np.float32)
    rewards[2, 2] = 1.0
    sides = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    game = Game(frames, live, np.arange(4), sides, rewards, 4, True)
    t = td_targets(net, game, lam=1.0)
    assert np.allclose(t[:, 2], [1.0, -1.0, 0.0, 0.0])  # won by seat 0's team, lost by seat 1's
    assert np.allclose(np.delete(t, 2, axis=1), 0.0)    # the other sets are never declared
    # with lam = 0 only the outcome of the very next move and the next prediction count
    v = net(torch.zeros(1, N_FEATURES)).item()  # every row looks the same here
    t0 = td_targets(net, game, lam=0.0)
    assert np.allclose(t0[:, 2], [-v, -1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(t0[:, 0], [-v, -v, -v, 0.0], atol=1e-6)
    # positions that were not recorded are skipped over: only seat 0's moves
    skipped = Game(frames[::2], live[::2], np.array([0, 2]), sides[::2], rewards, 4, True)
    assert np.allclose(td_targets(net, skipped, lam=1.0)[:, 2], [1.0, 0.0])


def test_training_game_with_other_players_in_some_seats():
    net = small_net()
    other = RuleBasedAgent()
    # the odd team is played by another player: only the even seats' moves are recorded
    g = play_training_game(net, np.random.default_rng(2), max_actions=40,
                           others=other, other_seats=(1, 3, 5), **FAST)
    assert len(g.frames) == len(g.when) == len(g.sides) <= g.length and (g.sides == 1.0).all()
    assert td_targets(net, g, lam=1.0).shape == (len(g.frames), N_SETS)
    # every seat is played by another player and all of them are recorded (the warm start)
    everyone = tuple(range(N_PLAYERS))
    g = play_training_game(net, np.random.default_rng(2), max_actions=40, others=other,
                           other_seats=everyone, learn_seats=everyone, **FAST)
    assert g.truncated and len(g.frames) == g.length == 40
    assert g.when.tolist() == list(range(40))
    with pytest.raises(ValueError):
        play_training_game(net, np.random.default_rng(2), other_seats=(1, 3, 5), **FAST)


def test_solver_ranks_moves_with_a_network():
    env = FishEnv()
    view = env.reset(seed=4, first_player=0)
    solver = Solver(0, mask_to_cards(view.hand), 0, n_particles=24, net=small_net())
    moves = solver.best_moves()
    assert moves and not moves[0].certain
    values = [m.value for m in moves]
    assert values == sorted(values, reverse=True)
    for m in moves:
        validate_action(env.state, m.action)


def test_replay_buffer_wraps_around():
    buf = ReplayBuffer(10)
    x = np.ones((4, N_SETS, N_FEATURES), dtype=np.float32)
    live = np.zeros((4, N_SETS), dtype=bool)
    live[:, :3] = True
    for i in range(4):
        buf.add(x, np.full((4, N_SETS), i, dtype=np.float32), live)
    assert buf.size == 10
    assert sorted(set(buf.y.ravel().tolist())) == [1.0, 2.0, 3.0]  # the oldest entries are gone
    bx, by, bm = buf.sample(32, np.random.default_rng(0))
    assert bx.shape == (32, N_SETS, N_FEATURES) and bx.dtype == np.float32
    assert by.shape == (32, N_SETS) and bm.shape == (32, N_SETS) and bm.dtype == np.float32
    assert (bm[:, :3] == 1).all() and (bm[:, 3:] == 0).all()


def test_net_agent_plays_whole_games_against_the_rule_player():
    agent = NetAgent(small_net(), **FAST)
    margin, wins = compare(agent, RuleBasedAgent(), games=2, seed=3)
    assert -9 <= margin <= 9 and 0.0 <= wins <= 1.0
