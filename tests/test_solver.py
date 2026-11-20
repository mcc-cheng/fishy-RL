"""The solver: following a game from one seat and recommending moves."""
from __future__ import annotations

import numpy as np
import pytest

from fish.baselines import RuleBasedAgent
from fish.cards import N_CARDS, N_PLAYERS, mask_to_cards
from fish.engine import Ask, AskObs, Declare, validate_action
from fish.env import FishEnv
from fish.solver import GAIN, BadInput, Solver, describe


def feed(solver: Solver, obs) -> None:
    """Enter one observed move into the solver."""
    if isinstance(obs, AskObs):
        solver.ask(obs.target, obs.card, obs.success)
    else:
        solver.declare(obs.set_id, obs.correct, lost=obs.removed)


def one_of_each() -> np.ndarray:
    """Deal where seat p holds card 6h + p of every set h."""
    return np.arange(N_CARDS) % N_PLAYERS


def test_solver_follows_a_whole_game_and_only_suggests_legal_moves():
    """Seat 0 plays the solver's advice; everyone else follows simple rules."""
    env = FishEnv()
    rng = np.random.default_rng(0)
    others = RuleBasedAgent()
    view = env.reset(seed=2)
    solver = Solver(0, mask_to_cards(env.state.hands[0]), env.turn, n_particles=60)
    my_moves = 0
    while not env.done:
        assert solver.turn == env.turn and solver.hand == env.state.hands[0]
        assert solver.hand_sizes == list(env.state.hand_sizes())
        assert solver.know.consistent_with(env.state)
        assert solver.public.consistent_with(env.state)
        if view.player == 0:
            moves = solver.best_moves()
            for m in moves:
                validate_action(env.state, m.action)
                assert 0.0 <= m.chance <= 1.0 + 1e-9
            action = moves[0].action
            my_moves += 1
        else:
            action = others.act(view, rng)
        view, obs = env.step(action)
        feed(solver, obs)
        assert len(env.history) < 3000
    assert solver.done and solver.score == env.score and my_moves > 0
    with pytest.raises(BadInput):
        solver.best_moves()


def test_a_card_seen_changing_hands_is_asked_for_first():
    env = FishEnv()
    env.reset(deal=one_of_each(), first_player=1)
    solver = Solver(0, mask_to_cards(env.state.hands[0]), 1, n_particles=100)
    assert not solver.my_turn
    with pytest.raises(BadInput):
        solver.best_moves()
    solver.ask(2, 2, True)   # seat 1 takes card 2 from seat 2, in plain sight
    solver.ask(0, 3)         # seat 1 asks us for card 3; we don't have it
    assert solver.my_turn
    best = solver.best_move()
    assert best.action == Ask(1, 2) and np.isclose(best.chance, 1.0)
    assert np.isclose(best.value, GAIN)
    assert describe(best) == "ask seat 1 for 4S"
    # every other ask is less likely and scores lower
    rest = [m for m in solver.best_moves() if m.action != best.action]
    assert rest and all(m.value < best.value for m in rest)
    # we know we don't hold card 3 and neither does seat 1
    P = solver.chances()
    assert P[0, 3] == 0 and P[1, 3] == 0 and np.isclose(P[:, 3].sum(), 1.0)


def test_certain_declaration_comes_first_and_midgame_start():
    # one set left: we hold five of its cards, opponent seat 1 holds the last
    solver = Solver(0, [0, 1, 2, 3, 4], first_player=0, hand_sizes=[5, 1, 0, 0, 0, 0],
                    declared=range(1, 9), score=(4, 4), n_particles=50)
    best = solver.best_move()
    assert best.action == Ask(1, 5) and np.isclose(best.chance, 1.0)
    solver.ask(1, 5, True)
    assert solver.my_turn and solver.hand_sizes == [6, 0, 0, 0, 0, 0]
    best = solver.best_move()
    assert best.certain and best.action == Declare(0, (0,) * 6)
    solver.declare(0, True, assignment=best.action.assignment)
    assert solver.done and solver.score == (5, 4)
    assert "game over" in solver.summary()


def test_forced_declaration_when_opponents_have_no_cards():
    # the last set is spread over our team; nobody is left to ask
    solver = Solver(0, [0, 1, 2], first_player=0, hand_sizes=[3, 0, 2, 0, 1, 0],
                    declared=range(1, 9), n_particles=100)
    moves = solver.best_moves()
    assert all(isinstance(m.action, Declare) for m in moves)
    sigma = moves[0].action.assignment
    assert sigma[:3] == (0, 0, 0) and sorted(sigma[3:]) == [2, 2, 4]
    assert 0.0 < moves[0].chance < 1.0 and not moves[0].certain


def test_bad_input_is_rejected_and_undo_restores():
    env = FishEnv()
    env.reset(deal=one_of_each(), first_player=0)
    solver = Solver(0, mask_to_cards(env.state.hands[0]), 0, n_particles=50)
    with pytest.raises(BadInput):
        solver.ask(2, 1, False)          # a teammate
    with pytest.raises(BadInput):
        solver.ask(1, 0, False)          # a card we hold
    with pytest.raises(BadInput):
        solver.ask(1, "ZZ", False)       # not a card
    with pytest.raises(BadInput):
        solver.ask(1, 1)                 # result missing
    with pytest.raises(BadInput):
        solver.declare(0, False, lost=[0, 2, 2, 2, 0, 0])   # we hold one card of the set
    with pytest.raises(BadInput):
        solver.declare(0, True, lost=[1, 1, 1, 1, 1, 1])    # correct needs one team
    with pytest.raises(BadInput):
        solver.undo()
    hand = solver.hand
    solver.ask(1, 1, True)               # we take card 1 from seat 1
    assert solver.hand == hand | 1 << 1 and solver.hand_sizes[:2] == [10, 8]
    solver.ask(1, 2, False)              # then miss; seat 1 is on turn
    assert solver.turn == 1
    with pytest.raises(BadInput):
        solver.ask(0, 1, False)          # we do hold card 1 now
    solver.ask(0, 1)                     # seat 1 takes it back; result filled in
    assert solver.hand == hand and solver.turn == 1
    solver.undo()
    solver.undo()
    assert solver.my_turn and solver.hand == hand | 1 << 1
    with pytest.raises(BadInput):
        Solver(0, [0, 1, 2], 0)          # a fresh game needs nine cards
    with pytest.raises(BadInput):
        Solver(0, [0] * 9, 0)            # duplicates


def test_same_seed_gives_same_advice():
    def advice():
        s = Solver(3, mask_to_cards(FishEnv().reset(seed=5, first_player=3).hand), 3,
                   n_particles=80, seed=7)
        return [(m.action, round(m.value, 9)) for m in s.best_moves()]
    assert advice() == advice()
