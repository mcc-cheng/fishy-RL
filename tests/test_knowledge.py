"""Stage 2: knowledge store and propagation."""
from __future__ import annotations

import numpy as np
import pytest

from fish.cards import ALL_SETS_MASK, N_PLAYERS, N_SETS, SET_MASKS, cards_to_mask, parse_card
from fish.engine import Ask, AskObs, Declare, DeclareObs, State, is_terminal, new_game, step
from fish.knowledge import Contradiction, Knowledge
from tests.test_engine import random_policy

C = parse_card


def tiny(hand_sizes: list[int], live_sets: list[int]) -> Knowledge:
    """Hand-made store: only ``live_sets`` remain, nothing else is known."""
    k = Knowledge()
    k.declared = ALL_SETS_MASK & ~sum(1 << h for h in live_sets)
    live = k.live_mask
    k.poss = [live if n else 0 for n in hand_sizes]
    k.hand_size = list(hand_sizes)
    assert sum(hand_sizes) == 6 * len(live_sets)
    return k


def bit(name: str) -> int:
    return 1 << C(name)


# ------------------------------------------------------------------ soak tests
def run_checked_game(seed: int, all_views: bool) -> int:
    rng = np.random.default_rng(seed)
    state = new_game(rng)
    public = Knowledge(state.turn)
    private = []
    for p in range(N_PLAYERS):
        k = Knowledge(state.turn)
        k.set_hand(p, state.hands[p])
        private.append(k)
    n = 0
    while True:
        assert public.violations(state) == []
        viewers = range(N_PLAYERS) if all_views else [state.turn]
        for p in viewers:
            v = public.view_for(p, state)
            assert v.know.violations(state) == [], (seed, n, p)
            assert v.know.known[p] == state.hands[p] == v.hand
            assert private[p].violations(state) == [], (seed, n, p)
        if is_terminal(state):
            break
        state, obs = step(state, random_policy(state, rng, p_declare=0.04))
        public.update(obs)
        for k in private:
            k.update(obs)
        n += 1
    assert public.is_terminal() and tuple(public.score) == state.score
    return n


def test_true_state_always_consistent_acting_view():
    total = sum(run_checked_game(seed, all_views=False) for seed in range(3000))
    assert total > 3000 * 9


def test_true_state_always_consistent_all_views():
    for seed in range(10_000, 10_300):
        run_checked_game(seed, all_views=True)


def test_knowledge_gets_informative():
    """Sanity: propagation actually pins cards down over a game."""
    rng = np.random.default_rng(5)
    state = new_game(rng)
    public = Knowledge(state.turn)
    most = 0
    while not is_terminal(state):
        state, obs = step(state, random_policy(state, rng, p_declare=0.01))
        public.update(obs)
        most = max(most, sum(k.bit_count() for k in public.known))
    assert most >= 20


# ------------------------------------------------------------- update rules
def test_failed_ask_updates():
    k = Knowledge(0)
    k.update(AskObs(0, 1, C("4S"), False, 1))
    assert not k.possible[0, C("4S")] and not k.possible[1, C("4S")]
    assert k.possible[2:, C("4S")].all()
    assert k.min_in_set[0][0] == 1 and k.min_in_set[1][0] == 0
    assert k.hand_size == [9] * 6 and k.turn == 1 and k.n_actions == 1


def test_successful_ask_updates():
    k = Knowledge(0)
    k.update(AskObs(1, 0, C("4S"), False, 0))  # player 1 holds >= 1 low spade
    assert k.min_in_set[1][0] == 1
    k.update(AskObs(0, 1, C("5S"), True, 0))
    assert k.known_owner(C("5S")) == 0
    assert k.possible[:, C("5S")].tolist() == [True] + [False] * 5
    assert k.min_in_set[0][0] == 2  # one it already had + the new card
    assert k.min_in_set[1][0] == 0  # max(0, 1 - 1)
    assert k.hand_size == [10, 8, 9, 9, 9, 9]
    k.update(AskObs(0, 3, C("6S"), True, 0))
    assert k.min_in_set[0][0] == 3 and k.min_in_set[3][0] == 0
    # the card leaves player 0 again
    k.update(AskObs(0, 3, C("7S"), False, 3))
    k.update(AskObs(3, 0, C("5S"), True, 3))
    assert k.known_owner(C("5S")) == 3 and k.min_in_set[0][0] == 2
    assert k.min_in_set[3][0] == 2 and k.hand_size[0] == 10 and k.hand_size[3] == 9


def test_declaration_removes_set():
    k = Knowledge(0)
    k.update(AskObs(0, 1, C("4S"), False, 1))
    k.update(DeclareObs(1, 0, (1,) * 6, False, (2, 1, 1, 0, 2, 0), 1))
    assert k.declared == 1 and k.score == [1, 0]
    assert not k.possible[:, :6].any() and k.possible[:, 6:].all()
    assert all(k.min_in_set[p][0] == 0 for p in range(6))
    assert k.hand_size == [7, 8, 8, 9, 7, 9]
    k.update(DeclareObs(1, 1, (1,) * 6, True, (0, 6, 0, 0, 0, 0), 1))
    assert k.score == [1, 1] and k.hand_size[1] == 2


# -------------------------------------------------------- propagation rules
def test_rule1_single_possible_holder_is_known():
    k = Knowledge(0)
    c = C("4S")
    for a in range(4):  # 0 asks 1, 1 asks 2, ... each fails
        k.update(AskObs(a, a + 1, c, False, a + 1))
        assert k.known_owner(c) == -1 or a == 3
    assert k.known_owner(c) == 5 and k.known[5] == 1 << c
    assert k.min_in_set[5][0] == 1  # min raised to the known count


def test_rule2_full_hand_known_means_nothing_else():
    k = tiny([1, 1, 1, 1, 1, 1], [0])
    for q in range(1, 6):
        k.poss[q] &= ~bit("2S")
    k.propagate()
    assert k.poss[0] == bit("2S") == k.known[0]
    # cascade: remove 3S from everyone but 1 as well
    for q in range(2, 6):
        k.poss[q] &= ~bit("3S")
    k.propagate()
    assert k.poss[1] == bit("3S")
    assert all(k.poss[q] == SET_MASKS[0] & ~bit("2S") & ~bit("3S") for q in range(2, 6))


def test_rule2_empty_hand_holds_nothing():
    k = Knowledge(0)
    k.update(DeclareObs(0, 0, (0,) * 6, True, (6, 0, 0, 0, 0, 0), 0))
    k.hand_size[0] = 0  # pretend the other three cards left too
    k.propagate()
    assert k.poss[0] == 0


def test_rule3_min_in_set_equals_possible_cards():
    k = Knowledge(0)
    # player 0 cannot hold 2S..5S (failed asks by 0 itself, each time to 1 / 3)
    k.update(AskObs(0, 1, C("2S"), False, 1))
    k.update(AskObs(1, 0, C("3S"), False, 0))
    k.update(AskObs(0, 1, C("4S"), False, 1))
    k.update(AskObs(1, 0, C("5S"), False, 0))
    # now 0 cannot hold 2S,3S,4S,5S and has >= 1 low spade: 6S or 7S
    assert k.min_in_set[0][0] == 1 and k.known[0] == 0
    k.update(AskObs(0, 3, C("6S"), False, 3))  # 0 does not hold 6S either
    assert k.known[0] == bit("7S")
    assert all(not k.possible[q, C("7S")] for q in range(1, 6))


def test_rule3_min_two_of_two():
    k = tiny([3, 3, 3, 3, 0, 0], [0, 1])
    k.min_in_set[2][0] = 2
    k.poss[2] &= ~(bit("2S") | bit("3S") | bit("4S") | bit("5S"))
    k.propagate()
    assert k.known[2] == bit("6S") | bit("7S")
    assert all(not k.poss[q] & (bit("6S") | bit("7S")) for q in (0, 1, 3))


def test_rule4_possible_count_equals_hand_size():
    k = tiny([2, 2, 2, 0, 0, 0], [0])
    k.poss[1] = bit("2S") | bit("3S")
    k.propagate()
    assert k.known[1] == bit("2S") | bit("3S")
    assert k.poss[0] == k.poss[2] == SET_MASKS[0] & ~k.known[1]
    k.poss[0] &= ~(bit("4S") | bit("5S"))
    k.propagate()
    assert k.known[0] == bit("6S") | bit("7S") and k.known[2] == bit("4S") | bit("5S")


def test_rule5_minimums_fill_hand():
    k = tiny([2, 4, 3, 3, 0, 0], [0, 1])
    k.min_in_set[0][0] = 1
    k.min_in_set[0][1] = 1
    for q in (1, 2, 3):
        k.poss[q] &= ~bit("9S")  # 0 is known to hold 9S
    k.propagate()
    assert k.known[0] == bit("9S")
    # exactly one high spade (the known one) and exactly one low spade
    assert k.poss[0] == SET_MASKS[0] | bit("9S")


def test_contradictions_are_detected():
    k = tiny([2, 2, 2, 0, 0, 0], [0])
    k.poss[0] = bit("2S")  # needs 2 cards, only 1 possible
    with pytest.raises(Contradiction):
        k.propagate()
    k = tiny([2, 2, 2, 0, 0, 0], [0])
    for q in range(3):
        k.poss[q] &= ~bit("2S")  # nobody can hold 2S
    with pytest.raises(Contradiction):
        k.propagate()
    k = tiny([2, 2, 2, 0, 0, 0], [0])
    k.min_in_set[0][0] = 3
    with pytest.raises(Contradiction):
        k.propagate()
    k = Knowledge(0)
    with pytest.raises(Contradiction):
        k.set_hand(0, 0b111)  # wrong size


# --------------------------------------------------------------------- views
def test_view_for_adds_own_hand():
    state = new_game(np.random.default_rng(3), first_player=2)
    public = Knowledge(state.turn)
    v = public.view_for(2, state)
    assert v.player == 2 and v.hand == state.hands[2] and v.public is public
    assert v.know.poss[2] == state.hands[2] == v.know.known[2]
    assert all(v.know.poss[q] & state.hands[2] == 0 for q in range(6) if q != 2)
    assert public.poss[2] == public.poss[0] and public.known == [0] * 6  # public untouched
    assert v.know.consistent_with(state) and public.consistent_with(state)


def test_view_endgame_deduction():
    """With one opponent left holding cards, a player's view pins everything."""
    owner = np.array([0] * 2 + [1] * 2 + [2] * 2 + [-1] * 48, dtype=np.int8)
    state = State.from_owner(owner, turn=0, score=(4, 4))
    public = tiny([2, 2, 2, 0, 0, 0], [0])
    public.score = [4, 4]
    v0 = public.view_for(0, state)
    assert v0.know.known[0] == state.hands[0] and v0.know.known[1] == 0
    state2, obs = step(state, Ask(1, C("4S")))
    public.update(obs)
    v0 = public.view_for(0, state2)
    # 0 holds 2S,3S,4S; 1 has one card left, 2 has two, among 5S,6S,7S
    assert v0.know.known[1] == 0
    state3, obs = step(state2, Ask(1, C("6S")))  # fails: 1 does not hold 6S
    public.update(obs)
    v0 = public.view_for(0, state3)
    assert v0.know.known[2] & bit("6S")
    assert v0.know.consistent_with(state3)
    # player 2 holds 6S,7S and knows 4S moved to 0, but 2S/3S/5S stay ambiguous
    # between 0 (two more cards) and 1 (one card): nothing more may be deduced.
    v2 = public.view_for(2, state3)
    assert v2.know.known == [bit("4S"), 0, state3.hands[2], 0, 0, 0]
    rest = bit("2S") | bit("3S") | bit("5S")
    assert v2.know.poss[0] == rest | bit("4S") and v2.know.poss[1] == rest
    # 1 asks 0 for 3S and gets it: 1 = {3S, one of 2S/5S}.
    state4, obs = step(state3, Ask(0, C("3S")))
    assert obs.success
    public.update(obs)
    v2 = public.view_for(2, state4)
    assert v2.know.known[1] == bit("3S") and v2.know.consistent_with(state4)
    # 1 asks 2 for 2S and fails: 1 lacks 2S, so 1 = {3S,5S} and 0 = {2S,4S}.
    state5, obs = step(state4, Ask(2, C("2S")))
    assert not obs.success
    public.update(obs)
    assert public.known[0] == state5.hands[0]  # public: 2S can only be with 0
    assert public.known[1] == bit("3S")  # public cannot place 5S vs 7S
    v2 = public.view_for(2, state5)
    assert v2.know.known == [state5.hands[p] for p in range(6)]


def test_violations_reports_wrong_state():
    state = new_game(np.random.default_rng(0), first_player=0)
    other = new_game(np.random.default_rng(1), first_player=0)
    v = Knowledge(0).view_for(0, state)
    assert v.know.consistent_with(state)
    assert not v.know.consistent_with(other)
