"""Stage 1: engine tests."""
from __future__ import annotations

import numpy as np
import pytest

from fish.cards import N_CARDS, N_PLAYERS, N_SETS, SET_MASKS, card_name, parse_card, set_of
from fish.engine import (
    Ask,
    AskObs,
    Declare,
    DeclareObs,
    IllegalAction,
    State,
    declarable_sets,
    is_terminal,
    legal_actions,
    new_game,
    resolve_turn,
    returns,
    step,
)


def check_invariants(state: State) -> None:
    """Card conservation and owner/hands/declared consistency."""
    owner = state.owner
    seen = 0
    for p in range(N_PLAYERS):
        assert seen & state.hands[p] == 0, "card held twice"
        seen |= state.hands[p]
        assert state.hands[p].bit_count() == int((owner == p).sum())
        for c in range(N_CARDS):
            assert bool(state.hands[p] >> c & 1) == (owner[c] == p)
    for h in range(N_SETS):
        gone = bool(state.declared >> h & 1)
        assert (seen & SET_MASKS[h] == 0) == gone or not gone
        assert all((owner[c] == -1) == gone for c in range(6 * h, 6 * h + 6))
    n_declared = bin(state.declared).count("1")
    assert seen.bit_count() == N_CARDS - 6 * n_declared
    assert sum(state.score) == n_declared
    if not is_terminal(state):
        assert state.hands[state.turn] != 0, "turn player must hold cards"


def random_policy(state: State, rng: np.random.Generator, p_declare: float = 0.1):
    """Engine-level random player used only for soak tests (peeks at the state)."""
    p = state.turn
    asks = legal_actions(state, p)
    if asks and rng.random() > p_declare:
        return asks[int(rng.integers(len(asks)))]
    sets = declarable_sets(state, p)
    h = sets[int(rng.integers(len(sets)))]
    mates = [q for q in range(N_PLAYERS) if q % 2 == p % 2]
    sigma = tuple(
        int(state.owner[c]) if state.owner[c] % 2 == p % 2 else mates[int(rng.integers(3))]
        for c in range(6 * h, 6 * h + 6)
    )
    return Declare(h, sigma)


def play_random(seed: int, full_checks: bool) -> tuple[State, int]:
    rng = np.random.default_rng(seed)
    state = new_game(rng)
    if full_checks:
        check_invariants(state)
    n = 0
    while not is_terminal(state):
        sizes = state.hand_sizes()
        p = state.turn
        action = random_policy(state, rng)
        state, obs = step(state, action)
        n += 1
        new_sizes = state.hand_sizes()
        if isinstance(obs, AskObs):
            assert obs.asker == p
            exp = list(sizes)
            if obs.success:
                exp[p] += 1
                exp[obs.target] -= 1
                assert state.owner[obs.card] == p and state.turn == p
            else:
                assert state.turn == obs.target
            assert tuple(exp) == new_sizes
        else:
            assert isinstance(obs, DeclareObs) and obs.declarer == p
            assert tuple(s - r for s, r in zip(sizes, obs.removed)) == new_sizes
            assert sum(obs.removed) == 6
        assert obs.next_turn == state.turn
        if full_checks:
            check_invariants(state)
        assert n < 100_000
    return state, n


def test_card_naming_roundtrip():
    assert len({card_name(c) for c in range(N_CARDS)}) == N_CARDS
    for c in range(N_CARDS):
        assert parse_card(card_name(c)) == c
    assert [set_of(parse_card(n)) for n in ("2S", "7S", "9S", "AS", "8S", "8C", "JK1", "JK2")] == [
        0, 0, 1, 1, 8, 8, 8, 8]
    assert set_of(parse_card("2C")) == 6 and set_of(parse_card("AC")) == 7


def test_deal():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = new_game(rng)
        assert s.hand_sizes() == (9,) * 6
        assert s.score == (0, 0) and s.declared == 0
        check_invariants(s)
    assert new_game(np.random.default_rng(1), first_player=3).turn == 3
    a, b = new_game(np.random.default_rng(7)), new_game(np.random.default_rng(7))
    assert (a.owner == b.owner).all() and a.turn == b.turn


def test_10000_random_games_complete():
    total = 0
    for seed in range(10_000):
        state, n = play_random(seed, full_checks=seed % 20 == 0)
        assert is_terminal(state)
        assert sum(state.score) == 9
        assert (state.owner == -1).all() and all(h == 0 for h in state.hands)
        assert returns(state)[0] == -returns(state)[1] == state.score[0] - state.score[1]
        assert legal_actions(state, state.turn) == []
        total += n
    assert total > 10_000 * 9


def _state(hands: dict[int, list[str]], turn: int, default: int | None = None) -> State:
    """State where the listed players hold the named cards; other cards go to
    ``default`` (or are spread so that nobody else is empty when None)."""
    owner = np.full(N_CARDS, -2, dtype=np.int8)
    for p, names in hands.items():
        for n in names:
            owner[parse_card(n)] = p
    rest = [c for c in range(N_CARDS) if owner[c] == -2]
    others = [q for q in range(N_PLAYERS) if q not in hands]
    for i, c in enumerate(rest):
        owner[c] = default if default is not None else others[i % len(others)]
    return State.from_owner(owner, turn)


def test_ask_success_and_failure():
    s = _state({0: ["2S", "3S"], 1: ["4S"]}, turn=0, default=3)
    s2, obs = step(s, Ask(1, parse_card("4S")))
    assert obs == AskObs(0, 1, parse_card("4S"), True, 0)
    assert s2.turn == 0 and s2.owner[parse_card("4S")] == 0
    assert s2.hand_size(0) == 3 and s2.hand_size(1) == 0
    # original state untouched (immutability)
    assert s.owner[parse_card("4S")] == 1 and s.hand_size(0) == 2
    s3, obs = step(s2, Ask(3, parse_card("2S") + 3))  # 5S is with player 3
    assert obs.success and s3.turn == 0
    s = _state({0: ["2S"], 1: ["4S"], 3: ["5S"]}, turn=0, default=5)
    s4, obs = step(s, Ask(1, parse_card("5S")))
    assert obs == AskObs(0, 1, parse_card("5S"), False, 1)
    assert s4.turn == 1 and (s4.owner == s.owner).all()


def test_illegal_actions_raise():
    s = _state({0: ["2S", "3S"], 1: ["4S"], 2: ["5S"]}, turn=0, default=3)
    with pytest.raises(IllegalAction):
        step(s, Ask(2, parse_card("5S")))  # teammate
    with pytest.raises(IllegalAction):
        step(s, Ask(0, parse_card("5S")))  # self
    with pytest.raises(IllegalAction):
        step(s, Ask(1, parse_card("2S")))  # holds the card
    with pytest.raises(IllegalAction):
        step(s, Ask(1, parse_card("9S")))  # no card in that set
    with pytest.raises(IllegalAction):
        step(s, Ask(5, parse_card("4S")))  # target has no cards
    with pytest.raises(IllegalAction):
        step(s, Ask(1, 54))
    with pytest.raises(IllegalAction):
        step(s, Ask(7, 3))
    with pytest.raises(IllegalAction):
        step(s, Declare(0, (0, 0, 1, 0, 0, 0)))  # opponent in assignment
    with pytest.raises(IllegalAction):
        step(s, Declare(0, (0, 0, 0)))  # wrong length
    with pytest.raises(IllegalAction):
        step(s, Declare(9, (0,) * 6))
    with pytest.raises(IllegalAction):
        step(s, "pass")  # type: ignore[arg-type]
    s2, _ = step(s, Declare(0, (0, 0, 0, 0, 0, 0)))
    with pytest.raises(IllegalAction):
        step(s2, Declare(0, (1,) * 6) if s2.turn % 2 else Declare(0, (0,) * 6))  # already declared
    # asking for a card of a declared set is illegal
    with pytest.raises(IllegalAction):
        step(s2, Ask(1 if s2.turn % 2 == 0 else 0, parse_card("4S")))
    # legal_actions is empty for players who are not on turn
    assert legal_actions(s, 1) == [] and declarable_sets(s, 1) == []
    for a in legal_actions(s, 0):
        step(s, a)


def test_legal_actions_enumeration():
    s = _state({0: ["2S", "3S", "9H"], 1: ["4S"]}, turn=0, default=3)
    acts = legal_actions(s, 0)
    # targets with cards: 1 and 3 (5 is empty); 4 missing low-S + 5 missing high-H
    assert len(acts) == (4 + 5) * 2 and len(set(acts)) == len(acts)
    assert all(isinstance(a, Ask) and a.target in (1, 3) for a in acts)
    assert {set_of(a.card) for a in acts} == {0, 3}
    assert declarable_sets(s, 0) == list(range(9))


def test_declare_correct_and_incorrect():
    low_s = ["2S", "3S", "4S", "5S", "6S", "7S"]
    s = _state({0: low_s[:2], 2: low_s[2:4], 4: low_s[4:]}, turn=0, default=1)
    good, obs = step(s, Declare(0, (0, 0, 2, 2, 4, 4)))
    assert obs.correct and good.score == (1, 0) and good.declared == 1
    assert obs.removed == (2, 0, 2, 0, 2, 0)
    assert (good.owner[:6] == -1).all()
    # one card misplaced within the team -> opponents win the set
    bad, obs = step(s, Declare(0, (0, 0, 2, 4, 4, 4)))
    assert not obs.correct and bad.score == (0, 1) and (bad.owner[:6] == -1).all()
    # an opponent holds one of the cards -> always wrong
    s = _state({0: low_s[:5], 1: low_s[5:]}, turn=0, default=3)
    bad, obs = step(s, Declare(0, (0,) * 6))
    assert not obs.correct and bad.score == (0, 1) and obs.removed == (5, 1, 0, 0, 0, 0)
    # declaring a set you hold nothing of is allowed
    s = _state({2: low_s}, turn=0)
    ok, obs = step(s, Declare(0, (2,) * 6))
    assert obs.correct and ok.score == (1, 0)


def test_turn_passes_to_next_teammate_clockwise_when_out_of_cards():
    low_s = ["2S", "3S", "4S", "5S", "6S", "7S"]
    # player 0 holds only the low spades and declares them: turn -> 2
    s = _state({0: low_s}, turn=0)
    s2, obs = step(s, Declare(0, (0,) * 6))
    assert s2.hand_size(0) == 0 and s2.turn == 2 == obs.next_turn
    # player 4 runs out: clockwise wraps to 0
    s = _state({4: low_s}, turn=4)
    assert step(s, Declare(0, (4,) * 6))[0].turn == 0
    # player 2 runs out and 4 is empty too: wraps to 0
    s = _state({2: low_s, 0: ["9S"]}, turn=2, default=1)
    assert step(s, Declare(0, (2,) * 6))[0].turn == 0
    assert resolve_turn((0, 1, 0, 1, 1, 1), 0) == 4
    assert resolve_turn((1, 1, 0, 1, 0, 1), 2) == 0
    assert resolve_turn((5, 1, 0, 1, 0, 1), 0) == 0


def test_whole_team_out_forces_other_team_to_declare():
    low_s = ["2S", "3S", "4S", "5S", "6S", "7S"]
    s = _state({1: low_s}, turn=1, default=2)  # team B only holds low spades
    s2, obs = step(s, Declare(0, (1,) * 6))
    assert obs.correct and s2.turn == 2  # next clockwise opponent with cards
    assert legal_actions(s2, 2) == []  # nobody to ask
    assert len(declarable_sets(s2, 2)) == 8
    n = 0
    while not is_terminal(s2):
        h = declarable_sets(s2, s2.turn)[0]
        s2, obs = step(s2, Declare(h, (2,) * 6))
        assert obs.correct
        n += 1
    assert n == 8 and s2.score == (8, 1)
    # same but the forced team's cards are split: turn moves on as players empty
    owner = np.array([0] * 6 + [2] * 6 + [4] * 6 + [1] * 6 + [-1] * 30, dtype=np.int8)
    s = State.from_owner(owner, turn=1, score=(3, 2))
    s, _ = step(s, Declare(3, (1,) * 6))
    assert s.turn == 2
    s, _ = step(s, Declare(1, (2,) * 6))
    assert s.turn == 4
    s, _ = step(s, Declare(0, (0,) * 6))  # 4 may declare a set held by 0
    assert s.turn == 4
    s, _ = step(s, Declare(2, (4,) * 6))
    assert is_terminal(s) and s.score == (6, 3)


def test_ask_emptying_target_then_failed_ask_never_targets_empty_player():
    s = _state({0: ["2S", "3S"], 1: ["4S"]}, turn=0, default=3)
    s, _ = step(s, Ask(1, parse_card("4S")))
    assert s.hand_size(1) == 0
    assert all(a.target != 1 for a in legal_actions(s, 0))
    with pytest.raises(IllegalAction):
        step(s, Ask(1, parse_card("5S")))


def test_from_owner_validation():
    owner = np.zeros(N_CARDS, dtype=np.int8)
    owner[0] = -1  # partially declared set
    with pytest.raises(ValueError):
        State.from_owner(owner, 0)
    with pytest.raises(ValueError):
        State.from_owner(np.zeros(10), 0)
