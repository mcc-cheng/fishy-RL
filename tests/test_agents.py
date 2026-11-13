"""Stage 4: agents."""
from __future__ import annotations

import inspect
import time

import numpy as np
import pytest

from fish.agents import (
    AGENT_TYPES,
    GreedyBeliefAgent,
    ISMCTSAgent,
    RandomAgent,
    RolloutAgent,
    RuleBasedAgent,
    certain_declaration,
    fast_policy,
    guess_declaration,
    heuristic_marginals,
    leaf_value,
    legal_asks,
    make_agent,
)
from fish.arena import play_game, play_match_game, summarize
from fish.belief import belief_scores, marginals, sample_deals
from fish.cards import N_CARDS, parse_card
from fish.engine import Ask, Declare, State, is_terminal, legal_actions, new_game, step, validate_action
from fish.knowledge import Knowledge, View
from tests.test_engine import _state, random_policy
from tests.test_knowledge import tiny

C = parse_card
ALL = [
    ("random", {}), ("rule", {}), ("greedy", {"n_particles": 100}),
    ("greedy", {"belief": "heuristic"}),
    ("rollout", {"n_particles": 60, "n_deals": 4, "depth": 6}),
    ("ismcts", {"n_particles": 60, "iterations": 30, "rollout_depth": 5}),
]


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


@pytest.mark.parametrize("spec", ALL, ids=lambda s: s[0] + ("-h" if s[1].get("belief") else ""))
def test_agents_play_full_legal_games(spec):
    """Self-play with engine validation and view-vs-truth consistency checks on."""
    for seed in range(2):
        agents = [make_agent(spec[0], **spec[1]) for _ in range(6)]
        res = play_game(agents, np.random.default_rng(seed), np.random.default_rng(seed + 100),
                        check=True)
        assert not res["truncated"] and sum(res["score"]) == 9


def test_agents_only_receive_views():
    for cls in AGENT_TYPES.values():
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
    for kind, kw in ALL:
        assert make_agent(kind, **kw).act(v, np.random.default_rng(0)) == want
    # a teammate who sees the public transfer does not know the other four cards
    assert certain_declaration(pub.view_for(2, s).know, 2) is None


def test_rule_based_and_greedy_ask_for_known_card():
    s, pub = known_card_scenario()
    v = pub.view_for(0, s)
    want = Ask(3, C("5S"))
    for seed in range(5):
        rng = np.random.default_rng(seed)
        assert RuleBasedAgent().act(v, rng) == want
        assert GreedyBeliefAgent(n_particles=100).act(v, rng) == want
        assert GreedyBeliefAgent(belief="heuristic").act(v, rng) == want
    picks = {RandomAgent().act(v, np.random.default_rng(i)) for i in range(40)}
    assert len(picks) > 5  # the random agent does not focus on it


def pass_or_declare(public_hand: bool):
    """Two live sets.  Player 0 holds 2S-5S, teammates 2 and 4 hold 6S / 7S (0
    cannot tell which), opponent 1 holds all six high spades.  0 can only make a
    hopeless ask (passing the turn to 1, who declares the high spades and leaves
    team B without cards, so teammate 2 must then declare the low spades) or
    declare the low spades now at 50%.

    * ``public_hand=True``: everyone knows 0's four cards, so teammate 2 will
      declare with certainty -> passing is worth a full set more than guessing.
    * ``public_hand=False``: teammate 2 would have to guess among five
      layouts (20%) -> declaring now at 50% is clearly better.
    """
    owner = np.full(N_CARDS, -1, dtype=np.int8)
    owner[:6] = [0, 0, 0, 0, 2, 4]
    owner[6:12] = 1
    s = State.from_owner(owner, turn=0, score=(4, 3))
    pub = tiny([4, 6, 1, 0, 1, 0], [0, 1])
    pub.score = [4, 3]
    low, high = 0b111111, 0b111111 << 6
    pub.poss = [0b001111 if public_hand else low, high, 0b110000 if public_hand else low, 0,
                0b110000 if public_hand else low, 0]
    pub.propagate()
    return s, pub.view_for(0, s)


PLANNERS = [
    lambda: RolloutAgent(n_particles=100, n_deals=40, depth=8),
    lambda: ISMCTSAgent(n_particles=100, iterations=300, rollout_depth=8),
]


@pytest.mark.parametrize("make", PLANNERS, ids=["rollout", "ismcts"])
def test_planners_pass_when_teammate_can_declare_safely(make):
    s, v = pass_or_declare(public_hand=True)
    for seed in range(3):
        agent = make()
        a = agent.act(v, np.random.default_rng(seed))
        assert isinstance(a, Ask), agent.last_values
        validate_action(s, a)
        vals = agent.last_values
        decl = [x for k, x in vals.items() if isinstance(k, Declare)]
        assert decl and vals[a] > max(decl) + 0.5


def test_private_variant_is_a_near_tie_and_guesses_respect_hand_sizes():
    """With 0's hand private, the hopeless ask still tells teammate 2 something
    (0 lacks the asked card), so passing and declaring at 50% are close; the
    planners must return a legal action and never rate either option as a
    sure thing."""
    s, v = pass_or_declare(public_hand=False)
    for make in PLANNERS:
        agent = make()
        a = agent.act(v, np.random.default_rng(0))
        validate_action(s, a)
        assert len(agent.last_values) == 2
        assert all(-2.0 < x < 1.0 for x in agent.last_values.values()), agent.last_values
    # teammate 2's forced guess never gives a player more cards than they hold
    owner = np.full(N_CARDS, -1, dtype=np.int8)
    owner[:6] = [0, 0, 0, 0, 2, 4]
    s2 = State.from_owner(owner, turn=2, score=(4, 4))
    pub = tiny([4, 0, 1, 0, 1, 0], [0])
    g = guess_declaration(pub.view_for(2, s2).know, 2)
    assert g.assignment[4] == 2
    assert sorted(g.assignment[:4] + g.assignment[5:]) == [0, 0, 0, 0, 4]


def ambiguous_endgame():
    """Team A holds all low spades; 0 has four, 6S/7S are split between 2 and 4."""
    s = _state({0: ["2S", "3S", "4S", "5S", "9S"], 2: ["6S", "TS"], 4: ["7S", "JS"],
                1: ["QS", "KS"], 3: ["AS"]}, turn=0, default=-1)
    return s


def _ambiguous_view():
    owner = np.full(N_CARDS, -1, dtype=np.int8)
    for p, names in {0: ["2S", "3S", "4S", "5S", "9S"], 2: ["6S", "TS"], 4: ["7S", "JS"],
                     1: ["QS", "KS"], 3: ["AS"]}.items():
        for n in names:
            owner[C(n)] = p
    s = State.from_owner(owner, turn=0, score=(4, 3))
    pub = tiny([5, 2, 2, 1, 2, 0], [0, 1])
    pub.score = [4, 3]
    for q in (1, 3):  # public: the opponents hold no low spades
        pub.poss[q] &= ~0b111111
    pub.propagate()
    return s, pub.view_for(0, s)


def test_greedy_declare_threshold():
    s, v = _ambiguous_view()
    parts = sample_deals(v.know, 400, np.random.default_rng(0))
    assert all((d[:6] % 2 == 0).all() for d in parts)  # low spades are on team A
    cautious = GreedyBeliefAgent(n_particles=200, declare_threshold=0.95)
    bold = GreedyBeliefAgent(n_particles=200, declare_threshold=0.05)
    for seed in range(5):
        a = cautious.act(v, np.random.default_rng(seed))
        assert isinstance(a, Ask)
        validate_action(s, a)
        b = bold.act(v, np.random.default_rng(seed))
        assert isinstance(b, Declare) and b.set_id == 0 and b.assignment[:4] == (0, 0, 0, 0)
        validate_action(s, b)


def test_forced_declaration_when_opponents_are_out():
    owner = np.full(N_CARDS, -1, dtype=np.int8)
    owner[:6] = [0, 0, 0, 2, 2, 4]
    s = State.from_owner(owner, turn=0, score=(4, 4))
    pub = tiny([3, 0, 2, 0, 1, 0], [0])
    pub.score = [4, 4]
    v = pub.view_for(0, s)
    assert legal_asks(v) == []
    for kind, kw in ALL:
        a = make_agent(kind, **kw).act(v, np.random.default_rng(1))
        assert isinstance(a, Declare) and a.set_id == 0 and a.assignment[:3] == (0, 0, 0)
        assert all(q in (2, 4) for q in a.assignment[3:])
        validate_action(s, a)
    g = guess_declaration(v.know, 0)
    assert g.set_id == 0 and g.assignment[:3] == (0, 0, 0)


def test_hopeless_ask_is_preferred_to_a_guess():
    """Nothing can succeed and no declaration is likely: pass the turn, don't guess."""
    s, v = _ambiguous_view()
    a = fast_policy(v.know, 0, v.hand, 0.3, declare_threshold=0.99)
    assert isinstance(a, Ask)
    validate_action(s, a)


def test_heuristic_marginals_are_distributions_and_sane():
    rng = np.random.default_rng(1)
    state = new_game(rng)
    pub = Knowledge(state.turn)
    h_scores, p_scores, u_scores = [], [], []
    t = 0
    while not is_terminal(state):
        if t % 10 == 0:
            v = pub.view_for(state.turn, state)
            H = heuristic_marginals(v.know)
            live = state.owner >= 0
            assert np.allclose(H[:, live].sum(0), 1.0) and (H[:, ~live] == 0).all()
            for p in range(6):
                for c in range(N_CARDS):
                    if not v.know.poss[p] >> c & 1:
                        assert H[p, c] == 0
            Pp = marginals(sample_deals(v.know, 300, rng))
            unif = v.know.possible / np.maximum(v.know.possible.sum(0), 1)
            h_scores.append(belief_scores(H, state, v.hand)[0])
            p_scores.append(belief_scores(Pp, state, v.hand)[0])
            u_scores.append(belief_scores(unif, state, v.hand)[0])
        state, obs = step(state, random_policy(state, rng, 0.02))
        pub.update(obs)
        t += 1
    # heuristic is no worse than "uniform over possible holders" and close to particles
    assert np.mean(h_scores) <= np.mean(u_scores) + 1e-9
    assert abs(np.mean(h_scores) - np.mean(p_scores)) < 0.05


def test_leaf_value():
    s = _state({0: ["2S", "3S", "4S", "5S", "6S", "7S"]}, turn=0, default=1)
    assert leaf_value(s, 0.0) == 0.0
    # team A holds 6 of set 0 and 0 of the other eight
    assert np.isclose(leaf_value(s, 0.6), 0.6 * (6 - 27) / 3)  # (cards - 3 per live set) / 3
    s2, _ = step(s, Declare(0, (0,) * 6))
    assert np.isclose(leaf_value(s2, 0.6), 1 + 0.6 * (0 - 24) / 3)


def test_agents_are_seeded():
    for spec in ALL[2:]:
        rows = [play_match_game(spec, ("rule", {}), g, seed=3) for g in (0, 0, 1)]
        keys = ("sets_x", "sets_y", "length", "decl_x", "decl_correct_x")
        assert all(rows[0][k] == rows[1][k] for k in keys)
        assert rows[0]["deal"] == rows[2]["deal"] and rows[0]["x_team"] != rows[2]["x_team"]


@pytest.mark.parametrize("kind", ["greedy", "rollout", "ismcts"])
def test_default_decision_time_under_one_second(kind):
    agents = [make_agent(kind) if p % 2 == 0 else make_agent("greedy") for p in range(6)]
    res = play_game(agents, np.random.default_rng(0), np.random.default_rng(1))
    worst = max(max(res["times"][p], default=0.0) for p in (0, 2, 4))
    assert worst < 1.0, f"{kind}: slowest decision took {worst:.2f}s"


def test_rule_based_beats_random():
    rows = [play_match_game(("rule", {}), ("random", {}), g, seed=11) for g in range(40)]
    s = summarize(rows)
    assert s["mean_diff"] - s["ci95"] > 0, s


def test_greedy_belief_beats_rule_based():
    rows = [play_match_game(("greedy", {"n_particles": 150}), ("rule", {}), g, seed=12)
            for g in range(60)]
    s = summarize(rows)
    assert s["mean_diff"] > 0, s
