"""Stage 3: particle belief tests."""
from __future__ import annotations

import time

import numpy as np
import pytest

from fish.belief import (
    ask_likelihood_weights,
    belief_scores,
    best_declaration,
    enumerate_deals,
    marginals,
    sample_deals,
    satisfies,
)
from fish.cards import ALL_SETS_MASK, N_CARDS, N_PLAYERS, parse_card
from fish.engine import State, is_terminal, new_game, step
from fish.knowledge import Knowledge
from tests.test_engine import random_policy
from tests.test_knowledge import tiny

C = parse_card


def deal_key(deal: np.ndarray) -> bytes:
    return deal.tobytes()


def compare_with_exact(k: Knowledge, n: int, seed: int, tol: float = 0.03) -> int:
    """Sample n deals and compare with brute-force enumeration; returns #deals."""
    exact = list(enumerate_deals(k))
    D = len(exact)
    assert D >= 1
    index = {deal_key(d): i for i, d in enumerate(exact)}
    parts = sample_deals(k, n, np.random.default_rng(seed))
    assert parts.shape == (n, N_CARDS)
    counts = np.zeros(D)
    for d in parts:
        key = deal_key(d)
        assert key in index, "sample is not a consistent deal"
        counts[index[key]] += 1
    exact_marg = marginals(np.array(exact))
    assert np.abs(marginals(parts) - exact_marg).max() < tol
    if D > 1:
        expected = n / D
        assert (counts > 0).all() or expected < 8, "some consistent deal is never sampled"
        chi2 = ((counts - expected) ** 2 / expected).sum()
        df = D - 1
        # generous bound (~6 sigma): catches non-uniformity, not noise
        assert chi2 < df + 6 * np.sqrt(2 * df) + 6, (D, chi2)
    return D


def endgame_positions(max_live_sets: int, lo: int, hi: int, want: int, seed0: int = 0):
    """Views from random games late enough that exact enumeration is feasible."""
    out = []
    seed = seed0
    while len(out) < want:
        rng = np.random.default_rng(seed)
        seed += 1
        state = new_game(rng)
        pub = Knowledge(state.turn)
        taken = False
        while not is_terminal(state):
            live = 9 - bin(state.declared).count("1")
            if live <= max_live_sets and not taken:
                for p in range(N_PLAYERS):
                    know = pub if p == 0 and live <= 1 else pub.view_for(p, state).know
                    unknown = sum(
                        1 for c in range(N_CARDS) if state.owner[c] >= 0 and know.known_owner(c) < 0
                    )
                    if 4 <= unknown <= 13:
                        try:
                            d = sum(1 for _ in enumerate_deals(know, limit=hi))
                        except Exception:
                            continue
                        if lo <= d <= hi:
                            out.append((know.copy(), state))
                            taken = True
                            break
            state, obs = step(state, random_policy(state, rng, 0.02))
            pub.update(obs)
    return out


def test_samples_satisfy_constraints_throughout_games():
    rng = np.random.default_rng(42)
    checked = 0
    for seed in range(40):
        g = np.random.default_rng(seed)
        state = new_game(g)
        pub = Knowledge(state.turn)
        t = 0
        while not is_terminal(state):
            if t % 7 == 0:
                for know in (pub, pub.view_for(state.turn, state).know):
                    parts = sample_deals(know, 24, rng)
                    assert parts.dtype == np.int8
                    for d in parts:
                        assert satisfies(know, d)
                        checked += 1
                    assert satisfies(know, state.owner)
            state, obs = step(state, random_policy(state, g, 0.03))
            pub.update(obs)
            t += 1
    assert checked > 5000


def test_matches_brute_force_on_endgames():
    positions = endgame_positions(max_live_sets=3, lo=3, hi=1500, want=12)
    sizes = []
    for i, (know, state) in enumerate(positions):
        assert satisfies(know, state.owner)
        sizes.append(compare_with_exact(know, 30_000, seed=i))
    assert max(sizes) >= 50  # at least one non-trivial position


def test_three_cycle_position_is_mixed():
    """Three cards / three players in a ring: the two deals differ by a 3-cycle only."""
    k = tiny([1, 1, 1, 1, 1, 1], [0])
    a, b, c = (1 << C("2S")), (1 << C("3S")), (1 << C("4S"))
    k.poss = [a | b, b | c, c | a, 1 << C("5S"), 1 << C("6S"), 1 << C("7S")]
    k.propagate()
    assert len(list(enumerate_deals(k))) == 2
    compare_with_exact(k, 4000, seed=1)


def test_min_in_set_constraints_are_respected_and_uniform():
    k = tiny([3, 3, 3, 3, 0, 0], [0, 1])
    k.min_in_set[0][0] = 2
    k.min_in_set[1][1] = 3
    k.min_in_set[2][0] = 1
    k.poss[3] &= ~(1 << C("2S"))
    k.propagate()
    D = compare_with_exact(k, 40_000, seed=2)
    assert D > 20
    for d in sample_deals(k, 200, np.random.default_rng(0)):
        assert (d[:6] == 0).sum() >= 2 and (d[6:12] == 1).sum() == 3 and (d[:6] == 2).sum() >= 1


def test_uniform_without_constraints_small():
    k = tiny([2, 1, 1, 1, 1, 0], [4])
    assert compare_with_exact(k, 40_000, seed=3) == 360  # 6!/2!


def test_fully_known_position():
    k = tiny([6, 0, 0, 0, 0, 0], [2])
    k.propagate()
    parts = sample_deals(k, 5, np.random.default_rng(0))
    assert (parts[:, 12:18] == 0).all() and (np.delete(parts, np.s_[12:18], axis=1) == -1).all()


def test_sampler_is_seeded():
    state = new_game(np.random.default_rng(0))
    k = Knowledge(state.turn).view_for(0, state).know
    a = sample_deals(k, 50, np.random.default_rng(9))
    b = sample_deals(k, 50, np.random.default_rng(9))
    c = sample_deals(k, 50, np.random.default_rng(10))
    assert (a == b).all() and not (a == c).all()


def test_initial_marginals_are_uniform():
    state = new_game(np.random.default_rng(0))
    v = Knowledge(state.turn).view_for(0, state)
    P = marginals(sample_deals(v.know, 4000, np.random.default_rng(1)))
    assert np.allclose(P.sum(0), 1.0)
    own = np.array([bool(v.hand >> c & 1) for c in range(N_CARDS)])
    assert (P[0, own] == 1).all() and (P[0, ~own] == 0).all()
    assert np.abs(P[1:, ~own] - 0.2).max() < 0.03
    assert np.allclose(P.sum(1), 9.0)


def test_marginals_and_weights():
    parts = np.array([[0, 1, 2] + [-1] * 51, [1, 1, 2] + [-1] * 51], dtype=np.int8)
    P = marginals(parts)
    assert P[0, 0] == 0.5 and P[1, 0] == 0.5 and P[1, 1] == 1.0 and P[2, 2] == 1.0
    assert P[:, 3:].sum() == 0
    Pw = marginals(parts, np.array([3.0, 1.0]))
    assert np.isclose(Pw[0, 0], 0.75) and np.isclose(Pw[1, 0], 0.25)


def test_best_declaration():
    base = np.full(N_CARDS, 1, dtype=np.int8)
    rows = []
    for sigma in ([0, 0, 2, 2, 4, 4],) * 5 + ([0, 0, 2, 2, 4, 0],) * 3 + ([0, 0, 2, 2, 4, 1],) * 2:
        r = base.copy()
        r[:6] = sigma
        rows.append(r)
    parts = np.array(rows)
    sigma, p = best_declaration(parts, 0, 0)
    assert sigma == (0, 0, 2, 2, 4, 4) and np.isclose(p, 0.5)
    assert best_declaration(parts, 1, 0) == (None, 0.0)
    sigma, p = best_declaration(parts, 1, 3)
    assert sigma == (1,) * 6 and p == 1.0
    w = np.array([1.0] * 5 + [4.0] * 3 + [1.0] * 2)
    sigma, p = best_declaration(parts, 0, 0, w)
    assert sigma == (0, 0, 2, 2, 4, 0) and np.isclose(p, 12 / 19)
    parts[:, 6:12] = -1
    assert best_declaration(parts, 1, 1) == (None, 0.0)


def test_belief_scores():
    owner = np.array([0, 1] + [-1] * 4 + [-1] * 48, dtype=np.int8)
    owner[:6] = [0, 1, 2, 3, 4, 5]
    state = State.from_owner(owner, 0)
    perfect = (owner[None, :] == np.arange(6)[:, None]).astype(float)
    assert belief_scores(perfect, state) == (0.0, 0.0, 6)
    uniform = np.zeros((6, N_CARDS))
    uniform[:, :6] = 1 / 6
    b, ll, n = belief_scores(uniform, state, viewer_hand=1)
    assert n == 5 and np.isclose(b, 5 / 36 + 25 / 36) and np.isclose(ll, np.log(6))


def test_likelihood_weights_shape_and_neutral_alpha():
    rng = np.random.default_rng(0)
    state = new_game(rng)
    pub = Knowledge(state.turn)
    for _ in range(30):
        state, obs = step(state, random_policy(state, rng, 0.0))
        pub.update(obs)
    k = pub.view_for(state.turn, state).know
    parts = sample_deals(k, 300, rng)
    w0 = ask_likelihood_weights(parts, k, alpha=0.0)
    assert np.allclose(w0, 1 / 300)
    w = ask_likelihood_weights(parts, k, alpha=1.0)
    assert w.shape == (300,) and np.isclose(w.sum(), 1.0) and (w > 0).all()
    # reconstructed "count when asking" is >= 1 in the true deal for every logged ask
    cnt = np.array([[sum(1 for c in range(6 * h, 6 * h + 6) if state.owner[c] == p)
                     for h in range(9)] for p in range(6)])
    assert all(cnt[a, h] + off >= 1 for a, h, off in k.ask_log)


def test_samples_per_second(capsys):
    state = new_game(np.random.default_rng(0))
    k = Knowledge(state.turn).view_for(0, state).know
    rng = np.random.default_rng(0)
    sample_deals(k, 200, rng)
    t0 = time.perf_counter()
    reps = 20
    for _ in range(reps):
        sample_deals(k, 200, rng)
    rate = reps * 200 / (time.perf_counter() - t0)
    with capsys.disabled():
        print(f"\n[belief] opening position (45 unknown cards): {rate:,.0f} samples/sec")
    assert rate > 2000
