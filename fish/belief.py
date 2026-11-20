"""Particle belief over deals consistent with a :class:`Knowledge` store.

A particle is a full owner array (``int8[54]``, -1 for declared cards).
:func:`sample_deals` draws particles with a vectorised constructive sampler
followed by symmetric MCMC moves (pair swaps and 3-cycles), whose stationary
distribution is uniform over the consistent deals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .cards import N_CARDS, N_PLAYERS, N_SETS, SET_MASKS, SET_SIZE, TEAM_MEMBERS
from .engine import State
from .knowledge import Knowledge, masks_to_matrix


class SamplingError(RuntimeError):
    """No consistent deal could be constructed."""


@dataclass(slots=True)
class _Problem:
    """Array form of the sampling problem for the still-unknown cards."""

    base: np.ndarray       # int8[54]: known owner, -1 otherwise
    cards: np.ndarray      # int[U]: unknown card ids
    set_u: np.ndarray      # int[U]: set of each unknown card
    poss: np.ndarray       # bool[6, U]
    cap: np.ndarray        # int[6]: free hand slots
    need: np.ndarray       # int[6, 9]: unknown cards still required per set
    known_cnt: np.ndarray  # int[6, 9]
    min_req: np.ndarray    # int[6, 9]: min_in_set (>= known_cnt)


def _problem(k: Knowledge) -> _Problem:
    possible = masks_to_matrix(k.poss)
    known = masks_to_matrix(k.known)
    base = np.full(N_CARDS, -1, dtype=np.int8)
    pl, cd = np.nonzero(known)
    base[cd] = pl
    live = np.array([not k.declared >> (c // SET_SIZE) & 1 for c in range(N_CARDS)])
    cards = np.flatnonzero(live & (base < 0))
    known_cnt = known.reshape(N_PLAYERS, N_SETS, SET_SIZE).sum(2)
    min_req = np.maximum(np.array(k.min_in_set, dtype=np.int64), known_cnt)
    cap = np.array(k.hand_size, dtype=np.int64) - known_cnt.sum(1)
    return _Problem(base, cards, cards // SET_SIZE, possible[:, cards], cap,
                    min_req - known_cnt, known_cnt, min_req)


def _order(pb: _Problem, rng: np.random.Generator) -> np.ndarray:
    """Most-constrained cards first (fewest possible holders), random ties."""
    key = pb.poss.sum(0) + rng.random(pb.cards.size)
    return np.argsort(key)


def _construct(
    pb: _Problem, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One vectorised constructive pass.

    Returns ``(owners[n, U], failed[n], logq[n])`` where ``logq`` is the log
    probability with which the pass generated each assignment.  Options are
    only excluded when they provably cannot be completed, so every consistent
    deal has positive probability and ``1/q`` is a valid importance weight for
    the uniform distribution over consistent deals.
    """
    U = pb.cards.size
    own = np.zeros((n, U), dtype=np.int8)
    rem = np.tile(pb.cap, (n, 1))
    need = np.tile(pb.need, (n, 1, 1))
    needsum = need.sum(2)
    avail = np.zeros((N_PLAYERS, N_SETS), dtype=np.int64)
    np.add.at(avail, (slice(None), pb.set_u), pb.poss)
    avail_tot = pb.poss.sum(1)
    left_in_set = np.bincount(pb.set_u, minlength=N_SETS)
    fail = np.zeros(n, dtype=bool)
    logq = np.zeros(n)
    ar = np.arange(n)
    r = rng.random((U, n))
    for t, ci in enumerate(_order(pb, rng)):
        h = pb.set_u[ci]
        pc = pb.poss[:, ci]
        nd = need[:, :, h]
        ok = pc & (rem > 0) & ((nd > 0) | (rem > needsum))
        # forced: the player needs every remaining candidate card (in the set / overall)
        forced = ok & ((nd >= avail[:, h]) | (rem >= avail_tot))
        # joint version: the set's outstanding needs use up all its remaining cards
        forced |= ok & (nd > 0) & (nd.sum(1) >= left_in_set[h])[:, None]
        w = rem * np.where(forced.any(1)[:, None], forced, ok)
        cum = np.cumsum(w, axis=1)
        tot = cum[:, -1]
        fail |= tot <= 0
        p = (cum > (r[t] * tot)[:, None]).argmax(1)
        own[:, ci] = p
        logq += np.log(np.maximum(w[ar, p], 1)) - np.log(np.maximum(tot, 1))
        rem[ar, p] -= 1
        had = need[ar, p, h] > 0
        need[ar, p, h] -= had
        needsum[ar, p] -= had
        avail[pc, h] -= 1
        avail_tot = avail_tot - pc
        left_in_set[h] -= 1
    fail |= (needsum > 0).any(1) | (rem != 0).any(1)
    return own, fail, logq


def _solve_one(pb: _Problem, rng: np.random.Generator, node_limit: int = 200_000) -> np.ndarray:
    """Backtracking search for a single consistent assignment (fallback)."""
    U = pb.cards.size
    order = _order(pb, rng).tolist()
    set_u = pb.set_u.tolist()
    poss = [[p for p in range(N_PLAYERS) if pb.poss[p, ci]] for ci in range(U)]
    rem = pb.cap.tolist()
    need = pb.need.tolist()
    needsum = [sum(row) for row in need]
    avail = [[0] * N_SETS for _ in range(N_PLAYERS)]
    for ci in range(U):
        for p in poss[ci]:
            avail[p][set_u[ci]] += 1
    avail_tot = [sum(row) for row in avail]
    left = np.bincount(pb.set_u, minlength=N_SETS).tolist()
    out = [0] * U
    nodes = 0

    def rec(t: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            raise SamplingError("backtracking node limit exceeded")
        if t == U:
            return True
        ci = order[t]
        h = set_u[ci]
        holders = poss[ci]
        cands = [p for p in holders
                 if rem[p] > 0 and (need[p][h] > 0 or rem[p] > needsum[p])]
        rng.shuffle(cands)
        cands.sort(key=lambda p: -((need[p][h] >= avail[p][h] and need[p][h] > 0)
                                   or rem[p] >= avail_tot[p]))
        for q in holders:
            avail[q][h] -= 1
            avail_tot[q] -= 1
        left[h] -= 1
        set_need = sum(need[q][h] for q in range(N_PLAYERS))
        for p in cands:
            if any(q != p and (need[q][h] > avail[q][h] or rem[q] > avail_tot[q])
                   for q in holders):
                continue
            if set_need - (need[p][h] > 0) > left[h]:
                continue
            had = need[p][h] > 0
            rem[p] -= 1
            need[p][h] -= had
            needsum[p] -= had
            out[ci] = p
            if rec(t + 1):
                return True
            rem[p] += 1
            need[p][h] += had
            needsum[p] += had
        for q in holders:
            avail[q][h] += 1
            avail_tot[q] += 1
        left[h] += 1
        return False

    if not rec(0) or any(rem) or any(needsum):
        raise SamplingError("no consistent deal exists")
    return np.array(out, dtype=np.int8)


def _counts(pb: _Problem, own: np.ndarray) -> np.ndarray:
    """cnt[n, p, h]: cards of set h held by p in each particle (known included)."""
    n = own.shape[0]
    cnt = np.tile(pb.known_cnt, (n, 1, 1))
    ar = np.arange(n)
    for ci in range(pb.cards.size):
        cnt[ar, own[:, ci], pb.set_u[ci]] += 1
    return cnt


def _mcmc(pb: _Problem, own: np.ndarray, sweeps: int, rng: np.random.Generator) -> None:
    """In-place MCMC: each sweep proposes one move per particle.

    Proposals (uniform ordered card pairs / triples) are symmetric, and a move
    is accepted iff the result is consistent, so the chain targets the uniform
    distribution over consistent deals.  Every fourth sweep uses 3-cycles,
    which connect deals that pair swaps alone cannot reach.
    """
    n, U = own.shape
    if U < 2 or sweeps <= 0:
        return
    cnt = _counts(pb, own)
    ar = np.arange(n)
    poss, set_u, min_req = pb.poss, pb.set_u, pb.min_req
    picks = rng.integers(0, U, size=(sweeps, 3, n))
    for s in range(sweeps):
        i, j = picks[s, 0], picks[s, 1]
        p1, p2 = own[ar, i], own[ar, j]
        hi, hj = set_u[i], set_u[j]
        if s % 4 != 3:
            ok = (p1 != p2) & poss[p2, i] & poss[p1, j]
            diff = hi != hj
            ok &= ~diff | ((cnt[ar, p1, hi] > min_req[p1, hi]) & (cnt[ar, p2, hj] > min_req[p2, hj]))
            a = np.flatnonzero(ok)
            if a.size:
                own[a, i[a]] = p2[a]
                own[a, j[a]] = p1[a]
                d = a[diff[a]]
                if d.size:
                    q1, q2, g1, g2 = p1[d], p2[d], hi[d], hj[d]
                    cnt[d, q1, g1] -= 1
                    cnt[d, q2, g1] += 1
                    cnt[d, q1, g2] += 1
                    cnt[d, q2, g2] -= 1
        else:
            # 3-cycle: card i -> p2, card j -> p3, card k -> p1.
            k = picks[s, 2]
            p3 = own[ar, k]
            hk = set_u[k]
            ok = (p1 != p2) & (p2 != p3) & (p1 != p3)
            ok &= poss[p2, i] & poss[p3, j] & poss[p1, k]
            # net change of the three entries that lose a card
            ok &= (cnt[ar, p1, hi] - 1 + (hk == hi)) >= min_req[p1, hi]
            ok &= (cnt[ar, p2, hj] - 1 + (hi == hj)) >= min_req[p2, hj]
            ok &= (cnt[ar, p3, hk] - 1 + (hj == hk)) >= min_req[p3, hk]
            a = np.flatnonzero(ok)
            if a.size:
                q1, q2, q3 = p1[a], p2[a], p3[a]
                own[a, i[a]] = q2
                own[a, j[a]] = q3
                own[a, k[a]] = q1
                np.subtract.at(cnt, (a, q1, hi[a]), 1)
                np.add.at(cnt, (a, q2, hi[a]), 1)
                np.subtract.at(cnt, (a, q2, hj[a]), 1)
                np.add.at(cnt, (a, q3, hj[a]), 1)
                np.subtract.at(cnt, (a, q3, hk[a]), 1)
                np.add.at(cnt, (a, q1, hk[a]), 1)


def default_sweeps(n_unknown: int) -> int:
    """Default number of MCMC sweeps for a problem with ``n_unknown`` free cards."""
    return 0 if n_unknown < 2 else 4 * n_unknown + 24


def sample_deals(
    knowledge: Knowledge,
    n: int,
    rng: np.random.Generator,
    sweeps: int | None = None,
    max_restarts: int = 30,
) -> np.ndarray:
    """Sample ``n`` deals (``int8[n, 54]`` owner arrays) satisfying ``knowledge``.

    1. Constructive passes (most-constrained card first, capacity-weighted,
       dead ends restart) until ``n`` consistent assignments exist.
    2. Importance resampling with weights ``1/q`` to correct the constructive
       sampler's bias towards the uniform distribution.
    3. ``sweeps`` MCMC sweeps (default :func:`default_sweeps`) to diversify
       duplicates and remove residual bias.
    """
    pb = _problem(knowledge)
    U = pb.cards.size
    out = np.tile(pb.base, (n, 1))
    if U == 0:
        return out
    owns: list[np.ndarray] = []
    logqs: list[np.ndarray] = []
    have = 0
    for _ in range(max_restarts):
        own, fail, logq = _construct(pb, max(n, 32), rng)
        good = ~fail
        k = int(good.sum())
        if k:
            owns.append(own[good])
            logqs.append(logq[good])
            have += k
        if have >= n:
            break
    if have == 0:
        own = np.tile(_solve_one(pb, rng), (n, 1))
        mix = 6
    else:
        pool = np.concatenate(owns)
        lw = -np.concatenate(logqs)
        w = np.exp(lw - lw.max())
        w /= w.sum()
        ess = 1.0 / float((w ** 2).sum())
        # systematic resampling
        pos = (rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(w), pos), pool.shape[0] - 1)
        own = pool[idx]
        mix = 1 if ess >= 0.5 * n else (2 if ess >= 0.1 * n else 4)
    _mcmc(pb, own, (default_sweeps(U) if sweeps is None else sweeps) * mix, rng)
    out[:, pb.cards] = own
    return out


# ------------------------------------------------------------------ summaries
def marginals(particles: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """``P[player, card]``: (weighted) fraction of particles where player holds card."""
    n = particles.shape[0]
    onehot = particles[:, None, :] == np.arange(N_PLAYERS, dtype=np.int8)[None, :, None]
    if weights is None:
        return onehot.sum(0) / n
    w = np.asarray(weights, dtype=float)
    return np.tensordot(w / w.sum(), onehot, axes=(0, 0))


_POW3 = 3 ** np.arange(SET_SIZE)


def best_declaration(
    particles: np.ndarray, team: int, set_id: int, weights: np.ndarray | None = None
) -> tuple[tuple[int, ...] | None, float]:
    """Most frequent full assignment of ``set_id`` within ``team`` and its probability.

    Particles where an opponent holds any card of the set count against every
    assignment.  Returns ``(None, 0.0)`` if no particle has the set on ``team``.
    """
    cols = particles[:, SET_SIZE * set_id : SET_SIZE * (set_id + 1)].astype(np.int64)
    if (cols < 0).any():
        return None, 0.0
    on_team = ((cols & 1) == team).all(1)
    if not on_team.any():
        return None, 0.0
    codes = (cols[on_team] >> 1) @ _POW3
    if weights is None:
        counts = np.bincount(codes, minlength=3 ** SET_SIZE).astype(float)
        total = float(particles.shape[0])
    else:
        w = np.asarray(weights, dtype=float)
        counts = np.bincount(codes, weights=w[on_team], minlength=3 ** SET_SIZE)
        total = float(w.sum())
    best = int(counts.argmax())
    sigma = tuple(2 * ((best // 3 ** i) % 3) + team for i in range(SET_SIZE))
    return sigma, float(counts[best] / total)


def ask_likelihood_weights(
    particles: np.ndarray, knowledge: Knowledge, alpha: float = 1.0, max_log: int = 60
) -> np.ndarray:
    """Importance weights from the logged asks under a simple opponent model.

    Model: a player asks in set ``h`` with probability proportional to
    ``count_h ** alpha`` where ``count_h`` is the number of set-``h`` cards they
    held when asking ("players fish in the sets they are long in").  Past
    counts are reconstructed exactly from each particle because every card
    transfer since the ask is public.  ``alpha = 0`` gives uniform weights.
    """
    n = particles.shape[0]
    log = knowledge.ask_log
    if not log or alpha == 0.0:
        return np.full(n, 1.0 / n)
    cnt = (
        (particles[:, None, :] == np.arange(N_PLAYERS, dtype=np.int8)[None, :, None])
        .reshape(n, N_PLAYERS, N_SETS, SET_SIZE)
        .sum(3)
    )
    logw = np.zeros(n)
    for a, h, off in log[-max_log:]:
        logw += alpha * np.log(np.maximum(cnt[:, a, h] + off, 1))
    w = np.exp(logw - logw.max())
    return w / w.sum()


def belief_scores(
    P: np.ndarray, state: State, viewer_hand: int = 0, eps: float = 1e-3
) -> tuple[float, float, int]:
    """Multi-class Brier score and log-loss of marginals against the true state.

    Averaged over live cards outside ``viewer_hand`` (the cards the viewer is
    actually uncertain about a priori).  Returns ``(brier, logloss, n_cards)``;
    probabilities are clipped to ``eps`` for the log-loss.
    """
    owner = state.owner
    cards = [c for c in range(N_CARDS) if owner[c] >= 0 and not viewer_hand >> c & 1]
    if not cards:
        return 0.0, 0.0, 0
    idx = np.array(cards)
    truth = owner[idx] == np.arange(N_PLAYERS)[:, None]
    sub = P[:, idx]
    brier = float(((sub - truth) ** 2).sum(0).mean())
    p_true = np.clip(sub[owner[idx].astype(int), np.arange(idx.size)], eps, 1.0)
    return brier, float(-np.log(p_true).mean()), len(cards)


# ------------------------------------------------------- exact enumeration
def enumerate_deals(knowledge: Knowledge, limit: int = 2_000_000) -> Iterator[np.ndarray]:
    """Yield every deal consistent with ``knowledge`` (brute force; tiny positions only)."""
    pb = _problem(knowledge)
    U = pb.cards.size
    set_u = pb.set_u.tolist()
    holders = [[p for p in range(N_PLAYERS) if pb.poss[p, ci]] for ci in range(U)]
    rem = pb.cap.tolist()
    need = pb.need.tolist()
    cur = [0] * U
    count = 0

    def rec(t: int) -> Iterator[np.ndarray]:
        nonlocal count
        if t == U:
            if not any(rem) and not any(any(r) for r in need):
                count += 1
                if count > limit:
                    raise SamplingError("too many deals to enumerate")
                out = pb.base.copy()
                out[pb.cards] = cur
                yield out
            return
        h = set_u[t]
        for p in holders[t]:
            if rem[p] > 0:
                had = need[p][h] > 0
                rem[p] -= 1
                need[p][h] -= had
                cur[t] = p
                yield from rec(t + 1)
                rem[p] += 1
                need[p][h] += had

    yield from rec(0)


def satisfies(knowledge: Knowledge, deal: np.ndarray) -> bool:
    """True when the owner array ``deal`` meets every constraint in ``knowledge``."""
    for p in range(N_PLAYERS):
        hand = 0
        for c in np.flatnonzero(deal == p).tolist():
            hand |= 1 << c
        if hand.bit_count() != knowledge.hand_size[p] or hand & ~knowledge.poss[p]:
            return False
        if knowledge.known[p] & ~hand:
            return False
        for h in range(N_SETS):
            if (hand & SET_MASKS[h]).bit_count() < knowledge.min_in_set[p][h]:
                return False
    live = knowledge.live_mask
    return all((deal[c] >= 0) == bool(live >> c & 1) for c in range(N_CARDS))


__all__ = [
    "SamplingError", "sample_deals", "marginals", "best_declaration", "belief_scores",
    "ask_likelihood_weights", "enumerate_deals", "satisfies", "default_sweeps", "TEAM_MEMBERS",
]
