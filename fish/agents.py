"""Agents: Random, RuleBased, GreedyBelief, Rollout, ISMCTS.

Every agent implements ``act(view, rng) -> Action`` and only ever touches its
:class:`~fish.knowledge.View` (own hand + public knowledge), never a ``State``.
The planning agents build *hypothetical* states from deals sampled out of
their own belief; simulated players inside those hypothetical games again only
get views derived from the simulated public knowledge.
"""
from __future__ import annotations

import math
import time
from typing import Any, Protocol

import numpy as np

from .belief import ask_likelihood_weights, best_declaration, marginals, sample_deals
from .cards import N_PLAYERS, N_SETS, OPPONENTS, SET_MASKS, SET_SIZE, TEAMMATES
from .engine import Action, Ask, Declare, State, enumerate_asks, step
from .knowledge import Knowledge, View

LIVE_ALL = (1 << N_SETS) - 1


class Agent(Protocol):
    """Interface shared by all agents."""

    name: str

    def act(self, view: View, rng: np.random.Generator) -> Action:
        """Choose an action for ``view.player`` (who is on turn)."""
        ...


# ============================================================ shared helpers
def legal_asks(view: View) -> list[Ask]:
    """Legal asks derivable from the view (own hand + public hand sizes)."""
    return enumerate_asks(view.player, view.hand, view.know.hand_size)


def certain_declaration(k: Knowledge, player: int) -> Declare | None:
    """A declaration that is provably correct under ``k`` (lowest set id), if any."""
    mates = TEAMMATES[player]
    team_known = k.known[mates[0]] | k.known[mates[1]] | k.known[mates[2]]
    for h in range(N_SETS):
        sm = SET_MASKS[h]
        if not k.declared >> h & 1 and team_known & sm == sm:
            return Declare(h, tuple(k.known_owner(c) for c in range(SET_SIZE * h, SET_SIZE * h + 6)))
    return None


def _densities(k: Knowledge) -> tuple[list[int], list[list[float]]]:
    """Per player: mask of unknown-but-possible cards and, per set, the
    heuristic probability that such a card is actually held.

    Base density is ``free slots / possible unknown cards``; in a set where the
    player provably holds more cards than are located, it is raised to
    ``missing / possible cards in the set``.
    """
    unk: list[int] = []
    dens: list[list[float]] = []
    for p in range(N_PLAYERS):
        kn = k.known[p]
        u = k.poss[p] & ~kn
        unk.append(u)
        free = k.hand_size[p] - kn.bit_count()
        if not u or free <= 0:
            dens.append([0.0] * N_SETS)
            continue
        d = free / u.bit_count()
        row = [d] * N_SETS
        mins = k.min_in_set[p]
        for h in range(N_SETS):
            mn = mins[h]
            if mn:
                sm = SET_MASKS[h]
                need = mn - (kn & sm).bit_count()
                if need > 0:
                    x = need / (u & sm).bit_count()
                    if x > d:
                        row[h] = x
        dens.append(row)
    return unk, dens


def heuristic_marginals(k: Knowledge) -> np.ndarray:
    """Particle-free approximation of ``P[player, card]`` (used inside rollouts)."""
    unk, dens = _densities(k)
    P = np.zeros((N_PLAYERS, N_SETS * SET_SIZE))
    for h in range(N_SETS):
        if k.declared >> h & 1:
            continue
        for c in range(SET_SIZE * h, SET_SIZE * h + SET_SIZE):
            bit = 1 << c
            owner = k.known_owner(c)
            if owner >= 0:
                P[owner, c] = 1.0
                continue
            ws = [dens[p][h] if unk[p] & bit else 0.0 for p in range(N_PLAYERS)]
            tot = sum(ws)
            if tot <= 0:  # degenerate: spread over possible holders
                ws = [1.0 if unk[p] & bit else 0.0 for p in range(N_PLAYERS)]
                tot = sum(ws)
            for p in range(N_PLAYERS):
                P[p, c] = ws[p] / tot
    return P


def _fast_asks(k: Knowledge, player: int, hand: int,
               unk: list[int], dens: list[list[float]]) -> list[tuple[float, int, int]]:
    """``(P(success), target, card)`` for every legal ask with P > 0 (heuristic belief)."""
    out: list[tuple[float, int, int]] = []
    opps = OPPONENTS[player]
    if not (k.hand_size[opps[0]] or k.hand_size[opps[1]] or k.hand_size[opps[2]]):
        return out
    known = k.known
    opp_known = known[opps[0]] | known[opps[1]] | known[opps[2]]
    mates = [q for q in TEAMMATES[player] if q != player]
    mate_known = known[mates[0]] | known[mates[1]]
    for h in range(N_SETS):
        sm = SET_MASKS[h]
        if not hand & sm:
            continue
        missing = sm & ~hand & ~mate_known
        while missing:
            bit = missing & -missing
            missing ^= bit
            c = bit.bit_length() - 1
            if opp_known & bit:
                for o in opps:
                    if known[o] & bit:
                        out.append((1.0, o, c))
                        break
                continue
            tot = 0.0
            for q in mates:
                if unk[q] & bit:
                    tot += dens[q][h]
            ws = []
            for o in opps:
                if unk[o] & bit:
                    w = dens[o][h]
                    ws.append((w, o))
                    tot += w
            if tot > 0:
                for w, o in ws:
                    if w > 0:
                        out.append((w / tot, o, c))
            else:
                for w, o in ws:
                    out.append((1e-6, o, c))
    return out


def _fast_declaration(k: Knowledge, player: int, h: int, unk: list[int],
                      dens: list[list[float]], u: float = 0.0) -> tuple[tuple[int, ...], float]:
    """Best per-card assignment of set ``h`` to the team and its (independence) probability.

    Unlocated cards go to the likeliest teammate that still has a free hand
    slot; ``u`` in [0, 1) picks which card is considered first, so that equally
    plausible guesses are not always resolved the same way.
    """
    mates = TEAMMATES[player]
    opps = OPPONENTS[player]
    known = k.known
    prob = 1.0
    sigma = [player] * SET_SIZE
    free = {q: k.hand_size[q] - known[q].bit_count() for q in mates}
    start = int(u * SET_SIZE)
    for step_i in range(SET_SIZE):
        i = (start + step_i) % SET_SIZE
        bit = 1 << (SET_SIZE * h + i)
        best_q, best_w, tot = player, 0.0, 0.0
        done = False
        for q in mates:
            if known[q] & bit:
                sigma[i] = q
                done = True
                break
            if unk[q] & bit:
                w = dens[q][h]
                tot += w
                if w > best_w and free[q] > 0:
                    best_q, best_w = q, w
        if done:
            continue
        free[best_q] -= 1
        for o in opps:
            if known[o] & bit:
                tot = 0.0
                best_w = 0.0
                break
            if unk[o] & bit:
                tot += dens[o][h]
        sigma[i] = best_q
        prob *= best_w / tot if tot > 0 else 0.0
    return tuple(sigma), prob


def guess_declaration(k: Knowledge, player: int, u: float = 0.0) -> Declare:
    """Best-effort declaration when one is forced (heuristic belief)."""
    unk, dens = _densities(k)
    best: tuple[float, int, tuple[int, ...]] | None = None
    for h in range(N_SETS):
        if k.declared >> h & 1:
            continue
        sigma, pr = _fast_declaration(k, player, h, unk, dens, u)
        if best is None or pr > best[0]:
            best = (pr, h, sigma)
    assert best is not None
    return Declare(best[1], best[2])


def fast_policy(k: Knowledge, player: int, hand: int, u: float,
                declare_threshold: float = 0.9) -> Action:
    """Cheap GreedyBelief stand-in: same decision rule, heuristic marginals.

    ``k`` is the acting player's private store; ``u`` in [0, 1) breaks ties.
    """
    cert = certain_declaration(k, player)
    if cert is not None:
        return cert
    unk, dens = _densities(k)
    asks = _fast_asks(k, player, hand, unk, dens)
    if declare_threshold < 1.0 or not asks:
        mates = TEAMMATES[player]
        opps = OPPONENTS[player]
        team_poss = k.poss[mates[0]] | k.poss[mates[1]] | k.poss[mates[2]]
        opp_known = k.known[opps[0]] | k.known[opps[1]] | k.known[opps[2]]
        best_p, best_d = -1.0, None
        for h in range(N_SETS):
            sm = SET_MASKS[h]
            if k.declared >> h & 1 or opp_known & sm or (team_poss & sm) != sm:
                continue
            sigma, pr = _fast_declaration(k, player, h, unk, dens, u)
            if pr > best_p:
                best_p, best_d = pr, Declare(h, sigma)
        if best_d is not None and best_p > declare_threshold:
            return best_d
        if not asks:
            # No ask can succeed.  A hopeless ask only passes the turn, which
            # beats guessing a declaration; guess only when no ask is legal.
            legal = enumerate_asks(player, hand, k.hand_size)
            if legal:
                return legal[int(u * len(legal))]
            return best_d if best_d is not None else guess_declaration(k, player, u)
    top = max(a[0] for a in asks) - 1e-9
    ties = [a for a in asks if a[0] >= top]
    _, t, c = ties[int(u * len(ties))]
    return Ask(t, c)


# ================================================================ baselines
class RandomAgent:
    """Uniformly random legal ask; declares only sets that are fully known."""

    def __init__(self, name: str = "random") -> None:
        self.name = name

    def act(self, view: View, rng: np.random.Generator) -> Action:
        cert = certain_declaration(view.know, view.player)
        if cert is not None:
            return cert
        asks = legal_asks(view)
        if not asks:
            return guess_declaration(view.know, view.player, float(rng.random()))
        return asks[int(rng.integers(len(asks)))]


class RuleBasedAgent:
    """Asks for a card known to be with an opponent if any, otherwise random."""

    def __init__(self, name: str = "rule") -> None:
        self.name = name

    def act(self, view: View, rng: np.random.Generator) -> Action:
        k = view.know
        cert = certain_declaration(k, view.player)
        if cert is not None:
            return cert
        asks = legal_asks(view)
        if not asks:
            return guess_declaration(k, view.player, float(rng.random()))
        sure = [a for a in asks if k.known[a.target] >> a.card & 1]
        pool = sure or asks
        return pool[int(rng.integers(len(pool)))]


# ============================================================ belief agents
class _BeliefAgent:
    """Shared particle-belief plumbing."""

    name: str
    n_particles: int
    reweight_alpha: float

    def _belief(self, view: View, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
        parts = sample_deals(view.know, self.n_particles, rng)
        w = None
        if self.reweight_alpha:
            w = ask_likelihood_weights(parts, view.know, self.reweight_alpha)
        self.last_marginals = marginals(parts, w)
        return parts, w

    def _declarations(self, view: View, parts: np.ndarray,
                      w: np.ndarray | None) -> list[tuple[float, Declare]]:
        """Best declaration per live set that has non-zero belief, most likely first."""
        k = view.know
        opps = OPPONENTS[view.player]
        opp_known = k.known[opps[0]] | k.known[opps[1]] | k.known[opps[2]]
        out = []
        for h in range(N_SETS):
            if k.declared >> h & 1 or opp_known & SET_MASKS[h]:
                continue
            sigma, pr = best_declaration(parts, view.player & 1, h, w)
            if sigma is not None:
                out.append((pr, Declare(h, sigma)))
        out.sort(key=lambda x: -x[0])
        return out


class GreedyBeliefAgent(_BeliefAgent):
    """Ask argmax P(success); declare once P(correct) exceeds a threshold.

    ``belief='particles'`` uses the particle filter; ``belief='heuristic'``
    uses the particle-free approximation (the policy used inside rollouts).
    """

    def __init__(self, n_particles: int = 300, declare_threshold: float = 0.9,
                 reweight_alpha: float = 0.0, belief: str = "particles",
                 name: str = "greedy") -> None:
        if belief not in ("particles", "heuristic"):
            raise ValueError(belief)
        self.name = name
        self.n_particles = n_particles
        self.declare_threshold = declare_threshold
        self.reweight_alpha = reweight_alpha
        self.belief = belief
        self.last_marginals: np.ndarray | None = None

    def act(self, view: View, rng: np.random.Generator) -> Action:
        k = view.know
        cert = certain_declaration(k, view.player)
        if cert is not None:
            return cert
        if self.belief == "heuristic":
            return fast_policy(k, view.player, view.hand, float(rng.random()),
                               self.declare_threshold)
        parts, w = self._belief(view, rng)
        asks = legal_asks(view)
        decls = self._declarations(view, parts, w)
        if decls and (decls[0][0] > self.declare_threshold or not asks):
            return decls[0][1]
        if not asks:
            return guess_declaration(k, view.player, float(rng.random()))
        P = self.last_marginals
        probs = np.array([P[a.target, a.card] for a in asks])
        best = np.flatnonzero(probs >= probs.max() - 1e-12)
        return asks[int(best[int(rng.integers(best.size))])]


# ------------------------------------------------------- simulation machinery
def leaf_value(state: State, weight: float) -> float:
    """Team-0 value of a (possibly unfinished) state: score difference plus
    ``weight * (cards_held_by_team0 - 3) / 3`` for every undeclared set."""
    v = float(state.score[0] - state.score[1])
    if weight and state.declared != LIVE_ALL:
        t0 = state.hands[0] | state.hands[2] | state.hands[4]
        acc = 0
        live = 0
        for h in range(N_SETS):
            if not state.declared >> h & 1:
                acc += (t0 & SET_MASKS[h]).bit_count()
                live += 1
        v += weight * (acc - 3 * live) / 3.0
    return v


def simulate(state: State, public: Knowledge, depth: int, rng: np.random.Generator,
             declare_threshold: float, leaf_weight: float) -> float:
    """Play ``depth`` actions with :func:`fast_policy` for everyone; return team-0 value.

    ``public`` is mutated.  Each simulated player decides from its own view:
    the simulated public knowledge plus its own hand in the hypothetical deal.
    """
    us = rng.random(depth) if depth > 0 else ()
    for i in range(depth):
        if state.declared == LIVE_ALL:
            break
        p = state.turn
        hand = state.hands[p]
        k = public.copy(False)
        k.set_hand(p, hand)
        state, obs = step(state, fast_policy(k, p, hand, float(us[i]), declare_threshold), False)
        public.update(obs)
    return leaf_value(state, leaf_weight)


class _Planner(_BeliefAgent):
    """Common root-candidate generation for Rollout and ISMCTS."""

    top_k: int
    declare_floor: float
    max_declares: int

    def _root(self, view: View, rng: np.random.Generator
              ) -> tuple[list[Action], np.ndarray, np.ndarray | None]:
        parts, w = self._belief(view, rng)
        P = self.last_marginals
        asks = legal_asks(view)
        scored = [(float(P[a.target, a.card]), i) for i, a in enumerate(asks)]
        scored = [s for s in scored if s[0] > 0]
        # random tie-break among equally likely asks, then keep the top k
        jitter = rng.random(len(scored))
        order = sorted(range(len(scored)), key=lambda i: (-scored[i][0], jitter[i]))
        cands: list[Action] = [asks[scored[i][1]] for i in order[: self.top_k]]
        decls = self._declarations(view, parts, w)
        keep = [d for pr, d in decls if pr >= self.declare_floor][: self.max_declares]
        if not cands:
            # nothing can succeed: compare passing the turn with the best guesses
            if asks:
                cands = [asks[int(rng.integers(len(asks)))]]
            if not keep:
                keep = [d for _, d in decls[: max(1, self.max_declares)]]
            if not cands and not keep:
                keep = [guess_declaration(view.know, view.player, float(rng.random()))]
        return cands + keep, parts, w

    @staticmethod
    def _draw(parts: np.ndarray, w: np.ndarray | None, n: int,
              rng: np.random.Generator) -> np.ndarray:
        """Indices of ``n`` particles drawn from the (weighted) belief."""
        if w is None:
            if n <= parts.shape[0]:
                return rng.permutation(parts.shape[0])[:n]
            return rng.integers(0, parts.shape[0], size=n)
        return rng.choice(parts.shape[0], size=n, p=w)


class RolloutAgent(_Planner):
    """Determinized Monte-Carlo rollouts over pruned candidate actions.

    Candidates: top-``top_k`` asks by P(success) plus the best declaration of
    each set with P(correct) >= ``declare_floor``.  Each candidate is played in
    ``n_deals`` deals sampled from the belief (the same deals for every
    candidate), followed by ``depth`` plies of greedy play and a heuristic leaf
    value; the best mean return wins.
    """

    def __init__(self, n_particles: int = 200, top_k: int = 4, n_deals: int = 24,
                 depth: int = 12, declare_floor: float = 0.5, max_declares: int = 2,
                 rollout_threshold: float = 0.9, leaf_weight: float = 0.5,
                 reweight_alpha: float = 0.0, name: str = "rollout") -> None:
        self.name = name
        self.n_particles = n_particles
        self.top_k = top_k
        self.n_deals = n_deals
        self.depth = depth
        self.declare_floor = declare_floor
        self.max_declares = max_declares
        self.rollout_threshold = rollout_threshold
        self.leaf_weight = leaf_weight
        self.reweight_alpha = reweight_alpha
        self.last_marginals: np.ndarray | None = None
        self.last_values: dict[Action, float] = {}

    def act(self, view: View, rng: np.random.Generator) -> Action:
        k = view.know
        me = view.player
        cert = certain_declaration(k, me)
        if cert is not None:
            return cert
        cands, parts, w = self._root(view, rng)
        if len(cands) == 1:
            return cands[0]
        sign = 1.0 if me % 2 == 0 else -1.0
        score = (k.score[0], k.score[1])
        totals = [0.0] * len(cands)
        for di in self._draw(parts, w, self.n_deals, rng):
            root = State.from_owner(parts[di], me, score)
            for ci, a in enumerate(cands):
                state, obs = step(root, a, False)
                public = view.public.copy(False)
                public.update(obs)
                totals[ci] += simulate(state, public, self.depth, rng,
                                       self.rollout_threshold, self.leaf_weight)
        means = [sign * t / self.n_deals for t in totals]
        self.last_values = dict(zip(cands, means))
        return cands[int(np.argmax(means))]


class _Node:
    """ISMCTS node: per-action [visits, total value, availability] and children."""

    __slots__ = ("stats", "children")

    def __init__(self) -> None:
        self.stats: dict[Action, list[float]] = {}
        self.children: dict[tuple[Action, bool], _Node] = {}


class ISMCTSAgent(_Planner):
    """Single-observer Information Set MCTS.

    Every iteration samples one deal from the belief and descends a tree whose
    nodes are public histories (action + public outcome).  Actions are pruned
    like in :class:`RolloutAgent`: at the root with the particle belief, deeper
    with each simulated player's heuristic view.  Selection is UCB1 with
    availability counts; one node is added per iteration and evaluated with a
    greedy rollout.  Each node's actor maximises its own team's return.
    """

    def __init__(self, n_particles: int = 200, iterations: int = 120, exploration: float = 1.0,
                 top_k: int = 4, tree_top_k: int = 3, rollout_depth: int = 10,
                 declare_floor: float = 0.5, max_declares: int = 2,
                 rollout_threshold: float = 0.9, leaf_weight: float = 0.5,
                 rollout_policy: str = "greedy", reweight_alpha: float = 0.0,
                 max_seconds: float | None = None, name: str = "ismcts") -> None:
        if rollout_policy not in ("greedy", "none"):
            raise ValueError(rollout_policy)
        self.name = name
        self.n_particles = n_particles
        self.iterations = iterations
        self.exploration = exploration
        self.top_k = top_k
        self.tree_top_k = tree_top_k
        self.rollout_depth = rollout_depth if rollout_policy == "greedy" else 0
        self.declare_floor = declare_floor
        self.max_declares = max_declares
        self.rollout_threshold = rollout_threshold
        self.leaf_weight = leaf_weight
        self.reweight_alpha = reweight_alpha
        self.max_seconds = max_seconds  # optional wall-clock cap (breaks reproducibility)
        self.last_marginals: np.ndarray | None = None
        self.last_values: dict[Action, float] = {}

    def _tree_candidates(self, k: Knowledge, player: int, hand: int) -> list[Action]:
        """Pruned actions for a simulated player from its heuristic view."""
        cert = certain_declaration(k, player)
        if cert is not None:
            return [cert]
        unk, dens = _densities(k)
        asks = _fast_asks(k, player, hand, unk, dens)
        asks.sort(reverse=True)
        out: list[Action] = [Ask(t, c) for _, t, c in asks[: self.tree_top_k]]
        mates = TEAMMATES[player]
        opps = OPPONENTS[player]
        team_poss = k.poss[mates[0]] | k.poss[mates[1]] | k.poss[mates[2]]
        opp_known = k.known[opps[0]] | k.known[opps[1]] | k.known[opps[2]]
        best_p, best_d = -1.0, None
        for h in range(N_SETS):
            sm = SET_MASKS[h]
            if k.declared >> h & 1 or opp_known & sm or (team_poss & sm) != sm:
                continue
            sigma, pr = _fast_declaration(k, player, h, unk, dens)
            if pr > best_p:
                best_p, best_d = pr, Declare(h, sigma)
        if not out:
            legal = enumerate_asks(player, hand, k.hand_size)
            if legal:
                out.append(legal[0])
        if best_d is not None and (best_p >= self.declare_floor or not asks):
            out.append(best_d)
        if not out:
            out.append(guess_declaration(k, player))
        return out

    def act(self, view: View, rng: np.random.Generator) -> Action:
        k = view.know
        me = view.player
        cert = certain_declaration(k, me)
        if cert is not None:
            return cert
        root_cands, parts, w = self._root(view, rng)
        if len(root_cands) == 1:
            return root_cands[0]
        score = (k.score[0], k.score[1])
        root = _Node()
        c_uct = self.exploration
        draws = self._draw(parts, w, self.iterations, rng)
        deadline = None if self.max_seconds is None else time.perf_counter() + self.max_seconds
        for it in range(self.iterations):
            if deadline is not None and time.perf_counter() > deadline:
                break
            state = State.from_owner(parts[draws[it]], me, score)
            public = view.public.copy(False)
            node = root
            path: list[tuple[list[float], int]] = []
            while state.declared != LIVE_ALL:
                p = state.turn
                if node is root:
                    cands = root_cands
                else:
                    kp = public.copy(False)
                    kp.set_hand(p, state.hands[p])
                    cands = self._tree_candidates(kp, p, state.hands[p])
                stats = node.stats
                best_a, best_s, untried = None, -math.inf, []
                for a in cands:
                    st = stats.get(a)
                    if st is None:
                        st = stats[a] = [0.0, 0.0, 0.0]
                    st[2] += 1.0
                    if st[0] == 0.0:
                        untried.append(a)
                    elif not untried:
                        s = st[1] / st[0] + c_uct * math.sqrt(math.log(st[2]) / st[0])
                        if s > best_s:
                            best_a, best_s = a, s
                if untried:
                    best_a = untried[int(rng.integers(len(untried)))]
                state, obs = step(state, best_a, False)
                public.update(obs)
                path.append((stats[best_a], p & 1))
                key = (best_a, obs.success if isinstance(best_a, Ask) else obs.correct)
                child = node.children.get(key)
                if child is None:
                    node.children[key] = _Node()
                    break
                node = child
            value = simulate(state, public, self.rollout_depth, rng,
                             self.rollout_threshold, self.leaf_weight)
            for st, team in path:
                st[0] += 1.0
                st[1] += value if team == 0 else -value
        visited = [(root.stats[a][0], root.stats[a][1] / max(root.stats[a][0], 1.0), i)
                   for i, a in enumerate(root_cands) if a in root.stats]
        self.last_values = {root_cands[i]: m for _, m, i in visited}
        return root_cands[max(visited)[2]] if visited else root_cands[0]


# ================================================================== registry
AGENT_TYPES: dict[str, type] = {
    "random": RandomAgent,
    "rule": RuleBasedAgent,
    "greedy": GreedyBeliefAgent,
    "rollout": RolloutAgent,
    "ismcts": ISMCTSAgent,
}


def make_agent(kind: str, **kwargs: Any) -> Agent:
    """Instantiate an agent by registry name (specs stay picklable for multiprocessing)."""
    return AGENT_TYPES[kind](**kwargs)
