# Fish (Literature): belief-space planning agents

A complete, tested implementation of six-player Fish and a family of agents that
(i) keep a probabilistic belief over where every hidden card is and (ii) plan with
it. The rules as implemented are in [RULES.md](RULES.md).

```
fish/
  cards.py      card ids, names, set mapping, pretty printing
  engine.py     State, legal_actions, step -> (state, PublicObs), is_terminal
  knowledge.py  public constraint store + propagation, per-player views
  belief.py     particle sampler over consistent deals, marginals, best declaration
  agents.py     Random, RuleBased, GreedyBelief, Rollout, ISMCTS
  arena.py      seeded matches, team rotation, metrics, CSV logging, multiprocessing
tests/          one test module per stage (70 tests)
scripts/        experiment runners + plots
results/        CSVs, logs and figures from the runs reported below
```

## Install

Python 3.11+ with NumPy; pytest for the tests and matplotlib for the plots.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python numpy pytest matplotlib
# or: python3 -m venv .venv && .venv/bin/pip install numpy pytest matplotlib
```

No install step is needed for the package itself: run everything from the repo root.

## Run the tests

```bash
.venv/bin/python -m pytest            # all 70 tests, about 4 minutes
.venv/bin/python -m pytest tests/test_engine.py      # stage 1 (10,000 random games)
.venv/bin/python -m pytest tests/test_knowledge.py   # stage 2 (soundness soak + rule scenarios)
.venv/bin/python -m pytest tests/test_belief.py -s   # stage 3 (prints samples/sec)
.venv/bin/python -m pytest tests/test_agents.py tests/test_arena.py   # stages 4-5
```

What the tests establish:

| Stage | Key checks |
|---|---|
| Engine | 10,000 random games finish; card conservation and hand sizes hold after every action; every kind of illegal action raises; turn passing with empty hands and the "whole team out, other team must declare" endgame. |
| Knowledge | In 3,300 simulated games the true deal satisfies the public store, every player's derived view and an incrementally updated private store at every step; one hand-made scenario per propagation rule; contradictions raise. |
| Belief | Every sample satisfies the constraints; on 12 brute-forced endgames marginals are within 0.03 of exact and deal frequencies pass a chi-square uniformity check; a position reachable only through 3-cycles mixes; sampler is seeded; samples/sec is printed. |
| Agents | Every agent plays full games with engine validation and view-vs-truth checks switched on; agents only receive a `View`; certain declarations are always taken; planners solve a clear-cut pass-or-declare endgame; default decisions take under 1 s; RuleBased beats Random and GreedyBelief beats RuleBased. |
| Arena | Parallel results equal serial results for the same seed; CSV round trip; paired-deal confidence intervals. |

## How it works

**Knowledge (hard constraints).** `Knowledge` stores, as bitmasks, which cards each
player can still hold, public hand sizes, and a lower bound on each player's count
per set (asking for a card proves you hold another card of that set). After each
public observation the rules are propagated to a fixed point. One sound rule was
added beyond the four in the brief: when a player's per-set minimums add up to
their hand size they hold nothing else. A player's view is the public store plus
their own hand (`Knowledge.view_for`); agents never see a `State`.

**Belief (particles).** `sample_deals` draws full deals consistent with a store:

1. a vectorised constructive pass assigns the most-constrained cards first, weighted
   by remaining hand capacity, only excluding choices that provably cannot be
   completed; dead ends are discarded and the pass restarts;
2. the surviving deals are importance-resampled with weights 1/q (q = the
   probability with which the pass built them), which corrects the pass's bias;
3. symmetric MCMC moves (pair swaps, and 3-cycles every fourth sweep) diversify
   the particles while leaving the uniform distribution over consistent deals invariant.

Step 2 is not in the brief. It was added because swap moves alone mix slowly in
tight endgames: the brute-force test showed marginals off by up to 0.27 after the
default number of sweeps without it, and within 0.01 with it.

`ask_likelihood_weights` is the optional importance reweighting: particles are
weighted by the likelihood of the logged asks under the model "a player asks in
set h with probability ∝ (cards of h they held)^alpha". Past counts are
reconstructed exactly because every later transfer is public. Agents take it as
`reweight_alpha` (default 0 = off).

**Agents.**

| Agent | Decision rule |
|---|---|
| `RandomAgent` | Uniform random legal ask; declares only what its knowledge proves. |
| `RuleBasedAgent` | Asks for a card known to be with an opponent if there is one, else random. |
| `GreedyBeliefAgent` | Asks argmax P(success) under the particle marginals; declares when the most frequent full assignment of a set has P(correct) above `declare_threshold` (default 0.9). |
| `RolloutAgent` | Candidates = top-4 asks by P(success) + best declarations with P ≥ 0.5. Each candidate is played in 24 deals drawn from the belief (same deals for all candidates), followed by 12 plies of greedy play for all six seats and a leaf evaluation. Highest mean return wins. |
| `ISMCTSAgent` | Single-observer information-set MCTS: 120 iterations, each on a fresh deal from the belief; UCB1 with availability counts over the same pruned actions; greedy rollout of 10 plies from the new node. |

Two implementation choices matter for reading the results:

- **Rollout policy.** The brief asks for rollouts with GreedyBelief players that
  each use their own view. Running a particle filter for every simulated decision
  would cost tens of seconds per move, so simulated players use `fast_policy`: the
  same decision rule on a particle-free approximation of the marginals
  (`heuristic_marginals`). Each simulated player still decides from its *own*
  view (simulated public knowledge + its own hand in the sampled deal), never
  from the sampled deal itself. As a real agent (`GreedyBeliefAgent(belief="heuristic")`)
  this policy is statistically level with the particle version (see results).
- **Depth-limited leaves.** Simulations stop after a fixed number of plies and
  add `leaf_weight * (cards held − 3) / 3` per undeclared set to the score
  difference. Longer rollouts did not help in tuning.

**Arena.** A match between two configurations plays every deal twice with the
teams swapped and a random first player per deal. Seeds are derived from
`(seed, deal)` and `(seed, game)`, so results do not depend on the worker count.
Confidence intervals treat the two games of a deal as one sample. Per-game CSV
columns: sets per team, set differential, game length, truncation flag,
declarations made / correct per side, mean belief Brier score and log-loss per
side, mean and max seconds per decision per side. `log_turns=True` adds a
per-turn log.

## Reproduce the experiments

Each script writes CSVs and a PNG to `results/` and prints a summary. All take
`--games`, `--seed`, `--workers`, `--out`.

```bash
.venv/bin/python scripts/round_robin.py --games 1000     # all agents, every pairing
.venv/bin/python scripts/threshold_sweep.py --games 400  # GreedyBelief declare threshold
.venv/bin/python scripts/budget_curve.py --games 160     # Rollout / ISMCTS budget vs strength
.venv/bin/python scripts/calibration.py --games 200      # reliability diagram + Brier vs turn
```

The numbers below come from exactly these commands with seed 0, except that the
round-robin was run with `--games 400` per pairing instead of 1000 to keep the
run to about an hour on a 14-core laptop. The script's default is 1000.

RESULTS_PLACEHOLDER
