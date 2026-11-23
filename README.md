# fishy-RL

A solver for the card game Fish (aka Literature) using reinforcement learning. You tell it your cards, then type in every move as it happens at the table. Whenever it is your turn, it tells you the best move.

The solver learns by playing games against itself and against a hand-written baseline. 

## Rules of fish

- Six players in two teams of three. 54 cards, split into 9 sets of 6.
- On your turn you **ask** one opponent for a specific card. You must already hold another card from the same set. If they have it, you take it and go again. If not, the turn passes to them.
- Instead of asking, you can **declare** a set: say exactly which teammate holds each of its six cards. Get it all right and your team scores the set. Get anything wrong and the other team scores it.
- You only see your own hand. Teammates cannot share information.
- The game ends when all 9 sets are declared. The team with more sets wins.

The exact rule choices for edge cases are in `RULES.md`.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pytest torch
```

## The three steps

### 1. Train the solver

```bash
.venv/bin/python scripts/train.py --games 2000
.venv/bin/python scripts/train.py --games 5000 --resume   # continue later
```

Every 100 games it saves `checkpoints/net.pt`, plays 20 games against the rule-based player and 20 against the heuristic, and prints both results. The same numbers go into `checkpoints/net_log.csv`, which gives a learning curve. Stop it any time with Ctrl-C; it saves before exiting.

The training logs are kept in this repository; the trained networks are not.

### 2. Measure it

```bash
.venv/bin/python scripts/evaluate.py --games 400
```

This plays the solver against the two baselines and prints its average set margin against each, with a confidence interval. Fewer games is quicker but less exact. Every deal is played twice with the teams swapped, so card luck mostly cancels. Results are saved to `results/evaluate.csv`; add `--resume` to continue a stopped run.

The players you can name with `--player` and `--against`:

| Name | What it is |
|---|---|
| `solver` | The trained network |
| `heuristic` | Hand-written scoring of moves; nothing is learned |
| `rule` | Asks for cards it knows about, otherwise asks at random |
| `random` | Asks at random |

### 3. Use it at the table

```bash
.venv/bin/python scripts/solve.py
```

It asks for your cards and who moves first. After that you type one line per move. The player on turn is always the one acting, so you never type who made the move.

| You type | Meaning |
|---|---|
| `a 3 QH y` | The player on turn asked seat 3 for the queen of hearts and got it |
| `a 3 QH n` | Same ask, but seat 3 did not have it |
| `a 0 QH` | The player on turn asked you (seat 0); the solver knows whether you have it |
| `d high-H y 2 0 3 0 1 0` | The player on turn declared the high hearts correctly; seats 0 to 5 held 2, 0, 3, 0, 1, 0 of those cards |
| `d high-H n 2 1 2 0 1 0` | A wrong declaration, with the same six counts |
| `best` | Show the best moves again |
| `show` | Your hand, how many cards everyone holds, and the score |
| `undo` | Take back the last line you typed |
| `help`, `quit` | |

**Seats** are numbered 0 to 5 clockwise. You are seat 0 unless you pass `--seat`. Your teammates are the seats 2 and 4 places after you.

**Cards** are written rank then suit: `2S`, `TH` (ten of hearts), `QD`, `AC`, and `JK1`, `JK2` for the jokers.

**Sets** are `low-S`, `high-S`, `low-H`, `high-H`, `low-D`, `high-D`, `low-C`, `high-C` (low is 2 to 7, high is 9 to ace) and `eights` (the four 8s and both jokers).

Enter every move at the table, not only your own. What the other players ask for is where the solver gets its information.

If there is no trained network, or you pass `--heuristic`, the hand-written heuristic gives the advice instead. To start in the middle of a game, add `--midgame`.

## How the solver decides

### 1. What is certain?

Each move rules things out:

- Someone who asks for the 7 of hearts doesn't have it, but does hold another card from that set.
- If the ask fails, the person asked doesn't have it either.
- If the ask succeeds, everyone now knows where that card is.

Facts are chained together: if five players can't hold a card, the sixth must.

### 2. Where are the other cards, probably?

The solver imagines many complete deals that fit every fact. If a player holds a card in 80% of them, that is an 80% chance.

It also reads behaviour. A player who keeps asking about one set probably holds several cards of it, so imagined deals where that is true count for more.

### 3. Who will end up with each set?

For every set still in play, a small neural network predicts who will win it: +1 for your team, −1 for the opponents. This is the part that is learned.

The network does not look at individual cards. It sees about twenty numbers that describe one set at a time, such as:

- how many of its cards your team probably holds, and the chance your team holds all six;
- how many of its cards you can place for certain, and how many everyone can;
- whose turn it is, and whether that player holds a card of the set;
- if it is your turn, your best chance of winning a card of the set.

The same network is used for all nine sets, so what it learns about one set carries over to the others. A position is worth the sets already scored plus the predictions for the rest.

### 4. Which move is best?

If a declaration is certain to be right, that comes first. Otherwise, for each move the solver works out the positions it can lead to (an ask works or fails; a declaration is right or wrong), scores each with the network, and weights them by how likely they are. The move with the best expected result wins.

## How the solver learns

The network starts knowing nothing. It improves by playing games and comparing its predictions with what happened: for every set, whether the team went on to win it (+1) or lose it (−1).

Who plays the training games:

1. **Warm start.** The first 200 games are played entirely by the heuristic, and the network learns how those games turn out.
2. **Mostly itself.** After that, about 70% of games are the network in all six seats.
3. **Sometimes the heuristic.** In the other 30% the network's team plays a team of three heuristic players.

`--warmup` and `--heuristic-share` change these numbers, and `--past-share 0.1` adds games against older saved copies of the network.

To compare training setups, train one network per setup and measure each the same way:

```bash
.venv/bin/python scripts/train.py --games 2000 --warmup 0 --heuristic-share 0 --out checkpoints/selfplay.pt
.venv/bin/python scripts/train.py --games 2000 --heuristic-share 1 --out checkpoints/heuristic_only.pt
.venv/bin/python scripts/evaluate.py --games 400 --checkpoint checkpoints/selfplay.pt --out results/selfplay.csv
```

## Results so far

The solver's average set margin per game (out of 9 sets) against each baseline, with 95% confidence intervals. Positive means the solver did better.

| Training setup | Against `rule` | Against `heuristic` |
|---|---|---|
| Self-play only, learning from the final result | −0.10 ± 0.61 (52% wins) | −3.52 ± 0.49 (9% wins) |
| Self-play only, each declared set as a reward | +0.54 ± 0.53 (56% wins) | −2.94 ± 0.60 (17% wins) |
| Warm start, mixed opponents, card probabilities as input (650 games) | about −0.2 | about −3.6 |
| **One prediction per set from a few numbers per set (current setup, 2000 games)** | **+3.40 ± 0.24 (92% wins)** | **+0.65 ± 0.31 (58% wins)** |
| The same network trained on to 3000 games | +3.42 ± 0.25 (90% wins) | +0.40 ± 0.31 (54% wins) |
| The same network trained on to 5000 games | +3.54 ± 0.27 (90% wins) | +0.39 ± 0.27 (56% wins) |

The first two rows are from 100 games against each baseline and the last three from 400. The third comes from 10-game checks during training, not from a full evaluation.

The first three rows all used one large network that read every card separately (725 or 1049 inputs) and predicted the final score in one number. None got past the rule-based player: with a few hundred games, that network memorises the games it has seen instead of learning what makes a position good.

The current setup replaced it and is the first that works. The solver beats the rule-based player by 3.4 sets a game and the hand-written heuristic by about two thirds of a set, and both intervals are clear of zero. One of the 400 games against the heuristic did not finish within 3000 moves and counts with the sets scored up to then. The earlier checkpoints no longer load.

Training longer made no measurable difference. The 2000, 3000 and 5000-game networks were measured on the same 200 deals, and the differences between them are within the noise (5000 against 2000 games: −0.27 ± 0.40 against the heuristic). Averaged over the three, the solver is +0.48 ± 0.18 sets ahead of the heuristic.

The training logs of these three networks are `checkpoints/net_log.csv` (2000 games), `checkpoints/games3000_log.csv` and `checkpoints/games5000_log.csv`. Each has one line per 100 games of training.

## The heuristic baseline

The heuristic uses steps 1 and 2 above, then scores moves with fixed formulas instead of a network:

- **An ask** is a gamble. If it works, your team gains a card. If it fails, the turn goes to that opponent, who may take a card back. So it prefers asks that are likely to work, aimed at opponents who can do the least damage with the turn.
- **A declaration** is compared with waiting: the chance of being right against what the set is already worth undeclared.

It is the opponent the solver trains against and the bar the solver has to clear.

## What's in the code

| File | What it does |
|---|---|
| `scripts/train.py` | Trains the network |
| `scripts/evaluate.py` | Plays one player against others and reports the results |
| `scripts/solve.py` | The interactive program |
| `fish/cards.py`, `fish/engine.py` | Card names and the rules of the game |
| `fish/env.py` | Runs a complete simulated game |
| `fish/knowledge.py` | Step 1: what is certain |
| `fish/belief.py` | Step 2: where the cards probably are |
| `fish/features.py` | Describes each set in a position with about twenty numbers |
| `fish/value_net.py` | Step 3: the network |
| `fish/chooser.py` | Step 4: picks a move with the network |
| `fish/selfplay.py` | Plays training games and builds the learning targets |
| `fish/solver.py` | Follows a game from your seat as you type moves in, and ranks your moves |
| `fish/heuristic.py` | The hand-written baseline |
| `fish/baselines.py` | The rule-based and random players |

Using the solver from Python:

```python
from fish.solver import Solver, describe
from fish.value_net import load

net, _ = load("checkpoints/net.pt")
solver = Solver(seat=0, hand=["2S", "3S", "QH", "KH", "8D", "JK1", "9C", "TC", "AC"],
                first_player=3, net=net)
solver.ask(target=0, card="4S")          # seat 3 asks you and misses; now it's your turn
for move in solver.best_moves(top=3):
    print(describe(move), f"{move.chance:.0%}")
```

## Tests

```bash
.venv/bin/python -m pytest
```

## Things to know

- **The game is played to all 9 sets.** Stopping at 5 would be a small change.
- **The solver looks one move ahead.** The network's predictions stand in for the rest of the game.
- **It assumes others ask about sets they hold a lot of.** A player who bluffs can mislead it. `--alpha 0` switches this off.
- **After a wrong declaration it needs the card counts,** meaning how many cards of that set each seat held. At a real table these are visible when the cards are handed in.
