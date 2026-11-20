# fishy-RL

A solver for the card game Fish (also called Literature). You tell it your cards, then type in every move as it happens at the table. Whenever it is your turn, it tells you the best move.

Nothing is trained. The solver works out each move on the spot from what it has seen.

## The game in one minute

- Six players in two teams of three. 54 cards, split into 9 sets of 6.
- On your turn you **ask** one opponent for a specific card. You must already hold another card from the same set. If they have it, you take it and go again. If not, the turn passes to them.
- Instead of asking, you can **declare** a set: say exactly which teammate holds each of its six cards. Get it all right and your team scores the set. Get anything wrong and the other team scores it.
- You only see your own hand. Teammates cannot share information.
- The game ends when all 9 sets are declared. The team with more sets wins.

The exact rule choices for edge cases are in `RULES.md`.

## Using the solver

Setup, once:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pytest
```

Start it:

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

When it is your turn the solver prints something like:

```
Your turn. Best moves:
  1. ask seat 3 for QH   [84% to get it, value +0.25]
  2. ask seat 1 for KH   [51% to get it, value +0.06]
  3. declare low-S (...) [62% to be right, value -0.31]
```

**Seats** are numbered 0 to 5 clockwise. You are seat 0 unless you pass `--seat`. Your teammates are the seats 2 and 4 places after you.

**Cards** are written rank then suit: `2S`, `TH` (ten of hearts), `QD`, `AC`, and `JK1`, `JK2` for the jokers.

**Sets** are `low-S`, `high-S`, `low-H`, `high-H`, `low-D`, `high-D`, `low-C`, `high-C` (low is 2 to 7, high is 9 to ace) and `eights` (the four 8s and both jokers).

Enter every move at the table, not only your own. What the other players ask for is where the solver gets its information.

To start in the middle of a game, add `--midgame`; it will also ask which sets are gone, how many cards each seat holds and the score.

## How it decides

Every time you enter a move, the solver updates what it knows. On your turn it goes through four steps.

### 1. What is certain?

Each move rules things out:

- Someone who asks for the 7 of hearts doesn't have it, but does hold another card from that set.
- If the ask fails, the person asked doesn't have it either.
- If the ask succeeds, everyone now knows where that card is.

Facts are chained together: if five players can't hold a card, the sixth must.

### 2. Where are the other cards, probably?

The solver imagines a few hundred complete deals that fit every fact. If a player holds a card in 80% of them, that is an 80% chance.

It also reads behaviour. A player who keeps asking about one set probably holds several cards of it, so imagined deals where that is true count for more.

### 3. How good is each set for us?

Each undeclared set scores from −1 to +1 by how much of it your team is expected to hold: all six cards is +1, an even split is 0, none is −1.

### 4. Which move is best?

If a declaration is certain to be right, that comes first. Otherwise every legal move gets a value and the highest wins.

- **An ask** is a gamble. If it works, your team gains a card. If it fails, the turn goes to that opponent, who may take a card back. So the solver prefers asks that are likely to work, aimed at opponents who can do the least damage with the turn.
- **A declaration** is compared with waiting: the chance of being right against what the set is already worth to you undeclared.

This looks one move ahead. It is the best choice for the current move given what you can know, not a plan for the whole game. Perfect play of full Fish cannot be computed.

## What's in the code

| File | What it does |
|---|---|
| `scripts/solve.py` | The interactive program described above |
| `fish/solver.py` | The solver: takes your hand and the moves, returns ranked moves |
| `fish/knowledge.py` | Step 1: what is certain |
| `fish/belief.py` | Step 2: where the cards probably are |
| `fish/cards.py`, `fish/engine.py` | Card names and the rules of the game |
| `fish/env.py` | Runs a complete simulated game (used for testing) |
| `fish/baselines.py`, `fish/arena.py` | Simple computer players and a game runner (used for testing) |

You can also use the solver from Python:

```python
from fish.solver import Solver, describe

solver = Solver(seat=0, hand=["2S", "3S", "QH", "KH", "8D", "JK1", "9C", "TC", "AC"], first_player=3)
solver.ask(target=0, card="4S")          # seat 3 asks you and misses; now it's your turn
for move in solver.best_moves(top=3):
    print(describe(move), f"{move.chance:.0%}")
```

## Tests

```bash
.venv/bin/python -m pytest
```

`tests/test_solver.py` plays a full simulated game with the solver in one seat and checks that it follows the game correctly and only ever suggests legal moves.

## Still to do

- Measure how good the advice is: play the solver against the simple computer players over many deals, and compare its choices with brute-force best moves on small endgames.
- Tune based on what that shows.

## Things to know

- **The game is played to all 9 sets.** Stopping at 5 would be a small change.
- **The solver looks one move ahead.** It will not suggest an ask just to learn something, which strong human players sometimes do.
- **It assumes others ask about sets they hold a lot of.** A player who bluffs can mislead it. `--alpha 0` switches this off.
- **After a wrong declaration it needs the card counts,** meaning how many cards of that set each seat held. At a real table these are visible when the cards are handed in.
