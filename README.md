# fishy-RL

A solver for the card game Fish (also called Literature). Given any starting deal, it decides each move for one player by working out where the cards probably are and then picking the move that is best for the team.

Nothing is trained. The solver works out each move on the spot, and you run everything yourself with the scripts below.

## The game in one minute

- Six players in two teams of three. 54 cards, split into 9 sets of 6.
- On your turn you **ask** one opponent for a specific card. You must already hold another card from the same set. If they have it, you take it and go again. If not, the turn passes to them.
- Instead of asking, you can **declare** a set: say exactly which teammate holds each of its six cards. Get it all right and your team scores the set. Get anything wrong and the other team scores it.
- You only see your own hand. Teammates cannot share information.
- The game ends when all 9 sets are declared. The team with more sets wins.

The exact rule choices for edge cases are in `RULES.md`.

## What "optimal" means here

Perfect play of full Fish cannot be computed: there are too many possible deals, and teammates who cannot talk break the methods used to solve games like poker. So the goal is the best move *given what the player can know*, found in a way we can check.

In Fish, most of the skill is working out where the cards are. Once you know that, the right move is usually clear. The method is built around that.

## The method, in five modules

Each module takes the output of the one before it. Each can be tested, and improved, on its own.

```
 game moves ──▶ 1. Rules ──▶ 2. Facts ──▶ 3. Beliefs ──▶ 4. Scoring ──▶ 5. Decision ──▶ move
                                                                             │
                                                              6. Checking ◀──┘
```

### Module 1: Rules

**Job:** know what is legal and what happens after each move.

- **In:** a game position and a move.
- **Out:** the next position, plus what everyone at the table saw.

This is the referee. Every other module relies on it.

### Module 2: Facts

**Job:** keep track of everything that is certain.

- **In:** the player's own hand and every public move so far.
- **Out:** for each player and each card: definitely has it, definitely doesn't, or unknown.

Each move rules things out. If someone asks for the 7 of hearts, they don't have it, but they do hold another card from that set. If the ask fails, the person asked doesn't have it either. Facts are chained together: if five players can't hold a card, the sixth must.

### Module 3: Beliefs

**Job:** turn "unknown" into probabilities.

- **In:** the facts from module 2 and the history of asks.
- **Out:** the chance that each player holds each card.

The solver imagines a few hundred complete deals that fit every fact. If a player holds a card in 80% of them, that is an 80% chance.

It also reads behaviour, not just facts. A player who keeps asking about one set probably holds several cards of it, so imagined deals where that is true count for more.

### Module 4: Scoring

**Job:** say how good a position is for the team.

- **In:** the beliefs from module 3.
- **Out:** one number for the position.

Each undeclared set scores from −1 to +1 by how much of it the team is expected to hold: all six cards is +1, an even split is 0, none is −1. The position's score is the sum over sets.

### Module 5: Decision

**Job:** pick the move.

- **In:** the legal moves (module 1), the beliefs (module 3) and the scoring (module 4).
- **Out:** the move to play.

Every legal move gets a number, and the solver plays the highest.

- **An ask** is a gamble. If it works, the team gains a card. If it fails, the turn goes to that opponent, who may take a card back. So the solver prefers asks that are likely to work, aimed at opponents who can do the least damage with the turn.
- **A declaration** is compared with waiting. The solver declares when the chance of being right is high enough that declaring beats every available ask.

This looks one move ahead. It is the best choice for the current move, not a plan for the whole game.

### Module 6: Checking

**Job:** measure how close the decisions are to the best possible.

- **In:** the solver, and positions to test it on.
- **Out:** how often it picks the best move, and how much it loses when it doesn't.

Two checks:

1. **Small endgames.** With two or three sets left there are few enough possibilities to work out the truly best move by brute force, and compare.
2. **Full games.** Over hundreds of random deals, the solver's team plays against simple opponents and should clearly win.

If the checks show a weakness, the fix goes into the module that caused it: wrong probabilities point to module 3, bad trade-offs to module 4 or 5.

## Where each module lives

| Module | File | State |
|---|---|---|
| 1. Rules | `fish/cards.py`, `fish/engine.py`, `fish/env.py` | done |
| 2. Facts | `fish/knowledge.py` | done |
| 3. Beliefs | `fish/belief.py` | done; reading behaviour to be switched on |
| 4. Scoring | `fish/value.py` | to write |
| 5. Decision | `fish/policy.py` | to write |
| 6. Checking | `fish/exact.py`, `fish/arena.py` | arena done; endgame check to write |
| Simple opponents | `fish/baselines.py` | done |

Every file has matching tests in `tests/`.

## How to run it

Setup, once:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pytest
```

| Command | What you get | State |
|---|---|---|
| `.venv/bin/python -m pytest` | Runs all the tests | works now |
| `python scripts/play.py --seed 0` | One full game, move by move, showing what the solver considered and why | to write |
| `python scripts/evaluate.py --games 200` | The solver's team against simple opponents over many random deals | to write |
| `python scripts/optimality_gap.py` | How often the solver's move matches the best move on small endgames | to write |

Runs use two processes by default and save after every game, so they stay light on memory and can be stopped and resumed.

## Build order

1. ~~Restore the rules, facts, beliefs and game runner; remove the old search agents.~~ Done.
2. Write the scoring (`value.py`).
3. Write the decision (`policy.py`).
4. Write `play.py` and `evaluate.py`.
5. Write the endgame check (`exact.py`, `optimality_gap.py`).
6. Add results here once they have been run.

## Things to know

- **The game is played to all 9 sets.** The project proposal says to stop at 5; that would be a small change.
- **The solver looks one move ahead.** It will not make an ask just to learn something, which strong human players sometimes do. Earlier experiments on this project found that looking further ahead added little compared with getting the beliefs right.
- **It assumes others ask about sets they hold a lot of.** An opponent who bluffs can mislead it.
