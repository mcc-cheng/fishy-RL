# fishy-RL

An agent that plays the card game Fish (also called Literature). Given any starting deal, it picks each move by asking: *based on everything I have seen so far, which action is best for my team right now?*

Nothing is trained. The agent works out its move on the spot, and you run everything yourself with the scripts below.

> **Status: this is the plan.** The code is not in the folder yet. Parts 1, 2 and the game runner already exist in an earlier commit and will be restored; the rest is new.

## The game in one minute

- Six players in two teams of three. 54 cards, split into 9 sets of 6.
- On your turn you **ask** one opponent for a specific card. You must already hold another card from the same set. If they have it, you take it and go again. If not, the turn passes to them.
- Instead of asking, you can **declare** a set: say exactly which teammate holds each of its six cards. Get it all right and your team scores the set. Get anything wrong and the other team scores it.
- You only see your own hand. Teammates cannot share information.
- The game ends when all 9 sets are declared. The team with more sets wins.

The exact rule choices for edge cases are in `RULES.md`.

## How the agent thinks

The agent answers four questions, in order. Each one is its own piece of code.

### 1. What do I know for sure?

Every move in the game is public, and each one rules things out:

- If someone asks for the 7 of hearts, they don't have it, but they do hold another card of that set.
- If the ask fails, the person asked doesn't have it either.
- If the ask succeeds, everyone now knows where that card is.

The agent keeps a running list of these facts for every player and every card.

### 2. Where are the other cards, probably?

Facts alone leave many possibilities. So the agent imagines a few hundred complete deals that fit everything it knows. If a player holds a card in 80% of those imagined deals, the agent treats that as an 80% chance.

It also pays attention to what people choose to ask for. A player who keeps asking about one set probably holds several cards of it, so deals where that is true count for more.

### 3. How good is my team's position?

Each undeclared set gets a score from −1 to +1 based on how much of it my team is expected to hold:

- we hold all six cards: **+1**
- an even split: **0**
- the opponents hold all six: **−1**

The position is worth the sum of those scores.

### 4. What should I do?

The agent gives every legal move a number and plays the highest.

**An ask** is a gamble:

- If it works, my team gains a card, which raises the score of that set.
- If it fails, the turn goes to that opponent, who may take a card from us. The more that opponent could take, the more a failed ask costs.

So the agent prefers asks that are likely to succeed, and between two equally likely asks it prefers the opponent who can do less damage with the turn.

**A declaration** is compared with waiting:

- Declaring now is worth the chance of getting it right minus the chance of getting it wrong.
- Not declaring is worth whatever the set was already scored at in step 3.

The agent declares only when that beats every available ask. In practice this means it declares when it is nearly sure, or when it has nothing better to do.

### What "optimal greedy" means

"Greedy" means the agent looks one move ahead: it picks the best move for right now, given its beliefs. It does not plan several turns in advance. That is the best possible choice for a one-move view, but not perfect play of the whole game, which nobody can compute for six-player Fish.

## What's in the code

| File | What it does | State |
|---|---|---|
| `fish/cards.py` | Card numbers, sets and teams | restore |
| `fish/engine.py` | The rules of the game | restore |
| `fish/knowledge.py` | Question 1: what is certain | restore |
| `fish/belief.py` | Question 2: where the cards probably are | restore |
| `fish/value.py` | Question 3: how good the position is | new |
| `fish/policy.py` | Question 4: the greedy agent | new |
| `fish/baselines.py` | Two simple opponents: one random, one following basic rules | restore |
| `fish/arena.py` | Plays full games between agents and saves the results | restore |
| `fish/exact.py` | Checks the agent's choices on tiny endgames | new |

Every file has matching tests in `tests/`.

## How to run it

Setup, once:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pytest
```

| Command | What you get |
|---|---|
| `python scripts/play.py --seed 0` | One full game, printed move by move, showing what the agent considered and why it chose its move |
| `python scripts/evaluate.py --games 200` | The agent's team against a baseline team over many random deals, with the average score |
| `python scripts/optimality_gap.py` | How often the agent's move matches the best move on small endgames |
| `pytest` | Runs all the tests |

Runs use two processes by default and save after every game, so they stay light on memory and can be stopped and resumed.

## How we check it's any good

1. **Against simple opponents.** Over hundreds of random deals, the agent should clearly beat both baselines.
2. **Against the best move on small endgames.** With only two or three sets left, there are few enough possibilities to work out better answers by brute force. We compare the agent's move against a version that looks several moves ahead, and against a player who can see everyone's cards.

## Build order

1. Restore the existing code (rules, knowledge, beliefs, game runner) from the earlier commit and remove the old search agents.
2. Write the position scoring (`value.py`).
3. Write the greedy agent (`policy.py`).
4. Write `play.py` and `evaluate.py`.
5. Write the endgame checker (`exact.py`, `optimality_gap.py`).
6. Add results to this README once you have run them.

## Things to know

- **The game is played to all 9 sets.** The project proposal says to stop at 5; that would be a small change.
- **The agent only looks one move ahead.** It will not make an ask just to learn something, which strong human players sometimes do.
- **It assumes others ask about sets they hold a lot of.** An opponent who bluffs can mislead it.
