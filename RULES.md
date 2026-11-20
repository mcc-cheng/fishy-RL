# Rule choices

The task statement fixes most of the game. Where it was silent or ambiguous, the
standard Literature/Fish convention was used. Everything below is implemented in
`fish/engine.py` and covered by `tests/test_engine.py`.

## Cards and seats
- Card ids 0–53, `set_id = card // 6`. For suit `s` in (♠, ♥, ♦, ♣): set `2s` is
  the low half (2–7), set `2s+1` the high half (9, 10, J, Q, K, A). Set 8 is the
  four 8s followed by the two jokers.
- Seats 0–5 sit clockwise in increasing order. Team A = {0, 2, 4}, team B = {1, 3, 5}.
- The first player is drawn uniformly at random from the game seed.

## Asking
- Exactly as specified: the target must be an opponent with ≥ 1 card; the asker
  must hold ≥ 1 card of the set and must not hold the card itself.
- Asking for a card that a *teammate* holds (or that the asker knows the target
  does not hold) is legal. It simply fails and passes the turn.
- A failed ask passes the turn to the player who was asked.

## Declaring
- Only the player whose turn it is may declare ("at any point on your turn").
- Any undeclared set may be declared, including one the declarer holds no card of.
- The assignment names one own-team seat per card (the declarer may name
  themself). It is not required to be consistent with the declarer's own hand;
  an inconsistent claim is simply wrong.
- Correct ⇔ every one of the six cards is held by exactly the named player. If an
  opponent holds any card of the set the declaration is necessarily wrong.
- Correct → declaring team scores the set. Wrong → the *other* team scores it.
  Either way the six cards leave the game.
- **After a declaration the declarer keeps the turn** (right or wrong). If they
  now hold no cards, the usual turn-passing rule below applies.
- **What a declaration reveals.** The declarer, set, claimed assignment and
  whether it was correct are public. After a wrong declaration the true holders
  are *not* announced; only how many cards each player lost is visible (hand
  sizes are public). This affects nothing else in the game because the set is gone.

## Turn passing with empty hands
- If the player who should move holds no cards, the turn goes to the next
  teammate clockwise who has cards (seat +2, then +4).
- If that whole team is out of cards, the turn goes to the next opponent
  clockwise who has cards (seat +1, +3, +5). That team now has nobody to ask, so
  its players must declare the remaining sets. The player on turn declares (any
  remaining set, including sets held by teammates) and keeps the turn until they
  run out of cards, then it passes to the next teammate clockwise.
- A player who holds cards always has a legal move: if no ask is legal (every set
  they hold is complete in their hand, or no opponent has cards) they can declare.

## End and reward
- The game ends when all nine sets are declared. Reward for a team = sets won −
  sets lost, so it lies in {−9, −7, …, 9}.

## Engine API notes
- `legal_actions(state, player)` returns the legal **asks** only (empty unless it
  is `player`'s turn). Declarations are too numerous to enumerate (up to 3⁶ per
  set); `declarable_sets` lists the sets that may be declared and
  `validate_action` / `step` check a concrete `Declare`.
- The arena stops a game after 3,000 actions and marks it `truncated` (remaining
  sets unscored). This is a safety net for random play.

## Agent conventions (not rules)
- Every agent declares immediately when its knowledge *proves* a declaration.
- A forced declaration without certainty is a best guess from the agent's belief.
