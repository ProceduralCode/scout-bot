# Scout Neural Network Bot — Design Document

## Overview

A neural network bot for the card game Scout (3-5 players). The bot uses an actor-critic architecture trained via self-play to learn strategies without explicit game knowledge or card counting.

## Goals

- See how good a bot can get at Scout through self-play
- Observe what strategies emerge
- No unfair advantages — the bot sees only what a casual human would track

---

## Architecture

### Single Shared Network

One network with shared hidden layers and multiple small readout heads.

```
Input (~460 floats)
    |
Hidden Layer 1 (size TBD, ~128-256)
    |
Hidden Layer 2 (size TBD, ~128)
    |
Hidden State [128]
    |
    +---> Value Head: hidden[128] → [1 output]
    |
    +---> Policy Heads: hidden[128] + conditioning[29] = [157 inputs]
              +---> Action Type Head → [9 outputs]
              +---> Play Start Head  → [20 outputs]
              +---> Play End Head    → [20 outputs]
              +---> Scout Insert Head → [21 outputs]
```

- The shared hidden layers compute a game state representation once per turn.
- Each readout head is a separate small `nn.Linear` module for code clarity.
- Policy readout heads take the hidden state plus conditioning from previous steps.
- The value head reads directly from the hidden state (no conditioning) — it evaluates the position, not the action.

### Readout Mechanism (Conditioned Cheap Readouts)

Instead of rerunning the full network for each decision step, the hidden state is computed once and reused. Each readout head takes:

```
hidden_state[128] + action_type[9] + first_index[20] = 157 inputs
```

- **Step 1 (action type):** Conditioning inputs are zeroed. Output: 9-way softmax.
- **Step 2 (play start):** Action type filled in, first_index zeroed. Output: 20-way softmax.
- **Step 3 (play end):** Both action type and first_index filled in. Output: 20-way softmax.
- **Step 2 (scout insert):** Action type filled in, first_index zeroed. Output: 21-way softmax.

Each step is a single matrix multiply (~3,000 multiply-adds). Negligible cost compared to the shared layer forward pass.

The **value head** is separate — it reads only the 128 hidden state with no conditioning and outputs a single float (`nn.Linear(128, 1)`). It evaluates the position, not the action.

### Hand Flip Decision (Round Start)

At the start of each round, each player may flip their entire hand. The network decides by running a forward pass for each orientation and comparing **value head outputs** — pick the orientation with higher estimated value. No new architecture needed. One extra forward pass per player per round.

### Scout-and-Show Handling

S&S is not a special action in the network. Instead:

1. The action type head picks the scout option (which encodes the end + orientation, e.g., "S&S left flipped"), then the scout insert head picks the insert position.
2. The hand is updated in the input array with the newly scouted card.
3. The network is rerun as a fresh turn with scout/S&S options masked — it must play.

This means the network never needs to learn S&S as a distinct concept. It just scouts, then plays. One extra full forward pass, but S&S happens at most once per round per player.

---

## Input Encoding

### Hand (~220 floats)

- **20 slots** (fixed size, accommodates max 15-card hand plus growth from scouting, with room for random placement offset).
- Each slot: **one-hot encoded, 11 values** (card values 1-10, plus empty).
- 20 x 11 = 220 floats.
- **Random placement:** During training, the hand is placed at a random offset within the 20 slots. Unused slots are zeroed (empty). This prevents positional bias and forces the network to learn position-agnostic card reading.

### Current Play (~220 floats)

- **10 slots** for cards in the current play.
- Each slot: **two one-hot encoded values** (both sides of the card), 11 values each = 22 per slot.
- 10 x 22 = 220 floats.
- Empty slots zeroed when fewer cards are in play, all zeroed when the table is empty (anything can be played).
- Both sides are encoded because the network needs to see alternate values when deciding scout orientation.

### Metadata (~30 floats)

- Your score (normalized, divided by ~20)
- Each opponent's score (up to 4, normalized) — zero-padded for fewer players
- Each opponent's hand size (up to 4, normalized)
- Your scout-and-show availability (binary)
- Each opponent's S&S availability (up to 4, binary)
- Player count (one-hot: 3/4/5 = 3 values)
- Scouts since current play (normalized: count / (num_players - 1)) — how close the round is to ending
- Relative position of play owner (one-hot, 4 slots: 1-4 seats away) — all zeros when table is empty
- Round number (normalized: round / total_rounds)

### Total Input: ~470 floats

---

## Output Encoding

### Step 1: Action Type (9 outputs)

```
0: Play
1: Scout left, normal orientation
2: Scout left, flipped orientation
3: Scout right, normal orientation
4: Scout right, flipped orientation
5: S&S left, normal orientation
6: S&S left, flipped orientation
7: S&S right, normal orientation
8: S&S right, flipped orientation
```

Illegal options masked before softmax.

### Step 2 (if play): Start Position (20 outputs)

One per hand slot. Illegal positions (empty slots, no valid play starting there) masked before softmax.

### Step 3 (if play): End Position (20 outputs)

One per hand slot. Masked so end >= start and the selected cards form a legal play (valid set or run that beats current play).

### Step 2 (if scout/S&S): Insert Position (21 outputs)

Where in the hand to insert the scouted card. 21 slots = 20 existing positions + 1 (can insert after the last card). Masked to valid insertion points.

### Value Head (1 output)

Single float predicting expected reward from the current position. Reads from hidden state only (`nn.Linear(128, 1)`), no conditioning — the value of a position doesn't depend on which action is being considered.

---

## Training

### Method

- **Actor-critic with advantage** (PPO or similar).
- Advantage = actual reward - value head's prediction. Per-move credit assignment rather than one signal for the whole game.

### Reward Signal

- **Score margin** vs the next-highest scoring player, plus a **win bonus of +5**.
- Normalized by dividing by ~20 to keep values in a manageable range for the value head.

### Self-Play

- Train by playing games against copies of itself.
- Maintain a **pool of older network versions** to play against, preventing overfitting to the current policy's style.

### Experience Replay

- Record games into a rolling replay buffer.
- Train multiple times on each game for better data efficiency.
- PPO's importance sampling handles the off-policy correction for stale data.
- **Prioritized replay** (weight surprising positions more heavily) as a later optimization.

### Network Sizing

- Start with 2 hidden layers, ~128-256 neurons each.
- Tune by trial and error: train, check if it plateaus quickly (too small) or trains fine.
- No automated hyperparameter search — manually try a few sizes and compare training curves.

---

## Information Available to the Bot

The bot sees only what a casual human player would notice:

- Its own hand (card values and positions)
- The current play on the table
- Each player's score
- Each player's remaining hand size
- Whether each player has used their scout-and-show this round

The bot does **NOT** see:

- What specific cards have been played previously
- What cards opponents scouted
- Discard/play history
- Opponents' hands

Note: since the bot has no play history, it cannot perform any form of card counting — implicit or explicit. It makes decisions purely from the current visible state.

---

## Tech Stack

- **Python** for everything (game engine, training, neural network).
- **PyTorch** for the neural network.
- Game logic is simple enough that Python is not the bottleneck — NN inference dominates training time.

---

## Scout Game Rules

### Components

- **45 double-sided cards.** Each card has two different values (1-10), one on each side. The deck contains every unique pair: 1/2, 1/3, ..., 1/10, 2/3, ..., 9/10. That's C(10,2) = 45 cards.
- Each value 1-10 appears on exactly 9 cards (paired with each of the other 9 values).
- Scout tokens (given to players when opponents scout from their play).
- One Scout & Show token per player (flipped after use, reset each round).

### Setup by Player Count

- **3 players:** Remove all cards with a 10 on either side (9 cards removed → 36 cards). Deal 12 each.
- **4 players:** Remove only the 9/10 card (1 card removed → 44 cards). Deal 11 each.
- **5 players:** Use all 45 cards. Deal 9 each.

### Round Setup

1. Shuffle and deal cards equally.
2. Players pick up their hand as a fan **without rearranging the order**.
3. Each player may **flip their entire hand once** — this reverses the card order and shows the other side's values. This is the only time card arrangement can change.
4. Each player gets their S&S token back (available once per round).
5. Starting player rotates between rounds.

### Turn Actions (choose one)

1. **Show (Play):** Play a group of consecutive cards from your hand onto the table.
   - Must beat the current play (or play anything if the table is empty).
   - Collect the previous play's cards face-down as points.
   - The player who made the previous play keeps any scout tokens already earned from it.

2. **Scout:** Take one card from **either end** of the current play.
   - Insert it **anywhere** in your hand, in **either orientation** (choose which side faces up).
   - The player whose play is on the table receives a scout token (+1 point).
   - The current play shrinks — remaining cards stay as the active play, and the next player must beat the **reduced** play.

3. **Scout & Show:** Scout a card, then immediately play. Once per round. Uses your S&S token.

### Legal Plays

Cards played must be **consecutive in your hand** and form one of:
- **Single card** (any value).
- **Set:** Two or more cards of the **same value** (e.g., 8-8-8).
- **Run:** Two or more cards in **strictly ascending or descending consecutive order** (e.g., 3-4-5 or 7-6-5). Mixed directions (e.g., 1-3-2) are not valid.

### Beat Hierarchy

A play beats the current play if:

1. **More cards** always beats fewer cards, regardless of type.
2. **Same card count, different type:** A set (matching values) beats a run (consecutive values).
3. **Same card count, same type:** The play with the **higher values wins.** For sets: higher set value (e.g., 8-8 beats 5-5). For runs: the run with the higher cards (e.g., 4-5-6 beats 3-4-5).
4. **Identical plays** (same type, same length, same values) **cannot be played.**

### Round End

A round ends when either:
- A player **empties their hand** (plays their last cards), OR
- A play goes **unbeaten** — every other player scouted (or couldn't play) instead of beating it.

### Scoring (per round)

- **+1 point** per card in your face-down collected pile (cards taken from the table by beating plays).
- **+1 point** per scout token received (from opponents scouting your plays).
- **-1 point** per card remaining in your hand.
- **Exception:** The player who **ended the round** (emptied hand or made the unbeaten play) does **not** lose points for remaining cards.

When the round ends due to an unbeaten play, that player does **not** collect their own play's cards — those cards are discarded.

### Game End

- Play **as many rounds as there are players** (3 rounds for 3 players, etc.).
- Highest total score wins.
- Ties are shared victories.

---

## Code Structure

```
Scout/
  game.py       — Game state, legal move masking, move application, scoring
  encoding.py   — Game state → tensor, network output → move
  network.py    — PyTorch model definition
  training.py   — Self-play, experience collection, PPO
  main.py       — Entry point, config, training orchestration
```

- Game engine is player-agnostic — accepts moves from network or human input.
- Legality is computed as **masks** per output step, not full move enumeration.
- Parallel games: maintain a batch of N game states, collect decisions, batch forward pass.
- Cards represented as `(showing_value, hidden_value)` tuples. Hand is an ordered list.

---

## Open Questions

1. **Hidden layer sizes** — Starting guess of 2 layers x 128-256 neurons, but needs empirical tuning.
2. **PPO hyperparameters** — Learning rate, batch size, clipping, entropy bonus. Standard defaults to start, tune from there.
3. **Replay buffer size** — How many games to store, how many times to replay each.
4. **Opponent pool management** — How often to snapshot the network, how many old versions to keep, how to sample opponents from the pool.
5. **Auxiliary prediction heads** — Opponent move prediction, round outcome prediction, etc. Could improve training signal. Worth experimenting with but not core.
