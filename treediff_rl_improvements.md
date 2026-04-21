# TreeDiff-RL Improvement Plan

## Context

Current state: diffusion policy + verifier + MCTS all integrated and training on 8x8 Othello. Win rate vs random fluctuates between 50-60%. Loss drops during training but win rate plateaus. The core issue is likely **credit assignment** — every move in a winning game gets equal positive weight, and losing games contribute no signal at all.

This document covers two categories of changes: diagnostics (to understand what's happening) and training improvements (to fix it). All changes are against the existing codebase: `trainer.py`, `diffusion.py`, `replay_buffer.py`, `denoiser.py`, `verifier.py`, `tree_search.py`, `node.py`.

---

## Part 1: Diagnostics

### 1.1 Action Entropy Tracking

**What:** Log the entropy of the policy's output distribution during each evaluation round.

**Why:** Determines whether the model is collapsing to a few favorite moves (low entropy) or playing diversely but badly (high entropy). These are different failure modes requiring different fixes.

**Where to add:** Inside `evaluate_vs_random` in `trainer.py`.

**Implementation details:**
- After the diffusion policy produces final logits and applies the valid mask, compute softmax probabilities over valid moves.
- Compute entropy: `H = -sum(p * log(p))` over valid moves.
- Accumulate per-game average entropy across all eval games.
- Log alongside the existing W/D/L line: `entropy=X.XX`
- Also track min and max entropy across the eval batch to see the range.

**What to look for:**
- Entropy collapsing toward 0 over iterations → model is overconfident, may need temperature or exploration fixes.
- Entropy staying high (~log(num_valid_moves)) → model hasn't learned meaningful preferences, state encoder may not be contributing.
- Entropy in a healthy middle range but win rate flat → credit assignment problem, not an entropy problem.

---

### 1.2 Action Heatmap Logging

**What:** Accumulate action counts across all eval games and print/save a board-shaped 8x8 heatmap showing where the model likes to play.

**Why:** Reveals whether the model has learned positional preferences (corners, edges) or is spraying moves uniformly. A model that ignores corners and edges in Othello is missing fundamental strategy.

**Where to add:** Inside `evaluate_vs_random` in `trainer.py`.

**Implementation details:**
- Maintain a `numpy.zeros((8, 8))` counter array across eval games.
- Each time the agent picks an action, increment the corresponding cell: `row, col = divmod(action, 8)`.
- After all eval games, normalize by total moves and print as an 8x8 grid.
- Log every `eval_every` iterations so you can see how preferences evolve.
- Format the output as a simple text grid with 2-decimal percentages, like:
  ```
  Action heatmap (% of moves):
   0.5  1.2  0.3  0.1  0.1  0.3  1.1  0.6
   1.0  2.3  1.5  0.8  0.7  1.4  2.1  0.9
   ...
  ```

**What to look for:**
- Corners (0,0), (0,7), (7,0), (7,7) should eventually get high weight — these are the most valuable positions in Othello.
- If the heatmap is nearly uniform, the model isn't learning positional value.
- If a few cells dominate regardless of board state, the model is ignoring the state encoder.

---

### 1.3 Win Rate by Color

**What:** Track win rate separately for when the agent plays as player 0 (black, moves first) vs player 1 (white).

**Why:** Large asymmetry could indicate a bug in the current-player POV encoding, or that the model is only learning to play one side effectively.

**Where to add:** Inside `evaluate_vs_random` in `trainer.py`.

**Implementation details:**
- The existing eval loop already alternates: even games → agent is player 0, odd games → agent is player 1.
- Track `wins_as_p0`, `wins_as_p1`, `games_as_p0`, `games_as_p1` separately.
- Log: `vs_random W=55% (p0=60% p1=50%)`

**What to look for:**
- Moderate asymmetry is normal (first-mover advantage exists in Othello).
- Extreme asymmetry (e.g., 80% as p0, 30% as p1) suggests the state encoding or observation flipping has an issue.

---

## Part 2: Training Improvements

### 2.1 Per-Move Advantage Weighting

**What:** Replace the flat game-outcome weight with a per-move advantage estimate. Each move gets weighted by how much better (or worse) the outcome was compared to what random play would have achieved from that position.

**Why:** This is the single highest-impact change. Currently every move in a winning game gets weight 1.0 — including bad moves that happened to be in a game the agent won. Advantage weighting gives the model a clear signal about which specific moves mattered.

**Where to add:** `trainer.py` — new method for computing baselines, modifications to `play_game` and `train_step`.

**Implementation details:**

Phase 1 — Random rollout baseline:
- After each self-play game, for each (state, action) in the game, estimate a baseline value by playing N random rollouts from that state (e.g., N=16).
- `baseline[i] = mean(outcome of N random games from state[i])`
- `advantage[i] = actual_outcome[i] - baseline[i]`
- Store advantage in the replay buffer instead of (or alongside) raw outcome.

This is expensive (N extra random games per move), so:
- Start with N=8 or N=16. Random games are cheap — no model inference.
- Only compute baselines for self-play games, not bootstrap random games.
- Can be parallelized since random games are independent.

Phase 2 — Verifier baseline (later swap):
- Once the verifier is accurate, replace random rollouts with `baseline[i] = verifier(state[i])`.
- Much cheaper — single forward pass instead of N rollout games.
- The advantage computation stays the same: `advantage[i] = actual_outcome[i] - verifier_value[i]`.

Changes to `train_step`:
- Replace the current `weights = (outcomes + 1.0) / 2.0` with the advantage values directly.
- The loss becomes: `loss = (advantages * per_sample_loss).mean()`
- Advantages can be negative (this is what enables learning from bad moves — see 2.2).

**Replay buffer change:**
- Add an `advantages` field alongside `outcomes`.
- `add_game` takes an additional `advantages` list.
- `sample` returns `(states, actions, outcomes, advantages)`.
- Keep `outcomes` stored for verifier training (verifier still trains on raw outcomes).

---

### 2.2 Negative Signal from Losing Games

**What:** Use losing moves as signal to push the model away from bad actions, instead of ignoring them (current behavior: weight=0 for losses).

**Why:** Currently half the training data is discarded. A move that led to a loss from a good position is valuable — it tells the model what not to do.

**Where to add:** This is mostly handled by 2.1 (advantage weighting), but there's an interim step.

**Interim implementation (before advantage weighting is ready):**
- Change the weight mapping in `train_step` from `(outcome + 1) / 2` → use raw outcome directly: `weights = outcomes` (so +1 for wins, -1 for losses, 0 for draws).
- Negate the loss contribution for negative weights. Concretely:
  ```python
  per_sample_loss = ((eps_pred - noise) ** 2).mean(dim=-1)
  loss = (weights * per_sample_loss).mean()
  ```
  When `weights` is negative, this pushes gradients in the opposite direction — the model learns to denoise *away* from the noise patterns that produced losing moves.

**Once advantage weighting (2.1) is implemented:**
- This happens naturally. Negative advantage → negative weight → model moves away from that action.
- Remove the interim raw-outcome weighting and use advantages exclusively.

**Caution:** Monitor training stability after this change. Negative weights can cause gradient oscillation if the balance of positive/negative samples is uneven. If training becomes unstable, consider clipping advantages to `[-1, 1]` or using `tanh(advantage)`.

---

### 2.3 Temperature Sampling During Self-Play

**What:** Add a temperature parameter to `select_action` in `diffusion.py`. Use temperature > 0 during self-play data collection, keep argmax (temperature=0) during evaluation.

**Why:** Currently `select_action` uses argmax, so the model always plays its single favorite move from each position. It never explores alternatives, so training data is self-reinforcing — the model only sees its own top picks and keeps training on them.

**Where to add:** `diffusion.py` (`select_action`), `trainer.py` (`_learned_action` and `play_game`).

**Implementation details:**

In `diffusion.py`, modify `select_action`:
```python
def select_action(self, state_np, valid_mask_np, temperature=0.0):
    # ... existing denoising loop ...
    
    logits = z.squeeze(0).cpu()
    valid = torch.as_tensor(valid_mask_np, dtype=torch.bool)
    logits[~valid] = float('-inf')
    
    if temperature > 0:
        probs = F.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())
    else:
        return int(torch.argmax(logits).item())
```

In `trainer.py`:
- `_learned_action` gets a `temperature` parameter, passed through to `select_action`.
- Self-play games in the training loop use `temperature=1.0` (or configurable via config).
- `evaluate_vs_random` continues using `temperature=0.0`.
- Consider an AlphaZero-style schedule: temperature=1.0 for the first ~15 moves of each game, then drop to 0.0 for the rest. This encourages opening diversity without adding noise to endgame play.

**Config additions:**
- `selfplay_temperature`: float, default 1.0
- `temp_drop_move`: int, default 15 (move number after which temperature drops to 0)

**If MCTS is used for self-play**, the same idea applies — sample from the MCTS visit distribution proportionally (temperature=1) rather than always picking the most-visited child. This means modifying `tree_search.py`'s `select_action` to support temperature over the root's visit counts.

---

### 2.4 Cosine Learning Rate Schedule

**What:** Replace the constant learning rate with cosine annealing.

**Why:** Constant LR can cause instability once the model is past the easy early gains. Cosine decay reduces the LR smoothly, allowing fine-grained learning in later iterations without overshooting.

**Where to add:** `trainer.py`, in `__init__` and the training loop.

**Implementation details:**
- Use `torch.optim.lr_scheduler.CosineAnnealingLR`.
- Set `T_max` to `config['num_iterations'] * config['steps_per_iter']` (total training steps).
- Set `eta_min` to something small like `1e-6`.
- Call `scheduler.step()` after each `train_step()`, not after each iteration.
- Log the current LR periodically for debugging.

```python
# In __init__:
self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    self.optim,
    T_max=config['num_iterations'] * config['steps_per_iter'],
    eta_min=1e-6
)

# In train_step, after optim.step():
self.scheduler.step()
```

**Checkpoint change:** Save and load `scheduler.state_dict()` alongside optimizer state so LR resumes correctly.

---

## Implementation Order

The recommended order, with validation checkpoints:

1. **Diagnostics (1.1, 1.2, 1.3)** — Add all three, run a few iterations, use the output to confirm your current failure mode before changing training.

2. **Temperature sampling (2.3)** — Low-risk change, improves data diversity immediately. Validate by checking that action entropy (from 1.1) increases during self-play.

3. **Cosine LR (2.4)** — Simple scheduler addition. Should stabilize late-training loss.

4. **Negative signal from losses (2.2 interim)** — Switch from `(outcome+1)/2` to raw outcome weighting. Monitor for training instability. Check if win rate improves.

5. **Per-move advantage weighting (2.1)** — Biggest change. Implement random-rollout baseline first. This subsumes the interim negative-signal weighting from step 4. Validate by checking that the advantage distribution has meaningful variance (not all close to 0).

6. **Swap to verifier-based advantage (2.1 phase 2)** — Once the verifier's predictions correlate well with actual outcomes (check via a scatter plot of predicted vs actual during eval), replace random rollouts with verifier forward passes for baseline estimation.

---

## Validation Checkpoints

After each change, run at least 30-50 iterations and check:

- **Loss:** Should still decrease, but more slowly with cosine LR.
- **Win rate vs random:** Target 70%+ after diagnostics + training improvements. 85%+ after advantage weighting is working.
- **Action entropy:** Should be in a healthy middle range (not collapsed, not uniform).
- **Action heatmap:** Should show emerging corner/edge preferences over time.
- **Color asymmetry:** Should be moderate (<15% difference).
