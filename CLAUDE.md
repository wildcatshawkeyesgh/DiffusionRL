# TreeDiff-RL — Implementation Plan

## Project Overview

Implement **TreeDiff-RL** for Othello 6×6: a diffusion policy where MCTS searches *inside* the denoising process rather than over game moves. The core innovation is that each MCTS node is a partially denoised action latent `z_t`, and the tree explores alternative noise trajectories to find the best final action.

Reference: `treediff_rl_implementation_spec.md` — read this before touching any code.

---

## What We're Building

```
Phase 1 (implement first):
    z_T → z_{T-1} → z_{T-2} → ... → z_0 → action
    (straight-line denoising, no branching — policy trained via self-play, not behavior cloning)

Phase 2 (add later):
    z_T → [MCTS explores multiple denoising paths] → z_0 → action
    (tree search OVER the denoising trajectory)
```

The search tree is built over **diffusion timesteps**, not game turns. MCTS selects which noise trajectory to follow. The verifier evaluates partially denoised latents without completing the full denoising.

### v2 Architecture Changes (from v1)
- **No behavior cloning.** Training signal comes from game outcomes, not imitating expert moves. Loss is standard diffusion MSE weighted by `(outcome + 1) / 2` — winning actions get stronger gradient.
- **Random game bootstrap.** Initial data: thousands of random self-play games. No heuristics, no game-specific knowledge. Takes seconds, no GPU needed.
- **MCTS deferred.** Diffusion policy with straight-line denoising ships first. MCTS is layered on without touching the policy or verifier internals.
- **Clean class separation.** Two primary classes — `DiffusionPolicy` (owns denoiser + noise schedule, knows nothing about rewards) and `SelfPlayTrainer` (owns policy + verifier + replay buffer, knows nothing about diffusion internals). MCTS sits between them at inference time.

### Two Primary Classes

```
DiffusionPolicy
├── Owns: denoiser network, noise schedule
├── Knows: how to turn noise into action logits
├── Does NOT know: rewards, game outcomes, value estimation
└── Key method: select_action(board_state, valid_mask) → action

SelfPlayTrainer
├── Owns: DiffusionPolicy (as a member), Verifier, replay buffer
├── Knows: game loop, training logic, self-improvement cycle
├── Does NOT know: internal diffusion mechanics
└── Key method: train(num_iterations) → improved policy
```

---

## Directory Structure

```
diffusion/
├── CLAUDE.md                          # This file
├── treediff_rl_implementation_spec.md # Reference architecture doc
├── config.yaml                        # All hyperparameters — edit this, not code
├── utils.py                           # pick_device(), get_best_gpu()
├── train.py                           # Entry point: construct → train → save → evaluation
├── requirements.txt
├── othello/
│   └── env.py                         # OthelloEnv: reset/step/outcome, (3,8,8) state
├── models/
│   ├── denoiser.py                    # MLPDenoiser: (z_t, t, board_state) → noise_pred
│   ├── diffusion.py                   # DiffusionPolicy: forward_diffuse + denoise_step + select_action
│   └── verifier.py                    # DualSpaceVerifier: (z_t, board_state, t) → value [-1,1]
├── mcts/
│   ├── node.py                        # MCTSNode with PUCT scoring
│   └── tree_search.py                 # TreeDiffMCTS: MCTS over denoising trajectories
├── training/
│   ├── trainer.py                     # SelfPlayTrainer: train/save/evaluation, injected deps
│   └── replay_buffer.py               # ReplayBuffer with state_dict for checkpointing
└── checkpoints/                       # Saved model weights
```

---

## Implementation Phases

### Phase 1 — Othello Environment + Baselines

**Goal:** Working game environment with the minimal interface, random and greedy baselines, evaluation harness.

**Files to create:**
- `othello/env.py` — `OthelloEnv` implementing the minimal game interface:
  - `reset() → state`
  - `get_valid_moves() → list[int]`
  - `step(action) → next_state`
  - `done → bool`
  - `get_outcome() → float` (+1 win, -1 loss, 0 draw)
  - Board state shape: `(3, 6, 6)` — my stones, opp stones, valid mask
- `othello/test_env.py` — edge cases: no valid moves (pass), game-ending boards, flip logic
- `experiments/baseline.py` — `RandomPolicy` and `GreedyPolicy` (maximize immediate disc count)
- `experiments/eval.py` — `evaluate(policy_a, policy_b, num_games=200) → win_rate`

**Acceptance criteria:**
- Greedy beats random ≥ 65% over 200 games
- Board state tensor shape is `(3, 6, 6)` throughout — never flattened before the CNN encoder

---

### Phase 2 — DiffusionPolicy (Straight-Line Denoising, No MCTS)

**Goal:** Build `DiffusionPolicy` as a standalone class. No RL dependencies, no training logic inside it.

**Files to create:**
- `config.yaml` — all hyperparameters (YAML, not Python)
- `models/denoiser.py` — `MLPDenoiser(action_dim, state_dim, hidden_dim)`:
  - Input: `z_t` (36,) + time embedding + board state encoding (flattened CNN features)
  - Output: predicted noise (36,)
  - Time embedding: sinusoidal or learned `nn.Embedding(T, hidden_dim)`
- `models/diffusion.py` — `DiffusionPolicy`:
  - `forward_diffuse(z_0, t) → z_t, noise` (add scheduled noise — used during training)
  - `denoise_step(z_t, t, board_state) → z_{t-1}` (single step — called by MCTS later)
  - `select_action(board_state, valid_mask) → action` (full straight-line denoising pass)
  - Linear beta schedule: `beta_t = linspace(1e-4, 0.02, T)`
  - **Does NOT know about rewards, game outcomes, or the trainer**

**Acceptance criteria:**
- `select_action` runs in ≤ 100ms (T=50 steps, no MCTS)
- `forward_diffuse` then `denoise_step × T` recovers approximately the original `z_0`

---

### Phase 3 — DualSpaceVerifier

**Goal:** Build the value prediction network. Train it alongside the policy from the start.

**Files to create:**
- `models/verifier.py` — `DualSpaceVerifier(state_dim, action_dim, hidden_dim=256)`:
  - `state_encoder`: Conv2d(3→32) → Conv2d(32→64) → Flatten → Linear → hidden_dim
  - `latent_encoder`: Linear(action_dim → hidden_dim) → ReLU → Linear
  - `time_embed`: `nn.Embedding(T, hidden_dim)` (handles all noise levels)
  - `fusion`: Linear(hidden_dim × 3 → hidden_dim) → ReLU → Linear
  - `value_head`: Linear(hidden_dim → 1) → Tanh (output in [-1, 1])
  - `forward(z_t, board_state, t) → value`

**Acceptance criteria:**
- Forward pass runs without error for all `t` in `[1, T]`
- Output is always in `[-1, 1]` due to Tanh

---

### Phase 4 — SelfPlayTrainer (Bootstrap + Self-Improvement Loop)

**Goal:** The full training pipeline. Random bootstrap → alternating verifier/policy training → self-play data collection.

**Files to create:**
- `training/replay_buffer.py` — `ReplayBuffer`:
  - `add(states, actions, outcome)` — stores full game trajectories
  - `sample_batches(weight_by_outcome=False)` — outcome-weighted sampling when flag is True
- `training/trainer.py` — `SelfPlayTrainer`:
  - Owns `DiffusionPolicy`, `DualSpaceVerifier`, `ReplayBuffer` as members
  - `generate_random_games(num_games)` — bootstrap with purely random play
  - `play_game(policy='learned'|'random') → (states, actions, outcome)`
  - `train_verifier()` — MSE loss on `(z_t, board_state, t) → outcome` tuples, `t` sampled uniformly across `[1, T]`
  - `train_policy()` — weighted MSE on noise prediction: `weights = (outcome + 1) / 2`
  - `train(num_iterations)` — bootstrap → repeat: train_verifier → train_policy → play new games

**Training loop (no behavior cloning):**
```
1. Play N random games → fill replay buffer
   Repeat:
2. Train verifier: MSE( verifier(z_t, state, t), outcome )
3. Train policy: weighted MSE( denoiser(z_t, t, state), noise ) weighted by (outcome+1)/2
4. Play new games with current policy → add to replay buffer
```

**Acceptance criteria:**
- Policy win rate vs random improves across training iterations
- Verifier MSE < 0.25 on held-out games after full training

---

### Phase 5 — Validate Scaling

**Goal:** Confirm the policy improves with training. Establish baseline numbers before MCTS.

**Files to create/modify:**
- `experiments/eval.py` — add training curve logging (win rate vs iteration)
- Run evaluation: policy vs random, policy vs greedy at multiple checkpoints

**Acceptance criteria:**
- Win rate vs random increases monotonically with training iterations
- By end of training, policy beats random ≥ 70% over 200 games

---

### Phase 6 — MCTS Over Denoising Paths

**Goal:** Add tree search inside `select_action`. Calls into `DiffusionPolicy` and `Verifier` with no changes to either class.

**Files to create:**
- `models/mcts.py`:
  - `MCTSNode(z, t, parent)` — `.visits`, `.value_sum`, `.value` property, `.children`
  - `puct_score(node, child, c=1.5)` — PUCT formula from AlphaZero
  - `select(root) → leaf` — descend tree via PUCT
  - `macro_step_denoise(z_t, t, k, board_state) → z_{t-k}` — denoise k steps with fresh noise samples
  - `backpropagate(node, value)` — walk to root updating visits and value_sum
  - `TreeDiffPolicy(diffusion_policy, verifier, num_iterations)`:
    - `select_action(board_state, valid_mask) → action`
    - MCTS loop: select → expand (macro_step) → evaluate with verifier → backprop
    - Terminal nodes: `t == 0`, extract action from `z_0` logits

**Acceptance criteria:**
- MCTS (32 iterations) beats straight-line diffusion policy ≥ 55% over 100 games
- Tree structure correct: root `t=T`, leaves `t=0` or `t < k`
- No shared state between iterations (fresh noise per expansion)

---

### Phase 7 — Dual-Space Denoising + Inference-Time Scaling

**Goal:** Add validity correction during MCTS expansion. Run the key scaling experiment.

**Files to create/modify:**
- `models/diffusion.py` — add `denoise_with_correction(z_t, t, board_state, n=4, m=1)`:
  - Denoise n steps in latent space
  - Decode to discrete action, check validity
  - If invalid: mask invalid moves, pick argmax of valid
  - Re-encode corrected action → guidance vector `(z_corrected - z_latent) / sigma²`
- `models/diffusion.py` — add `guided_denoise_step(z_t, t, guidance, h=1.0)`:
  - Standard denoise + `h * beta_t * guidance`
- `experiments/scaling.py` — sweep MCTS iterations `[1, 4, 8, 16, 32, 64, 128]`:
  - For each count, evaluate 200 games vs fixed greedy baseline
  - Plot win rate vs compute (this is the key result)

**Acceptance criteria:**
- Win rate increases monotonically with MCTS iterations (the scaling plot)
- Dual-space correction reduces invalid move rate vs uncorrected version
- At 64 iterations, TreeDiff-RL beats diffusion-only policy by ≥ 10pp

---

## Hyperparameters (from Architecture Doc v2)

All hyperparameters live in `config.yaml` at the project root. Edit that file to change any value — no Python changes needed. `train.py` reads it at startup via `yaml.safe_load`. Key fields: `T`, `hidden_dim`, `num_bootstrap_games`, `num_iterations`, `lr`, `batch_size`, `mcts_iterations`, `macro_step_k`, `puct_c`.

---

## Dependencies

```
torch>=2.0
numpy
```

No heavy deps. No sentence-transformers, no HuggingFace. Pure PyTorch + numpy. Keep it lean — the whole project should run on CPU if needed.

---

## Key Implementation Notes

### Class Boundary — Do Not Cross
`DiffusionPolicy` must never import from `training/` or reference rewards/outcomes. `SelfPlayTrainer` must never call diffusion internals directly — it goes through `policy.forward_diffuse()` and `policy.denoise_step()` only. This separation is what lets MCTS slot in at inference time without refactoring.

### Board State Encoding
Always use `(3, 6, 6)` shape:
- Channel 0: current player's stones
- Channel 1: opponent's stones
- Channel 2: valid moves mask

Never flatten before the CNN encoder in the verifier. The Conv2d layers need the spatial structure.

### Action Encoding
Actions are integers 0–35 (flattened 6×6 grid):
- `z_0 = one_hot(action, num_classes=36).float()` — clean latent
- Forward diffuse to `z_t` for training the denoiser

### Training Signal — Outcome-Weighted, Not Behavior Cloning
The denoiser loss is standard diffusion MSE, but each sample is weighted by `(outcome + 1) / 2` (maps `[-1, 1]` to `[0, 1]`). Winning game actions get weight ~1, losing actions get weight ~0. This is the full training signal — there is no separate behavior cloning phase.

### Noise Schedule
Use linear beta schedule. DDPM-style:
```
alpha_t = 1 - beta_t
alpha_bar_t = cumprod(alpha_t)
z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * noise
```

### MCTS Branching
Each expansion samples **fresh Gaussian noise** to create a new denoising path. The branching factor is effectively infinite — MCTS doesn't enumerate all children, it samples new ones each iteration. Expansion always creates exactly one new child node.

### Verifier Training — Critical Detail
Sample `t` uniformly across `[1, T]` when creating training pairs. The verifier must learn to handle noisy latents at all noise levels, not just clean `z_0`. Without this, it will only work at the end of denoising.

---

## Evaluation Protocol

All evaluation: 200 games, alternating colors, report win rate excluding draws.

| Comparison | Expected Result |
|-----------|----------------|
| Random vs Greedy | Greedy ≥ 65% |
| Random vs Diffusion (self-play trained) | Diffusion ≥ 70% |
| Diffusion vs MCTS+Verifier (32 iter) | MCTS ≥ 55% |
| MCTS 8 iter vs MCTS 64 iter | 64 iter ≥ 55% |

The last comparison is the **key result** — inference-time scaling.

Note: there is no "Greedy vs Diffusion (BC)" comparison — v2 does not use behavior cloning. The diffusion policy is trained via the self-improvement loop on random game bootstrap data.

---

## Session Log

### 2026-04-13

**What was done:**
- Read `treediff_rl_architecture.md` in full
- Created this `CLAUDE.md` with phased implementation plan
- No code written yet

**Starting point:** empty directory, only the architecture reference doc exists

**Pick up next:** Phase 1 — Othello environment (`othello/env.py`) + baselines

---

### 2026-04-14

**What was done:**
- Read `treediff_rl_architecture_v2.md` in full
- Updated `CLAUDE.md` to reflect v2 architecture throughout:
  - Updated reference doc pointer to v2
  - Added "v2 Changes" summary and two-class design (`DiffusionPolicy` / `SelfPlayTrainer`) to "What We're Building"
  - Updated directory structure (removed `collect.py`, `train_diffusion.py`, `train_verifier.py`; added `trainer.py`)
  - Rewrote all 5 implementation phases → 7 phases matching v2 implementation order
  - Updated hyperparameters config (new params: `num_bootstrap_games`, `games_per_iteration`, `num_iterations`; removed `behavior_clone_steps`, `verifier_train_steps`, `num_games`)
  - Added "Class Boundary" and "Training Signal" notes to Key Implementation Notes
  - Updated evaluation protocol (no BC comparison; note explains why)
- No code written yet

**Key v2 changes to keep in mind:**
- No behavior cloning — training signal is outcome-weighted noise MSE
- `DiffusionPolicy` knows nothing about rewards; `SelfPlayTrainer` knows nothing about diffusion internals
- Bootstrap from random games (not greedy), then self-improve
- Verifier trained alongside policy from the start, not as a separate later phase

**Pick up next:** Phase 1 — Othello environment (`othello/env.py`) + baselines (`experiments/baseline.py`, `experiments/eval.py`)

---

### 2026-04-16

**What was done:**
- Reworked architecture to mirror hw03wildcatshawkeyes patterns:
  - `SelfPlayTrainer` now receives all dependencies injected (policy, verifier, mcts, optimizers, buffer) — no internal construction
  - `save_onnx()` renamed to `save()` matching hw03's `ClassTrainer.save()` pattern
  - Added `evaluation(plot=False)` method — prints win rates + training summary to terminal; optional plots saved as `loss_curve.png` / `winrate_curve.png` when `plot=True`
  - Added per-iteration loss and win-rate history (`loss_history_policy`, `loss_history_verifier`, `winrate_history`) tracked in trainer and saved/restored via checkpoints
- Switched config from `config.py` Python dict → `config.yaml`; `train.py` loads it via `yaml.safe_load`; `config.py` deleted
- Created `utils.py` with `get_best_gpu()` (reused from hw03) and `pick_device()` wrapper (MPS → best CUDA → CPU)
- Fixed checkpoint buffer bug: `save_checkpoint` now serializes last 50k buffer entries; `load_checkpoint` restores them; `bootstrap_more(n)` added for appending more random games at any time
- Deleted `__main__` test blocks from `models/denoiser.py`, `models/diffusion.py`, `models/verifier.py`
- Deleted all `.DS_Store` files
- Updated `CLAUDE.md`: fixed stale doc references, updated directory structure, replaced Python config dict with YAML note

**Later in the day — import-style refactor to match hw03 exactly:**
- Converted the package layout from flat modules to `src/diffusion/<submodule>/` with a `pyproject.toml` and editable install (`pip install -e .`). Submodules: `models`, `mcts`, `rl`, `othello`, plus `utils.py` at the package root.
- Every module declares `__all__` (`MLPDenoiser`, `DiffusionPolicy`, `DualSpaceVerifier`, `MCTSNode`, `TreeDiffMCTS`, `ReplayBuffer`, `SelfPlayTrainer`, `OthelloEnv`, `get_best_gpu`, `pick_device`).
- Each submodule `__init__.py` cascades star-imports from its files (e.g., `models/__init__.py` → `from .denoiser import *` / `from .diffusion import *` / `from .verifier import *`). Top-level `diffusion/__init__.py` star-imports all submodules.
- Internal cross-module imports rewritten to relative (`from .denoiser import MLPDenoiser`, `from ..othello import OthelloEnv`).
- `scripts/train.py` now mirrors hw03's `scripts/imagenet_impl.py` style: `from diffusion import models, rl, mcts, utils` then namespace access — `models.DiffusionPolicy(cfg)`, `rl.SelfPlayTrainer(...)`, `mcts.TreeDiffMCTS(...)`, `utils.pick_device()`. Local variable renamed from `mcts` → `tree` to avoid colliding with the submodule name.
- Smoke-tested: both access paths work — `from diffusion import models; models.DiffusionPolicy` and flat `from diffusion import DiffusionPolicy` (via `__all__` cascade).

**Pick up next:** continue Phase implementation — Othello baselines and eval harness
