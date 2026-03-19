# 2048-RL

A deep reinforcement learning agent that learns to play the 2048 game using
**MaskablePPO** (Proximal Policy Optimization with action masking).

The agent uses a custom CNN feature extractor, superlinear reward shaping,
and reward normalization via VecNormalize. After 30M training steps, it
**consistently reaches the 2048 tile** and has achieved the **4096 tile**,
with an average score of ~27,500.

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Play as a human
python play.py

# Train the agent (best config)
python training/train.py --config v3

# Watch the agent play
python watch_agent.py --model models/ppo_2048_v3_final.zip --config v3

# Monitor training metrics
tensorboard --logdir ./logs
```

## Tech Stack

| Tool | Role |
|------|------|
| Python 3.10+ | Language |
| Gymnasium | RL environment interface |
| stable-baselines3 + sb3-contrib | MaskablePPO implementation |
| PyTorch | Deep learning backend |
| NumPy | Board logic |
| Pygame | UI and live training visualization |
| TensorBoard | Training metrics |
| pytest | Unit tests (117 tests) |

Device auto-selection: Apple Silicon MPS > CUDA > CPU.

## Project Structure

```
2048-rl/
├── env/
│   ├── board.py               # Pure game logic (Board class)
│   ├── game2048_env.py        # Gymnasium environment wrapper
│   └── constants.py           # Action IDs, board size, tile probabilities
│
├── rl/
│   ├── preprocessing.py       # One-hot encoding: (4,4) -> (16,4,4)
│   ├── reward.py              # Reward shaping (merge, empty, monotonicity)
│   ├── wrappers.py            # Gymnasium wrappers (3 chained)
│   ├── network.py             # CNN2048FeaturesExtractor
│   └── callbacks.py           # RenderCallback, MetricsCallback
│
├── training/
│   ├── config.py              # TrainingConfig / V2 / V3 dataclasses
│   ├── train.py               # Main training script (--config v1|v2|v3)
│   └── compare_runs.py        # Launch sequential comparison runs
│
├── ui/
│   └── pygame_ui.py           # Human-playable Pygame UI
│
├── play.py                    # Entry point: play as human
├── watch_agent.py             # Entry point: watch trained agent
│
├── tests/                     # Unit tests
│   ├── test_board.py
│   ├── test_env.py
│   ├── test_preprocessing.py
│   └── test_reward.py
│
├── models/                    # Saved models (.zip + VecNormalize stats)
└── logs/                      # TensorBoard event files
```

## Architecture

### Wrapper Stack

```
Game2048Env                    # Raw (4,4) board, score reward
  -> RewardShapingWrapper      # Shaped reward (merge + empty + mono + survival)
  -> PreprocessingWrapper      # (4,4) int board -> (16,4,4) one-hot float32
  -> ActionMaskWrapper         # Exposes action_masks() for MaskablePPO
```

### Observation Encoding

Raw board values (0, 2, 4, ..., 2048) are encoded as a `(16, 4, 4)` one-hot
tensor. Each tile value maps to channel `log2(value)`: empty cells go to
channel 0, tile 2 to channel 1, tile 2048 to channel 11, etc.

### CNN Feature Extractor

```
Input (16, 4, 4) one-hot
  -> Conv2d(16, 128, kernel=2, padding=1) + ReLU    # (128, 5, 5)
  -> Conv2d(128, 128, kernel=2) + ReLU              # (128, 4, 4)
  -> Flatten                                         # 2048
  -> Linear(2048, 256) + ReLU                        # 256-dim features
```

Actor head: `[128, 128] -> 4 actions`
Critic head: `[256, 256] -> 1 value`

### Reward Shaping

```
reward = w_merge * r_merge + w_empty * r_empty + w_mono * r_mono + r_survival
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| r_merge | `log2(score_gained)^2` | Incentivize large merges (superlinear) |
| r_empty | empty_cells / 16 | Keep the board open |
| r_mono | monotonicity_score | Encourage ordered tile layout |
| r_survival | constant bonus | Reward longer games |

Invalid moves receive a flat penalty (`w_invalid = -1.0`).

## Training Configurations

Three configurations were developed iteratively:

| Parameter | V1 (baseline) | V2 (improved) | V3 (best) |
|-----------|:---:|:---:|:---:|
| Learning rate | 3e-4 | 3e-4 -> 5e-5 (schedule) | 2.5e-4 (fixed) |
| n_steps | 2048 | 2048 | 4096 |
| n_epochs | 10 | 10 | 4 |
| gae_lambda | 0.95 | 0.95 | 0.9 |
| ent_coef | 0.0 | 0.01 | 0.01 |
| Merge reward | `log2(s)` | `log2(s)^2` | `log2(s)^2` |
| VecNormalize | No | No | **Yes** |
| vf_arch | default | [256, 256] | [256, 256] |

### Why V3 works

The superlinear reward `log2(score)^2` creates values ranging from 0 to 121.
Without normalization, the critic's value_loss explodes (2,153 in V2 vs
**0.03** in V3). VecNormalize rescales rewards to zero-mean, unit-variance
using running statistics, making the critic's learning problem tractable.

## Results

### Comparison Runs (3M steps each)

| Run | Config | score_mean_100 | max_tile | value_loss | explained_var |
|-----|--------|---------------:|---------:|-----------:|--------------:|
| A - entropy | V2 | 3,634 | 256-512 | 137 | 0.70 |
| B - reward | V2 | 3,735 | 256-512 | 2,153 | 0.68 |
| C - global | V2 | ~2,500 | 128-256 | ~1,000 | 0.72 |
| **D - v3 baseline** | **V3** | **9,474** | **512-709** | **0.03** | **0.86** |
| E - no vecnorm | V3 | 2,533 | ~267 | 2,044 | 0.78 |
| F - ent 0.02 | V3 | ~4,500 | 256-512 | low | 0.79 |

VecNormalize alone (D vs E) accounts for a **3.7x score improvement** and a
**68,000x reduction in value_loss**.

### Full V3 Run (3M -> 10M -> 30M steps)

| Metric | At 3M | At 10M | At 30M |
|--------|------:|-------:|-------:|
| score_mean_100 | 9,474 | 15,892 | **27,461** |
| max_tile | 512-709 | 846-1024 | **2,030-4096** |
| ep_len_mean | ~598 | ~923 | **~1,456** |
| value_loss | 0.03 | 0.024 | **0.0185** |
| explained_variance | 0.86 | 0.87 | **0.9025** |

At 30M steps the agent **consistently reaches 2048** and has achieved the
**4096 tile** (score 60,496 in a single game). No plateau observed - all
metrics are still improving.

## Commands

```bash
# Play as human
python play.py

# Train (v1/v2/v3)
python training/train.py --config v3

# Compare multiple configs (3M steps each)
python training/compare_runs.py

# Watch trained agent
python watch_agent.py --model models/ppo_2048_v3_final.zip --config v3 --episodes 10 --speed 200

# TensorBoard
tensorboard --logdir ./logs

# Tests
pytest tests/ -v
```

## Tests

117 unit tests covering board logic, Gymnasium environment, observation
encoding, reward shaping (linear, superlinear, survival bonus), config
inheritance, and learning rate schedules.

```bash
pytest tests/ -v
```
