"""Centralised hyperparameter configuration for Step 2 training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """Return a linear learning-rate schedule compatible with SB3.

    Args:
        initial_value: Learning rate at the start of training (progress=1.0).
        final_value:   Learning rate at the end of training (progress=0.0).

    Returns:
        A callable ``schedule(progress_remaining) -> float``.
    """
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule


@dataclass
class TrainingConfig:
    """All hyperparameters and training knobs in one place.

    PPO parameters are forwarded directly to ``MaskablePPO`` via
    :meth:`to_dict`.  Reward-shaping weights and visualisation settings
    are consumed by wrappers and callbacks respectively.
    """

    # ------------------------------------------------------------------
    # PPO hyperparameters
    # ------------------------------------------------------------------
    learning_rate: float = 3e-4
    n_steps: int = 2048      # rollout buffer size (per env)
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.99
    clip_range: float = 0.2

    # ------------------------------------------------------------------
    # Reward-shaping weights
    # ------------------------------------------------------------------
    w_merge: float = 1.0    # weight for log2(score_gained) component
    w_empty: float = 0.1    # weight for empty-cells fraction
    w_mono: float = 0.3     # weight for monotonicity score
    w_invalid: float = -1.0  # penalty for an invalid (no-op) action

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    render_freq: int = 500      # render a game every N completed episodes
    render_delay_ms: int = 150  # delay (ms) between displayed steps

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    total_timesteps: int = 10_000_000
    n_envs: int = 8  # number of parallel environments

    def to_dict(self) -> dict:
        """Return the subset of fields that are valid MaskablePPO kwargs."""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "clip_range": self.clip_range,
        }


@dataclass
class TrainingConfigV2(TrainingConfig):
    """Extended configuration for training run v2, targeting higher tiles.

    Fixes four issues observed after 10M steps of training:
      1. Collapsed entropy  → ``ent_coef=0.01`` forces ongoing exploration.
      2. Flat merge reward  → super-linear ``log2(score)^2`` creates steeper gradient.
      3. Weak critic        → wider value-function head ``[256, 256]``.
      4. Aggressive updates → linear LR decay from 3e-4 down to 5e-5.

    Backward-compatible: ``train.py --config v1`` still uses :class:`TrainingConfig`.
    """

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------
    ent_coef: float = 0.01

    # ------------------------------------------------------------------
    # Learning rate schedule
    # ------------------------------------------------------------------
    use_lr_schedule: bool = True
    lr_start: float = 3e-4
    lr_end: float = 5e-5

    # ------------------------------------------------------------------
    # Network architecture (actor / critic head sizes)
    # ------------------------------------------------------------------
    vf_arch: list = field(default_factory=lambda: [256, 256])
    pi_arch: list = field(default_factory=lambda: [128, 128])

    # ------------------------------------------------------------------
    # Reward shaping v2
    # ------------------------------------------------------------------
    use_superlinear_merge_reward: bool = True
    w_merge: float = 1.0
    w_mono: float = 0.5       # was 0.3 in v1
    w_empty: float = 0.1
    w_invalid: float = -1.0
    w_survival: float = 0.01  # bonus per valid step

    def to_dict(self) -> dict:
        """Return MaskablePPO kwargs, including ent_coef."""
        d = super().to_dict()
        d["ent_coef"] = self.ent_coef
        return d


@dataclass
class TrainingConfigV3(TrainingConfigV2):
    """V3 configuration: reward normalisation and tuned PPO hyperparameters.

    Key changes over V2 (motivated by TensorBoard analysis of 3M-step runs):
      1. VecNormalize on rewards — fixes value_loss explosion caused by
         superlinear rewards spanning 0–121.
      2. n_epochs 10→4    — reduces approx_kl (policy was overfitting per batch).
      3. n_steps 2048→4096 — larger rollout for better gradient estimates.
      4. gae_lambda 0.95→0.9 — lower variance advantage estimates.
      5. No LR schedule  — stable LR at 2.5e-4 (schedule decayed too fast for 3M).
    """

    # ------------------------------------------------------------------
    # PPO hyperparameters (overrides)
    # ------------------------------------------------------------------
    learning_rate: float = 2.5e-4
    n_steps: int = 4096
    n_epochs: int = 4
    gae_lambda: float = 0.9
    max_grad_norm: float = 0.5
    ent_coef: float = 0.01

    # LR schedule: disabled for V3
    use_lr_schedule: bool = False

    # ------------------------------------------------------------------
    # Reward shaping (overrides)
    # ------------------------------------------------------------------
    w_survival: float = 0.005  # halved from v2's 0.01
    w_mono: float = 0.4        # between v1 (0.3) and v2 (0.5)

    # ------------------------------------------------------------------
    # VecNormalize settings (consumed by train.py, NOT passed to PPO)
    # ------------------------------------------------------------------
    use_vec_normalize: bool = True
    vec_norm_obs: bool = False      # obs already one-hot [0, 1]
    vec_norm_reward: bool = True
    vec_clip_reward: float = 10.0

    def to_dict(self) -> dict:
        """Return MaskablePPO kwargs, adding gae_lambda and max_grad_norm."""
        d = super().to_dict()
        d["gae_lambda"] = self.gae_lambda
        d["max_grad_norm"] = self.max_grad_norm
        return d
