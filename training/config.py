"""Centralised hyperparameter configuration for Step 2 training."""

from __future__ import annotations

from dataclasses import dataclass, field


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
