"""Main training script: MaskablePPO + CNN extractor + live Pygame render."""

from __future__ import annotations

import os
import sys

# Add the project root to sys.path so that ``env``, ``rl``, ``training``
# are importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

from env.game2048_env import Game2048Env
from rl.network import CNN2048FeaturesExtractor
from rl.wrappers import ActionMaskWrapper, PreprocessingWrapper, RewardShapingWrapper
from rl.callbacks import MetricsCallback, RenderCallback
from training.config import TrainingConfig


def make_env_fn(config: TrainingConfig):
    """Return a factory that builds one fully-wrapped environment."""
    def _init() -> ActionMaskWrapper:
        env = Game2048Env()
        env = RewardShapingWrapper(env, config)
        env = PreprocessingWrapper(env)
        env = ActionMaskWrapper(env)
        return env
    return _init


def main() -> None:
    config = TrainingConfig()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ------------------------------------------------------------------
    # Vectorised environments
    # ------------------------------------------------------------------
    vec_env = make_vec_env(
        make_env_fn(config),
        n_envs=config.n_envs,
        vec_env_cls=DummyVecEnv,
    )

    # ------------------------------------------------------------------
    # Policy with custom CNN feature extractor
    # ------------------------------------------------------------------
    policy_kwargs = dict(
        features_extractor_class=CNN2048FeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = MaskablePPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs",
        verbose=1,
        **config.to_dict(),
    )

    print("=" * 60)
    print("  2048 RL — Step 2: PPO training")
    print(f"  total_timesteps : {config.total_timesteps:,}")
    print(f"  n_envs          : {config.n_envs}")
    print(f"  render_freq     : every {config.render_freq} episodes")
    print("  Press Ctrl+C to interrupt and save the model.")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        RenderCallback(
            config,
            model_save_path="models/ppo_2048_checkpoint",
            verbose=1,
        ),
        MetricsCallback(config, verbose=0),
    ]

    # ------------------------------------------------------------------
    # Training loop — handle Ctrl+C gracefully
    # ------------------------------------------------------------------
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        save_path = "models/ppo_2048_final"
        model.save(save_path)
        print(f"Model saved to {save_path}.zip")
        vec_env.close()


if __name__ == "__main__":
    main()
