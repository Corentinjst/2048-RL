"""Main training script: MaskablePPO + CNN extractor + live Pygame render."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# Add the project root to sys.path so that ``env``, ``rl``, ``training``
# are importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO

from env.game2048_env import Game2048Env
from rl.network import CNN2048FeaturesExtractor
from rl.wrappers import ActionMaskWrapper, PreprocessingWrapper, RewardShapingWrapper
from rl.callbacks import MetricsCallback, RenderCallback
from training.config import TrainingConfig, TrainingConfigV2, TrainingConfigV3, linear_schedule


def make_env_fn(config: TrainingConfig):
    """Return a factory that builds one fully-wrapped environment."""
    def _init() -> ActionMaskWrapper:
        env = Game2048Env()
        env = RewardShapingWrapper(env, config)
        env = PreprocessingWrapper(env)
        env = ActionMaskWrapper(env)
        return env
    return _init


def _select_device() -> str:
    """Pick the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_policy_kwargs(config: TrainingConfig) -> dict:
    """Build policy_kwargs for MaskablePPO from config."""
    kwargs: dict = dict(
        features_extractor_class=CNN2048FeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    pi_arch = getattr(config, "pi_arch", None)
    vf_arch = getattr(config, "vf_arch", None)
    if pi_arch is not None or vf_arch is not None:
        kwargs["net_arch"] = dict(
            pi=pi_arch if pi_arch is not None else [64, 64],
            vf=vf_arch if vf_arch is not None else [64, 64],
        )
    return kwargs


def _save_vec_normalize(vec_env, path: str) -> None:
    """Save VecNormalize stats if *vec_env* is a VecNormalize wrapper."""
    if isinstance(vec_env, VecNormalize):
        stats_path = path + "_vecnormalize.pkl"
        vec_env.save(stats_path)
        print(f"VecNormalize stats saved to {stats_path}")


def train(
    config: TrainingConfig,
    config_version: str = "v1",
    run_name: str | None = None,
    save_path: str | None = None,
) -> None:
    """Run a full training loop with the given config.

    Args:
        config:         Hyperparameter config to use.
        config_version: Label used in TensorBoard log dir (``"v1"`` or ``"v2"``).
        run_name:       Override the TensorBoard sub-folder name. If None, a
                        timestamped name is generated automatically.
        save_path:      Where to save the final model (without ``.zip``).
    """
    device = _select_device()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_run = run_name if run_name is not None else f"run_{config_version}_{timestamp}"
    tb_log_dir = os.path.join("logs", tb_run)

    final_save = save_path if save_path is not None else f"models/ppo_2048_{config_version}_final"

    # ------------------------------------------------------------------
    # Vectorised environments
    # ------------------------------------------------------------------
    vec_env = make_vec_env(
        make_env_fn(config),
        n_envs=config.n_envs,
        vec_env_cls=DummyVecEnv,
    )

    # ------------------------------------------------------------------
    # VecNormalize (reward normalisation)
    # ------------------------------------------------------------------
    if getattr(config, "use_vec_normalize", False):
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config.vec_norm_obs,
            norm_reward=config.vec_norm_reward,
            clip_reward=config.vec_clip_reward,
            gamma=config.gamma,
        )

    # ------------------------------------------------------------------
    # Learning rate: schedule or fixed
    # ------------------------------------------------------------------
    ppo_kwargs = config.to_dict()
    if getattr(config, "use_lr_schedule", False):
        ppo_kwargs["learning_rate"] = linear_schedule(config.lr_start, config.lr_end)

    # ------------------------------------------------------------------
    # Policy with custom CNN feature extractor
    # ------------------------------------------------------------------
    policy_kwargs = build_policy_kwargs(config)

    model = MaskablePPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log_dir,
        verbose=1,
        device=device,
        **ppo_kwargs,
    )

    use_vecnorm = getattr(config, "use_vec_normalize", False)
    print()
    print(f"2048 RL : PPO training  [{config_version}]")
    print(f"    - device          : {device}")
    print(f"    - total_timesteps : {config.total_timesteps:,}")
    print(f"    - n_envs          : {config.n_envs}")
    print(f"    - n_steps         : {ppo_kwargs.get('n_steps', '?')}")
    print(f"    - n_epochs        : {ppo_kwargs.get('n_epochs', '?')}")
    print(f"    - vec_normalize   : {use_vecnorm}")
    print(f"    - render_freq     : every {config.render_freq} episodes")
    print(f"    - tensorboard     : {tb_log_dir}")
    print("Press Ctrl+C to interrupt and save the model.")
    print()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        RenderCallback(
            config,
            model_save_path=f"models/ppo_2048_{config_version}_checkpoint",
            vec_env=vec_env if use_vecnorm else None,
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
        model.save(final_save)
        print(f"Model saved to {final_save}.zip")
        _save_vec_normalize(vec_env, final_save)
        vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 2048 PPO agent")
    parser.add_argument(
        "--config",
        choices=["v1", "v2", "v3"],
        default="v1",
        help="Training configuration version (default: v1)",
    )
    args = parser.parse_args()

    config: TrainingConfig
    if args.config == "v3":
        config = TrainingConfigV3()
    elif args.config == "v2":
        config = TrainingConfigV2()
    else:
        config = TrainingConfig()

    train(config, config_version=args.config)


if __name__ == "__main__":
    main()
