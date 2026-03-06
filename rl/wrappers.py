"""Gymnasium wrappers for preprocessing, reward shaping, and action masking."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.board import Board
from env.constants import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP
from rl.preprocessing import encode_observation
from rl.reward import compute_reward


# ---------------------------------------------------------------------------
# PreprocessingWrapper
# ---------------------------------------------------------------------------

class PreprocessingWrapper(gym.ObservationWrapper):
    """Transform raw (4, 4) board observations into (16, 4, 4) one-hot tensors.

    Applies ``encode_observation`` to every observation returned by reset()
    and step(). The updated observation space is Box(0, 1, (16, 4, 4), float32).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(16, 4, 4), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return encode_observation(obs)


# ---------------------------------------------------------------------------
# RewardShapingWrapper
# ---------------------------------------------------------------------------

class RewardShapingWrapper(gym.Wrapper):
    """Replace the raw score-delta reward with a shaped multi-component reward.

    Components are controlled by ``config`` (a :class:`TrainingConfig`
    instance).  The wrapper also injects ``"max_tile"`` into every info dict
    so downstream callbacks can track episode statistics without accessing
    the unwrapped env directly.

    The raw observation (before preprocessing) is used for the reward
    computation, so this wrapper must sit *between* ``Game2048Env`` and
    ``PreprocessingWrapper``.
    """

    def __init__(self, env: gym.Env, config) -> None:
        super().__init__(env)
        self._config = config
        self._prev_score: int = 0
        self._episode_max_tile: int = 0

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._prev_score = 0
        self._episode_max_tile = int(np.max(obs))
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, _raw_reward, terminated, truncated, info = self.env.step(action)

        score_gained = info["score"] - self._prev_score
        self._prev_score = info["score"]

        current_max = int(np.max(obs))
        self._episode_max_tile = max(self._episode_max_tile, current_max)
        info["max_tile"] = self._episode_max_tile

        reward = compute_reward(obs, score_gained, info["moved"], self._config)
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# ActionMaskWrapper
# ---------------------------------------------------------------------------

class ActionMaskWrapper(gym.Wrapper):
    """Add an ``action_masks()`` method compatible with MaskablePPO.

    For each of the four possible actions, the wrapper tests whether applying
    that action to a *copy* of the current board state would change anything.
    Actions that leave the board unchanged are masked out (False).

    This wrapper must be the outermost wrapper so that MaskablePPO (via
    ``DummyVecEnv.env_method("action_masks")``) can call it directly.
    """

    _ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)

    def action_masks(self) -> np.ndarray:
        """Return a boolean array of shape (4,) indicating valid actions.

        An action is valid if applying it would move at least one tile.
        """
        inner_board: Board = self.unwrapped._board
        masks = np.zeros(4, dtype=bool)
        for action in self._ACTIONS:
            # Test on a temporary copy — never mutates real state
            test = Board()
            test._board = inner_board.get_board()
            test._score = inner_board.get_score()
            masks[action] = test.move(action)
        return masks
