"""Gymnasium compliance tests and integration tests for Game2048Env."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env.game2048_env import Game2048Env


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_returns_tuple_of_two(self) -> None:
        env = Game2048Env()
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2

    def test_obs_shape(self) -> None:
        env = Game2048Env()
        obs, _ = env.reset()
        assert obs.shape == (4, 4)

    def test_obs_dtype(self) -> None:
        env = Game2048Env()
        obs, _ = env.reset()
        assert obs.dtype == np.int32

    def test_obs_in_observation_space(self) -> None:
        env = Game2048Env()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_info_contains_score(self) -> None:
        env = Game2048Env()
        _, info = env.reset()
        assert "score" in info
        assert info["score"] == 0

    def test_seed_reproducibility(self) -> None:
        env = Game2048Env()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        assert np.array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# action_space / observation_space
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_action_space_size(self) -> None:
        env = Game2048Env()
        assert env.action_space.n == 4

    def test_observation_space_shape(self) -> None:
        env = Game2048Env()
        assert env.observation_space.shape == (4, 4)

    def test_observation_space_dtype(self) -> None:
        env = Game2048Env()
        assert env.observation_space.dtype == np.int32


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestStep:
    def test_returns_five_elements(self) -> None:
        env = Game2048Env()
        env.reset()
        result = env.step(0)
        assert len(result) == 5

    def test_obs_shape_and_dtype(self) -> None:
        env = Game2048Env()
        env.reset()
        obs, *_ = env.step(0)
        assert obs.shape == (4, 4)
        assert obs.dtype == np.int32

    def test_reward_is_float(self) -> None:
        env = Game2048Env()
        env.reset()
        _, reward, *_ = env.step(0)
        assert isinstance(reward, float)

    def test_terminated_is_bool(self) -> None:
        env = Game2048Env()
        env.reset()
        _, _, terminated, truncated, _ = env.step(0)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_truncated_always_false(self) -> None:
        env = Game2048Env()
        env.reset()
        for action in range(4):
            env.reset()
            _, _, _, truncated, _ = env.step(action)
            assert truncated is False

    def test_info_contains_moved_and_score(self) -> None:
        env = Game2048Env()
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "moved" in info
        assert "score" in info

    def test_obs_in_observation_space(self) -> None:
        env = Game2048Env()
        env.reset()
        obs, *_ = env.step(0)
        assert env.observation_space.contains(obs)

    def test_invalid_move_sets_moved_false(self) -> None:
        """When an action does not change the board, info['moved'] is False."""
        env = Game2048Env()
        env.reset()
        # Row 0 already left-aligned → moving left changes nothing there;
        # force a board where LEFT is guaranteed to be a no-op.
        env._board._board = np.array(
            [
                [2, 4, 8, 16],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        _, _, _, _, info = env.step(2)  # ACTION_LEFT
        assert info["moved"] is False

    def test_invalid_move_does_not_terminate_game(self) -> None:
        """An invalid move should not end the game if valid moves remain."""
        env = Game2048Env()
        env.reset()
        env._board._board = np.array(
            [
                [2, 4, 8, 16],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        _, _, terminated, _, _ = env.step(2)  # ACTION_LEFT — no move
        assert terminated is False

    def test_reward_equals_score_gain(self) -> None:
        """Reward must match the score gained on this step."""
        env = Game2048Env()
        env.reset()
        env._board._board = np.array(
            [
                [2, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        _, reward, _, _, info = env.step(2)  # ACTION_LEFT → merge 2+2=4
        assert reward == 4.0
        assert info["score"] == 4


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

class TestTermination:
    def test_episode_terminates_when_board_is_blocked(self) -> None:
        env = Game2048Env()
        env.reset()
        # Build a fully blocked board (no empty cell, no adjacent equal tiles)
        env._board._board = np.array(
            [
                [2, 4, 2, 4],
                [4, 2, 4, 2],
                [2, 4, 2, 4],
                [4, 2, 4, 2],
            ],
            dtype=np.int32,
        )
        # Any action will fail to change this board → terminated=True
        _, _, terminated, _, _ = env.step(0)
        assert terminated is True


# ---------------------------------------------------------------------------
# Random agent integration test
# ---------------------------------------------------------------------------

class TestRandomAgent:
    def test_episode_completes_without_error(self) -> None:
        """A random agent must be able to play through a full episode."""
        env = Game2048Env()
        obs, info = env.reset()

        assert obs.shape == (4, 4)
        assert info["score"] == 0

        terminated = False
        truncated = False
        step_count = 0
        max_steps = 20_000

        while not (terminated or truncated) and step_count < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            step_count += 1

        assert terminated or step_count >= max_steps

    def test_multiple_episodes_without_error(self) -> None:
        env = Game2048Env()
        for _ in range(3):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            steps = 0
            while not (terminated or truncated) and steps < 5_000:
                obs, _, terminated, truncated, _ = env.step(
                    env.action_space.sample()
                )
                steps += 1


# ---------------------------------------------------------------------------
# Gymnasium env_checker
# ---------------------------------------------------------------------------

class TestGymnasiumCompliance:
    def test_check_env_passes(self) -> None:
        """The environment must satisfy Gymnasium's built-in env_checker."""
        env = Game2048Env()
        check_env(env, warn=True, skip_render_check=True)
