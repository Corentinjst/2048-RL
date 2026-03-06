"""Gymnasium environment wrapping the 2048 Board. No Pygame dependency."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.board import Board
from env.constants import BOARD_SIZE


class Game2048Env(gym.Env):
    """A Gymnasium-compatible environment for the 2048 game.

    Observation space:
        Box(low=0, high=2**17, shape=(4, 4), dtype=np.int32)
        Raw tile values (0, 2, 4, 8, …, 131072).

    Action space:
        Discrete(4): 0=up, 1=down, 2=left, 3=right.

    Reward:
        Score gained on the current step (sum of merged tile values).
        Returns 0.0 for invalid moves.

    Termination:
        The episode ends when no valid move exists (``has_moves() == False``).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode {render_mode!r} not supported. "
                f"Choose from {self.metadata['render_modes']}."
            )
        self.render_mode = render_mode
        self._board = Board()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=2**17,
            shape=(BOARD_SIZE, BOARD_SIZE),
            dtype=np.int32,
        )

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and return the initial observation.

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Unused; kept for API compatibility.

        Returns:
            A tuple ``(observation, info)`` where *observation* is the
            initial board and *info* contains ``{"score": 0}``.
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._board.reset()
        obs = self._board.get_board()
        info: dict = {"score": self._board.get_score()}
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply *action* and advance the environment by one step.

        Args:
            action: Integer in {0, 1, 2, 3}.

        Returns:
            A tuple ``(observation, reward, terminated, truncated, info)``.
            *info* always contains ``"moved"`` (bool) and ``"score"`` (int).
        """
        old_score = self._board.get_score()
        moved = self._board.move(int(action))
        new_score = self._board.get_score()

        reward = float(new_score - old_score)
        terminated = bool(not self._board.has_moves())
        truncated = False
        obs = self._board.get_board()
        info: dict = {
            "moved": moved,
            "score": new_score,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the current board state as ASCII art in the terminal."""
        board = self._board.get_board()
        sep = "+" + "+".join(["------"] * BOARD_SIZE) + "+"
        print(f"\nScore: {self._board.get_score()}")
        print(sep)
        for row in board:
            print("|" + "|".join(f"{v:^6d}" for v in row) + "|")
            print(sep)

    def close(self) -> None:
        """Clean up resources (no-op for this environment)."""
        pass
