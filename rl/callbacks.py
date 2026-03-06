"""SB3 callbacks: live Pygame rendering and TensorBoard metrics."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from training.config import TrainingConfig

# ---------------------------------------------------------------------------
# Shared Pygame rendering helpers (standalone, no PygameUI dependency)
# ---------------------------------------------------------------------------

_WIN_W = 500
_WIN_H = 600
_HEADER_H = 100
_GAP = 10
_CELL = (_WIN_W - 5 * _GAP) // 4  # 112 px

_C_BG = (250, 248, 239)
_C_GRID = (187, 173, 160)
_C_EMPTY = (205, 193, 180)
_C_DARK = (119, 110, 101)
_C_LIGHT = (249, 246, 242)

_TILE_COLORS: dict[int, tuple[int, int, int]] = {
    0:     (205, 193, 180),
    2:     (238, 228, 218),
    4:     (237, 224, 200),
    8:     (242, 177, 121),
    16:    (245, 149, 99),
    32:    (246, 124, 95),
    64:    (246, 94, 59),
    128:   (237, 207, 114),
    256:   (237, 204, 97),
    512:   (237, 200, 80),
    1024:  (237, 197, 63),
    2048:  (237, 194, 46),
    4096:  (60, 58, 50),
}


def _render_frame(
    screen,
    board: np.ndarray,
    score: int,
    episode: int,
    total_reward: float,
    max_tile: int,
    fonts: dict,
) -> None:
    """Draw one game frame onto *screen*."""
    import pygame  # late import — only when rendering

    screen.fill(_C_BG)

    # --- Header ---
    title = fonts["title"].render("2048", True, _C_DARK)
    screen.blit(title, (_GAP, 15))

    info_text = fonts["small"].render(
        f"Ep {episode}  |  Score {score}  |  Reward {total_reward:.1f}  |  Best tile {max_tile}",
        True,
        _C_DARK,
    )
    screen.blit(info_text, (_GAP, 65))

    # --- Board background ---
    pygame.draw.rect(screen, _C_GRID, (0, _HEADER_H, _WIN_W, _WIN_H - _HEADER_H))

    # --- Cells ---
    for r in range(4):
        for c in range(4):
            value = int(board[r, c])
            x = _GAP + c * (_CELL + _GAP)
            y = _HEADER_H + _GAP + r * (_CELL + _GAP)
            color = _TILE_COLORS.get(value, (60, 58, 50))
            pygame.draw.rect(screen, color, (x, y, _CELL, _CELL), border_radius=6)

            if value == 0:
                continue

            text_color = _C_LIGHT if value > 4 else _C_DARK
            digits = len(str(value))
            font_key = f"tile_{min(digits, 4)}"
            font = fonts.get(font_key, fonts["tile_4"])
            surf = font.render(str(value), True, text_color)
            screen.blit(surf, (x + _CELL // 2 - surf.get_width() // 2,
                                y + _CELL // 2 - surf.get_height() // 2))


def _build_fonts() -> dict:
    """Build font objects for the render window (call after pygame.init())."""
    import pygame
    return {
        "title": pygame.font.SysFont("Arial", 40, bold=True),
        "small":  pygame.font.SysFont("Arial", 16),
        "tile_1": pygame.font.SysFont("Arial", 48, bold=True),
        "tile_2": pygame.font.SysFont("Arial", 40, bold=True),
        "tile_3": pygame.font.SysFont("Arial", 32, bold=True),
        "tile_4": pygame.font.SysFont("Arial", 24, bold=True),
    }


# ---------------------------------------------------------------------------
# RenderCallback
# ---------------------------------------------------------------------------

class RenderCallback(BaseCallback):
    """Play and display a full game episode every *render_freq* episodes.

    The render runs synchronously in the main thread (training is paused
    briefly).  If the user closes the Pygame window during a render episode,
    the render is aborted and training resumes immediately.

    Args:
        config:          TrainingConfig with render_freq / render_delay_ms.
        model_save_path: If given, save the model to this path after each
                         rendered episode.  Pass None to disable.
        verbose:         SB3 verbosity level.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model_save_path: str | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._config = config
        self._model_save_path = model_save_path
        self._episode_count = 0

    def _on_step(self) -> bool:
        dones: np.ndarray = self.locals.get("dones", np.array([]))
        for done in dones:
            if done:
                self._episode_count += 1
                if self._episode_count % self._config.render_freq == 0:
                    self._run_render_episode(self._episode_count)
                    if self._model_save_path:
                        self.model.save(self._model_save_path)
        return True

    def _run_render_episode(self, episode_num: int) -> None:
        """Play one greedy episode and display it with Pygame."""
        import pygame  # local import — no pygame needed for training-only runs

        from env.game2048_env import Game2048Env
        from rl.wrappers import (
            ActionMaskWrapper,
            PreprocessingWrapper,
            RewardShapingWrapper,
        )

        # Build a fresh single-env stack (same as training envs)
        env = Game2048Env()
        env = RewardShapingWrapper(env, self._config)
        env = PreprocessingWrapper(env)
        env = ActionMaskWrapper(env)

        try:
            if not pygame.get_init():
                pygame.init()

            screen = pygame.display.set_mode((_WIN_W, _WIN_H))
            pygame.display.set_caption(f"2048 Agent — Episode {episode_num}")
            clock = pygame.time.Clock()
            fonts = _build_fonts()

            obs, _ = env.reset()
            total_reward = 0.0
            max_tile = 0
            done = False

            while not done:
                # ---- handle events (allow user to close window) ----
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return  # abort render; training continues

                # ---- greedy action ----
                masks = env.action_masks()
                action, _ = self.model.predict(
                    obs[np.newaxis],
                    action_masks=masks[np.newaxis],
                    deterministic=True,
                )
                action = int(action[0])

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                max_tile = max(max_tile, info.get("max_tile", 0))
                done = terminated or truncated

                # ---- render ----
                raw_board = env.unwrapped._board.get_board()
                score = env.unwrapped._board.get_score()
                _render_frame(screen, raw_board, score, episode_num,
                               total_reward, max_tile, fonts)
                pygame.display.flip()

                if self._config.render_delay_ms > 0:
                    clock.tick(1000 // self._config.render_delay_ms)

            # Brief pause so the player can see the final state
            pygame.time.wait(800)
            pygame.quit()

        except Exception as exc:  # noqa: BLE001
            if self.verbose >= 1:
                print(f"[RenderCallback] rendering error (training continues): {exc}")
            try:
                pygame.quit()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# MetricsCallback
# ---------------------------------------------------------------------------

class MetricsCallback(BaseCallback):
    """Log custom metrics to TensorBoard at each episode end.

    Logged metrics:
        metrics/score_mean_100       — mean score over the last 100 episodes.
        metrics/max_tile             — max tile reached in the latest episode.
        metrics/invalid_action_rate  — fraction of invalid actions since start.

    Args:
        config:  TrainingConfig (currently unused, reserved for future use).
        verbose: SB3 verbosity level.
    """

    def __init__(self, config: TrainingConfig, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._config = config
        self._recent_scores: deque[int] = deque(maxlen=100)
        self._total_steps: int = 0
        self._invalid_steps: int = 0

    def _on_step(self) -> bool:
        dones: np.ndarray = self.locals.get("dones", np.array([]))
        infos: list[dict] = self.locals.get("infos", [])

        for i, done in enumerate(dones):
            info = infos[i] if i < len(infos) else {}
            moved = info.get("moved", True)

            self._total_steps += 1
            if not moved:
                self._invalid_steps += 1

            if done:
                score = info.get("score", 0)
                max_tile = info.get("max_tile", 0)
                self._recent_scores.append(score)

                if self._recent_scores:
                    self.logger.record(
                        "metrics/score_mean_100",
                        float(np.mean(self._recent_scores)),
                    )
                self.logger.record("metrics/max_tile", max_tile)
                invalid_rate = (
                    self._invalid_steps / self._total_steps
                    if self._total_steps > 0
                    else 0.0
                )
                self.logger.record(
                    "metrics/invalid_action_rate", invalid_rate
                )

        return True
