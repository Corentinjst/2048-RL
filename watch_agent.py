"""CLI script to watch a trained MaskablePPO agent play 2048.

Usage::

    python watch_agent.py --model models/ppo_2048_final.zip
    python watch_agent.py --model models/ppo_2048_final.zip --speed 300 --episodes 10
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pygame
from sb3_contrib import MaskablePPO

from env.game2048_env import Game2048Env
from rl.wrappers import ActionMaskWrapper, PreprocessingWrapper, RewardShapingWrapper
from training.config import TrainingConfig, TrainingConfigV2, TrainingConfigV3

# ---------------------------------------------------------------------------
# Pygame layout / colours (mirror of the callback renderer)
# ---------------------------------------------------------------------------
_WIN_W = 500
_WIN_H = 600
_HEADER_H = 100
_GAP = 10
_CELL = (_WIN_W - 5 * _GAP) // 4  # 112 px

_C_BG = (250, 248, 239)
_C_GRID = (187, 173, 160)
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


def _draw(screen, board, score, episode, ep_total, speed_ms, fonts):
    screen.fill(_C_BG)

    title = fonts["title"].render("2048", True, _C_DARK)
    screen.blit(title, (_GAP, 15))

    sub = fonts["small"].render(
        f"Episode {episode}  |  Score {score}  |  Speed {speed_ms} ms/step",
        True,
        _C_DARK,
    )
    screen.blit(sub, (_GAP, 62))

    pygame.draw.rect(screen, _C_GRID, (0, _HEADER_H, _WIN_W, _WIN_H - _HEADER_H))

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
            font = fonts.get(f"tile_{min(digits, 4)}", fonts["tile_4"])
            surf = font.render(str(value), True, text_color)
            screen.blit(
                surf,
                (x + _CELL // 2 - surf.get_width() // 2,
                 y + _CELL // 2 - surf.get_height() // 2),
            )


def _build_fonts() -> dict:
    return {
        "title":  pygame.font.SysFont("Arial", 40, bold=True),
        "small":  pygame.font.SysFont("Arial", 16),
        "tile_1": pygame.font.SysFont("Arial", 48, bold=True),
        "tile_2": pygame.font.SysFont("Arial", 40, bold=True),
        "tile_3": pygame.font.SysFont("Arial", 32, bold=True),
        "tile_4": pygame.font.SysFont("Arial", 24, bold=True),
    }


def _make_env(config: TrainingConfig) -> ActionMaskWrapper:
    env = Game2048Env()
    env = RewardShapingWrapper(env, config)
    env = PreprocessingWrapper(env)
    env = ActionMaskWrapper(env)
    return env


_CONFIG_MAP: dict[str, type] = {
    "v1": TrainingConfig,
    "v2": TrainingConfigV2,
    "v3": TrainingConfigV3,
}


def watch(model_path: str, n_episodes: int, speed_ms: int, config_version: str = "v1") -> None:
    """Load a model and watch it play for *n_episodes*."""
    if not os.path.exists(model_path):
        sys.exit(f"Model file not found: {model_path}")

    config = _CONFIG_MAP[config_version]()
    model = MaskablePPO.load(model_path, device="cpu")

    pygame.init()
    screen = pygame.display.set_mode((_WIN_W, _WIN_H))
    pygame.display.set_caption("2048 — Watch Agent")
    clock = pygame.time.Clock()
    fonts = _build_fonts()

    for episode in range(1, n_episodes + 1):
        env = _make_env(config)
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            # ---- event handling ----
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print(f"\nClosed after episode {episode - 1}/{n_episodes}.")
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    pygame.quit()
                    return

            # ---- greedy action ----
            masks = env.action_masks()
            action, _ = model.predict(
                obs[np.newaxis],
                action_masks=masks[np.newaxis],
                deterministic=True,
            )
            action = int(action[0])

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

            # ---- render ----
            raw_board = env.unwrapped._board.get_board()
            score = env.unwrapped._board.get_score()
            _draw(screen, raw_board, score, episode, total_reward, speed_ms, fonts)
            pygame.display.flip()

            if speed_ms > 0:
                clock.tick(1000 // speed_ms)

        final_score = env.unwrapped._board.get_score()
        max_tile = int(np.max(env.unwrapped._board.get_board()))
        print(
            f"Episode {episode:3d}/{n_episodes} — "
            f"score={final_score:7,}  max_tile={max_tile:5}  "
            f"reward={total_reward:.1f}"
        )

        # Pause between episodes so the user can see the final board
        pygame.time.wait(1000)

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch a trained MaskablePPO agent play 2048."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the saved model (.zip file, with or without extension).",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=200,
        help="Delay in milliseconds between each displayed step (default: 200).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to watch (default: 5).",
    )
    parser.add_argument(
        "--config",
        choices=["v1", "v2", "v3"],
        default="v1",
        help="Config version used during training (default: v1).",
    )
    args = parser.parse_args()

    # Normalise model path: SB3 accepts both with and without .zip
    model_path = args.model
    if not model_path.endswith(".zip") and not os.path.exists(model_path):
        model_path = model_path + ".zip"

    watch(model_path=model_path, n_episodes=args.episodes, speed_ms=args.speed, config_version=args.config)


if __name__ == "__main__":
    main()
