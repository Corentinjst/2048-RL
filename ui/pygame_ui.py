"""Pygame-based UI for playing 2048 with keyboard controls."""

from __future__ import annotations

import sys

import numpy as np
import pygame

from env.constants import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, WIN_TILE
from env.game2048_env import Game2048Env

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 600
HEADER_HEIGHT = 100          # space above the board for score / title
GRID_GAP = 10                # gap between cells (also used as outer padding)
CELL_SIZE = (WINDOW_WIDTH - 5 * GRID_GAP) // 4  # = 112 px

# ---------------------------------------------------------------------------
# Colour palette (faithful to the original 2048 design)
# ---------------------------------------------------------------------------
C_BG = (250, 248, 239)           # page background
C_GRID_BG = (187, 173, 160)      # grid / board background
C_EMPTY = (205, 193, 180)        # empty cell
C_TEXT_DARK = (119, 110, 101)    # text on light tiles (≤4)
C_TEXT_LIGHT = (249, 246, 242)   # text on dark tiles (>4)
C_HEADER_TEXT = (119, 110, 101)

TILE_COLORS: dict[int, tuple[int, int, int]] = {
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
_TILE_COLOR_HIGH = (60, 58, 50)  # fallback for values beyond 4096

# Tile font sizes indexed by number of digits (1–4)
_TILE_FONT_SIZES = {1: 52, 2: 44, 3: 36, 4: 28}


class PygameUI:
    """Interactive Pygame window for playing 2048.

    Key bindings:
        Arrow keys  – move tiles (up / down / left / right)
        R           – restart the game
        Q / window close – quit
    """

    def __init__(self, env: Game2048Env) -> None:
        self._env = env
        self._best_score: int = 0
        self._score: int = 0
        self._obs: np.ndarray
        self._game_over: bool = False
        self._won: bool = False

        pygame.init()
        self._screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("2048")
        self._clock = pygame.time.Clock()

        # Fonts
        self._font_title = pygame.font.SysFont("Arial", 52, bold=True)
        self._font_score_label = pygame.font.SysFont("Arial", 14, bold=True)
        self._font_score_value = pygame.font.SysFont("Arial", 22, bold=True)
        self._font_overlay = pygame.font.SysFont("Arial", 52, bold=True)
        self._font_hint = pygame.font.SysFont("Arial", 18)
        self._tile_fonts: dict[int, pygame.font.Font] = {
            digits: pygame.font.SysFont("Arial", size, bold=True)
            for digits, size in _TILE_FONT_SIZES.items()
        }

        obs, _ = self._env.reset()
        self._obs = obs

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the main game loop (blocks until the window is closed)."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if not self._handle_key(event.key):
                        running = False

            self._draw()
            pygame.display.flip()
            self._clock.tick(60)

        pygame.quit()
        sys.exit()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _handle_key(self, key: int) -> bool:
        """Process a key-down event. Return False to quit."""
        if key == pygame.K_q:
            return False

        if key == pygame.K_r:
            self._restart()
            return True

        if self._game_over:
            return True

        action_map: dict[int, int] = {
            pygame.K_UP: ACTION_UP,
            pygame.K_DOWN: ACTION_DOWN,
            pygame.K_LEFT: ACTION_LEFT,
            pygame.K_RIGHT: ACTION_RIGHT,
        }
        action = action_map.get(key)
        if action is not None:
            self._apply_action(action)

        return True

    def _restart(self) -> None:
        obs, _ = self._env.reset()
        self._obs = obs
        self._score = 0
        self._game_over = False
        self._won = False

    def _apply_action(self, action: int) -> None:
        obs, _reward, terminated, _truncated, info = self._env.step(action)
        self._obs = obs
        self._score = info["score"]
        self._best_score = max(self._best_score, self._score)
        if not self._won and np.any(obs >= WIN_TILE):
            self._won = True
        if terminated:
            self._game_over = True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        self._screen.fill(C_BG)
        self._draw_header()
        self._draw_board()
        if self._game_over:
            self._draw_overlay("Game Over!", color=(238, 228, 218, 200))
        elif self._won:
            self._draw_overlay("You Win!", color=(237, 194, 46, 200))

    def _draw_header(self) -> None:
        # Title
        title_surf = self._font_title.render("2048", True, C_HEADER_TEXT)
        self._screen.blit(title_surf, (GRID_GAP, 18))

        # Score boxes
        self._draw_score_box(280, 15, "SCORE", str(self._score))
        self._draw_score_box(390, 15, "BEST", str(self._best_score))

    def _draw_score_box(
        self, x: int, y: int, label: str, value: str
    ) -> None:
        box = pygame.Rect(x, y, 100, 60)
        pygame.draw.rect(self._screen, C_GRID_BG, box, border_radius=6)

        label_surf = self._font_score_label.render(label, True, C_TEXT_LIGHT)
        value_surf = self._font_score_value.render(value, True, C_TEXT_LIGHT)

        lx = box.centerx - label_surf.get_width() // 2
        vx = box.centerx - value_surf.get_width() // 2
        self._screen.blit(label_surf, (lx, y + 10))
        self._screen.blit(value_surf, (vx, y + 30))

    def _draw_board(self) -> None:
        # Board background
        board_rect = pygame.Rect(0, HEADER_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT - HEADER_HEIGHT)
        pygame.draw.rect(self._screen, C_GRID_BG, board_rect)

        for row in range(4):
            for col in range(4):
                self._draw_cell(row, col, int(self._obs[row, col]))

    def _draw_cell(self, row: int, col: int, value: int) -> None:
        x = GRID_GAP + col * (CELL_SIZE + GRID_GAP)
        y = HEADER_HEIGHT + GRID_GAP + row * (CELL_SIZE + GRID_GAP)
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

        color = TILE_COLORS.get(value, _TILE_COLOR_HIGH)
        pygame.draw.rect(self._screen, color, rect, border_radius=6)

        if value == 0:
            return

        text_color = C_TEXT_LIGHT if value > 4 else C_TEXT_DARK
        digits = len(str(value))
        font = self._tile_fonts.get(min(digits, 4), self._tile_fonts[4])
        surf = font.render(str(value), True, text_color)

        tx = x + CELL_SIZE // 2 - surf.get_width() // 2
        ty = y + CELL_SIZE // 2 - surf.get_height() // 2
        self._screen.blit(surf, (tx, ty))

    def _draw_overlay(
        self,
        message: str,
        color: tuple[int, int, int, int],
    ) -> None:
        overlay = pygame.Surface(
            (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA
        )
        overlay.fill(color)
        self._screen.blit(overlay, (0, 0))

        msg_surf = self._font_overlay.render(message, True, C_TEXT_DARK)
        mx = WINDOW_WIDTH // 2 - msg_surf.get_width() // 2
        my = WINDOW_HEIGHT // 2 - msg_surf.get_height() // 2 - 20
        self._screen.blit(msg_surf, (mx, my))

        hint_surf = self._font_hint.render("Press R to restart", True, C_TEXT_DARK)
        hx = WINDOW_WIDTH // 2 - hint_surf.get_width() // 2
        self._screen.blit(hint_surf, (hx, my + 70))
