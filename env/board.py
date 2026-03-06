"""Game logic for 2048. No Gymnasium or Pygame dependencies."""

import numpy as np

from env.constants import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    BOARD_SIZE,
    TILE_2_PROB,
    WIN_TILE,
)


def _slide_row_left(row: np.ndarray) -> tuple[np.ndarray, int]:
    """Slide a single row to the left, merging equal adjacent tiles.

    Tiles are compacted left, then each pair of equal adjacent tiles is merged
    once from left to right. A merged tile cannot be merged again on the same
    move.

    Args:
        row: 1-D integer array of length BOARD_SIZE.

    Returns:
        A tuple (new_row, score) where new_row is the resulting row and score
        is the sum of all merged tile values.
    """
    tiles = row[row != 0].tolist()
    score = 0
    result: list[int] = []
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] * 2
            result.append(merged)
            score += merged
            i += 2
        else:
            result.append(tiles[i])
            i += 1

    out = np.zeros(BOARD_SIZE, dtype=np.int32)
    out[: len(result)] = result
    return out, score


def _move_left(board: np.ndarray) -> tuple[np.ndarray, int]:
    """Slide every row of *board* to the left.

    Args:
        board: 2-D integer array of shape (BOARD_SIZE, BOARD_SIZE).

    Returns:
        A tuple (new_board, total_score).
    """
    new_board = np.zeros_like(board)
    total_score = 0
    for i in range(BOARD_SIZE):
        new_board[i], row_score = _slide_row_left(board[i])
        total_score += row_score
    return new_board, total_score


class Board:
    """Manages the 2048 board state and all game logic.

    Attributes are intentionally private; callers use the public methods.
    No dependency on Gymnasium or Pygame.
    """

    def __init__(self) -> None:
        self._board: np.ndarray = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        self._score: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the board to a blank state and place two starting tiles."""
        self._board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        self._score = 0
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self) -> None:
        """Place a 2 (90 %) or 4 (10 %) tile on a random empty cell.

        Does nothing if the board is full.
        """
        empty = list(zip(*np.where(self._board == 0)))
        if not empty:
            return
        idx = np.random.randint(len(empty))
        row, col = empty[idx]
        value = 2 if np.random.random() < TILE_2_PROB else 4
        self._board[row, col] = value

    # ------------------------------------------------------------------
    # Game actions
    # ------------------------------------------------------------------

    def move(self, direction: int) -> bool:
        """Apply a move in *direction* and return whether the board changed.

        A new random tile is added only when the board actually changes.

        Args:
            direction: One of ACTION_UP (0), ACTION_DOWN (1),
                       ACTION_LEFT (2), ACTION_RIGHT (3).

        Returns:
            True if at least one tile moved or merged, False otherwise.

        Raises:
            ValueError: If *direction* is not in {0, 1, 2, 3}.
        """
        original = self._board.copy()

        if direction == ACTION_LEFT:
            new_board, score = _move_left(self._board)

        elif direction == ACTION_RIGHT:
            # Flip → slide left → flip back
            new_board, score = _move_left(np.fliplr(self._board))
            new_board = np.fliplr(new_board).copy()

        elif direction == ACTION_UP:
            # Transpose → slide left → transpose back
            new_board, score = _move_left(self._board.T)
            new_board = new_board.T.copy()

        elif direction == ACTION_DOWN:
            # Transpose + flip → slide left → flip + transpose back
            new_board, score = _move_left(np.fliplr(self._board.T))
            new_board = np.fliplr(new_board).T.copy()

        else:
            raise ValueError(f"Invalid direction: {direction!r}")

        moved = not np.array_equal(original, new_board)
        if moved:
            self._board = new_board
            self._score += score
            self.add_random_tile()
        return moved

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def has_moves(self) -> bool:
        """Return True if at least one valid move exists.

        A move is valid when it would change the board (i.e. there is an
        empty cell or two equal adjacent tiles in some direction).
        """
        # Any empty cell means tiles can slide somewhere
        if np.any(self._board == 0):
            return True

        # Check for equal neighbours horizontally and vertically
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE - 1):
                if self._board[i, j] == self._board[i, j + 1]:  # horizontal
                    return True
                if self._board[j, i] == self._board[j + 1, i]:  # vertical
                    return True

        return False

    def is_won(self) -> bool:
        """Return True if a tile equal to WIN_TILE (2048) is present."""
        return bool(np.any(self._board >= WIN_TILE))

    def get_board(self) -> np.ndarray:
        """Return a copy of the current board as a (4, 4) int32 array."""
        return self._board.copy()

    def get_score(self) -> int:
        """Return the current accumulated score."""
        return self._score
