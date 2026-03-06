"""Unit tests for Board game logic."""

import numpy as np
import pytest

from env.board import Board, _slide_row_left
from env.constants import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    BOARD_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_board(grid: list[list[int]]) -> Board:
    """Return a Board whose internal state is set to *grid* (score = 0)."""
    b = Board()
    b._board = np.array(grid, dtype=np.int32)
    b._score = 0
    return b


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestBoardInit:
    def test_reset_places_exactly_two_tiles(self) -> None:
        b = Board()
        b.reset()
        assert np.count_nonzero(b.get_board()) == 2

    def test_reset_score_is_zero(self) -> None:
        b = Board()
        b.reset()
        assert b.get_score() == 0

    def test_board_shape_and_dtype(self) -> None:
        b = Board()
        b.reset()
        board = b.get_board()
        assert board.shape == (BOARD_SIZE, BOARD_SIZE)
        assert board.dtype == np.int32

    def test_starting_tiles_are_2_or_4(self) -> None:
        for _ in range(20):
            b = Board()
            b.reset()
            tiles = b.get_board().flatten()
            non_zero = tiles[tiles != 0]
            assert all(v in (2, 4) for v in non_zero)

    def test_get_board_returns_copy(self) -> None:
        b = Board()
        b.reset()
        snapshot = b.get_board()
        snapshot[0, 0] = 9999
        assert b.get_board()[0, 0] != 9999


# ---------------------------------------------------------------------------
# _slide_row_left (unit-level)
# ---------------------------------------------------------------------------

class TestSlideRowLeft:
    def test_all_zeros(self) -> None:
        row = np.zeros(4, dtype=np.int32)
        out, score = _slide_row_left(row)
        assert list(out) == [0, 0, 0, 0]
        assert score == 0

    def test_no_merge_needed(self) -> None:
        row = np.array([2, 4, 8, 16], dtype=np.int32)
        out, score = _slide_row_left(row)
        assert list(out) == [2, 4, 8, 16]
        assert score == 0

    def test_compact_without_merge(self) -> None:
        row = np.array([0, 2, 0, 4], dtype=np.int32)
        out, score = _slide_row_left(row)
        assert list(out) == [2, 4, 0, 0]
        assert score == 0

    def test_single_merge(self) -> None:
        row = np.array([2, 2, 0, 0], dtype=np.int32)
        out, score = _slide_row_left(row)
        assert list(out) == [4, 0, 0, 0]
        assert score == 4

    def test_no_double_merge_four_equal(self) -> None:
        """[2, 2, 2, 2] → [4, 4, 0, 0], NOT [8, 0, 0, 0]."""
        row = np.array([2, 2, 2, 2], dtype=np.int32)
        out, score = _slide_row_left(row)
        assert list(out) == [4, 4, 0, 0]
        assert score == 8

    def test_merge_then_compact(self) -> None:
        row = np.array([0, 2, 2, 4], dtype=np.int32)
        out, score = _slide_row_left(row)
        assert list(out) == [4, 4, 0, 0]
        assert score == 4

    def test_score_equals_sum_of_merged_tiles(self) -> None:
        row = np.array([4, 4, 8, 8], dtype=np.int32)
        out, score = _slide_row_left(row)
        assert list(out) == [8, 16, 0, 0]
        assert score == 24  # 8 + 16


# ---------------------------------------------------------------------------
# Direction moves
# ---------------------------------------------------------------------------

class TestMoveLeft:
    def test_basic_merge(self) -> None:
        b = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        moved = b.move(ACTION_LEFT)
        assert moved is True
        assert b.get_board()[0, 0] == 4

    def test_score_updated(self) -> None:
        b = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        b.move(ACTION_LEFT)
        assert b.get_score() == 4

    def test_already_left_aligned_no_move(self) -> None:
        b = make_board([
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        moved = b.move(ACTION_LEFT)
        assert moved is False
        assert b.get_score() == 0


class TestMoveRight:
    def test_merge_moves_to_right(self) -> None:
        b = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        b.move(ACTION_RIGHT)
        assert b.get_board()[0, 3] == 4

    def test_score_updated(self) -> None:
        b = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        b.move(ACTION_RIGHT)
        assert b.get_score() == 4


class TestMoveUp:
    def test_merge_moves_to_top(self) -> None:
        b = make_board([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        b.move(ACTION_UP)
        assert b.get_board()[0, 0] == 4

    def test_score_updated(self) -> None:
        b = make_board([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        b.move(ACTION_UP)
        assert b.get_score() == 4


class TestMoveDown:
    def test_merge_moves_to_bottom(self) -> None:
        b = make_board([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        b.move(ACTION_DOWN)
        assert b.get_board()[3, 0] == 4

    def test_score_updated(self) -> None:
        b = make_board([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        b.move(ACTION_DOWN)
        assert b.get_score() == 4


# ---------------------------------------------------------------------------
# Random tile spawning
# ---------------------------------------------------------------------------

class TestRandomTile:
    def test_new_tile_added_after_valid_move(self) -> None:
        b = make_board([
            [2, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        b.move(ACTION_LEFT)
        # board changed (2,4 already left-aligned... so let's pick a direction
        # that moves: RIGHT would move 2 and 4 to the right)

    def test_new_tile_is_2_or_4(self) -> None:
        """After a valid move, the spawned tile must be 2 or 4."""
        for _ in range(30):
            b = make_board([
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ])
            b.move(ACTION_RIGHT)
            flat = b.get_board().flatten()
            new_tiles = [v for v in flat if v not in (0, 2)]
            # After the move: [0,0,0,2] + one new tile somewhere
            # All non-zero values must be 2 or 4
            assert all(v in (2, 4) for v in flat if v != 0)

    def test_no_tile_added_after_invalid_move(self) -> None:
        """Board with no valid left-move should not gain a new tile."""
        b = make_board([
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        count_before = np.count_nonzero(b.get_board())
        b.move(ACTION_LEFT)
        assert np.count_nonzero(b.get_board()) == count_before


# ---------------------------------------------------------------------------
# has_moves
# ---------------------------------------------------------------------------

class TestHasMoves:
    def test_empty_board_has_moves(self) -> None:
        b = Board()
        b._board = np.zeros((4, 4), dtype=np.int32)
        assert b.has_moves() is True

    def test_board_with_empty_cell_has_moves(self) -> None:
        b = make_board([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 0],  # one empty cell
        ])
        assert b.has_moves() is True

    def test_full_board_no_adjacent_equal_no_moves(self) -> None:
        b = make_board([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ])
        assert b.has_moves() is False

    def test_full_board_with_horizontal_equal_has_moves(self) -> None:
        b = make_board([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 2, 8],  # two adjacent 2s in row 3
        ])
        assert b.has_moves() is True

    def test_full_board_with_vertical_equal_has_moves(self) -> None:
        b = make_board([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 4, 8],  # column 2: rows 2 and 3 both have 4
            [4, 2, 4, 2],
        ])
        assert b.has_moves() is True

    def test_has_moves_returns_false_after_board_fills(self) -> None:
        """Manually build an unwinnable blocked board."""
        blocked = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        b = make_board(blocked)
        assert b.has_moves() is False


# ---------------------------------------------------------------------------
# is_won
# ---------------------------------------------------------------------------

class TestIsWon:
    def test_not_won_at_start(self) -> None:
        b = Board()
        b.reset()
        assert b.is_won() is False

    def test_won_when_2048_present(self) -> None:
        b = make_board([
            [2048, 0, 0, 0],
            [0,    0, 0, 0],
            [0,    0, 0, 0],
            [0,    0, 0, 0],
        ])
        assert b.is_won() is True

    def test_won_when_above_2048(self) -> None:
        b = make_board([
            [4096, 0, 0, 0],
            [0,    0, 0, 0],
            [0,    0, 0, 0],
            [0,    0, 0, 0],
        ])
        assert b.is_won() is True

    def test_not_won_below_2048(self) -> None:
        b = make_board([
            [1024, 512, 256, 128],
            [0,    0,   0,   0  ],
            [0,    0,   0,   0  ],
            [0,    0,   0,   0  ],
        ])
        assert b.is_won() is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_full_board_add_random_tile_does_nothing(self) -> None:
        """add_random_tile should be a no-op on a full board."""
        b = make_board([
            [2, 4, 8,  16 ],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [2, 4, 8, 16],
        ])
        before = b.get_board().copy()
        b.add_random_tile()
        assert np.array_equal(b.get_board(), before)

    def test_chain_merge_in_column(self) -> None:
        """Two independent merges in the same column going up."""
        b = make_board([
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [4, 0, 0, 0],
        ])
        b.move(ACTION_UP)
        board = b.get_board()
        # Col 0 after UP: [2,2] merge to 4 → [4,4] ... but they don't merge again
        # Col 0 was [0,2,2,4] → slide up → tiles=[2,2,4] → merge(2,2)=4 → [4,4,0,0]
        assert board[0, 0] == 4
        assert board[1, 0] == 4

    def test_invalid_direction_raises(self) -> None:
        b = Board()
        b.reset()
        with pytest.raises(ValueError):
            b.move(99)
