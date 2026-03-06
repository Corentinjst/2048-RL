"""Unit tests for the one-hot observation encoding."""

import numpy as np
import pytest

from rl.preprocessing import encode_observation


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

class TestShape:
    def test_output_shape(self) -> None:
        board = np.zeros((4, 4), dtype=np.int32)
        out = encode_observation(board)
        assert out.shape == (16, 4, 4)

    def test_output_dtype(self) -> None:
        board = np.zeros((4, 4), dtype=np.int32)
        out = encode_observation(board)
        assert out.dtype == np.float32

    def test_shape_with_full_board(self) -> None:
        board = np.array(
            [[2, 4, 8, 16], [32, 64, 128, 256],
             [512, 1024, 2048, 4096], [2, 4, 8, 16]],
            dtype=np.int32,
        )
        out = encode_observation(board)
        assert out.shape == (16, 4, 4)


# ---------------------------------------------------------------------------
# One-hot correctness: each cell has exactly one active channel
# ---------------------------------------------------------------------------

class TestOneHot:
    def test_each_cell_has_exactly_one_active_channel(self) -> None:
        board = np.array(
            [[2, 4, 8, 16], [32, 64, 128, 256],
             [512, 1024, 2048, 4096], [2, 4, 8, 16]],
            dtype=np.int32,
        )
        out = encode_observation(board)
        # Sum over channel axis must be 1 for every (row, col)
        assert np.all(out.sum(axis=0) == 1.0)

    def test_empty_board_all_on_channel_0(self) -> None:
        board = np.zeros((4, 4), dtype=np.int32)
        out = encode_observation(board)
        assert np.all(out[0] == 1.0), "Every empty cell should activate channel 0"
        assert out[1:].sum() == 0.0

    def test_values_are_binary(self) -> None:
        board = np.array(
            [[0, 2, 0, 4], [8, 0, 16, 0], [0, 32, 0, 64], [128, 0, 256, 0]],
            dtype=np.int32,
        )
        out = encode_observation(board)
        unique = np.unique(out)
        assert set(unique.tolist()) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# Channel mapping for known values
# ---------------------------------------------------------------------------

class TestChannelMapping:
    """Verify that each tile value maps to the correct log2 channel."""

    @pytest.mark.parametrize("value,expected_channel", [
        (0,    0),   # empty cell → channel 0
        (2,    1),   # log2(2) = 1
        (4,    2),   # log2(4) = 2
        (8,    3),
        (16,   4),
        (32,   5),
        (64,   6),
        (128,  7),
        (256,  8),
        (512,  9),
        (1024, 10),
        (2048, 11),
        (4096, 12),
    ])
    def test_single_value_channel(self, value: int, expected_channel: int) -> None:
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, 0] = value
        out = encode_observation(board)
        assert out[expected_channel, 0, 0] == 1.0, (
            f"Value {value} should activate channel {expected_channel}"
        )
        # All other channels at (0, 0) must be 0
        for ch in range(16):
            if ch != expected_channel:
                assert out[ch, 0, 0] == 0.0

    def test_2048_on_channel_11(self) -> None:
        board = np.zeros((4, 4), dtype=np.int32)
        board[2, 3] = 2048
        out = encode_observation(board)
        assert out[11, 2, 3] == 1.0

    def test_zero_on_channel_0(self) -> None:
        board = np.zeros((4, 4), dtype=np.int32)
        board[1, 1] = 0
        out = encode_observation(board)
        assert out[0, 1, 1] == 1.0


# ---------------------------------------------------------------------------
# Known full-board encoding
# ---------------------------------------------------------------------------

class TestKnownBoard:
    def test_mixed_board_encoding(self) -> None:
        board = np.array(
            [[2, 4, 0, 8], [0, 16, 0, 0], [0, 0, 2048, 0], [0, 0, 0, 4]],
            dtype=np.int32,
        )
        out = encode_observation(board)

        # Check specific (channel, row, col) activations
        assert out[1, 0, 0] == 1.0   # 2  → channel 1
        assert out[2, 0, 1] == 1.0   # 4  → channel 2
        assert out[0, 0, 2] == 1.0   # 0  → channel 0
        assert out[3, 0, 3] == 1.0   # 8  → channel 3
        assert out[0, 1, 0] == 1.0   # 0  → channel 0
        assert out[4, 1, 1] == 1.0   # 16 → channel 4
        assert out[11, 2, 2] == 1.0  # 2048 → channel 11
        assert out[2, 3, 3] == 1.0   # 4  → channel 2

    def test_same_value_multiple_cells(self) -> None:
        board = np.array(
            [[2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int32,
        )
        out = encode_observation(board)
        # Channel 1 should be active for all four cells in row 0
        assert np.all(out[1, 0, :] == 1.0)
        # Channel 0 should be active for rows 1–3
        assert np.all(out[0, 1:, :] == 1.0)
