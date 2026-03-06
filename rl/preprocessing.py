"""One-hot observation encoding for the 2048 board."""

import numpy as np


def encode_observation(board: np.ndarray) -> np.ndarray:
    """Encode a raw (4, 4) board into a (16, 4, 4) one-hot float32 tensor.

    Each cell value *v* is mapped to its log2 level *k*:
        - v = 0  → channel 0  (empty cell)
        - v = 2  → channel 1  (log2(2) = 1)
        - v = 4  → channel 2  (log2(4) = 2)
        - …
        - v = 2048 → channel 11
        - v = 4096 → channel 12
        - (values beyond channel 15 are clamped to 15)

    The resulting tensor has exactly one 1.0 per spatial position (one-hot)
    and all other entries are 0.0.

    Args:
        board: Integer array of shape (4, 4) with raw tile values.

    Returns:
        Float32 array of shape (16, 4, 4).
    """
    out = np.zeros((16, 4, 4), dtype=np.float32)

    # Compute log2 level for every cell; empty cells → 0 (log2(1) = 0 via mask)
    levels = np.where(
        board > 0,
        np.log2(np.maximum(board, 1)).astype(np.int32),
        0,
    )
    levels = np.clip(levels, 0, 15)  # guard against future tiles > 2^15

    # Fancy-index to set the active channel for each (row, col) position
    rows = np.arange(4).reshape(4, 1)  # (4, 1) → broadcasts to (4, 4)
    cols = np.arange(4).reshape(1, 4)  # (1, 4) → broadcasts to (4, 4)
    out[levels, rows, cols] = 1.0

    return out
