"""Reward shaping functions for the 2048 RL agent."""

from __future__ import annotations

import math

import numpy as np


def monotonicity_score(board: np.ndarray) -> float:
    """Measure how well tiles are ordered along rows and columns.

    For each row and column, we check how many adjacent pairs are already
    in non-decreasing or non-increasing order, then take the better direction.
    The result is averaged over all 8 sequences (4 rows + 4 columns).

    Returns:
        A float in [0, 1].  1.0 means every row and column is fully sorted
        in some direction; 0.0 means no adjacent pair is in order anywhere.
    """
    scores: list[float] = []
    for i in range(4):
        for seq in (board[i, :], board[:, i]):
            n = len(seq)
            asc = sum(1 for j in range(n - 1) if seq[j] <= seq[j + 1])
            desc = sum(1 for j in range(n - 1) if seq[j] >= seq[j + 1])
            scores.append(max(asc, desc) / (n - 1))
    return float(np.mean(scores))


def compute_reward(
    board: np.ndarray,
    score_gained: int,
    moved: bool,
    config,  # TrainingConfig — not type-annotated to avoid circular import
) -> float:
    """Compute the shaped reward for one environment step.

    Components (each weighted by the corresponding config field):
        r_merge  : log2(score_gained) when a merge occurred (0 if no merge).
        r_empty  : fraction of empty cells on the board  ∈ [0, 1].
        r_mono   : monotonicity_score(board)              ∈ [0, 1].

    When the move is invalid (no tile moved), returns config.w_invalid
    immediately without computing the other components.

    Args:
        board:        The board state *after* the step (raw (4,4) int array).
        score_gained: Score delta gained on this step.
        moved:        True if any tile actually moved.
        config:       TrainingConfig instance holding the weights.

    Returns:
        A shaped reward as a Python float.
    """
    if not moved:
        return float(config.w_invalid)

    if score_gained > 0:
        log_merge = math.log2(score_gained)
        if getattr(config, "use_superlinear_merge_reward", False):
            r_merge = log_merge ** 2
        else:
            r_merge = log_merge
    else:
        r_merge = 0.0

    r_empty = float(np.sum(board == 0)) / 16.0
    r_mono = monotonicity_score(board)
    r_survival = float(getattr(config, "w_survival", 0.0))

    return float(
        config.w_merge * r_merge
        + config.w_empty * r_empty
        + config.w_mono * r_mono
        + r_survival
    )
