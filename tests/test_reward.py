"""Unit tests for reward shaping and monotonicity scoring."""

import math

import numpy as np
import pytest

from rl.reward import compute_reward, monotonicity_score
from training.config import TrainingConfig, TrainingConfigV2, TrainingConfigV3, linear_schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> TrainingConfig:
    """Return a TrainingConfig with selective field overrides."""
    return TrainingConfig(**overrides)


# ---------------------------------------------------------------------------
# monotonicity_score
# ---------------------------------------------------------------------------

class TestMonotonicityScore:
    def test_returns_float_in_unit_interval(self) -> None:
        board = np.random.randint(0, 2048, size=(4, 4), dtype=np.int32)
        score = monotonicity_score(board)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_fully_sorted_board_scores_one(self) -> None:
        """A board where every row and column is monotone → score = 1.0."""
        board = np.array(
            [
                [2048, 1024, 512, 256],
                [1024,  512, 256, 128],
                [ 512,  256, 128,  64],
                [ 256,  128,  64,  32],
            ],
            dtype=np.int32,
        )
        assert monotonicity_score(board) == pytest.approx(1.0)

    def test_monotone_board_beats_random_board(self) -> None:
        """Monotone board should consistently score higher than a shuffled one."""
        monotone = np.array(
            [
                [2048, 1024, 512, 256],
                [1024,  512, 256, 128],
                [ 512,  256, 128,  64],
                [ 256,  128,  64,  32],
            ],
            dtype=np.int32,
        )
        random_board = np.array(
            [
                [  2, 512,   4, 128],
                [ 64,   8, 256,  16],
                [ 32, 128,   2, 512],
                [  4,  16,  64,   8],
            ],
            dtype=np.int32,
        )
        assert monotonicity_score(monotone) > monotonicity_score(random_board)

    def test_all_zeros_board(self) -> None:
        """A blank board is trivially monotone (0 ≤ 0 in every direction)."""
        board = np.zeros((4, 4), dtype=np.int32)
        assert monotonicity_score(board) == pytest.approx(1.0)

    def test_alternating_board_is_less_than_one(self) -> None:
        """An alternating board cannot be fully sorted."""
        board = np.array(
            [
                [2, 4096, 2, 4096],
                [4096, 2, 4096, 2],
                [2, 4096, 2, 4096],
                [4096, 2, 4096, 2],
            ],
            dtype=np.int32,
        )
        assert monotonicity_score(board) < 1.0

    def test_ascending_row_scores_one(self) -> None:
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, :] = [2, 4, 8, 16]
        # Row 0 is ascending, columns are all [x, 0, 0, 0] → also monotone.
        # Overall score should be 1.0.
        assert monotonicity_score(board) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_reward — invalid moves
# ---------------------------------------------------------------------------

class TestInvalidMoveReward:
    def test_invalid_move_returns_w_invalid(self) -> None:
        config = _cfg(w_invalid=-1.0)
        board = np.zeros((4, 4), dtype=np.int32)
        reward = compute_reward(board, score_gained=0, moved=False, config=config)
        assert reward == pytest.approx(config.w_invalid)

    def test_invalid_move_custom_penalty(self) -> None:
        config = _cfg(w_invalid=-5.0)
        board = np.zeros((4, 4), dtype=np.int32)
        reward = compute_reward(board, score_gained=0, moved=False, config=config)
        assert reward == pytest.approx(-5.0)

    def test_invalid_move_ignores_board_content(self) -> None:
        config = _cfg(w_invalid=-2.0)
        # Even a "good" board gives the fixed penalty for an invalid move
        board = np.array(
            [[2048, 1024, 512, 256], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int32,
        )
        reward = compute_reward(board, score_gained=100, moved=False, config=config)
        assert reward == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# compute_reward — merge component
# ---------------------------------------------------------------------------

class TestMergeComponent:
    def test_merge_of_two_1024s_gives_log2_2048(self) -> None:
        """Merging two 1024-tiles yields score_gained=2048; r_merge = log2(2048) = 11."""
        config = _cfg(w_merge=1.0, w_empty=0.0, w_mono=0.0)
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, 0] = 2048  # result of the merge
        reward = compute_reward(board, score_gained=2048, moved=True, config=config)
        assert reward == pytest.approx(math.log2(2048))  # = 11.0

    def test_no_merge_gives_zero_merge_component(self) -> None:
        config = _cfg(w_merge=1.0, w_empty=0.0, w_mono=0.0)
        board = np.zeros((4, 4), dtype=np.int32)
        reward = compute_reward(board, score_gained=0, moved=True, config=config)
        assert reward == pytest.approx(0.0)

    def test_merge_weight_scales_proportionally(self) -> None:
        config_1 = _cfg(w_merge=1.0, w_empty=0.0, w_mono=0.0)
        config_2 = _cfg(w_merge=2.0, w_empty=0.0, w_mono=0.0)
        board = np.zeros((4, 4), dtype=np.int32)
        r1 = compute_reward(board, score_gained=64, moved=True, config=config_1)
        r2 = compute_reward(board, score_gained=64, moved=True, config=config_2)
        assert r2 == pytest.approx(2.0 * r1)


# ---------------------------------------------------------------------------
# compute_reward — empty-cells component
# ---------------------------------------------------------------------------

class TestEmptyComponent:
    def test_full_empty_board_gives_one(self) -> None:
        config = _cfg(w_merge=0.0, w_empty=1.0, w_mono=0.0)
        board = np.zeros((4, 4), dtype=np.int32)
        reward = compute_reward(board, score_gained=0, moved=True, config=config)
        assert reward == pytest.approx(1.0)  # 16/16 = 1.0

    def test_half_empty_board(self) -> None:
        config = _cfg(w_merge=0.0, w_empty=1.0, w_mono=0.0)
        board = np.zeros((4, 4), dtype=np.int32)
        board[:2, :] = 2  # 8 cells non-zero
        reward = compute_reward(board, score_gained=0, moved=True, config=config)
        assert reward == pytest.approx(0.5)  # 8/16


# ---------------------------------------------------------------------------
# compute_reward — combined reward
# ---------------------------------------------------------------------------

class TestCombinedReward:
    def test_reward_is_float(self) -> None:
        config = TrainingConfig()
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, 0] = 4
        reward = compute_reward(board, score_gained=4, moved=True, config=config)
        assert isinstance(reward, float)

    def test_higher_merge_gives_higher_reward(self) -> None:
        """Merging into 512 should reward more than merging into 4."""
        config = _cfg(w_merge=1.0, w_empty=0.0, w_mono=0.0)
        board = np.zeros((4, 4), dtype=np.int32)
        r_small = compute_reward(board, score_gained=4, moved=True, config=config)
        r_large = compute_reward(board, score_gained=512, moved=True, config=config)
        assert r_large > r_small

    def test_all_weights_zero_valid_move_returns_zero(self) -> None:
        config = _cfg(w_merge=0.0, w_empty=0.0, w_mono=0.0)
        board = np.zeros((4, 4), dtype=np.int32)
        reward = compute_reward(board, score_gained=0, moved=True, config=config)
        assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_reward — superlinear merge reward (TrainingConfigV2)
# ---------------------------------------------------------------------------

class TestSuperlinearMergeReward:
    def _v2(self, **overrides) -> TrainingConfigV2:
        return TrainingConfigV2(
            w_merge=1.0, w_empty=0.0, w_mono=0.0, w_survival=0.0, **overrides
        )

    def test_superlinear_reward_scales_with_tile_value(self) -> None:
        """log2(score)^2 must grow much faster than linearly."""
        config = self._v2(use_superlinear_merge_reward=True)
        board = np.zeros((4, 4), dtype=np.int32)
        # fusion 4+4 → score_gained=8, log2(8)^2 = 9
        r_small = compute_reward(board, score_gained=8, moved=True, config=config)
        # fusion 512+512 → score_gained=1024, log2(1024)^2 = 100
        r_large = compute_reward(board, score_gained=1024, moved=True, config=config)
        ratio = r_large / r_small
        assert ratio > 10

    def test_superlinear_larger_than_linear_for_big_merges(self) -> None:
        """For a 256+256 merge, superlinear reward must exceed linear."""
        cfg_linear = self._v2(use_superlinear_merge_reward=False)
        cfg_super = self._v2(use_superlinear_merge_reward=True)
        board = np.zeros((4, 4), dtype=np.int32)
        r_lin = compute_reward(board, score_gained=512, moved=True, config=cfg_linear)
        r_sup = compute_reward(board, score_gained=512, moved=True, config=cfg_super)
        assert r_sup > r_lin

    def test_v1_config_unchanged_by_superlinear_flag(self) -> None:
        """TrainingConfig (v1) must behave exactly as before — linear merge reward."""
        config = _cfg(w_merge=1.0, w_empty=0.0, w_mono=0.0)
        board = np.zeros((4, 4), dtype=np.int32)
        reward = compute_reward(board, score_gained=2048, moved=True, config=config)
        assert reward == pytest.approx(math.log2(2048))  # = 11.0, not 121.0


# ---------------------------------------------------------------------------
# compute_reward — survival bonus
# ---------------------------------------------------------------------------

class TestSurvivalBonus:
    def test_survival_bonus_on_valid_move(self) -> None:
        """A valid move with w_survival set must include the bonus."""
        config = TrainingConfigV2(
            w_merge=0.0, w_empty=0.0, w_mono=0.0,
            w_survival=0.01, use_superlinear_merge_reward=False,
        )
        board = np.zeros((4, 4), dtype=np.int32)
        reward = compute_reward(board, score_gained=0, moved=True, config=config)
        assert reward >= config.w_survival

    def test_no_survival_bonus_on_invalid_move(self) -> None:
        """Invalid moves must NOT receive the survival bonus."""
        config = TrainingConfigV2(w_survival=0.01)
        board = np.zeros((4, 4), dtype=np.int32)
        reward = compute_reward(board, score_gained=0, moved=False, config=config)
        assert reward == pytest.approx(config.w_invalid)

    def test_v1_config_no_survival_bonus(self) -> None:
        """TrainingConfig (v1) has no w_survival — reward must not change."""
        config_no_survival = _cfg(w_merge=0.0, w_empty=0.0, w_mono=0.0)
        board = np.zeros((4, 4), dtype=np.int32)
        reward = compute_reward(board, score_gained=0, moved=True, config=config_no_survival)
        assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# linear_schedule
# ---------------------------------------------------------------------------

class TestLinearSchedule:
    def test_schedule_at_start_equals_initial_value(self) -> None:
        schedule = linear_schedule(3e-4, 5e-5)
        assert schedule(1.0) == pytest.approx(3e-4)

    def test_schedule_at_end_equals_final_value(self) -> None:
        schedule = linear_schedule(3e-4, 5e-5)
        assert schedule(0.0) == pytest.approx(5e-5)

    def test_schedule_is_monotone_decreasing(self) -> None:
        schedule = linear_schedule(3e-4, 5e-5)
        values = [schedule(p) for p in [1.0, 0.75, 0.5, 0.25, 0.0]]
        assert values == sorted(values, reverse=True)

    def test_schedule_midpoint(self) -> None:
        schedule = linear_schedule(1.0, 0.0)
        assert schedule(0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TrainingConfigV3 — defaults and to_dict
# ---------------------------------------------------------------------------

class TestTrainingConfigV3:
    def test_inherits_from_v2(self) -> None:
        config = TrainingConfigV3()
        assert isinstance(config, TrainingConfigV2)
        assert isinstance(config, TrainingConfig)

    def test_v3_defaults(self) -> None:
        config = TrainingConfigV3()
        assert config.learning_rate == 2.5e-4
        assert config.n_steps == 4096
        assert config.n_epochs == 4
        assert config.gae_lambda == 0.9
        assert config.max_grad_norm == 0.5
        assert config.ent_coef == 0.01
        assert config.use_lr_schedule is False
        assert config.w_survival == 0.005
        assert config.w_mono == 0.4
        assert config.use_vec_normalize is True
        assert config.vec_norm_obs is False
        assert config.vec_norm_reward is True

    def test_to_dict_includes_gae_lambda_and_max_grad_norm(self) -> None:
        config = TrainingConfigV3()
        d = config.to_dict()
        assert "gae_lambda" in d
        assert d["gae_lambda"] == 0.9
        assert "max_grad_norm" in d
        assert d["max_grad_norm"] == 0.5

    def test_to_dict_includes_ent_coef(self) -> None:
        config = TrainingConfigV3()
        d = config.to_dict()
        assert d["ent_coef"] == 0.01

    def test_to_dict_excludes_vec_normalize_settings(self) -> None:
        """VecNormalize settings are NOT PPO kwargs."""
        config = TrainingConfigV3()
        d = config.to_dict()
        assert "use_vec_normalize" not in d
        assert "vec_norm_obs" not in d
        assert "vec_norm_reward" not in d
        assert "vec_clip_reward" not in d

    def test_v3_reward_backward_compat_with_v1(self) -> None:
        """V1 config must still produce linear merge reward without survival bonus."""
        config_v1 = _cfg(w_merge=1.0, w_empty=0.0, w_mono=0.0)
        board = np.zeros((4, 4), dtype=np.int32)
        reward = compute_reward(board, score_gained=2048, moved=True, config=config_v1)
        assert reward == pytest.approx(math.log2(2048))

    def test_v3_superlinear_reward(self) -> None:
        """V3 inherits superlinear reward from V2."""
        config = TrainingConfigV3(w_merge=1.0, w_empty=0.0, w_mono=0.0, w_survival=0.0)
        board = np.zeros((4, 4), dtype=np.int32)
        reward = compute_reward(board, score_gained=2048, moved=True, config=config)
        assert reward == pytest.approx(math.log2(2048) ** 2)  # 121.0
