"""Sequential comparison of training configurations.

Launches comparison runs one after the other so that TensorBoard logs
can be compared side-by-side.  Each run saves its model under ``models/``
and its logs under ``logs/``.

Usage::

    python training/compare_runs.py

Ctrl+C during any run saves the model immediately and exits cleanly.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import TrainingConfigV3
from training.train import train

# ---------------------------------------------------------------------------
# Run definitions
# ---------------------------------------------------------------------------

COMPARISON_TIMESTEPS = 3_000_000

_BASE: dict = dict(
    total_timesteps=COMPARISON_TIMESTEPS,
    n_envs=8,
    # Disable live render during comparison runs for speed
    render_freq=99_999_999,
)

RUNS: list[dict] = [
    # Run D — V3 full (VecNormalize + tuned PPO)
    {
        "name": "runD_v3_baseline",
        "cls": TrainingConfigV3,
        "overrides": {},
    },
    # Run E — V3 without VecNormalize (ablation)
    {
        "name": "runE_v3_no_vecnorm",
        "cls": TrainingConfigV3,
        "overrides": dict(use_vec_normalize=False),
    },
    # Run F — V3 with more exploration (ent_coef=0.02)
    {
        "name": "runF_v3_ent02",
        "cls": TrainingConfigV3,
        "overrides": dict(ent_coef=0.02),
    },
]


def _build_config(overrides: dict, cls=TrainingConfigV3):
    """Merge base settings and per-run overrides into a config instance."""
    merged = {**_BASE, **overrides}
    return cls(**merged)


def main() -> None:
    results: list[dict] = []

    for i, run in enumerate(RUNS):
        name: str = run["name"]
        cls = run.get("cls", TrainingConfigV3)
        config = _build_config(run["overrides"], cls=cls)

        print("=" * 60)
        print(f"  Run {i + 1}/{len(RUNS)}: {name}")
        print("=" * 60)

        t_start = time.time()
        try:
            train(
                config=config,
                config_version="v3",
                run_name=name,
                save_path=f"models/{name}_3M",
            )
            status = "completed"
        except SystemExit:
            status = "interrupted"

        elapsed = time.time() - t_start
        results.append({"name": name, "status": status, "elapsed_s": elapsed})

        if status == "interrupted":
            print(f"\nRun {name} was interrupted — stopping comparison.")
            break

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Comparison summary")
    print("=" * 60)
    print(f"{'Run':<28}  {'Status':<12}  {'Duration'}")
    print("-" * 56)
    for r in results:
        mins, secs = divmod(int(r["elapsed_s"]), 60)
        print(f"{r['name']:<28}  {r['status']:<12}  {mins}m {secs:02d}s")
    print()
    print("TensorBoard: tensorboard --logdir ./logs")
    print()


if __name__ == "__main__":
    main()
