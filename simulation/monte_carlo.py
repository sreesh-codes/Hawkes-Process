"""Monte-Carlo engine for validating the Hawkes MLE estimator.

Every trial is fully independent: simulate one synthetic Hawkes path from known
("ground truth") parameters, then re-estimate the parameters from that path. The
spread of the recovered estimates around the truth quantifies the estimator's
bias and variance.

Parallelism
-----------
Trials are embarrassingly parallel. We use :class:`joblib.Parallel` (loky
process backend) and auto-detect cores. The same per-trial function is exposed
through a command-line interface designed for an **HPC SLURM job array**: each
array task computes a disjoint slice of trials (seeds are derived from a single
:class:`numpy.random.SeedSequence`, so results are independent *and* reproducible
regardless of how the work is partitioned) and writes a part file; a ``--merge``
mode concatenates the parts.

Example (SLURM array of 50 tasks, 10000 trials)::

    #SBATCH --array=0-49
    python -m simulation.monte_carlo run --experiment main \\
        --n-trials 10000 --task-id $SLURM_ARRAY_TASK_ID --n-tasks 50
    # then, once all tasks finish:
    python -m simulation.monte_carlo merge --experiment main --n-trials 10000
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed

# Allow running as a plain script (python simulation/monte_carlo.py ...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawkes import GROUND_TRUTH, HawkesParams  # noqa: E402
from hawkes.estimator import fit_hawkes  # noqa: E402
from simulation.hawkes_simulator import simulate_hawkes  # noqa: E402

__all__ = ["TrialResult", "horizon_for_target_events", "run_trial",
           "run_monte_carlo", "detect_cores"]

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


@dataclass(frozen=True)
class TrialResult:
    """Estimates from one Monte-Carlo trial."""

    trial: int
    seed: int
    n_events: int
    mu_hat: float
    alpha_hat: float
    beta_hat: float
    rho_hat: float
    log_likelihood: float
    success: bool


def detect_cores() -> int:
    """Return the number of usable CPU cores.

    Honours the SLURM allocation (``SLURM_CPUS_PER_TASK``) when present, else
    falls back to :func:`os.cpu_count`.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm is not None and slurm.isdigit():
        return int(slurm)
    return os.cpu_count() or 1


def horizon_for_target_events(params: HawkesParams, n_target: int) -> float:
    """Observation horizon ``T`` giving ``n_target`` expected events.

    Uses the stationary mean intensity ``mu / (1 - rho)`` so that sample size is
    decoupled from the branching ratio (essential for the stress test, where
    ``rho`` varies but expected sample size is held fixed).

    Parameters
    ----------
    params : HawkesParams
        Data-generating parameters.
    n_target : int
        Desired expected number of events.

    Returns
    -------
    float
        Horizon ``T`` in seconds.
    """
    return n_target / params.mean_intensity


def run_trial(seed: int, params: HawkesParams, T: float, trial: int = 0) -> TrialResult:
    """Run one independent simulate-then-estimate trial.

    Parameters
    ----------
    seed : int
        Random seed for the simulation (well-dispersed, from a SeedSequence).
    params : HawkesParams
        Ground-truth data-generating parameters.
    T : float
        Simulation/observation horizon.
    trial : int, optional
        Trial index (for bookkeeping).

    Returns
    -------
    TrialResult
        Recovered estimates and a success flag.
    """
    events = simulate_hawkes(params.mu, params.alpha, params.beta, T, seed)
    fit = fit_hawkes(events, T)
    return TrialResult(
        trial=trial, seed=int(seed), n_events=len(events),
        mu_hat=fit.mu, alpha_hat=fit.alpha, beta_hat=fit.beta,
        rho_hat=fit.rho, log_likelihood=fit.log_likelihood,
        success=fit.success,
    )


def _make_seeds(base_seed: int, n: int) -> npt.NDArray[np.uint32]:
    """Generate ``n`` well-dispersed, reproducible seeds from ``base_seed``."""
    return np.random.SeedSequence(base_seed).generate_state(n)


def run_monte_carlo(
    params: HawkesParams,
    n_trials: int,
    T: float,
    *,
    n_jobs: int | None = None,
    base_seed: int = 20240621,
    indices: npt.NDArray[np.int64] | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """Run ``n_trials`` independent trials in parallel.

    Parameters
    ----------
    params : HawkesParams
        Ground-truth parameters.
    n_trials : int
        Total number of trials in the experiment (defines the seed space).
    T : float
        Observation horizon per trial.
    n_jobs : int, optional
        Worker processes; ``None`` auto-detects cores.
    base_seed : int, optional
        Master seed; the per-trial seeds are spawned from it deterministically.
    indices : numpy.ndarray, optional
        Subset of trial indices to run (for SLURM array partitioning). ``None``
        runs all ``n_trials``.
    verbose : int, optional
        joblib verbosity.

    Returns
    -------
    pandas.DataFrame
        One row per trial with the recovered estimates.
    """
    if n_jobs is None:
        n_jobs = detect_cores()
    all_seeds = _make_seeds(base_seed, n_trials)
    if indices is None:
        indices = np.arange(n_trials)

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_trial)(int(all_seeds[i]), params, T, int(i)) for i in indices
    )
    return pd.DataFrame([asdict(r) for r in results])


# --------------------------------------------------------------------------- #
# Experiment registry (so the CLI and the analysis code agree on definitions)
# --------------------------------------------------------------------------- #
def main_experiment_config() -> dict:
    """Definition of the headline Monte-Carlo experiment (Task 2)."""
    return {
        "params": GROUND_TRUTH,
        "n_target_events": 2000,
        "n_trials": 10000,
        "base_seed": 20240621,
    }


# --------------------------------------------------------------------------- #
# Command-line interface (SLURM-array friendly)
# --------------------------------------------------------------------------- #
def _cli() -> None:
    parser = argparse.ArgumentParser(description="Hawkes Monte-Carlo engine")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run (a slice of) the trials")
    run_p.add_argument("--experiment", default="main", choices=["main"])
    run_p.add_argument("--n-trials", type=int, default=None)
    run_p.add_argument("--task-id", type=int, default=0,
                       help="SLURM_ARRAY_TASK_ID (0-based)")
    run_p.add_argument("--n-tasks", type=int, default=1,
                       help="size of the SLURM array")
    run_p.add_argument("--n-jobs", type=int, default=None)
    run_p.add_argument("--out-dir", default=RESULTS_DIR)

    merge_p = sub.add_parser("merge", help="merge SLURM part files")
    merge_p.add_argument("--experiment", default="main", choices=["main"])
    merge_p.add_argument("--n-trials", type=int, default=None)
    merge_p.add_argument("--out-dir", default=RESULTS_DIR)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = main_experiment_config()
    n_trials = args.n_trials or cfg["n_trials"]
    params = cfg["params"]
    T = horizon_for_target_events(params, cfg["n_target_events"])

    if args.command == "run":
        # contiguous block partition: task k handles a near-equal slice
        idx = np.array_split(np.arange(n_trials), args.n_tasks)[args.task_id]
        print(f"[task {args.task_id}/{args.n_tasks}] {len(idx)} trials, "
              f"T={T:.1f}s, cores={args.n_jobs or detect_cores()}")
        df = run_monte_carlo(params, n_trials, T, n_jobs=args.n_jobs,
                             base_seed=cfg["base_seed"], indices=idx, verbose=5)
        out = os.path.join(args.out_dir,
                           f"{args.experiment}_part{args.task_id:04d}.parquet")
        df.to_parquet(out)
        print(f"wrote {out}  ({len(df)} rows, {df.success.mean():.1%} success)")

    elif args.command == "merge":
        import glob
        parts = sorted(glob.glob(
            os.path.join(args.out_dir, f"{args.experiment}_part*.parquet")))
        df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
        out = os.path.join(args.out_dir, f"{args.experiment}_results.parquet")
        df.to_parquet(out)
        print(f"merged {len(parts)} parts -> {out}  ({len(df)} rows)")


if __name__ == "__main__":
    _cli()
