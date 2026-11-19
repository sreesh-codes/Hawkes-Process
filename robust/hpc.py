r"""HPC uncertainty simulation: propagate :math:`\theta\sim\mathcal N(\hat\theta,\Sigma)`.

Part 8. The point-estimate backtest answers "what happens on the calibrated
market". The decision-relevant question is instead: **given that we do not know
the true Hawkes parameters, how does the robust market maker's performance
*distribute*?** Each Monte-Carlo replicate therefore

1. draws a *true* market :math:`\theta^{(k)}\sim\mathcal N(\hat\theta,\Sigma_\theta)`
   per side (rejecting draws that are non-stationary or non-positive);
2. generates order flow from :math:`\theta^{(k)}`;
3. runs the **robust** strategy (Method B, variance-penalised, quoting on the
   central estimate :math:`\hat\theta`); and
4. records PnL, terminal-inventory variance proxy, Sharpe and maximum drawdown.

The result is the *predictive* distribution of each performance metric under
parameter uncertainty. Replicates are embarrassingly parallel and dispatched with
:mod:`joblib` (``ProcessPoolExecutor``-style); the driver scales to the
:math:`10^5` replicates requested by the brief, and the number actually run is
recorded with the output so the reported distributions are never overstated.

Sampling is done in the natural parameterisation with a Cholesky factor of
:math:`\Sigma_\theta`; because the marginal standard errors are small relative to
the estimates, the vast majority of draws are valid and the rejection rate is
negligible (reported in the summary).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from market_maker import ASParameters
from analysis.robust_backtest import (StrategySpec, make_belief_engines,
                                       run_robust_backtest)

__all__ = ["HPCResult", "sample_theta", "run_hpc"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HPCResult:
    """Distributions from the HPC uncertainty simulation.

    Attributes
    ----------
    pnl, inv_var, sharpe, max_drawdown : numpy.ndarray
        One value per successful replicate.
    n_requested, n_success : int
        Replicates requested and successfully simulated.
    n_rejected_draws : int
        Parameter draws rejected for non-stationarity / non-positivity.
    """

    pnl: FloatArray
    inv_var: FloatArray
    sharpe: FloatArray
    max_drawdown: FloatArray
    n_requested: int
    n_success: int
    n_rejected_draws: int

    def summary(self) -> dict[str, dict[str, float]]:
        """Mean / std / 5-50-95 percentiles for each metric."""
        out = {}
        for name, arr in [("pnl", self.pnl), ("inv_var", self.inv_var),
                          ("sharpe", self.sharpe), ("max_drawdown", self.max_drawdown)]:
            a = arr[np.isfinite(arr)]
            p5, p50, p95 = np.percentile(a, [5, 50, 95])
            out[name] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                         "p5": float(p5), "p50": float(p50), "p95": float(p95)}
        return out


def sample_theta(theta_hat: FloatArray, chol: FloatArray, rng: np.random.Generator,
                 *, max_tries: int = 20) -> tuple[FloatArray, int]:
    r"""Draw one valid :math:`\theta\sim\mathcal N(\hat\theta,\Sigma)`.

    A draw is valid iff :math:`\mu,\alpha,\beta>0` and the branching ratio
    :math:`\alpha/\beta<1` (stationarity). Returns ``(theta, n_rejected)``; falls
    back to :math:`\hat\theta` if no valid draw is found within ``max_tries``.
    """
    rejected = 0
    for _ in range(max_tries):
        z = rng.standard_normal(3)
        theta = theta_hat + chol @ z
        if theta[0] > 0 and theta[1] > 0 and theta[2] > 0 and theta[1] < theta[2]:
            return theta, rejected
        rejected += 1
    return theta_hat.copy(), rejected


def _one_sim(k: int, theta_buy_hat, theta_sell_hat, chol_buy, chol_sell,
             cov_buy, cov_sell, params, c, impact, T, s0):
    """One HPC replicate: sample a true market, run the robust strategy."""
    from analysis.comparison import compute_metrics
    from robust.uncertainty import SidedBeliefs

    rng = np.random.default_rng(2_000_000 + k)
    true_buy, r1 = sample_theta(theta_buy_hat, chol_buy, rng)
    true_sell, r2 = sample_theta(theta_sell_hat, chol_sell, rng)

    from market_maker.hawkes_execution import HawkesIntensity
    from market_maker.historical_replay import generate_synthetic_events
    events = generate_synthetic_events(HawkesIntensity(*true_buy),
                                       HawkesIntensity(*true_sell), T, seed=k)

    # Maker quotes robustly on the CENTRAL estimate (it does not know the draw).
    beliefs = SidedBeliefs.from_covariances(theta_buy_hat, cov_buy,
                                            theta_sell_hat, cov_sell)
    from robust.delta_method import lambda_bar
    from robust.robust_quotes import RobustHawkesASPolicy
    lbb = lambda_bar(theta_buy_hat); lbs = lambda_bar(theta_sell_hat)
    policy = RobustHawkesASPolicy(params, lbb, lbs, c=c)
    sr, sv = make_belief_engines(theta_buy_hat, theta_sell_hat, cov_buy, cov_sell)
    res = run_robust_backtest(events, policy, sr, sv, params, label="Robust",
                              synthetic=True, use_variance=True, impact=impact,
                              s0=s0, seed=1000 + k)
    m = compute_metrics(res)
    return (m["total_pnl"], m["inv_var"], m["sharpe"], m["max_drawdown"],
            r1 + r2)


def run_hpc(theta_buy_hat: FloatArray, theta_sell_hat: FloatArray,
            cov_buy: FloatArray, cov_sell: FloatArray, params: ASParameters, *,
            n_sims: int = 100_000, c: float = 1.0, impact: float = 0.002,
            T: float = 600.0, s0: float = 585.0, n_jobs: int = -1,
            batch: int | None = None) -> HPCResult:
    r"""Run the HPC uncertainty simulation.

    Parameters
    ----------
    theta_buy_hat, theta_sell_hat : numpy.ndarray
        Central per-side estimates :math:`\hat\theta`.
    cov_buy, cov_sell : numpy.ndarray
        Per-side covariances :math:`\Sigma_\theta` to sample from.
    params : ASParameters
        AS parameters.
    n_sims : int, optional
        Number of replicates (default :math:`10^5`; the brief's target).
    c : float, optional
        Robustness weight for the robust strategy (default 1.0).
    impact, T, s0 : float, optional
        Permanent impact ($), horizon (s), initial mid ($).
    n_jobs : int, optional
        Worker processes for :mod:`joblib`.
    batch : int, optional
        If given, dispatch in batches of this size (limits peak memory for very
        large ``n_sims``).

    Returns
    -------
    HPCResult
        The metric distributions and bookkeeping.
    """
    from joblib import Parallel, delayed

    chol_buy = np.linalg.cholesky(cov_buy)
    chol_sell = np.linalg.cholesky(cov_sell)

    def _dispatch(ks):
        return Parallel(n_jobs=n_jobs)(
            delayed(_one_sim)(k, theta_buy_hat, theta_sell_hat, chol_buy,
                              chol_sell, cov_buy, cov_sell, params, c, impact,
                              T, s0) for k in ks)

    ks_all = range(n_sims)
    if batch is None:
        out = _dispatch(ks_all)
    else:
        out = []
        for start in range(0, n_sims, batch):
            out.extend(_dispatch(range(start, min(start + batch, n_sims))))

    arr = np.array(out, dtype=float)
    return HPCResult(
        pnl=arr[:, 0], inv_var=arr[:, 1], sharpe=arr[:, 2],
        max_drawdown=arr[:, 3], n_requested=n_sims, n_success=arr.shape[0],
        n_rejected_draws=int(arr[:, 4].sum()),
    )
