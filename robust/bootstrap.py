r"""Parametric bootstrap for the Hawkes estimation covariance (Phase 6, Part 1).

Given the fitted parameters :math:`\hat\theta=(\hat\mu,\hat\alpha,\hat\beta)` and
the observation horizon :math:`T`, the parametric bootstrap treats the fitted
model as ground truth and repeats the whole estimation experiment:

1. simulate an independent event stream :math:`\{t_i^{(b)}\}` from
   :math:`\text{Hawkes}(\hat\theta)` on :math:`[0, T]` (Ogata thinning,
   :func:`simulation.hawkes_simulator.simulate_hawkes`);
2. re-estimate :math:`\hat\theta^{(b)}` by maximum likelihood
   (:func:`hawkes.estimator.fit_hawkes`);
3. repeat for :math:`b = 1,\dots,B` and take the empirical mean and covariance of
   the :math:`\{\hat\theta^{(b)}\}`.

The empirical covariance

.. math::

    \Sigma_\theta^{\text{boot}}
      = \frac{1}{B-1}\sum_{b}(\hat\theta^{(b)}-\bar\theta)(\hat\theta^{(b)}-\bar\theta)^\top

is a distribution-free estimate of the sampling covariance: unlike the Fisher
approximation it does not assume Gaussianity and it exposes finite-sample
**bias** (:math:`\bar\theta-\hat\theta`) and skew. Its cost is :math:`B` full MLE
refits, which is why the refits are parallelised with :mod:`joblib`.

This is the *same machinery* as the Phase-3 Monte-Carlo recovery study, reused
here at the fitted operating point to quantify estimation uncertainty rather than
to validate the estimator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hawkes.estimator import fit_hawkes
from robust.covariance import PARAM_NAMES, CovarianceResult
from simulation.hawkes_simulator import simulate_hawkes

__all__ = ["BootstrapResult", "run_parametric_bootstrap"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class BootstrapResult:
    r"""Outcome of a parametric-bootstrap covariance estimate.

    Attributes
    ----------
    theta_hat : numpy.ndarray
        The fitted parameters the bootstrap was centred on.
    estimates : numpy.ndarray
        ``(B_success, 3)`` array of successful bootstrap estimates.
    n_requested, n_success : int
        Number of bootstrap datasets requested and successfully refit.
    T : float
        Horizon each dataset was simulated on.
    """

    theta_hat: FloatArray
    estimates: FloatArray
    n_requested: int
    n_success: int
    T: float

    @property
    def mean(self) -> FloatArray:
        """Bootstrap mean :math:`\\bar\\theta` of the successful estimates."""
        return self.estimates.mean(axis=0)

    @property
    def bias(self) -> FloatArray:
        r"""Estimated bias :math:`\bar\theta - \hat\theta`."""
        return self.mean - self.theta_hat

    @property
    def cov(self) -> FloatArray:
        r"""Empirical covariance :math:`\Sigma_\theta^{\text{boot}}` (ddof=1)."""
        return np.cov(self.estimates, rowvar=False, ddof=1)

    @property
    def percentiles(self) -> dict[str, tuple[float, float, float]]:
        """Per-parameter ``(2.5%, 50%, 97.5%)`` bootstrap percentiles."""
        qs = np.percentile(self.estimates, [2.5, 50, 97.5], axis=0)
        return {name: (float(qs[0, i]), float(qs[1, i]), float(qs[2, i]))
                for i, name in enumerate(PARAM_NAMES)}

    def as_covariance_result(self) -> CovarianceResult:
        """Wrap as a :class:`~robust.covariance.CovarianceResult` for uniform use."""
        return CovarianceResult(
            theta=self.theta_hat, cov=self.cov, method="bootstrap",
            n_events=int(round(self.theta_hat[0])) if False else -1,
            extra={"n_success": self.n_success, "bias": self.bias,
                   "mean": self.mean, "T": self.T},
        )


def _one_bootstrap(theta_hat: FloatArray, T: float, seed: int,
                   n_starts: int) -> tuple[float, float, float] | None:
    """Simulate one dataset from ``theta_hat`` and refit; ``None`` on failure."""
    mu, alpha, beta = float(theta_hat[0]), float(theta_hat[1]), float(theta_hat[2])
    t = simulate_hawkes(mu, alpha, beta, T, seed)
    if t.shape[0] < 10:
        return None
    res = fit_hawkes(t, T, n_starts=n_starts)
    if not res.success:
        return None
    return res.mu, res.alpha, res.beta


def run_parametric_bootstrap(
    theta_hat: FloatArray,
    T: float,
    *,
    n_boot: int = 1000,
    n_starts: int = 3,
    n_jobs: int = -1,
    seed0: int = 100_000,
) -> BootstrapResult:
    r"""Run a parametric bootstrap of the Hawkes MLE.

    Parameters
    ----------
    theta_hat : numpy.ndarray
        Fitted parameters :math:`(\hat\mu, \hat\alpha, \hat\beta)` to simulate from.
    T : float
        Observation horizon (matching the original fit).
    n_boot : int, optional
        Number of bootstrap datasets (default ``1000``; the Phase-6 spec asks for
        at least this many).
    n_starts : int, optional
        Multi-start count for each refit (default 3; fewer than the headline fit
        keeps :math:`B` refits affordable while retaining robustness).
    n_jobs : int, optional
        Worker processes for :mod:`joblib` (default all cores).
    seed0 : int, optional
        Base simulation seed; dataset ``b`` uses ``seed0 + b``.

    Returns
    -------
    BootstrapResult
        The successful estimates and their empirical mean / covariance.
    """
    from joblib import Parallel, delayed

    theta_hat = np.asarray(theta_hat, dtype=float)
    out = Parallel(n_jobs=n_jobs)(
        delayed(_one_bootstrap)(theta_hat, T, seed0 + b, n_starts)
        for b in range(n_boot)
    )
    estimates = np.array([r for r in out if r is not None], dtype=float)
    return BootstrapResult(
        theta_hat=theta_hat, estimates=estimates, n_requested=n_boot,
        n_success=estimates.shape[0], T=T,
    )
