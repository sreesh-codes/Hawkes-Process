"""Maximum-likelihood estimator for the univariate exponential Hawkes process.

The intensity is

.. math::

    \\lambda(t) = \\mu + \\alpha \\sum_{t_i < t} e^{-\\beta (t - t_i)}.

The negative log-likelihood and its gradient are evaluated in :math:`O(N)` via
the standard exponential-kernel recursion (see Phase 2, ``hawkes_mle.py``). The
hot loop is JIT-compiled with Numba so that the tens of thousands of fits in the
Phase-3 Monte-Carlo study run in minutes rather than hours.

The optimiser is ``scipy.optimize.minimize`` with **L-BFGS-B** and positivity
box constraints -- no Hawkes library is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.optimize import minimize

__all__ = ["EstimationResult", "neg_log_likelihood", "fit_hawkes"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EstimationResult:
    """Outcome of a single Hawkes MLE fit.

    Attributes
    ----------
    mu, alpha, beta : float
        Estimated parameters.
    rho : float
        Estimated branching ratio ``alpha / beta``.
    log_likelihood : float
        Maximised log-likelihood.
    success : bool
        Whether the optimiser reported convergence *and* the estimate is finite,
        interior (off the lower bound) and stationary (``rho < 1``).
    n_events : int
        Number of events the fit was based on.
    n_starts_used : int
        Number of multi-start initialisations attempted.
    """

    mu: float
    alpha: float
    beta: float
    rho: float
    log_likelihood: float
    success: bool
    n_events: int
    n_starts_used: int


@njit(cache=True, fastmath=True)
def _nll_and_grad(
    mu: float, alpha: float, beta: float, t: FloatArray, T: float
) -> tuple[float, float, float, float]:
    """Negative log-likelihood and its gradient, in a single O(N) pass.

    Parameters
    ----------
    mu, alpha, beta : float
        Model parameters (assumed strictly positive by the caller).
    t : numpy.ndarray
        Sorted, strictly increasing event times on ``[0, T]``.
    T : float
        Observation horizon.

    Returns
    -------
    tuple of float
        ``(nll, d_nll_d_mu, d_nll_d_alpha, d_nll_d_beta)``.

    Notes
    -----
    Uses the recursions :math:`A_i = e^{-\\beta\\Delta_i}(1 + A_{i-1})` and
    :math:`B_i = \\partial A_i/\\partial\\beta = -\\Delta_i A_i +
    e^{-\\beta\\Delta_i} B_{i-1}`, accumulating all likelihood and gradient sums
    on the fly.
    """
    n = t.shape[0]
    A = 0.0          # excitation sum at current event
    B = 0.0          # dA/dbeta at current event
    sum_log = 0.0    # sum log lambda(t_i)
    sum_inv = 0.0    # sum 1 / lambda(t_i)
    sum_A_inv = 0.0  # sum A_i / lambda(t_i)
    sum_aB_inv = 0.0  # sum alpha B_i / lambda(t_i)
    sum_decay = 0.0  # sum (1 - exp(-beta (T - t_i)))
    sum_texp = 0.0   # sum (T - t_i) exp(-beta (T - t_i))

    for i in range(n):
        if i > 0:
            d = t[i] - t[i - 1]
            e = np.exp(-beta * d)
            A = e * (1.0 + A)
            B = -d * A + e * B
        lam = mu + alpha * A
        if lam < 1e-300:
            lam = 1e-300
        inv = 1.0 / lam
        sum_log += np.log(lam)
        sum_inv += inv
        sum_A_inv += A * inv
        sum_aB_inv += alpha * B * inv

        rem = T - t[i]
        ed = np.exp(-beta * rem)
        sum_decay += 1.0 - ed
        sum_texp += rem * ed

    compensator = mu * T + (alpha / beta) * sum_decay
    nll = -(sum_log - compensator)

    d_mu = -(sum_inv - T)
    d_alpha = -(sum_A_inv - sum_decay / beta)
    dcomp_dbeta = alpha * (-sum_decay / (beta * beta) + sum_texp / beta)
    d_beta = -(sum_aB_inv - dcomp_dbeta)
    return nll, d_mu, d_alpha, d_beta


def neg_log_likelihood(
    params: FloatArray, t: FloatArray, T: float
) -> tuple[float, FloatArray]:
    """SciPy-compatible objective: returns ``(nll, gradient)``.

    Parameters
    ----------
    params : numpy.ndarray
        Array ``[mu, alpha, beta]``.
    t : numpy.ndarray
        Sorted event times.
    T : float
        Observation horizon.

    Returns
    -------
    tuple
        Negative log-likelihood and its gradient as a length-3 array.
    """
    mu, alpha, beta = params
    nll, g0, g1, g2 = _nll_and_grad(mu, alpha, beta, t, T)
    return nll, np.array([g0, g1, g2])


def fit_hawkes(
    t: FloatArray,
    T: float,
    *,
    n_starts: int = 4,
    max_iter: int = 500,
) -> EstimationResult:
    """Fit ``(mu, alpha, beta)`` by maximum likelihood (L-BFGS-B, multi-start).

    Positivity is enforced with box constraints. Because the likelihood surface
    can be multi-modal in ``beta``, several scale-spanning starting points are
    tried and the best (lowest negative log-likelihood) is kept.

    Parameters
    ----------
    t : numpy.ndarray
        Sorted, strictly increasing event times on ``[0, T]``.
    T : float
        Observation horizon.
    n_starts : int, optional
        Number of initialisations (default 4), spanning ``beta`` scales.
    max_iter : int, optional
        Maximum L-BFGS-B iterations per start.

    Returns
    -------
    EstimationResult
        The best fit and its convergence/quality flag.
    """
    n = t.shape[0]
    if n < 5:
        return EstimationResult(np.nan, np.nan, np.nan, np.nan, np.nan,
                                False, n, 0)
    rate = n / T
    # Starts span several beta scales so a fast-decaying kernel is reachable.
    beta_scales = [rate, 5.0, 50.0, 500.0, 5000.0]
    starts = [
        np.array([0.5 * rate, 0.5 * b0, b0], dtype=np.float64)
        for b0 in beta_scales[:n_starts]
    ]
    bounds = [(1e-8, None), (1e-8, None), (1e-8, None)]  # mu, alpha, beta > 0

    best = None
    for x0 in starts:
        try:
            res = minimize(
                neg_log_likelihood, x0, args=(t, T), method="L-BFGS-B",
                jac=True, bounds=bounds,
                options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
            )
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        return EstimationResult(np.nan, np.nan, np.nan, np.nan, np.nan,
                                False, n, len(starts))

    mu, alpha, beta = best.x
    rho = alpha / beta
    # A fit is "good" if the optimiser converged, the estimate is finite,
    # stationary, and not pinned to the lower positivity bound.
    ok = bool(
        best.success
        and np.all(np.isfinite(best.x))
        and rho < 1.0
        and mu > 1e-7
        and beta > 1e-7
    )
    return EstimationResult(
        mu=float(mu), alpha=float(alpha), beta=float(beta), rho=float(rho),
        log_likelihood=float(-best.fun), success=ok, n_events=n,
        n_starts_used=len(starts),
    )
