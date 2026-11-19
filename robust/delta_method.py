r"""Delta-method propagation of parameter uncertainty into the intensity.

The Phase-2 intensity is a smooth function of the parameters,

.. math::

    \lambda(t) \;=\; f(\mu,\alpha,\beta)
              \;=\; \mu + \alpha\,S_\beta(t),\qquad
    S_\beta(t) = \sum_{t_i<t} e^{-\beta (t-t_i)} .

Conditioning on the observed event times :math:`\{t_i\}` (so the only randomness
is in :math:`\hat\theta`), the **delta method** [APPROX] gives a first-order
variance for the intensity:

.. math::

    \operatorname{Var}\!\big(\lambda(t)\big)
      \;\approx\; \nabla f(t)^\top \, \Sigma_\theta \, \nabla f(t),

with the **analytic** gradient

.. math::

    \nabla f(t) =
      \Big(\; \tfrac{\partial\lambda}{\partial\mu},\;
              \tfrac{\partial\lambda}{\partial\alpha},\;
              \tfrac{\partial\lambda}{\partial\beta} \;\Big)
      = \big(\,1,\; S_\beta(t),\; -\alpha\, Q_\beta(t)\,\big),\qquad
    Q_\beta(t) = \sum_{t_i<t} (t-t_i)\,e^{-\beta (t-t_i)} .

Here :math:`\partial\lambda/\partial\beta = \alpha\,\partial S_\beta/\partial\beta`
and :math:`\partial S_\beta/\partial\beta = -\sum (t-t_i)e^{-\beta(t-t_i)} = -Q_\beta`.

An approximate two-sided :math:`(1-a)` confidence band is then

.. math::  \lambda(t) \pm z_{1-a/2}\,\sqrt{\operatorname{Var}(\lambda(t))}.

**Status.** The gradient is *exact*; the variance and the band are a *first-order
(delta-method) approximation* [APPROX] -- they neglect the curvature of
:math:`f` and the non-Gaussian tails of :math:`\hat\theta`, and they condition on
the event path (they do not add the intrinsic Poisson variability of the counting
process itself). Both :math:`S_\beta` and :math:`Q_\beta` are accumulated on a
grid in :math:`O(G+N)` via the exponential-kernel recursions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

__all__ = [
    "IntensityBand",
    "s_and_q_on_grid",
    "intensity_gradient",
    "intensity_variance_on_grid",
    "confidence_band",
    "lambda_bar",
    "lambda_bar_gradient",
    "lambda_bar_variance",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class IntensityBand:
    r"""Intensity path with a delta-method confidence band.

    Attributes
    ----------
    grid : numpy.ndarray
        Query times.
    intensity : numpy.ndarray
        Central intensity :math:`\lambda(t)` at :math:`\hat\theta`.
    std : numpy.ndarray
        Delta-method standard deviation :math:`\sqrt{\operatorname{Var}\lambda(t)}`.
    lower, upper : numpy.ndarray
        Confidence-band bounds (``lower`` clipped at 0).
    level : float
        Nominal coverage (e.g. 0.95).
    """

    grid: FloatArray
    intensity: FloatArray
    std: FloatArray
    lower: FloatArray
    upper: FloatArray
    level: float


def s_and_q_on_grid(grid: FloatArray, events: FloatArray,
                    beta: float) -> tuple[FloatArray, FloatArray]:
    r"""Accumulate :math:`S_\beta(g)` and :math:`Q_\beta(g)` on a sorted grid.

    Uses the forward recursions for advancing time by :math:`\delta` with no new
    event, :math:`S \leftarrow e^{-\beta\delta}S`,
    :math:`Q \leftarrow e^{-\beta\delta}(Q + \delta S)`, and, at an event,
    :math:`S \leftarrow S + 1`, :math:`Q \leftarrow Q` (its own age is zero).

    Parameters
    ----------
    grid : numpy.ndarray
        Sorted query times.
    events : numpy.ndarray
        Sorted event times.
    beta : float
        Decay rate.

    Returns
    -------
    tuple of numpy.ndarray
        ``(S, Q)`` evaluated at each grid point.
    """
    grid = np.asarray(grid, dtype=float)
    events = np.asarray(events, dtype=float)
    S = np.empty(grid.shape[0])
    Q = np.empty(grid.shape[0])
    s = 0.0
    q = 0.0
    j = 0
    prev = grid[0]
    for k, g in enumerate(grid):
        delta = g - prev
        if delta > 0:
            e = np.exp(-beta * delta)
            q = e * (q + delta * s)
            s = e * s
        while j < events.shape[0] and events[j] <= g:
            # advance internal clock to the event, fold it in
            de = g - events[j]  # remaining age relative to g handled by decay below
            # decay from prev-handled state is already applied to g; add contribution
            # of this event as if placed at events[j] and aged to g:
            w = np.exp(-beta * de)
            s += w
            q += de * w
            j += 1
        S[k] = s
        Q[k] = q
        prev = g
    return S, Q


def intensity_gradient(S: FloatArray, Q: FloatArray,
                       alpha: float) -> FloatArray:
    r"""Return the :math:`(\,\partial\lambda/\partial\mu,\partial\alpha,\partial\beta)` stack.

    Parameters
    ----------
    S, Q : numpy.ndarray
        :math:`S_\beta` and :math:`Q_\beta` on a grid.
    alpha : float
        Excitation parameter.

    Returns
    -------
    numpy.ndarray
        ``(G, 3)`` array whose rows are :math:`(1, S, -\alpha Q)`.
    """
    ones = np.ones_like(S)
    return np.stack([ones, S, -alpha * Q], axis=1)


def intensity_variance_on_grid(grid: FloatArray, events: FloatArray,
                               theta: FloatArray, cov: FloatArray,
                               ) -> tuple[FloatArray, FloatArray]:
    r"""Central intensity and its delta-method variance on a grid.

    Parameters
    ----------
    grid : numpy.ndarray
        Sorted query times.
    events : numpy.ndarray
        Sorted event times up to (and within) the grid.
    theta : numpy.ndarray
        Parameter vector :math:`(\mu, \alpha, \beta)`.
    cov : numpy.ndarray
        Parameter covariance :math:`\Sigma_\theta`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(lambda, variance)`` at each grid point.
    """
    mu, alpha, beta = float(theta[0]), float(theta[1]), float(theta[2])
    S, Q = s_and_q_on_grid(grid, events, beta)
    lam = mu + alpha * S
    grad = intensity_gradient(S, Q, alpha)          # (G, 3)
    # var_g = grad_g^T Sigma grad_g, vectorised over the grid
    var = np.einsum("gi,ij,gj->g", grad, cov, grad)
    return lam, np.clip(var, 0.0, np.inf)


def confidence_band(grid: FloatArray, events: FloatArray, theta: FloatArray,
                    cov: FloatArray, *, level: float = 0.95) -> IntensityBand:
    r"""Delta-method confidence band for :math:`\lambda(t)` over a grid.

    Parameters
    ----------
    grid : numpy.ndarray
        Sorted query times.
    events : numpy.ndarray
        Sorted event times.
    theta : numpy.ndarray
        Parameter vector :math:`(\mu, \alpha, \beta)`.
    cov : numpy.ndarray
        Parameter covariance.
    level : float, optional
        Nominal coverage (default 0.95).

    Returns
    -------
    IntensityBand
        Central path, standard deviation and band bounds.
    """
    lam, var = intensity_variance_on_grid(grid, events, theta, cov)
    std = np.sqrt(var)
    z = float(norm.ppf(0.5 + level / 2.0))
    lower = np.clip(lam - z * std, 0.0, np.inf)
    upper = lam + z * std
    return IntensityBand(grid=np.asarray(grid, float), intensity=lam, std=std,
                         lower=lower, upper=upper, level=level)


# --------------------------------------------------------------------------- #
# Delta method for the stationary mean intensity lambda_bar = mu / (1 - alpha/beta)
# --------------------------------------------------------------------------- #
def lambda_bar(theta: FloatArray) -> float:
    r"""Stationary mean intensity :math:`\bar\lambda = \mu/(1-\alpha/\beta)`."""
    mu, alpha, beta = float(theta[0]), float(theta[1]), float(theta[2])
    return mu / (1.0 - alpha / beta)


def lambda_bar_gradient(theta: FloatArray) -> FloatArray:
    r"""Analytic gradient of :math:`\bar\lambda` w.r.t. :math:`(\mu,\alpha,\beta)`.

    With :math:`\rho=\alpha/\beta`,
    :math:`\partial\bar\lambda/\partial\mu = 1/(1-\rho)`,
    :math:`\partial\bar\lambda/\partial\alpha = \mu/[\beta(1-\rho)^2]`,
    :math:`\partial\bar\lambda/\partial\beta = -\mu\alpha/[\beta^2(1-\rho)^2]`.
    """
    mu, alpha, beta = float(theta[0]), float(theta[1]), float(theta[2])
    rho = alpha / beta
    one_m = 1.0 - rho
    d_mu = 1.0 / one_m
    d_alpha = mu / (beta * one_m**2)
    d_beta = -mu * alpha / (beta**2 * one_m**2)
    return np.array([d_mu, d_alpha, d_beta])


def lambda_bar_variance(theta: FloatArray, cov: FloatArray) -> float:
    r"""Delta-method variance of :math:`\bar\lambda`: :math:`\nabla^\top\Sigma\nabla`."""
    g = lambda_bar_gradient(theta)
    return float(g @ cov @ g)
