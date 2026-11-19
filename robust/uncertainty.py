r"""Uncertainty propagation into intensity *beliefs* and an online variance engine.

Two things live here (Phase 6, Parts 3 and 4):

**1. Belief construction.**
    From the point estimate :math:`\hat\theta` and covariance :math:`\Sigma_\theta`
    we build *lower / central / upper* parameter beliefs. Rather than perturbing a
    single parameter, we shift :math:`\theta` along the direction that moves the
    stationary mean intensity :math:`\bar\lambda(\theta)=\mu/(1-\alpha/\beta)` by
    :math:`\pm z\,\operatorname{SE}(\bar\lambda)`, using the **minimum-Mahalanobis**
    shift consistent with :math:`\Sigma_\theta`:

    .. math::

        \theta_\pm = \hat\theta \pm z\, u,\qquad
        u = \frac{\Sigma_\theta\,\nabla\bar\lambda}
                 {\sqrt{\nabla\bar\lambda^\top \Sigma_\theta \nabla\bar\lambda}} .

    Then :math:`\nabla\bar\lambda^\top(\theta_\pm-\hat\theta)=\pm z\,
    \operatorname{SE}(\bar\lambda)` to first order [APPROX]: the *upper* belief is a
    market that is genuinely burstier (higher mean intensity) than the point
    estimate, the *lower* one calmer, each the most plausible such market given the
    estimation covariance.

**2. Online intensity-plus-variance engine.**
    :class:`VarianceHawkes` extends the Phase-5 :class:`HawkesIntensity` recursion
    with the companion sum :math:`Q_\beta(t)=\sum(t-t_i)e^{-\beta(t-t_i)}`, so that
    at any query time it returns both :math:`\lambda(t)` and the delta-method
    standard deviation :math:`\sqrt{\nabla f^\top\Sigma_\theta\nabla f}` in
    :math:`O(1)` per event -- exactly what the variance-penalised quoting policy
    (:mod:`robust.robust_quotes`, Method B) consumes during a backtest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from market_maker.hawkes_execution import HawkesIntensity
from robust.delta_method import lambda_bar, lambda_bar_gradient

__all__ = [
    "build_beliefs",
    "SidedBeliefs",
    "VarianceHawkes",
    "SidedVarianceHawkes",
]

FloatArray = npt.NDArray[np.float64]
BELIEF_NAMES = ("lower", "central", "upper")


def build_beliefs(theta: FloatArray, cov: FloatArray, *, z: float = 1.96,
                  ) -> dict[str, FloatArray]:
    r"""Construct ``lower``/``central``/``upper`` parameter beliefs.

    Parameters
    ----------
    theta : numpy.ndarray
        Point estimate :math:`(\hat\mu, \hat\alpha, \hat\beta)`.
    cov : numpy.ndarray
        Parameter covariance :math:`\Sigma_\theta`.
    z : float, optional
        Confidence multiplier (default ``1.96`` for a 95% shift in
        :math:`\bar\lambda`).

    Returns
    -------
    dict[str, numpy.ndarray]
        Beliefs keyed by name. Each is clipped to remain a valid stationary Hawkes
        model (:math:`\mu,\alpha,\beta>0`, branching ratio ``< 1``).
    """
    theta = np.asarray(theta, dtype=float)
    g = lambda_bar_gradient(theta)
    quad = float(g @ cov @ g)
    if quad <= 0:                       # degenerate covariance -> no spread
        return {name: theta.copy() for name in BELIEF_NAMES}
    u = (cov @ g) / np.sqrt(quad)       # minimum-Mahalanobis unit direction
    beliefs = {
        "lower": theta - z * u,
        "central": theta.copy(),
        "upper": theta + z * u,
    }
    return {name: _sanitize(th) for name, th in beliefs.items()}


def _sanitize(theta: FloatArray) -> FloatArray:
    """Clip a belief to a valid, stationary Hawkes parameterisation."""
    mu, alpha, beta = float(theta[0]), float(theta[1]), float(theta[2])
    mu = max(mu, 1e-8)
    beta = max(beta, 1e-8)
    alpha = min(max(alpha, 1e-8), 0.999 * beta)   # enforce rho < 1
    return np.array([mu, alpha, beta])


@dataclass(frozen=True)
class SidedBeliefs:
    r"""Per-side (buy/sell) lower/central/upper beliefs and their :math:`\bar\lambda`.

    Attributes
    ----------
    buy, sell : dict[str, numpy.ndarray]
        Belief triples keyed by ``lower``/``central``/``upper``.
    """

    buy: dict[str, FloatArray]
    sell: dict[str, FloatArray]

    @classmethod
    def from_covariances(cls, theta_buy: FloatArray, cov_buy: FloatArray,
                         theta_sell: FloatArray, cov_sell: FloatArray,
                         *, z: float = 1.96) -> "SidedBeliefs":
        """Build per-side beliefs from per-side estimates and covariances."""
        return cls(build_beliefs(theta_buy, cov_buy, z=z),
                   build_beliefs(theta_sell, cov_sell, z=z))

    def lam_bar(self, side: str, belief: str) -> float:
        r"""Stationary mean intensity :math:`\bar\lambda` for one side/belief."""
        th = (self.buy if side == "buy" else self.sell)[belief]
        return lambda_bar(th)


@dataclass
class VarianceHawkes:
    r"""Stateful intensity **and** delta-method variance evaluator.

    Extends :class:`~market_maker.hawkes_execution.HawkesIntensity` with the
    companion sum :math:`Q_\beta` needed for :math:`\partial\lambda/\partial\beta`,
    plus the parameter covariance so it can report the intensity's standard
    deviation online.

    Parameters
    ----------
    mu, alpha, beta : float
        Hawkes parameters (the belief the maker quotes on).
    cov : numpy.ndarray
        Parameter covariance :math:`\Sigma_\theta` for this side.

    Attributes
    ----------
    S : float
        Decayed excitation sum :math:`\sum e^{-\beta(t-t_i)}`.
    Q : float
        Companion sum :math:`\sum (t-t_i) e^{-\beta(t-t_i)}`.
    last_t : float
        Time to which ``S`` and ``Q`` are decayed.
    """

    mu: float
    alpha: float
    beta: float
    cov: FloatArray
    S: float = 0.0
    Q: float = 0.0
    last_t: float = 0.0

    @property
    def rho(self) -> float:
        return self.alpha / self.beta

    @property
    def mean_intensity(self) -> float:
        r"""Stationary mean :math:`\mu/(1-\rho)`."""
        return self.mu / (1.0 - self.rho)

    def _decay_to(self, t: float) -> None:
        if t > self.last_t:
            delta = t - self.last_t
            e = np.exp(-self.beta * delta)
            self.Q = e * (self.Q + delta * self.S)
            self.S = e * self.S
            self.last_t = t

    def intensity_at(self, t: float) -> float:
        r"""Instantaneous intensity :math:`\mu+\alpha S(t)`."""
        self._decay_to(t)
        return self.mu + self.alpha * self.S

    def intensity_and_std(self, t: float) -> tuple[float, float]:
        r"""Return :math:`(\lambda(t),\ \sqrt{\operatorname{Var}\lambda(t)})`.

        The standard deviation is the delta-method value
        :math:`\sqrt{\nabla f^\top\Sigma\nabla f}` with
        :math:`\nabla f=(1, S, -\alpha Q)`.
        """
        self._decay_to(t)
        lam = self.mu + self.alpha * self.S
        grad = np.array([1.0, self.S, -self.alpha * self.Q])
        var = float(grad @ self.cov @ grad)
        return lam, np.sqrt(max(var, 0.0))

    def register_event(self, t: float) -> None:
        """Record an event at ``t`` (decays, then adds unit excitation)."""
        self._decay_to(t)
        self.S += 1.0   # Q unchanged: the new event has zero age

    def reset(self, t0: float = 0.0) -> None:
        self.S = 0.0
        self.Q = 0.0
        self.last_t = t0


@dataclass
class SidedVarianceHawkes:
    """Buy/sell pair of :class:`VarianceHawkes` engines (mirrors ``SidedHawkes``)."""

    buy: VarianceHawkes
    sell: VarianceHawkes

    def intensities_at(self, t: float) -> tuple[float, float]:
        return self.buy.intensity_at(t), self.sell.intensity_at(t)

    def intensity_and_std(self, t: float) -> tuple[float, float, float, float]:
        """Return ``(lam_buy, sd_buy, lam_sell, sd_sell)`` at ``t``."""
        lb, sb = self.buy.intensity_and_std(t)
        ls, ss = self.sell.intensity_and_std(t)
        return lb, sb, ls, ss

    def register(self, t: float, side: str) -> None:
        if side == "buy":
            self.buy.register_event(t)
        elif side == "sell":
            self.sell.register_event(t)
        else:  # pragma: no cover - defensive
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    def reset(self, t0: float = 0.0) -> None:
        self.buy.reset(t0)
        self.sell.reset(t0)
