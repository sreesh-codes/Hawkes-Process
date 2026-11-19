"""Core Hawkes-process package: shared estimator and ground-truth parameters.

This package is reused across Phase 2 (calibration) and Phase 3 (Monte-Carlo
validation). Keeping the estimator in one place guarantees that the *same* code
that fitted the real AAPL data also fits every synthetic data set, so the
validation is self-consistent.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HawkesParams:
    """Parameters of a univariate exponential Hawkes process.

    Attributes
    ----------
    mu : float
        Baseline (exogenous) intensity, events per second.
    alpha : float
        Excitation jump: instantaneous increase in intensity after each event.
    beta : float
        Exponential decay rate of the excitation, per second.
    """

    mu: float
    alpha: float
    beta: float

    @property
    def branching_ratio(self) -> float:
        """Branching ratio ``rho = alpha / beta`` (stationary iff ``rho < 1``)."""
        return self.alpha / self.beta

    @property
    def mean_intensity(self) -> float:
        """Stationary mean intensity ``mu / (1 - rho)`` (events per second)."""
        return self.mu / (1.0 - self.branching_ratio)


# --------------------------------------------------------------------------- #
# Ground truth: the MLE fit to the real LOBSTER AAPL 2012-06-21 trade flow
# (Phase 2, hawkes_mle.py). Treated as the data-generating truth for Phase 3.
# --------------------------------------------------------------------------- #
GROUND_TRUTH = HawkesParams(mu=0.567867, alpha=210.583156, beta=452.349905)
