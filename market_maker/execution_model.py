r"""Exponential (Poisson) execution model for Avellaneda--Stoikov.

Why this model?
---------------
Avellaneda--Stoikov assume that **market orders arrive as a Poisson process** and
that the probability a posted quote is reached decays exponentially with its
distance from the mid. Concretely, a limit order resting at distance ``delta``
from the mid is filled with intensity

    lambda(delta) = A exp(-kappa delta).                                 (2)

Two facts motivate this choice:

1. *Distance decay (the ``exp(-kappa delta)`` core).* The deeper a quote sits in
   the book, the larger the market order needed to reach it, and large market
   orders are exponentially rarer. So fill probability falls off like
   ``exp(-kappa delta)`` -- exactly the empirical shape we fitted from LOBSTER.
2. *Poisson / memorylessness.* Treating arrivals as Poisson makes them
   **independent with a constant (per-distance) rate**, which is what makes the
   stochastic-control problem analytically tractable and yields the closed-form
   reservation price and spread.

Over a short step ``dt`` the probability of *at least one* fill is, for a Poisson
process of rate ``lambda``,

    P(fill) = 1 - exp(-lambda(delta) dt).                                (6)

The pure ``exp(-kappa delta)`` term is the relative fill-likelihood across
distances; ``A`` and ``dt`` set the absolute rate.

**This Poisson assumption is the key limitation of the classical model**: real
trade arrivals *cluster* (self-excitation), which Phases 1--3 demonstrated
statistically. Relaxing it via a Hawkes intensity is the subject of the next
phase; here we keep arrivals Poisson to reproduce the original framework exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ExecutionModel", "FillResult"]


@dataclass(frozen=True)
class FillResult:
    """Outcome of one execution step.

    Attributes
    ----------
    bid_filled : bool
        Whether the market maker's bid was hit (the MM **buys** one unit).
    ask_filled : bool
        Whether the market maker's ask was lifted (the MM **sells** one unit).
    """

    bid_filled: bool
    ask_filled: bool


@dataclass(frozen=True)
class ExecutionModel:
    """Exponential market-order arrival / fill model, eq. (2).

    Attributes
    ----------
    A : float
        Base intensity at the mid (events/second).
    kappa : float
        Distance-decay parameter (1/$).
    """

    A: float
    kappa: float

    def intensity(self, delta: float) -> float:
        r"""Arrival intensity ``lambda(delta) = A exp(-kappa delta)`` (events/s).

        Parameters
        ----------
        delta : float
            Quote distance from the mid ($). Negative distances (quote through the
            mid) are clipped to 0, giving the maximal intensity ``A``.

        Returns
        -------
        float
            Fill intensity (events/second).
        """
        return self.A * np.exp(-self.kappa * max(delta, 0.0))

    def fill_probability(self, delta: float, dt: float) -> float:
        r"""Probability of at least one fill within ``dt``, eq. (6).

        ``P(fill) = 1 - exp(-lambda(delta) dt)``.

        Parameters
        ----------
        delta : float
            Quote distance from the mid ($).
        dt : float
            Time-step length (seconds).

        Returns
        -------
        float
            Fill probability in ``[0, 1]``.
        """
        return 1.0 - np.exp(-self.intensity(delta) * dt)

    def sample_fills(self, delta_bid: float, delta_ask: float, dt: float,
                     rng: np.random.Generator) -> FillResult:
        r"""Draw bid/ask fills over one step using independent Poisson arrivals.

        Parameters
        ----------
        delta_bid : float
            Distance of the bid below the mid ($), ``mid - bid``.
        delta_ask : float
            Distance of the ask above the mid ($), ``ask - mid``.
        dt : float
            Step length (seconds).
        rng : numpy.random.Generator
            Random generator (for reproducibility).

        Returns
        -------
        FillResult
            Booleans for whether each side traded this step.
        """
        bid_filled = rng.random() < self.fill_probability(delta_bid, dt)
        ask_filled = rng.random() < self.fill_probability(delta_ask, dt)
        return FillResult(bid_filled=bid_filled, ask_filled=ask_filled)
