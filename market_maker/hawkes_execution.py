r"""Hawkes intensity engine for the dynamic market maker.

This module turns the *calibrated* exponential-Hawkes parameters from Phase 2
into a running estimate of instantaneous market activity ``lambda(t)`` that the
quoting policy can react to.

Provenance of the mathematics (kept explicit throughout the project):

* **[HAWKES THEORY]** The conditional intensity of an exponential Hawkes process
  is ``lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta (t - t_i))`` and its
  stationary mean is ``lambda_bar = mu / (1 - rho)`` with ``rho = alpha / beta``.
  These are standard results (Hawkes 1971; Phase 2 estimated the parameters).

Nothing in this file is an Avellaneda--Stoikov result; it only *evaluates* the
Hawkes intensity efficiently, both online (event-driven, O(1) per event via the
exponential-kernel recursion) and offline (on a grid, O(G + N)).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["HawkesIntensity", "SidedHawkes", "intensity_on_grid"]

FloatArray = npt.NDArray[np.float64]


@dataclass
class HawkesIntensity:
    r"""Stateful evaluator of a univariate exponential-Hawkes intensity.

    Maintains the decayed excitation sum ``S(t) = sum_{t_i <= t} exp(-beta(t-t_i))``
    so that ``lambda(t) = mu + alpha * S(t)`` is available in O(1) per update.

    Parameters
    ----------
    mu, alpha, beta : float
        Calibrated Hawkes parameters (baseline, excitation, decay).

    Attributes
    ----------
    S : float
        Current decayed excitation sum (excludes ``mu``).
    last_t : float
        Time to which ``S`` has been decayed.
    """

    mu: float
    alpha: float
    beta: float
    S: float = 0.0
    last_t: float = 0.0

    @property
    def rho(self) -> float:
        """Branching ratio ``alpha / beta`` (**[HAWKES THEORY]**)."""
        return self.alpha / self.beta

    @property
    def mean_intensity(self) -> float:
        """Stationary mean intensity ``mu / (1 - rho)`` (**[HAWKES THEORY]**)."""
        return self.mu / (1.0 - self.rho)

    def _decay_to(self, t: float) -> None:
        """Decay the excitation sum forward to time ``t`` (internal)."""
        if t > self.last_t:
            self.S *= np.exp(-self.beta * (t - self.last_t))
            self.last_t = t

    def intensity_at(self, t: float) -> float:
        r"""Return ``lambda(t)`` without registering an event.

        Parameters
        ----------
        t : float
            Query time (>= ``last_t``).

        Returns
        -------
        float
            Instantaneous intensity ``mu + alpha * S(t)``.
        """
        self._decay_to(t)
        return self.mu + self.alpha * self.S

    def expected_rate(self, t: float, horizon: float) -> float:
        r"""Expected arrival rate over the near future ``(t, t+horizon]`` [HAWKES].

        The *instantaneous* intensity of an exponential Hawkes process jumps by
        ``alpha`` at every event and decays on the (very short) ``1/beta`` scale,
        so it is far too spiky to quote on directly. A market maker instead reacts
        to activity over a **quote-reaction window** ``horizon``. Using the
        conditional-expectation identity
        ``E[lambda(u) | F_t] = lambda_bar + (lambda(t) - lambda_bar)
        exp(-(beta - alpha)(u - t))`` (valid for a stationary process,
        ``beta > alpha``), the expected number of events in ``(t, t+h]`` is

            E[N] = lambda_bar h + (lambda(t) - lambda_bar)
                   (1 - exp(-(beta-alpha) h)) / (beta - alpha),

        and this method returns ``E[N] / h``. It equals ``lambda_bar`` at baseline
        and rises smoothly (and boundedly) during sustained bursts.

        Parameters
        ----------
        t : float
            Current time (>= ``last_t``).
        horizon : float
            Forecast / quote-reaction window (seconds).

        Returns
        -------
        float
            Expected arrival rate (events/second) over the window.
        """
        lam = self.intensity_at(t)
        gap = self.beta - self.alpha
        lam_bar = self.mean_intensity
        if gap <= 0:  # non-stationary safeguard; fall back to instantaneous
            return lam
        expected_n = lam_bar * horizon + (lam - lam_bar) * (
            1.0 - np.exp(-gap * horizon)) / gap
        return expected_n / horizon

    def register_event(self, t: float) -> None:
        """Record an event at time ``t`` (decays, then adds a unit excitation)."""
        self._decay_to(t)
        self.S += 1.0

    def reset(self, t0: float = 0.0) -> None:
        """Reset the state to the baseline at time ``t0``."""
        self.S = 0.0
        self.last_t = t0


@dataclass
class SidedHawkes:
    """Container for two **independent** self-exciting intensities (buy/sell).

    Note
    ----
    **[PROJECT ASSUMPTION]** We model the buy- and sell-initiated flows as two
    *marginal* univariate Hawkes processes. A fully general bivariate Hawkes
    model would add cross-excitation kernels (buy->sell and sell->buy); those are
    deliberately omitted here for tractability and transparency. Each side is
    calibrated on its own aggressor stream (Phase 5 calibration).
    """

    buy: HawkesIntensity
    sell: HawkesIntensity

    def intensities_at(self, t: float) -> tuple[float, float]:
        """Return the instantaneous ``(lambda_buy(t), lambda_sell(t))``."""
        return self.buy.intensity_at(t), self.sell.intensity_at(t)

    def expected_rates(self, t: float, horizon: float) -> tuple[float, float]:
        """Return the smoothed near-future ``(rate_buy, rate_sell)`` [HAWKES].

        See :meth:`HawkesIntensity.expected_rate`. This is the signal the quoting
        policy reacts to (bounded, quote-timescale), whereas
        :meth:`intensities_at` returns the raw spiky instantaneous intensity used
        for diagnostics/plots.
        """
        return (self.buy.expected_rate(t, horizon),
                self.sell.expected_rate(t, horizon))

    def register(self, t: float, side: str) -> None:
        """Register an event of ``side in {'buy', 'sell'}`` at time ``t``."""
        if side == "buy":
            self.buy.register_event(t)
        elif side == "sell":
            self.sell.register_event(t)
        else:  # pragma: no cover - defensive
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    def reset(self, t0: float = 0.0) -> None:
        """Reset both sides."""
        self.buy.reset(t0)
        self.sell.reset(t0)


def intensity_on_grid(grid: FloatArray, events: FloatArray, mu: float,
                      alpha: float, beta: float) -> FloatArray:
    r"""Evaluate ``lambda(g) = mu + alpha sum_{t_i <= g} e^{-beta(g-t_i)}`` on a grid.

    O(G + N) via a carried, decayed excitation sum (**[HAWKES THEORY]**).

    Parameters
    ----------
    grid : numpy.ndarray
        Sorted query times.
    events : numpy.ndarray
        Sorted event times.
    mu, alpha, beta : float
        Hawkes parameters.

    Returns
    -------
    numpy.ndarray
        Intensity at each grid point.
    """
    grid = np.asarray(grid, dtype=float)
    lam = np.empty(grid.shape[0])
    s = 0.0
    j = 0
    prev = grid[0]
    for k, g in enumerate(grid):
        s *= np.exp(-beta * (g - prev))
        while j < events.shape[0] and events[j] <= g:
            s += np.exp(-beta * (g - events[j]))
            j += 1
        lam[k] = mu + alpha * s
        prev = g
    return lam
