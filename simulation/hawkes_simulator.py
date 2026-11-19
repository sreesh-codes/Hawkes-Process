"""Exponential Hawkes simulator via Ogata's thinning algorithm (from scratch).

Ogata's modified thinning algorithm draws candidate points from a homogeneous
Poisson process whose rate is an *upper bound* of the conditional intensity, then
accepts each candidate with probability ``lambda(t) / lambda_bar``. For the
exponential kernel the intensity only *decreases* between events, so the value
immediately after the current state,

.. math::

    \\bar\\lambda = \\mu + \\alpha \\sum_{t_i \\le t} e^{-\\beta (t - t_i)},

is a valid upper bound until the next accepted/rejected candidate. The decaying
excitation sum is maintained recursively, giving an :math:`O(N)` simulator with
no external Hawkes library.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

__all__ = ["simulate_hawkes"]

FloatArray = npt.NDArray[np.float64]


@njit(cache=True, fastmath=True)
def _simulate(mu: float, alpha: float, beta: float, T: float,
              seed: int) -> FloatArray:
    """Numba core for :func:`simulate_hawkes` (see that function's docstring)."""
    np.random.seed(seed)

    rho = alpha / beta
    mean_rate = mu / (1.0 - rho) if rho < 1.0 else mu + alpha
    cap = int(mean_rate * T * 4.0) + 1024  # generous initial capacity
    times = np.empty(cap, dtype=np.float64)
    n = 0

    t = 0.0
    s = 0.0  # decayed excitation sum: sum_i exp(-beta (t - t_i))
    while True:
        lam_bar = mu + alpha * s            # upper bound, valid going forward
        # waiting time of a homogeneous Poisson process with rate lam_bar
        w = -np.log(np.random.random()) / lam_bar
        t += w
        if t >= T:
            break
        s *= np.exp(-beta * w)              # decay excitation to candidate time
        lam = mu + alpha * s               # true intensity at candidate
        if np.random.random() <= lam / lam_bar:
            if n >= cap:                    # grow buffer if under-allocated
                cap *= 2
                grown = np.empty(cap, dtype=np.float64)
                grown[:n] = times[:n]
                times = grown
            times[n] = t
            n += 1
            s += 1.0                         # new event contributes e^0 = 1
    return times[:n]


def simulate_hawkes(
    mu: float, alpha: float, beta: float, T: float, seed: int
) -> FloatArray:
    """Simulate an exponential Hawkes process on ``[0, T]`` (Ogata thinning).

    Parameters
    ----------
    mu : float
        Baseline intensity (events per second), ``> 0``.
    alpha : float
        Excitation jump per event, ``>= 0``.
    beta : float
        Excitation decay rate (per second), ``> 0``. For a stationary process
        require ``alpha < beta`` (branching ratio below one).
    T : float
        Simulation horizon (seconds).
    seed : int
        Seed for the (Numba) random stream; distinct seeds give independent draws.

    Returns
    -------
    numpy.ndarray
        Sorted event timestamps in ``[0, T)``.

    Raises
    ------
    ValueError
        If parameters are non-positive or ``T <= 0``.
    """
    if mu <= 0.0 or beta <= 0.0 or alpha < 0.0 or T <= 0.0:
        raise ValueError("require mu>0, beta>0, alpha>=0, T>0")
    return _simulate(float(mu), float(alpha), float(beta), float(T), int(seed))
