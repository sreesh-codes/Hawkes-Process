r"""Core Avellaneda--Stoikov mathematics: reservation price, optimal spread, quotes.

This module reproduces the closed-form results of Avellaneda & Stoikov (2008),
"High-frequency trading in a limit order book", with the derivations written out
in the comments. It also calibrates the two data-driven inputs -- the volatility
``sigma`` and the arrival-decay ``kappa`` -- directly from LOBSTER data.

Model setup
-----------
The mid-price follows an arithmetic Brownian motion

    dS_t = sigma dW_t,                                                   (1)

i.e. *no drift* (the market maker has no directional view) and *constant
volatility*. The agent quotes a bid ``S - delta^b`` and an ask ``S + delta^a``
and earns the half-spreads when market orders arrive. Market sell/buy orders
arrive as Poisson processes whose intensity decays exponentially with the quote's
distance from the mid,

    lambda(delta) = A exp(-kappa delta).                                (2)

The agent maximises expected exponential (CARA) utility of terminal wealth
``U(x) = -exp(-gamma x)`` over ``[0, T]``.

Reservation price (derivation sketch)
-------------------------------------
Let ``v(x, s, q, t)`` be the value function with cash ``x``, mid ``s`` and
inventory ``q``. With CARA utility the value function factorises as
``v = -exp(-gamma x) exp(-gamma theta(s, q, t))`` and the HJB equation for
``theta`` is, to leading order, linear. The **reservation (indifference) price**
is the price at which the agent is indifferent to adding/removing one unit:

    r(s, q, t) = (v(x, s, q-1, t) ... )  ->   r = s - q gamma sigma^2 (T - t). (3)

Intuitively, a long inventory (``q > 0``) lowers the price at which the agent is
willing to hold risk, shifting *all* quotes **down** so the agent is keener to
sell than to buy; the shift grows with risk aversion ``gamma``, with variance
``sigma^2`` and with the time ``(T - t)`` left to carry the risk.

Optimal spread (derivation sketch)
----------------------------------
The agent chooses the half-spreads ``delta^a, delta^b`` (measured from the
reservation price) to maximise expected utility. The arrival of a market order at
the ask gives a first-order condition that, combined with (2), yields the optimal
**total** spread (Avellaneda--Stoikov eq. for ``delta^a + delta^b``):

    delta*(t) = gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / kappa).  (4)

The first term is the **inventory-risk** premium (widens as the horizon and
volatility grow); the second is the **market-microstructure / monopoly** term
coming from the arrival elasticity ``kappa`` (as gamma -> 0 it tends to the
risk-neutral spread ``2 / kappa``). The agent then posts symmetric half-spreads
``delta*/2`` around the reservation price, giving

    bid = r - delta*/2,     ask = r + delta*/2.                          (5)
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from market_maker import ASParameters

__all__ = [
    "reservation_price", "optimal_spread", "compute_quotes",
    "calibrate_sigma", "calibrate_kappa",
]

FloatArray = npt.NDArray[np.float64]


def reservation_price(s: float, q: float, t: float, p: ASParameters) -> float:
    r"""Inventory-adjusted reservation price, eq. (3).

    ``r(s, q, t) = s - q * gamma * sigma^2 * (T - t)``.

    Parameters
    ----------
    s : float
        Current mid-price ($).
    q : float
        Current inventory (shares; signed).
    t : float
        Current time (seconds), ``0 <= t <= T``.
    p : ASParameters
        Model parameters.

    Returns
    -------
    float
        Reservation price ($). Equals ``s`` when ``q == 0`` and shifts linearly
        with inventory.
    """
    return s - q * p.gamma * p.sigma**2 * (p.T - t)


def optimal_spread(t: float, p: ASParameters) -> float:
    r"""Optimal total bid-ask spread, eq. (4).

    ``delta*(t) = gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/kappa)``.

    Parameters
    ----------
    t : float
        Current time (seconds).
    p : ASParameters
        Model parameters.

    Returns
    -------
    float
        Optimal total spread ($); half of it is quoted on each side.
    """
    inventory_term = p.gamma * p.sigma**2 * (p.T - t)
    microstructure_term = (2.0 / p.gamma) * np.log(1.0 + p.gamma / p.kappa)
    return inventory_term + microstructure_term


def compute_quotes(s: float, q: float, t: float, p: ASParameters
                   ) -> tuple[float, float, float, float]:
    r"""Reservation price, optimal spread and the two quotes, eqs. (3)--(5).

    Parameters
    ----------
    s : float
        Mid-price ($).
    q : float
        Inventory (shares).
    t : float
        Time (seconds).
    p : ASParameters
        Model parameters.

    Returns
    -------
    tuple of float
        ``(reservation, spread, bid, ask)`` all in $.
    """
    r = reservation_price(s, q, t, p)
    spread = optimal_spread(t, p)
    bid = r - 0.5 * spread
    ask = r + 0.5 * spread
    return r, spread, bid, ask


# --------------------------------------------------------------------------- #
# Data calibration of the two empirical inputs
# --------------------------------------------------------------------------- #
def calibrate_sigma(times: FloatArray, mids: FloatArray,
                    sample_dt: float = 1.0) -> float:
    r"""Estimate the volatility ``sigma`` ($/sqrt(s)) of ``dS = sigma dW``.

    The mid-price is interpolated onto a uniform ``sample_dt``-second grid; under
    the arithmetic-BM assumption the grid increments are i.i.d.
    ``Normal(0, sigma^2 sample_dt)``, so ``sigma = std(increments)/sqrt(sample_dt)``.

    Parameters
    ----------
    times : numpy.ndarray
        Event timestamps (seconds), strictly increasing after de-duplication.
    mids : numpy.ndarray
        Mid-prices ($) aligned with ``times``.
    sample_dt : float, optional
        Sampling interval in seconds (default 1.0).

    Returns
    -------
    float
        Estimated ``sigma`` in $ per sqrt(second).
    """
    order = np.argsort(times)
    t_u, idx = np.unique(times[order], return_index=True)
    m_u = mids[order][idx]
    grid = np.arange(np.floor(t_u[0]), np.ceil(t_u[-1]), sample_dt)
    mg = np.interp(grid, t_u, m_u)
    incr = np.diff(mg)
    return float(np.std(incr) / np.sqrt(sample_dt))


def calibrate_kappa(trade_prices: FloatArray, trade_times: FloatArray,
                    mid_times: FloatArray, mids: FloatArray,
                    n_bins: int = 25) -> tuple[float, float]:
    r"""Estimate ``(A, kappa)`` in ``lambda(delta) = A exp(-kappa delta)``.

    For each executed trade we measure its distance ``delta = |price - mid|`` from
    the prevailing mid. The empirical distribution of these distances decays
    exponentially; fitting ``log(count) = log(A0) - kappa * delta`` recovers the
    decay rate ``kappa``. ``A`` (the per-second intensity scale) is set from the
    overall trade rate, since the histogram heights are counts, not rates.

    Parameters
    ----------
    trade_prices : numpy.ndarray
        Executed-trade prices ($).
    trade_times : numpy.ndarray
        Executed-trade timestamps (seconds).
    mid_times, mids : numpy.ndarray
        Mid-price series used to look up the mid at each trade time.
    n_bins : int, optional
        Number of histogram bins for the fit.

    Returns
    -------
    tuple of float
        ``(A, kappa)`` with ``A`` in events/second and ``kappa`` in 1/$.
    """
    mid_at_trade = np.interp(trade_times, mid_times, mids)
    dist = np.abs(trade_prices - mid_at_trade)
    dist = dist[dist > 0]
    edges = np.linspace(0.0, np.percentile(dist, 95), n_bins)
    counts, edges = np.histogram(dist, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mask = counts > 0
    slope, _ = np.polyfit(centres[mask], np.log(counts[mask]), 1)
    kappa = float(-slope)
    horizon = trade_times.max() - trade_times.min()
    A = float(len(dist) / horizon)  # base intensity scale (events/s)
    return A, kappa
