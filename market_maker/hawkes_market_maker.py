r"""Hawkes-driven Avellaneda--Stoikov quoting policy.

This is the heart of Phase 5. It builds a quoting rule that reacts to the
instantaneous Hawkes intensity, and it is *scrupulous* about the provenance of
every formula. Three tags are used throughout:

    [AS-EXACT]     : an exact result of Avellaneda & Stoikov (2008).
    [HAWKES]       : a standard result of Hawkes process theory (Phase 2).
    [APPROX]       : an approximation / modelling assumption introduced by THIS
                     project. Never claimed to be an exact theoretical result.

---------------------------------------------------------------------------
Reservation price                                                 [AS-EXACT]
---------------------------------------------------------------------------
    r(s, q, t) = s - q gamma sigma^2 (T - t).

Unchanged from Phase 4: it is a statement about *inventory risk under Brownian
mid-price dynamics* and does not involve the arrival process at all, so it
remains exactly valid under Hawkes arrivals.

---------------------------------------------------------------------------
Why we do NOT "just replace kappa by lambda(t)"                     [APPROX]
---------------------------------------------------------------------------
The classical microstructure spread term is ``(2/gamma) ln(1 + gamma/kappa)``
where ``kappa`` (units 1/$) is the *price-elasticity* of market-order arrivals
in ``lambda(delta) = A exp(-kappa delta)``. The Hawkes intensity ``lambda(t)``
(units 1/s) is a *rate*, a completely different quantity. Substituting
``kappa -> lambda(t)`` directly is (i) dimensionally inconsistent and (ii) badly
mis-scaled (``lambda(t) ~ 1/s`` vs ``kappa ~ 24/$`` inflates the spread ~10x).

Instead we let the intensity modulate ``kappa`` through the **dimensionless
normalised intensity** ``R(t) = lambda(t) / lambda_bar`` (=1 on average). We
define an *effective elasticity* ``kappa_eff(t) = kappa / R(t)`` and substitute
it into the AS term:

    delta(t) = gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma R(t) / kappa).  [APPROX]

Properties (all easy to check):
* At ``R(t) = 1`` it reduces **exactly** to the classical AS spread [AS-EXACT].
* When activity spikes (``R > 1``) the spread **widens** -- the market maker
  demands more compensation precisely when bursts (often informed) raise the risk
  of being adversely filled.

This is an *approximation motivated by* replacing the constant arrival regime
with the instantaneous Hawkes regime; it is **not** an exact solution of a
Hawkes-driven HJB equation (which is not available in closed form). The naive
literal form ``(2/gamma) ln(1 + gamma/lambda(t))`` is also provided (``mode=
"naive"``) purely for comparison and is flagged as dimensionally unsound.

---------------------------------------------------------------------------
Buy/sell asymmetry                                                 [APPROX]
---------------------------------------------------------------------------
Split the spread into two half-spreads driven by the *opposing* flow:
the **ask** is lifted by **buy** market orders, so it reacts to
``R_buy = lambda_buy/lambda_bar_buy``; the **bid** is hit by **sell** market
orders, so it reacts to ``R_sell``:

    delta_ask(t) = 1/2 gamma sigma^2 (T-t) + (1/gamma) ln(1 + gamma R_buy(t)/kappa)
    delta_bid(t) = 1/2 gamma sigma^2 (T-t) + (1/gamma) ln(1 + gamma R_sell(t)/kappa)
    ask = r + delta_ask,   bid = r - delta_bid.

Economic reasoning: a surge in buy market orders signals upward pressure /
informed buying, so the maker makes its **ask less aggressive** (raises it) to
avoid selling just before the price rises; symmetrically a sell surge makes the
**bid less aggressive**. At ``R_buy = R_sell = 1`` this collapses to the
symmetric classical quotes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from market_maker import ASParameters

__all__ = ["Quote", "PoissonASPolicy", "HawkesASPolicy"]


@dataclass(frozen=True)
class Quote:
    """A pair of quotes with diagnostics.

    Attributes
    ----------
    bid, ask : float
        Quoted prices ($).
    reservation : float
        Reservation price ($).
    spread : float
        Total quoted spread ``ask - bid`` ($).
    """

    bid: float
    ask: float
    reservation: float
    spread: float


def _reservation(s: float, q: float, t: float, p: ASParameters) -> float:
    """Reservation price ``s - q gamma sigma^2 (T-t)`` [AS-EXACT]."""
    return s - q * p.gamma * p.sigma**2 * (p.T - t)


@dataclass(frozen=True)
class PoissonASPolicy:
    """Baseline (Strategy A): classical, constant-``kappa`` AS quotes [AS-EXACT].

    Parameters
    ----------
    params : ASParameters
        AS parameters (``gamma, sigma, kappa, T, ...``).
    """

    params: ASParameters

    def quote(self, s: float, q: float, t: float,
              lam_buy: float = 0.0, lam_sell: float = 0.0) -> Quote:
        """Return classical symmetric quotes (intensity arguments ignored)."""
        p = self.params
        r = _reservation(s, q, t, p)
        spread = p.gamma * p.sigma**2 * (p.T - t) + (2.0 / p.gamma) * np.log(
            1.0 + p.gamma / p.kappa)
        return Quote(bid=r - 0.5 * spread, ask=r + 0.5 * spread,
                     reservation=r, spread=spread)


@dataclass(frozen=True)
class HawkesASPolicy:
    """Hawkes-driven AS quotes (Strategy B): intensity-modulated spread [APPROX].

    Parameters
    ----------
    params : ASParameters
        AS parameters (its ``kappa`` is the calibrated elasticity).
    lam_bar_buy, lam_bar_sell : float
        Stationary mean intensities ``mu/(1-rho)`` per side, used to normalise
        the instantaneous intensity to ``R(t) = lambda(t)/lambda_bar`` [HAWKES].
    mode : str
        ``"normalized"`` (default, dimensionally consistent) or ``"naive"`` (the
        literal ``ln(1+gamma/lambda(t))`` substitution, for comparison only).
    """

    params: ASParameters
    lam_bar_buy: float
    lam_bar_sell: float
    mode: str = "normalized"

    def _half_micro(self, R: float) -> float:
        """One-sided microstructure half-term as a function of normalised ``R``."""
        p = self.params
        if self.mode == "normalized":
            # kappa_eff = kappa / R  =>  gamma/kappa_eff = gamma R / kappa
            return (1.0 / p.gamma) * np.log(1.0 + p.gamma * R / p.kappa)
        # "naive": literal substitution kappa -> lambda (dimensionally unsound)
        lam = max(R * (self.lam_bar_buy + self.lam_bar_sell) / 2.0, 1e-9)
        return (1.0 / p.gamma) * np.log(1.0 + p.gamma / lam)

    def quote(self, s: float, q: float, t: float,
              lam_buy: float, lam_sell: float) -> Quote:
        r"""Return asymmetric Hawkes-driven quotes.

        Parameters
        ----------
        s : float
            Mid-price ($).
        q : float
            Inventory (shares).
        t : float
            Time (seconds).
        lam_buy, lam_sell : float
            Instantaneous buy/sell Hawkes intensities (events/s) [HAWKES].

        Returns
        -------
        Quote
            Reservation price, bid, ask and total spread.
        """
        p = self.params
        r = _reservation(s, q, t, p)  # [AS-EXACT]
        inv_half = 0.5 * p.gamma * p.sigma**2 * (p.T - t)  # half inventory-risk [AS-EXACT]

        R_buy = lam_buy / self.lam_bar_buy if self.lam_bar_buy > 0 else 1.0
        R_sell = lam_sell / self.lam_bar_sell if self.lam_bar_sell > 0 else 1.0

        delta_ask = inv_half + self._half_micro(R_buy)    # ask reacts to buy flow
        delta_bid = inv_half + self._half_micro(R_sell)   # bid reacts to sell flow
        ask = r + delta_ask
        bid = r - delta_bid
        return Quote(bid=bid, ask=ask, reservation=r, spread=ask - bid)
