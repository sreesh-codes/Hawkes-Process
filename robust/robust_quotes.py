r"""Robust Hawkes quoting: two uncertainty-aware alternatives to :math:`\delta_H`.

The Phase-5 Hawkes half-spread reacts to the *point-estimate* normalised
intensity :math:`\hat R(t)=\hat\lambda(t)/\hat{\bar\lambda}` and quotes as if the
parameters were known exactly. Phase 6 replaces that with two heuristics that
account for estimation uncertainty. **Neither is claimed to be optimal**
(there is no closed-form Hawkes-driven HJB solution); both are proposed
*heuristics* [HEURISTIC] evaluated empirically.

Method A -- Conservative (worst-case belief)
--------------------------------------------
Quote with the **upper** intensity belief instead of the central estimate. The
maker assumes the market is at the burstier edge of its 95% confidence region, so
the intensity ratio -- and hence the spread -- is systematically wider, hedging
against the adverse-selection risk of *under*-estimating clustering.

.. note::

   The Phase-6 brief phrases Method A as "replace :math:`\hat\lambda` with the
   lower confidence estimate". We interpret *conservative* in the robust-control
   sense of guarding against the **worst case** for a market maker, which is an
   *under*-estimated intensity (quotes too tight during a burst). That is the
   **upper** intensity bound. The opposite (lower-belief) strategy is the
   *aggressive* variant and is available through the same belief machinery
   (:func:`robust.uncertainty.build_beliefs`); Part 3 runs all three beliefs.

Method B -- Variance-penalised spread
-------------------------------------
Widen each half-spread by :math:`c` standard deviations of *its own* estimation
uncertainty. The Phase-5 microstructure half-term
:math:`m(R)=\tfrac1\gamma\ln(1+\gamma R/\kappa)` depends on the parameters only
through :math:`R=\lambda/\bar\lambda`, so the delta method gives its standard
deviation as :math:`\big|\partial m/\partial\lambda\big|\,
\sqrt{\operatorname{Var}\lambda}`, and we add :math:`c` of them:

.. math::

    \delta_{\text{robust}}(t) \;=\; \delta_H(t)
        \;+\; c\,\Big|\frac{\partial m}{\partial\lambda}\Big|\,
              \sqrt{\operatorname{Var}\lambda(t)},\qquad
    \frac{\partial m}{\partial\lambda}
      = \frac{1}{\bar\lambda\,(\kappa+\gamma R)} .

Equivalently the penalty is
:math:`c\,\sqrt{\operatorname{Var}\lambda}\,/\,[\bar\lambda(\kappa+\gamma R)]`,
with :math:`\sqrt{\operatorname{Var}\lambda}` from the delta method
(:mod:`robust.delta_method`) and :math:`c\ge 0` a tunable robustness weight.

**Dimensional note.** The brief writes :math:`\delta_H + c\sqrt{\operatorname{Var}
\lambda}`, but :math:`\sqrt{\operatorname{Var}\lambda}` has units of intensity
(events/s) whereas :math:`\delta_H` is a price ($) -- adding them directly is
dimensionally inconsistent, the very error Phase 5 was careful to avoid in the
:math:`\kappa\!\to\!\lambda` substitution. Propagating
:math:`\sqrt{\operatorname{Var}\lambda}` through the spread map
:math:`\partial m/\partial\lambda` (units $ per event/s) yields a genuine **price**
penalty on the correct scale. At :math:`c=0` this collapses **exactly** to
:math:`\delta_H`; larger :math:`c` widens the quotes precisely where the intensity
(and therefore the fill/adverse-selection risk) is most uncertain. Investigated at
:math:`c\in\{0, 0.5, 1, 2\}`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from market_maker import ASParameters
from market_maker.hawkes_market_maker import Quote, _reservation

__all__ = ["RobustHawkesASPolicy", "conservative_lam_bars"]


@dataclass(frozen=True)
class RobustHawkesASPolicy:
    r"""Variance-penalised Hawkes-driven AS quotes (Method B) [HEURISTIC].

    The policy reproduces the Phase-5 :class:`HawkesASPolicy` half-spread and adds
    the variance penalty :math:`(c/\gamma)\,\sigma_R`. Its :meth:`quote` takes the
    per-side intensity standard deviations in addition to the intensities, so it
    must be driven by a loop that tracks the delta-method variance (see
    :func:`analysis.robust_backtest.run_robust_backtest`). With ``c = 0`` and the
    central belief it is identical to the Phase-5 Hawkes policy.

    Parameters
    ----------
    params : ASParameters
        AS parameters (its ``kappa`` is the calibrated elasticity).
    lam_bar_buy, lam_bar_sell : float
        Stationary mean intensities used to normalise :math:`R(t)`.
    c : float
        Robustness weight (``>= 0``). ``c = 0`` recovers :math:`\delta_H`.
    """

    params: ASParameters
    lam_bar_buy: float
    lam_bar_sell: float
    c: float = 1.0

    def _half_micro(self, R: float) -> float:
        """Phase-5 dimensionally-consistent microstructure half-term."""
        p = self.params
        return (1.0 / p.gamma) * np.log(1.0 + p.gamma * R / p.kappa)

    def _half_penalty(self, sd: float, lam_bar: float, R: float) -> float:
        r"""Price penalty :math:`c\,|\partial m/\partial\lambda|\,\sqrt{\operatorname{Var}\lambda}`.

        With :math:`\partial m/\partial\lambda = 1/[\bar\lambda(\kappa+\gamma R)]`
        this is :math:`c\,\text{sd}/[\bar\lambda(\kappa+\gamma R)]` (units $).
        """
        if self.c <= 0.0 or lam_bar <= 0.0:
            return 0.0
        p = self.params
        return self.c * sd / (lam_bar * (p.kappa + p.gamma * R))

    def quote(self, s: float, q: float, t: float, lam_buy: float,
              lam_sell: float, sd_buy: float = 0.0, sd_sell: float = 0.0,
              ) -> Quote:
        r"""Return asymmetric, variance-penalised quotes.

        Parameters
        ----------
        s, q, t : float
            Mid-price ($), inventory (shares), time (s).
        lam_buy, lam_sell : float
            Belief intensities (events/s) the quotes react to.
        sd_buy, sd_sell : float, optional
            Delta-method standard deviations of the buy/sell intensities.

        Returns
        -------
        Quote
            Reservation price, bid, ask and total spread.
        """
        p = self.params
        r = _reservation(s, q, t, p)                       # [AS-EXACT]
        inv_half = 0.5 * p.gamma * p.sigma**2 * (p.T - t)  # [AS-EXACT]

        R_buy = lam_buy / self.lam_bar_buy if self.lam_bar_buy > 0 else 1.0
        R_sell = lam_sell / self.lam_bar_sell if self.lam_bar_sell > 0 else 1.0

        # ask reacts to buy flow; bid reacts to sell flow (Phase-5 asymmetry)
        delta_ask = (inv_half + self._half_micro(R_buy)
                     + self._half_penalty(sd_buy, self.lam_bar_buy, R_buy))
        delta_bid = (inv_half + self._half_micro(R_sell)
                     + self._half_penalty(sd_sell, self.lam_bar_sell, R_sell))
        ask = r + delta_ask
        bid = r - delta_bid
        return Quote(bid=bid, ask=ask, reservation=r, spread=ask - bid)


def conservative_lam_bars(sided_beliefs, belief: str = "upper",
                          ) -> tuple[float, float]:
    r"""Per-side :math:`\bar\lambda` for a Method-A conservative belief.

    Parameters
    ----------
    sided_beliefs : robust.uncertainty.SidedBeliefs
        Per-side lower/central/upper beliefs.
    belief : str, optional
        Which belief to quote on (``"upper"`` = conservative/worst-case).

    Returns
    -------
    tuple of float
        ``(lam_bar_buy, lam_bar_sell)`` under the chosen belief.
    """
    return (sided_beliefs.lam_bar("buy", belief),
            sided_beliefs.lam_bar("sell", belief))
