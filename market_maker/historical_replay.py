r"""Event-driven backtest engine: synthetic Hawkes paths and LOBSTER replay.

One engine drives two experiments so that Strategy A (Poisson AS) and Strategy B
(Hawkes AS) are compared on *identical* order flow:

* **Synthetic** -- buy/sell market orders are generated from the calibrated
  per-side Hawkes processes (Phase 3 simulator); the mid-price is an arithmetic
  Brownian motion **plus a permanent price impact** from each market order, so
  bursts move the price and create genuine *adverse selection*. Fills use the AS
  exponential model ``P(fill) = exp(-kappa * delta)`` -- identical for both
  strategies, so any difference is due solely to *where each policy places its
  quotes*.
* **Replay** -- the actual LOBSTER event stream is replayed on historical
  timestamps and mid-prices; a resting quote is deemed executed if a market order
  trades *through* it (a crossing rule).

Provenance tags: [AS-EXACT], [HAWKES], [APPROX] as in ``hawkes_market_maker``.

**[PROJECT ASSUMPTION]** The replay fill rule ignores queue position (it assumes
the maker is filled whenever its quote is at least as aggressive as the executing
market order), so replay fill rates are an optimistic upper bound.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from market_maker import ASParameters
from market_maker.hawkes_execution import HawkesIntensity, SidedHawkes
from market_maker.hawkes_market_maker import Quote
from market_maker.inventory import Portfolio
from simulation.hawkes_simulator import simulate_hawkes

__all__ = ["QuotingPolicy", "BacktestResult", "generate_synthetic_events",
           "run_backtest", "load_lobster_events"]

FloatArray = npt.NDArray[np.float64]


class QuotingPolicy(Protocol):
    """Structural type for a quoting policy (Poisson or Hawkes)."""

    def quote(self, s: float, q: float, t: float, lam_buy: float,
              lam_sell: float) -> Quote:  # pragma: no cover - protocol
        ...


@dataclass
class BacktestResult:
    """Outcome of one backtest run.

    Attributes
    ----------
    history : pandas.DataFrame
        Per-event state (time, mid, bid, ask, spread, inventory, cash, wealth,
        intensities, fill flags).
    fills : pandas.DataFrame
        One row per executed fill (time, side, price, mid, markout).
    terminal_wealth : float
        Mark-to-market wealth after terminal liquidation.
    label : str
        Human-readable strategy label.
    """

    history: pd.DataFrame
    fills: pd.DataFrame
    terminal_wealth: float
    label: str


# --------------------------------------------------------------------------- #
# Synthetic event generation (Hawkes flow + impact-driven mid)
# --------------------------------------------------------------------------- #
def generate_synthetic_events(
    buy: HawkesIntensity, sell: HawkesIntensity, T: float, seed: int,
) -> pd.DataFrame:
    """Generate a merged buy/sell market-order stream from two Hawkes processes.

    Parameters
    ----------
    buy, sell : HawkesIntensity
        Calibrated per-side Hawkes parameters (only ``mu, alpha, beta`` are used).
    T : float
        Horizon (seconds).
    seed : int
        Base seed (buy uses ``seed``, sell uses ``seed + 1``).

    Returns
    -------
    pandas.DataFrame
        Columns ``time`` (sorted) and ``side`` in {'buy', 'sell'}.
    """
    tb = simulate_hawkes(buy.mu, buy.alpha, buy.beta, T, seed)
    ts = simulate_hawkes(sell.mu, sell.alpha, sell.beta, T, seed + 1)
    df = pd.concat([
        pd.DataFrame({"time": tb, "side": "buy"}),
        pd.DataFrame({"time": ts, "side": "sell"}),
    ], ignore_index=True).sort_values("time").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Core event loop
# --------------------------------------------------------------------------- #
def _markout(fill_times: FloatArray, fill_prices: FloatArray,
             fill_signs: FloatArray, mid_times: FloatArray,
             mid_values: FloatArray, horizon: float) -> FloatArray:
    """Signed adverse-selection mark-out ``sign * (mid_{t+h} - fill_price)``.

    A positive value means the mid moved *against* the maker after the fill
    (adverse). ``sign = +1`` for a maker **buy** (bid fill), ``-1`` for a maker
    **sell** (ask fill): in both cases a positive result is adverse.
    """
    future_mid = np.interp(fill_times + horizon, mid_times, mid_values)
    return fill_signs * (future_mid - fill_prices)


def run_backtest(
    events: pd.DataFrame,
    policy: QuotingPolicy,
    sided: SidedHawkes,
    params: ASParameters,
    *,
    label: str,
    synthetic: bool,
    kappa_fill: float | None = None,
    impact: float = 0.0,
    sigma: float | None = None,
    s0: float = 585.0,
    seed: int = 0,
    markout_horizon: float = 5.0,
    quote_horizon: float = 1.0,
) -> BacktestResult:
    r"""Run the event-driven backtest for one strategy on one event stream.

    Parameters
    ----------
    events : pandas.DataFrame
        Columns ``time, side`` (and, for replay, ``mid, exec_price``).
    policy : QuotingPolicy
        Strategy A (:class:`PoissonASPolicy`) or B (:class:`HawkesASPolicy`).
    sided : SidedHawkes
        Buy/sell Hawkes state used to compute the intensities the policy reacts
        to (reset internally). [HAWKES]
    params : ASParameters
        AS parameters (``gamma, sigma, kappa, T, q_max``).
    label : str
        Strategy label for the result.
    synthetic : bool
        If ``True`` the mid is generated online (BM + impact) and fills use the
        exponential model; if ``False`` the mid/exec_price come from ``events``
        (LOBSTER replay) and fills use the crossing rule.
    kappa_fill : float, optional
        Elasticity for the exponential fill model (synthetic). Defaults to
        ``params.kappa``.
    impact : float, optional
        Permanent price impact per market order ($) (synthetic).
    sigma : float, optional
        Mid volatility ($/sqrt(s)) for the synthetic BM. Defaults to
        ``params.sigma``.
    s0 : float, optional
        Initial mid ($) (synthetic).
    seed : int, optional
        Seed for fills / BM noise (synthetic).
    markout_horizon : float, optional
        Horizon (s) for the adverse-selection mark-out.

    Returns
    -------
    BacktestResult
        History, fills and terminal wealth.
    """
    sided.reset(0.0)
    book = Portfolio(q_max=params.q_max, order_size=params.order_size)
    rng = np.random.default_rng(seed)
    kappa_fill = params.kappa if kappa_fill is None else kappa_fill
    sigma = params.sigma if sigma is None else sigma

    ev_t = events["time"].to_numpy()
    ev_side = events["side"].to_numpy()
    ev_mid = events["mid"].to_numpy() if "mid" in events else None
    ev_exec = events["exec_price"].to_numpy() if "exec_price" in events else None

    mid = s0
    prev_t = 0.0
    rows = []
    fills = []  # (time, side, price, mid_at_fill, sign)
    prev_bid = prev_ask = np.nan
    n_quote_changes = 0

    for i in range(len(ev_t)):
        t = float(ev_t[i])
        side = ev_side[i]
        # --- diffuse the mid to this event time (synthetic) --------------
        if synthetic:
            dt = t - prev_t
            if dt > 0:
                mid += sigma * np.sqrt(dt) * rng.standard_normal()
        else:
            mid = float(ev_mid[i])

        # --- intensities from PAST events (this event not yet registered) [HAWKES]
        lam_buy, lam_sell = sided.intensities_at(t)          # instantaneous (diagnostic)
        rate_buy, rate_sell = sided.expected_rates(t, quote_horizon)  # smoothed (quoting)

        # --- quote ------------------------------------------------------- [AS/APPROX]
        qte = policy.quote(mid, book.inventory, t, rate_buy, rate_sell)

        # count quote changes (for average quote lifetime)
        if not np.isnan(prev_bid):
            if abs(qte.bid - prev_bid) + abs(qte.ask - prev_ask) > 1e-4:
                n_quote_changes += 1
        prev_bid, prev_ask = qte.bid, qte.ask

        # --- determine fill ---------------------------------------------
        bid_fill = ask_fill = False
        if side == "buy":  # buy MO lifts the ask
            delta_ask = max(qte.ask - mid, 0.0)
            if synthetic:
                hit = rng.random() < np.exp(-kappa_fill * delta_ask)
            else:
                hit = qte.ask <= float(ev_exec[i])
            if hit and book.can_sell():
                book.on_ask_fill(qte.ask)
                fills.append((t, "sell", qte.ask, mid, -1.0))
                ask_fill = True
        else:  # sell MO hits the bid
            delta_bid = max(mid - qte.bid, 0.0)
            if synthetic:
                hit = rng.random() < np.exp(-kappa_fill * delta_bid)
            else:
                hit = qte.bid >= float(ev_exec[i])
            if hit and book.can_buy():
                book.on_bid_fill(qte.bid)
                fills.append((t, "buy", qte.bid, mid, 1.0))
                bid_fill = True

        # --- permanent impact of the market order (synthetic) -----------
        if synthetic and impact != 0.0:
            mid += impact if side == "buy" else -impact

        # --- register the event in the Hawkes state ---------------------- [HAWKES]
        sided.register(t, side)

        rows.append({
            "time": t, "mid": mid, "bid": qte.bid, "ask": qte.ask,
            "spread": qte.spread, "reservation": qte.reservation,
            "lam_buy": lam_buy, "lam_sell": lam_sell,
            "rate_buy": rate_buy, "rate_sell": rate_sell,
            "inventory": book.inventory, "cash": book.cash,
            "wealth": book.mark_to_market(mid),
            "bid_fill": bid_fill, "ask_fill": ask_fill,
        })
        prev_t = t

    terminal_wealth = book.liquidate(mid)
    history = pd.DataFrame(rows)
    history.attrs["n_quote_changes"] = n_quote_changes
    history.attrs["horizon"] = float(ev_t[-1]) if len(ev_t) else params.T

    fills_df = pd.DataFrame(fills, columns=["time", "side", "price", "mid", "sign"])
    if len(fills_df):
        fills_df["markout"] = _markout(
            fills_df["time"].to_numpy(), fills_df["price"].to_numpy(),
            fills_df["sign"].to_numpy(), history["time"].to_numpy(),
            history["mid"].to_numpy(), markout_horizon)
    else:
        fills_df["markout"] = pd.Series(dtype=float)

    return BacktestResult(history=history, fills=fills_df,
                          terminal_wealth=terminal_wealth, label=label)


# --------------------------------------------------------------------------- #
# LOBSTER event loading for replay
# --------------------------------------------------------------------------- #
def load_lobster_events(data_prefix: str) -> pd.DataFrame:
    """Load the LOBSTER trade stream for historical replay.

    Parameters
    ----------
    data_prefix : str
        Path prefix, e.g. ``".../AAPL_2012-06-21_34200000_57600000_"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``time, side, exec_price, mid`` for each executed trade
        (event types 4 and 5). ``side`` is the aggressor: a buy market order
        (``dir == -1``, buyer-initiated) or a sell market order (``dir == +1``).
    """
    cols = ["time", "etype", "oid", "size", "price", "dir"]
    msg = pd.read_csv(data_prefix + "message_10.csv", header=None, names=cols)
    ob_names = sum([[f"ap{i}", f"as{i}", f"bp{i}", f"bs{i}"] for i in range(1, 11)], [])
    ob = pd.read_csv(data_prefix + "orderbook_10.csv", header=None, names=ob_names)
    mid = (ob["ap1"] + ob["bp1"]) / 2.0 / 10000.0

    is_trade = msg["etype"].isin([4, 5]).to_numpy()
    out = pd.DataFrame({
        "time": msg["time"].to_numpy()[is_trade] - msg["time"].to_numpy()[0],
        "side": np.where(msg["dir"].to_numpy()[is_trade] == -1, "buy", "sell"),
        "exec_price": msg["price"].to_numpy()[is_trade] / 10000.0,
        "mid": mid.to_numpy()[is_trade],
    })
    return out.reset_index(drop=True)
