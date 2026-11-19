r"""Event-driven Avellaneda--Stoikov market-making simulator + episode plots.

Each step ``dt`` of the trading horizon ``[0, T]``:

1. **Update the price.** The mid evolves as an arithmetic Brownian motion,
   ``S_{t+dt} = S_t + sigma sqrt(dt) Z``, ``Z ~ N(0, 1)`` (eq. 1).
2. **Quote.** Compute the reservation price (eq. 3), optimal spread (eq. 4) and
   the bid/ask (eq. 5); respect the inventory cap by withdrawing the side that
   would breach ``q_max``.
3. **Generate fills.** Draw bid/ask executions from the exponential Poisson model
   (eq. 2/6).
4. **Update inventory & cash**, then record the mark-to-market wealth (eq. 7).

At ``T`` the residual inventory is liquidated. Every state variable is stored in a
tidy :class:`pandas.DataFrame` for analysis and plotting.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from market_maker import ASParameters
from market_maker.avellaneda_stoikov import compute_quotes
from market_maker.execution_model import ExecutionModel
from market_maker.inventory import Portfolio

__all__ = ["SimulationResult", "simulate_episode", "plot_episode"]


@dataclass
class SimulationResult:
    """Container for one simulated episode.

    Attributes
    ----------
    history : pandas.DataFrame
        Per-step state: ``t, mid, reservation, bid, ask, spread, delta_bid,
        delta_ask, inventory, cash, wealth, bid_fill, ask_fill``.
    terminal_wealth : float
        Mark-to-market wealth after terminal liquidation ($).
    params : ASParameters
        Parameters used.
    """

    history: pd.DataFrame
    terminal_wealth: float
    params: ASParameters


def simulate_episode(p: ASParameters, *, dt: float = 1.0, s0: float = 585.0,
                     seed: int = 0) -> SimulationResult:
    r"""Run one Avellaneda--Stoikov trading episode.

    Parameters
    ----------
    p : ASParameters
        Model parameters (``gamma, sigma, kappa, A, T, q_max, order_size``).
    dt : float, optional
        Time-step (seconds). Default 1.0.
    s0 : float, optional
        Initial mid-price ($). Default 585 (AAPL scale).
    seed : int, optional
        Random seed for the price path and the fills.

    Returns
    -------
    SimulationResult
        The recorded episode and its terminal wealth.
    """
    rng = np.random.default_rng(seed)
    exec_model = ExecutionModel(A=p.A, kappa=p.kappa)
    book = Portfolio(q_max=p.q_max, order_size=p.order_size)

    n_steps = int(np.floor(p.T / dt))
    s = s0
    rows = []
    sqrt_dt = np.sqrt(dt)

    for k in range(n_steps):
        t = k * dt
        # --- (2) quotes from the AS formulas -----------------------------
        r, spread, bid, ask = compute_quotes(s, book.inventory, t, p)
        delta_bid = s - bid          # distance of bid below mid
        delta_ask = ask - s          # distance of ask above mid

        # --- (3) fills (respect inventory risk limit) --------------------
        fills = exec_model.sample_fills(delta_bid, delta_ask, dt, rng)
        bid_fill = fills.bid_filled and book.can_buy()
        ask_fill = fills.ask_filled and book.can_sell()
        if bid_fill:
            book.on_bid_fill(bid)
        if ask_fill:
            book.on_ask_fill(ask)

        # --- (4) record state (mark-to-market at current mid) ------------
        rows.append({
            "t": t, "mid": s, "reservation": r, "bid": bid, "ask": ask,
            "spread": spread, "delta_bid": delta_bid, "delta_ask": delta_ask,
            "inventory": book.inventory, "cash": book.cash,
            "wealth": book.mark_to_market(s),
            "bid_fill": bid_fill, "ask_fill": ask_fill,
        })

        # --- (1) price update: arithmetic Brownian motion ----------------
        s = s + p.sigma * sqrt_dt * rng.standard_normal()

    # --- terminal liquidation at T ---------------------------------------
    terminal_wealth = book.liquidate(s)
    history = pd.DataFrame(rows)
    return SimulationResult(history=history, terminal_wealth=terminal_wealth,
                            params=p)


def plot_episode(result: SimulationResult, out_dir: str,
                 prefix: str = "episode") -> list[str]:
    r"""Render the publication figures for a single episode.

    Produces: a combined quotes/price panel, inventory, cash, PnL (MtM wealth),
    spread, quote distances and a trade-execution scatter.

    Parameters
    ----------
    result : SimulationResult
        Episode to plot.
    out_dir : str
        Output directory (created if needed).
    prefix : str, optional
        File-name prefix.

    Returns
    -------
    list of str
        Paths of the saved figures.
    """
    os.makedirs(out_dir, exist_ok=True)
    h = result.history
    t = h["t"].to_numpy()
    saved = []

    def _save(fig, name):
        path = os.path.join(out_dir, f"{prefix}_{name}.png")
        fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig); saved.append(path)

    # 1. Price, reservation price and quotes (with executions)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, h["mid"], color="black", lw=1.1, label="mid-price $S_t$")
    ax.plot(t, h["reservation"], color="purple", lw=1.0, ls="--",
            label="reservation price $r(t,q)$")
    ax.plot(t, h["bid"], color="green", lw=0.6, alpha=0.7, label="bid")
    ax.plot(t, h["ask"], color="red", lw=0.6, alpha=0.7, label="ask")
    bf, af = h[h.bid_fill], h[h.ask_fill]
    ax.scatter(bf["t"], bf["bid"], s=14, color="green", marker="^",
               label="bid fill (buy)", zorder=5)
    ax.scatter(af["t"], af["ask"], s=14, color="red", marker="v",
               label="ask fill (sell)", zorder=5)
    ax.set(xlabel="time (s)", ylabel="price ($)",
           title="Avellaneda--Stoikov quotes, reservation price and executions")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    _save(fig, "quotes_and_price")

    # 2. Inventory
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, h["inventory"], color="navy", lw=1.0)
    ax.axhline(0, color="gray", ls=":")
    ax.axhline(result.params.q_max, color="red", ls="--", lw=0.8, label="$\\pm q_{max}$")
    ax.axhline(-result.params.q_max, color="red", ls="--", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="inventory (shares)",
           title="Inventory trajectory"); ax.legend()
    _save(fig, "inventory")

    # 3. Cash
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, h["cash"], color="darkgreen", lw=1.0)
    ax.set(xlabel="time (s)", ylabel="cash ($)", title="Cash balance")
    _save(fig, "cash")

    # 4. PnL (mark-to-market wealth)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, h["wealth"], color="black", lw=1.1)
    ax.axhline(0, color="gray", ls=":")
    ax.set(xlabel="time (s)", ylabel="MtM wealth ($)",
           title=f"PnL (mark-to-market); terminal = ${result.terminal_wealth:,.2f}")
    _save(fig, "pnl")

    # 5. Spread
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, h["spread"], color="orange", lw=1.1)
    ax.set(xlabel="time (s)", ylabel="optimal spread ($)",
           title=r"Optimal spread $\delta^*(t)=\gamma\sigma^2(T-t)+\frac{2}{\gamma}\ln(1+\gamma/\kappa)$")
    _save(fig, "spread")

    # 6. Quote distances from mid
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, h["delta_bid"], color="green", lw=0.8, label=r"$\delta^b=S-\mathrm{bid}$")
    ax.plot(t, h["delta_ask"], color="red", lw=0.8, label=r"$\delta^a=\mathrm{ask}-S$")
    ax.set(xlabel="time (s)", ylabel="distance from mid ($)",
           title="Quote distances from the mid (asymmetry = inventory skew)")
    ax.legend()
    _save(fig, "quote_distances")

    # 7. Combined dashboard
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, h["mid"], "k", lw=1, label="mid")
    axes[0].plot(t, h["reservation"], "purple", ls="--", lw=1, label="reservation")
    axes[0].fill_between(t, h["bid"], h["ask"], color="gray", alpha=0.2, label="quoted spread")
    axes[0].legend(fontsize=9); axes[0].set_ylabel("price ($)")
    axes[0].set_title("Episode dashboard")
    axes[1].plot(t, h["inventory"], "navy"); axes[1].axhline(0, color="gray", ls=":")
    axes[1].set_ylabel("inventory")
    axes[2].plot(t, h["wealth"], "k"); axes[2].axhline(0, color="gray", ls=":")
    axes[2].set_ylabel("MtM wealth ($)"); axes[2].set_xlabel("time (s)")
    _save(fig, "dashboard")

    return saved
