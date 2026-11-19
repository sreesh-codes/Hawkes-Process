r"""Uncertainty-aware Hawkes quoting -- the project's original contribution (Part 10).

Phases 2--3 did not just *point-estimate* the Hawkes parameters; Phase 3's Monte
-Carlo study delivered their **sampling distribution** and 95% confidence
intervals. A standard Hawkes market maker throws that information away and quotes
as if ``(mu, alpha, beta)`` were known exactly. Here we **propagate parameter
uncertainty into the quotes**.

Idea
----
Build three Hawkes intensity models from the Phase-3 CIs of the excitation:

* **lower** (aggressive)  -- excitation at the 2.5% bound: assume the market is
  *calmer* than the point estimate;
* **central**             -- the point estimate (Strategy B as before);
* **upper** (conservative) -- excitation at the 97.5% bound: assume the market is
  *burstier* than the point estimate.

The market maker then quotes using each belief while the *true* order flow is
generated from the central (ground-truth) parameters. A **conservative** maker
over-estimates clustering, so it widens sooner during bursts. The question this
answers empirically: **does quoting with the conservative (upper-CI) intensity
reduce adverse selection and/or inventory risk?** -- a robust-control style hedge
against estimation error.

This is an *empirical* contribution (a proposed heuristic evaluated by
simulation), not a new theorem: labelled [PROJECT CONTRIBUTION] throughout.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from market_maker import ASParameters
from market_maker.hawkes_execution import (HawkesIntensity, SidedHawkes,
                                           intensity_on_grid)
from market_maker.hawkes_market_maker import HawkesASPolicy
from market_maker.historical_replay import generate_synthetic_events, run_backtest

__all__ = ["excitation_multipliers", "run_robust_study", "plot_robust"]

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPACT = 0.002
T_EPISODE = 600.0
S0 = 585.0


def excitation_multipliers() -> tuple[float, float]:
    r"""Relative 95% CI multipliers on the excitation from Phase-3 Monte-Carlo.

    Loads the Phase-3 headline results and returns
    ``(m_lower, m_upper) = (alpha_2.5% , alpha_97.5%) / alpha_mean`` so that a
    per-side ``alpha`` can be scaled to its lower/upper confidence bound. Falls
    back to +/-10% if the file is missing.

    Returns
    -------
    tuple of float
        Lower and upper multipliers (e.g. ``(0.90, 1.10)``).
    """
    path = os.path.join(BASE, "results", "main_results.parquet")
    try:
        df = pd.read_parquet(path)
        ok = df[df["success"]]["alpha_hat"].to_numpy()
        lo, hi = np.percentile(ok, [2.5, 97.5])
        mean = ok.mean()
        return float(lo / mean), float(hi / mean)
    except (FileNotFoundError, OSError):  # pragma: no cover
        return 0.90, 1.10


def _scaled_params(sp: dict, mult: float) -> dict:
    """Deep-copy the side params with per-side ``alpha`` scaled by ``mult``."""
    out = json.loads(json.dumps(sp))
    for side in ("buy", "sell"):
        out[side]["alpha"] = sp[side]["alpha"] * mult
        rho = out[side]["alpha"] / out[side]["beta"]
        out[side]["lam_bar"] = out[side]["mu"] / (1.0 - rho)
    return out


def _make_sided(sp: dict) -> SidedHawkes:
    return SidedHawkes(
        HawkesIntensity(sp["buy"]["mu"], sp["buy"]["alpha"], sp["buy"]["beta"]),
        HawkesIntensity(sp["sell"]["mu"], sp["sell"]["alpha"], sp["sell"]["beta"]))


def _run_scenario(seed: int, params: ASParameters, sp_true: dict,
                  sp_belief: dict, impact: float, T: float) -> dict:
    """One path: flow from ``sp_true``, quotes from ``sp_belief``."""
    from analysis.comparison import compute_metrics  # avoid circular at import time
    buy = HawkesIntensity(sp_true["buy"]["mu"], sp_true["buy"]["alpha"], sp_true["buy"]["beta"])
    sell = HawkesIntensity(sp_true["sell"]["mu"], sp_true["sell"]["alpha"], sp_true["sell"]["beta"])
    events = generate_synthetic_events(buy, sell, T, seed)
    pol = HawkesASPolicy(params, sp_belief["buy"]["lam_bar"], sp_belief["sell"]["lam_bar"])
    res = run_backtest(events, pol, _make_sided(sp_belief), params, label="robust",
                       synthetic=True, impact=impact, s0=S0, seed=1000 + seed)
    return compute_metrics(res)


def run_robust_study(params: ASParameters, sp: dict, n_paths: int,
                     n_jobs: int = -1) -> pd.DataFrame:
    """Compare lower/central/upper intensity beliefs on central-truth flow.

    Parameters
    ----------
    params : ASParameters
        AS parameters.
    sp : dict
        Central (calibrated) per-side Hawkes parameters.
    n_paths : int
        Paired paths per scenario (shared seeds across scenarios).
    n_jobs : int
        Worker processes.

    Returns
    -------
    pandas.DataFrame
        Long table with a ``scenario`` column and the full metric panel.
    """
    from joblib import Parallel, delayed
    m_lo, m_hi = excitation_multipliers()
    beliefs = {"lower": _scaled_params(sp, m_lo), "central": sp,
               "upper": _scaled_params(sp, m_hi)}
    frames = []
    for name, belief in beliefs.items():
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_run_scenario)(s, params, sp, belief, IMPACT, T_EPISODE)
            for s in range(n_paths))
        d = pd.DataFrame(rows)
        d["scenario"] = name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def plot_robust(df: pd.DataFrame, out_dir: str) -> None:
    """Figure 11: CI-propagated intensity bands + robustness outcomes."""
    os.makedirs(out_dir, exist_ok=True)
    sp = json.load(open(os.path.join(BASE, "results", "hawkes_side_params.json")))
    m_lo, m_hi = excitation_multipliers()

    # (a) intensity bands from the CI on excitation, on one representative path
    buy = HawkesIntensity(sp["buy"]["mu"], sp["buy"]["alpha"], sp["buy"]["beta"])
    sell = HawkesIntensity(sp["sell"]["mu"], sp["sell"]["alpha"], sp["sell"]["beta"])
    ev = generate_synthetic_events(buy, sell, 120.0, seed=7)
    tb = ev[ev.side == "buy"]["time"].to_numpy()
    grid = np.linspace(0, 120, 3000)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for mult, name, c in [(m_lo, "lower", "#1f77b4"), (1.0, "central", "black"),
                          (m_hi, "upper", "#d62728")]:
        lam = intensity_on_grid(grid, tb, sp["buy"]["mu"], sp["buy"]["alpha"] * mult,
                                sp["buy"]["beta"])
        axes[0].plot(grid, lam, color=c, lw=0.8, label=f"{name} (alpha x {mult:.2f})")
    axes[0].set(xlabel="time (s)", ylabel=r"$\lambda_{buy}(t)$",
                title="Fig 11a. CI-propagated intensity (buy side)")
    axes[0].legend(fontsize=8)

    # (b) adverse selection & inventory variance vs scenario
    order = ["lower", "central", "upper"]
    g = df.groupby("scenario")
    adv = g["adverse_selection_ratio"].mean().reindex(order)
    ivar = g["inv_var"].mean().reindex(order)
    ax2 = axes[1]
    ax2.plot(order, adv.values, "o-", color="crimson", label="adverse-selection ratio")
    ax2.set_ylabel("adverse-selection ratio", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax3 = ax2.twinx()
    ax3.plot(order, ivar.values, "s--", color="navy", label="inventory variance")
    ax3.set_ylabel("inventory variance", color="navy")
    ax3.tick_params(axis="y", labelcolor="navy")
    ax2.set(xlabel="intensity belief", title="Fig 11b. Conservative quoting vs risk")
    fig.suptitle("Fig 11. Uncertainty-aware quoting [PROJECT CONTRIBUTION]",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig11_ci_propagation.png"), bbox_inches="tight")
    plt.close(fig)
