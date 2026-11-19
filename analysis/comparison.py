r"""Comparative backtest: classical (Poisson) vs Hawkes-driven Avellaneda--Stoikov.

Strategy A (Poisson AS) and Strategy B (Hawkes AS) are run on *identical*,
Hawkes-clustered synthetic order flow with a permanent-impact mid-price
(common random numbers => a low-variance **paired** comparison). We compute a
full metric panel per path, aggregate it into publication tables/figures, run the
statistical tests (``analysis.stat_tests``) and the uncertainty-aware extension
(``analysis.robust_quotes``), and write the report.

Metric provenance: PnL/inventory/drawdown/spread are standard; the
*adverse-selection* decomposition ``quoted_edge = realized (mark-out) +
adverse_selection`` is the classic Glosten--Harris / effective-vs-realized-spread
identity from market-microstructure. Run ``python -m analysis.comparison``
(``--quick`` for a smoke test).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_maker import ASParameters  # noqa: E402
from market_maker.hawkes_execution import HawkesIntensity, SidedHawkes  # noqa: E402
from market_maker.hawkes_market_maker import HawkesASPolicy, PoissonASPolicy  # noqa: E402
from market_maker.historical_replay import (  # noqa: E402
    BacktestResult, generate_synthetic_events, load_lobster_events, run_backtest,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(BASE, "plots", "hawkes_market_maker")
RESULTS = os.path.join(BASE, "results")
REPORT = os.path.join(BASE, "reports", "hawkes_extension_report.md")
DATA = os.path.join(BASE, "LOBSTER_SampleFile_AAPL_2012-06-21_10",
                    "AAPL_2012-06-21_34200000_57600000_")
SIDE_PARAMS = os.path.join(RESULTS, "hawkes_side_params.json")
for d in (PLOTS, RESULTS, os.path.dirname(REPORT)):
    os.makedirs(d, exist_ok=True)

sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "axes.titleweight": "bold"})

IMPACT = 0.002          # permanent price impact per market order ($) [PROJECT ASSUMPTION]
T_EPISODE = 600.0       # synthetic episode horizon (s)
S0 = 585.0


# --------------------------------------------------------------------------- #
# Parameter / policy construction
# --------------------------------------------------------------------------- #
def load_side_params() -> dict:
    """Load the per-side Hawkes calibration (Phase-5 calibration JSON)."""
    with open(SIDE_PARAMS) as f:
        return json.load(f)


def make_sided(sp: dict) -> SidedHawkes:
    """Fresh :class:`SidedHawkes` state from the calibration dict."""
    return SidedHawkes(
        HawkesIntensity(sp["buy"]["mu"], sp["buy"]["alpha"], sp["buy"]["beta"]),
        HawkesIntensity(sp["sell"]["mu"], sp["sell"]["alpha"], sp["sell"]["beta"]),
    )


# --------------------------------------------------------------------------- #
# Metric panel
# --------------------------------------------------------------------------- #
def compute_metrics(res: BacktestResult) -> dict[str, float]:
    r"""Full metric panel for one backtest.

    Returns
    -------
    dict
        ``total_pnl, sharpe, inv_var, max_drawdown, avg_spread, realized_spread,
        fill_rate, adverse_selection_ratio, inventory_half_life, avg_quote_lifetime``.
    """
    h = res.history
    w = h["wealth"].to_numpy()
    inv = h["inventory"].to_numpy()
    T = float(h.attrs.get("horizon", h["time"].iloc[-1]))

    dw = np.diff(w)
    sharpe = float(dw.mean() / dw.std() * np.sqrt(len(dw))) if dw.std() > 0 else 0.0
    running_max = np.maximum.accumulate(w)
    max_dd = float(np.max(running_max - w))

    # adverse-selection decomposition on fills
    f = res.fills
    if len(f):
        quoted_edge = np.abs(f["price"].to_numpy() - f["mid"].to_numpy())
        markout = f["markout"].to_numpy()             # realized (mark-out) edge
        adverse = quoted_edge - markout               # adverse-selection component
        qe = quoted_edge.mean()
        realized = float(markout.mean())
        adverse_ratio = float(adverse.mean() / qe) if qe > 0 else np.nan
    else:
        realized, adverse_ratio = np.nan, np.nan

    # inventory half-life from an AR(1) fit  inv[k+1] = phi inv[k] + c
    if len(inv) > 10 and np.std(inv) > 0:
        phi = np.polyfit(inv[:-1], inv[1:], 1)[0]
        mean_dt = T / len(inv)
        half_life = float(-np.log(2) / np.log(phi) * mean_dt) if 0 < phi < 1 else np.nan
    else:
        half_life = np.nan

    n_changes = max(h.attrs.get("n_quote_changes", len(h)), 1)
    return {
        "total_pnl": float(res.terminal_wealth),
        "sharpe": sharpe,
        "inv_var": float(np.var(inv)),
        "max_drawdown": max_dd,
        "avg_spread": float(h["spread"].mean()),
        "realized_spread": realized,
        "fill_rate": float(len(f) / T),
        "adverse_selection_ratio": adverse_ratio,
        "inventory_half_life": half_life,
        "avg_quote_lifetime": float(T / n_changes),
    }


# --------------------------------------------------------------------------- #
# One paired path (Strategy A and B on identical flow + common random numbers)
# --------------------------------------------------------------------------- #
def _run_pair(seed: int, params: ASParameters, sp: dict,
              impact: float, T: float) -> dict:
    buy = HawkesIntensity(sp["buy"]["mu"], sp["buy"]["alpha"], sp["buy"]["beta"])
    sell = HawkesIntensity(sp["sell"]["mu"], sp["sell"]["alpha"], sp["sell"]["beta"])
    events = generate_synthetic_events(buy, sell, T, seed)
    polA = PoissonASPolicy(params)
    polB = HawkesASPolicy(params, sp["buy"]["lam_bar"], sp["sell"]["lam_bar"])
    rA = run_backtest(events, polA, make_sided(sp), params, label="Poisson",
                      synthetic=True, impact=impact, s0=S0, seed=1000 + seed)
    rB = run_backtest(events, polB, make_sided(sp), params, label="Hawkes",
                      synthetic=True, impact=impact, s0=S0, seed=1000 + seed)
    mA, mB = compute_metrics(rA), compute_metrics(rB)
    row = {"seed": seed}
    row.update({f"{k}_A": v for k, v in mA.items()})
    row.update({f"{k}_B": v for k, v in mB.items()})
    return row


def run_comparison(params: ASParameters, sp: dict, n_paths: int, *,
                   impact: float = IMPACT, T: float = T_EPISODE,
                   n_jobs: int = -1) -> pd.DataFrame:
    """Run ``n_paths`` paired A/B backtests and return the metric table."""
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_run_pair)(s, params, sp, impact, T) for s in range(n_paths))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
METRIC_LABELS = {
    "total_pnl": "Total PnL ($)", "sharpe": "Sharpe", "inv_var": "Inventory variance",
    "max_drawdown": "Max drawdown ($)", "avg_spread": "Avg spread ($)",
    "realized_spread": "Realized spread ($)", "fill_rate": "Fill rate (/s)",
    "adverse_selection_ratio": "Adverse-selection ratio",
    "inventory_half_life": "Inventory half-life (s)",
    "avg_quote_lifetime": "Avg quote lifetime (s)",
}


def plot_episode_diagnostics(res: BacktestResult, sp: dict) -> None:
    """Figures 1--6: intensity, dynamic spread, bid/ask, inventory, cash, PnL."""
    h = res.history
    t = h["time"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, h["lam_buy"], color="green", lw=0.5, alpha=0.7, label=r"$\lambda_{buy}(t)$")
    ax.plot(t, h["lam_sell"], color="red", lw=0.5, alpha=0.7, label=r"$\lambda_{sell}(t)$")
    ax.axhline(sp["buy"]["lam_bar"], color="green", ls="--", lw=0.8)
    ax.axhline(sp["sell"]["lam_bar"], color="red", ls="--", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="intensity (events/s)",
           title="Fig 1. Hawkes intensity (instantaneous) — spikes during clustering")
    ax.legend(); _savefig(fig, "fig01_intensity")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, h["spread"], color="orange", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="spread ($)",
           title="Fig 2. Dynamic Hawkes-driven spread")
    _savefig(fig, "fig02_dynamic_spread")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, h["mid"], "k", lw=0.8, label="mid")
    ax.plot(t, h["bid"], "green", lw=0.5, alpha=0.7, label="bid")
    ax.plot(t, h["ask"], "red", lw=0.5, alpha=0.7, label="ask")
    ax.set(xlabel="time (s)", ylabel="price ($)", title="Fig 3. Bid/ask evolution")
    ax.legend(); _savefig(fig, "fig03_bid_ask")

    for name, col, color, title in [
        ("fig04_inventory", "inventory", "navy", "Fig 4. Inventory"),
        ("fig05_cash", "cash", "darkgreen", "Fig 5. Cash"),
        ("fig06_pnl", "wealth", "black", "Fig 6. PnL (mark-to-market wealth)")]:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t, h[col], color=color, lw=0.9)
        ax.axhline(0, color="gray", ls=":")
        ax.set(xlabel="time (s)", ylabel=col, title=title)
        _savefig(fig, name)


def plot_comparison(df: pd.DataFrame) -> None:
    """Figures 7--9: Hawkes vs Poisson spread, inventory variance and Sharpe."""
    for name, metric, title in [
        ("fig07_spread_compare", "avg_spread", "Fig 7. Average spread: Hawkes vs Poisson"),
        ("fig08_inventory_compare", "inv_var", "Fig 8. Inventory variance: Hawkes vs Poisson"),
        ("fig09_sharpe_compare", "sharpe", "Fig 9. Sharpe ratio: Hawkes vs Poisson")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        data = pd.DataFrame({"Poisson (A)": df[f"{metric}_A"], "Hawkes (B)": df[f"{metric}_B"]})
        sns.boxplot(data=data, ax=ax, showfliers=False,
                    palette=["#1f77b4", "#d62728"], hue=data.columns.tolist() if False else None)
        sns.stripplot(data=data, ax=ax, size=2, alpha=0.25, color="black")
        ax.set(ylabel=METRIC_LABELS[metric], title=title)
        _savefig(fig, name)


def _savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric mean A vs mean B with the mean paired difference (B - A)."""
    rows = {}
    for m in METRIC_LABELS:
        a, b = df[f"{m}_A"], df[f"{m}_B"]
        rows[m] = {"Poisson_A": a.mean(), "Hawkes_B": b.mean(),
                   "diff_BmA": (b - a).mean()}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(quick: bool = False) -> None:
    """Run the full Phase-5 comparative study, tests, robustness and report."""
    from analysis import robust_quotes, stat_tests  # local import (same package)

    sp = load_side_params()
    params = ASParameters(gamma=0.1, sigma=0.03477, kappa=23.96, A=1.445,
                          T=T_EPISODE, q_max=50)
    n_paths = 40 if quick else 300
    n_jobs = 1 if quick else -1

    print("== representative Hawkes-MM episode (Figs 1-6) ==")
    buy = HawkesIntensity(sp["buy"]["mu"], sp["buy"]["alpha"], sp["buy"]["beta"])
    sell = HawkesIntensity(sp["sell"]["mu"], sp["sell"]["alpha"], sp["sell"]["beta"])
    ev = generate_synthetic_events(buy, sell, T_EPISODE, seed=7)
    polB = HawkesASPolicy(params, sp["buy"]["lam_bar"], sp["sell"]["lam_bar"])
    repB = run_backtest(ev, polB, make_sided(sp), params, label="Hawkes",
                        synthetic=True, impact=IMPACT, s0=S0, seed=1007)
    plot_episode_diagnostics(repB, sp)

    print(f"== paired comparison ({n_paths} paths) ==")
    df = run_comparison(params, sp, n_paths, n_jobs=n_jobs)
    df.to_parquet(os.path.join(RESULTS, "hawkes_vs_poisson.parquet"))
    tbl = summary_table(df)
    print(tbl.round(4).to_string())
    plot_comparison(df)

    print("== statistical tests (Part 8) ==")
    tests = stat_tests.run_all_tests(df)
    stat_tests.plot_significance(df, tests, PLOTS)
    for k, v in tests.items():
        print(f"  {k}: dmean={v['mean_diff']:+.4f} t_p={v['t_p']:.3g} "
              f"w_p={v['wilcoxon_p']:.3g} CI=[{v['boot_lo']:.4f},{v['boot_hi']:.4f}]")

    print("== sensitivity heatmaps (Part 9) ==")
    heat = run_heatmaps(sp)

    print("== uncertainty-aware extension (Part 10) ==")
    robust = robust_quotes.run_robust_study(params, sp, n_paths=n_paths, n_jobs=n_jobs)
    robust_quotes.plot_robust(robust, PLOTS)

    print("== LOBSTER historical replay (Part 6) ==")
    replay = run_replay(params, sp)

    write_report(params, sp, tbl, tests, heat, robust, replay,
                 {"n_paths": n_paths, "impact": IMPACT, "T": T_EPISODE})
    print(f"DONE. Report -> {os.path.relpath(REPORT, BASE)}")


def run_heatmaps(sp: dict, n_paths: int = 60) -> dict:
    """Part 9: gamma x rho heatmaps of PnL, inventory variance and Sharpe.

    ``rho`` is swept by rescaling the excitation ``alpha = rho * beta`` on both
    sides (holding ``beta`` fixed); ``gamma`` is the AS risk aversion.
    """
    gammas = [0.02, 0.05, 0.1, 0.2, 0.4]
    rhos = [0.1, 0.3, 0.5, 0.7, 0.9]
    pnl = np.zeros((len(gammas), len(rhos)))
    ivar = np.zeros_like(pnl)
    shp = np.zeros_like(pnl)
    for i, g in enumerate(gammas):
        for j, rho in enumerate(rhos):
            spj = json.loads(json.dumps(sp))  # deep copy
            for side in ("buy", "sell"):
                spj[side]["alpha"] = rho * spj[side]["beta"]
                spj[side]["lam_bar"] = spj[side]["mu"] / (1 - rho)
            params = ASParameters(gamma=g, sigma=0.03477, kappa=23.96, A=1.445,
                                  T=T_EPISODE, q_max=50)
            rows = [_run_pair(s, params, spj, IMPACT, T_EPISODE) for s in range(n_paths)]
            d = pd.DataFrame(rows)
            pnl[i, j] = d["total_pnl_B"].mean()
            ivar[i, j] = d["inv_var_B"].mean()
            shp[i, j] = d["sharpe_B"].mean()
    _plot_heatmaps(gammas, rhos, pnl, ivar, shp)
    return {"gammas": gammas, "rhos": rhos, "pnl": pnl, "inv_var": ivar, "sharpe": shp}


def _plot_heatmaps(gammas, rhos, pnl, ivar, shp) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, M, title, cmap in [
        (axes[0], pnl, "Fig 10a. Mean PnL ($)", "viridis"),
        (axes[1], ivar, "Fig 10b. Inventory variance", "magma_r"),
        (axes[2], shp, "Fig 10c. Sharpe", "viridis")]:
        sns.heatmap(M, ax=ax, annot=True, fmt=".2f", cmap=cmap,
                    xticklabels=[f"{r:.1f}" for r in rhos],
                    yticklabels=[f"{g:.2f}" for g in gammas])
        ax.set(xlabel=r"branching ratio $\rho$", ylabel=r"risk aversion $\gamma$", title=title)
    fig.suptitle(r"Fig 10. Hawkes-MM sensitivity across $\gamma \times \rho$", fontweight="bold")
    _savefig(fig, "fig10_heatmap_gamma_rho")


def run_replay(params: ASParameters, sp: dict) -> dict:
    """Part 6: replay the actual LOBSTER stream for both strategies."""
    lob = load_lobster_events(DATA)
    pr = ASParameters(gamma=params.gamma, sigma=params.sigma, kappa=params.kappa,
                      A=params.A, T=float(lob["time"].max()), q_max=params.q_max)
    out = {}
    for label, pol in [("Poisson_A", PoissonASPolicy(pr)),
                       ("Hawkes_B", HawkesASPolicy(pr, sp["buy"]["lam_bar"], sp["sell"]["lam_bar"]))]:
        res = run_backtest(lob, pol, make_sided(sp), pr, label=label, synthetic=False)
        out[label] = compute_metrics(res)
    return out


def write_report(params, sp, tbl, tests, heat, robust, replay, meta) -> None:
    """Delegate to the report writer (kept in a separate function for clarity)."""
    from analysis._hawkes_report import write_hawkes_report
    write_hawkes_report(REPORT, params, sp, tbl, tests, heat, robust, replay, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
