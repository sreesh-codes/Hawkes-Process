r"""Phase 6 orchestrator: robust (uncertainty-aware) Hawkes market making.

Runs Parts 1--8 end to end -- parameter covariance (observed Fisher + parametric
bootstrap), the delta-method intensity band, uncertainty propagation through the
market maker, the robust quote construction, the robust backtest (Classical vs
Hawkes vs Robust), the significance tests, the :math:`c\times\gamma\times\rho`
sensitivity analysis and the HPC uncertainty simulation -- saves every figure to
``plots/robust_market_making/`` and the numeric artefacts to ``results/robust/``,
then writes the mathematical report ``reports/robust_market_making.md`` (Part 9).

Run ``python -m analysis.robust_market_making`` (add ``--quick`` for a fast smoke
run). The heavy knobs (bootstrap size, backtest paths, HPC replicates) are scaled
down under ``--quick`` and reported honestly in the output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_maker import ASParameters  # noqa: E402
from robust.data import load_streams  # noqa: E402
from robust.covariance import PARAM_NAMES, fisher_covariance  # noqa: E402
from robust.bootstrap import run_parametric_bootstrap  # noqa: E402
from robust.delta_method import confidence_band, lambda_bar, lambda_bar_variance  # noqa: E402
from robust.uncertainty import SidedBeliefs  # noqa: E402
from robust.robust_quotes import RobustHawkesASPolicy  # noqa: E402
from robust.hpc import run_hpc  # noqa: E402
from market_maker.hawkes_execution import HawkesIntensity  # noqa: E402
from market_maker.historical_replay import (generate_synthetic_events,  # noqa: E402
                                            load_lobster_events)
from analysis.robust_backtest import (StrategySpec, make_belief_engines,  # noqa: E402
                                      run_paired_paths, run_robust_backtest)
from analysis.comparison import compute_metrics  # noqa: E402
from analysis import robust_sensitivity as rs  # noqa: E402
from analysis import statistics as st_tests  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(BASE, "plots", "robust_market_making")
RESULTS = os.path.join(BASE, "results", "robust")
REPORT = os.path.join(BASE, "reports", "robust_market_making.md")
DATA = os.path.join(BASE, "LOBSTER_SampleFile_AAPL_2012-06-21_10",
                    "AAPL_2012-06-21_34200000_57600000_")
for d in (PLOTS, RESULTS, os.path.dirname(REPORT)):
    os.makedirs(d, exist_ok=True)

sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150,
                     "axes.titleweight": "bold"})

IMPACT = 0.002
T_EPISODE = 600.0
S0 = 585.0
C_GRID = [0.0, 0.5, 1.0, 2.0]
AS_PARAMS = ASParameters(gamma=0.1, sigma=0.03477, kappa=23.96, A=1.445,
                         T=T_EPISODE, q_max=50)


def _savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Part 1 -- covariance (Fisher + bootstrap)
# --------------------------------------------------------------------------- #
def part1_covariance(streams: dict, n_boot: int, n_jobs: int) -> dict:
    """Observed-Fisher and parametric-bootstrap covariance of the 'all' fit."""
    fisher = {name: fisher_covariance(s.theta, s.t, s.T)
              for name, s in streams.items()}
    s_all = streams["all"]
    t0 = time.time()
    boot = run_parametric_bootstrap(s_all.theta, s_all.T, n_boot=n_boot,
                                    n_starts=3, n_jobs=n_jobs)
    boot_secs = time.time() - t0

    fc = fisher["all"]
    tbl = pd.DataFrame({
        "estimate": s_all.theta,
        "fisher_se": fc.std_errors,
        "boot_se": np.sqrt(np.diag(boot.cov)),
        "boot_bias": boot.bias,
    }, index=list(PARAM_NAMES))
    _plot_covariance(fc, boot)
    out = {"fisher": fisher, "boot": boot, "table": tbl, "boot_secs": boot_secs}
    tbl.to_csv(os.path.join(RESULTS, "covariance_table.csv"))
    np.savez(os.path.join(RESULTS, "covariances.npz"),
             fisher_all=fc.cov, boot_all=boot.cov,
             fisher_buy=fisher["buy"].cov, fisher_sell=fisher["sell"].cov)
    return out


def _plot_covariance(fc, boot) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(3)
    axes[0].bar(x - 0.2, fc.std_errors, 0.4, label="observed Fisher", color="#1f77b4")
    axes[0].bar(x + 0.2, np.sqrt(np.diag(boot.cov)), 0.4, label="bootstrap",
                color="#d62728")
    axes[0].set(xticks=x, title="Fig 1a. Standard errors: Fisher vs bootstrap",
                ylabel="standard error")
    axes[0].set_xticklabels(PARAM_NAMES)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    sns.heatmap(fc.correlation, ax=axes[1], annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, xticklabels=PARAM_NAMES, yticklabels=PARAM_NAMES,
                cbar=False)
    axes[1].set_title("Fig 1b. Fisher correlation matrix")
    # bootstrap scatter alpha vs beta with the Fisher 95% ellipse
    est = boot.estimates
    axes[2].scatter(est[:, 1], est[:, 2], s=6, alpha=0.25, color="gray",
                    label="bootstrap")
    _ellipse(axes[2], fc.theta[1:], fc.cov[1:, 1:], color="#1f77b4")
    axes[2].scatter([fc.theta[1]], [fc.theta[2]], color="black", zorder=5,
                    label=r"$\hat\theta$")
    axes[2].set(xlabel=r"$\alpha$", ylabel=r"$\beta$",
                title="Fig 1c. Bootstrap cloud + Fisher 95% ellipse")
    axes[2].legend(fontsize=8)
    fig.suptitle("Fig 1. Parameter estimation covariance [Part 1]", fontweight="bold")
    _savefig(fig, "fig01_covariance")


def _ellipse(ax, mean, cov2, *, color, nsig=2.4477) -> None:
    """Draw an ``nsig`` (≈95% for 2 dof) covariance ellipse."""
    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    from matplotlib.patches import Ellipse
    w, h = 2 * nsig * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=theta, fill=False,
                         edgecolor=color, lw=2, label="Fisher 95%"))


# --------------------------------------------------------------------------- #
# Part 2 -- delta-method intensity band
# --------------------------------------------------------------------------- #
def part2_delta_band(streams: dict, fisher: dict) -> dict:
    """Delta-method 95% confidence band for lambda(t) on a representative window."""
    s = streams["all"]
    fc = fisher["all"]
    window = 120.0
    grid = np.linspace(0, window, 4000)
    ev = s.t[s.t <= window]
    band = confidence_band(grid, ev, s.theta, fc.cov, level=0.95)

    lb = lambda_bar(s.theta)
    lb_se = np.sqrt(lambda_bar_variance(s.theta, fc.cov))

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.fill_between(grid, band.lower, band.upper, color="#1f77b4", alpha=0.3,
                    label="95% delta-method band")
    ax.plot(grid, band.intensity, color="black", lw=0.7, label=r"$\hat\lambda(t)$")
    ax.plot(ev, np.full_like(ev, -0.4), "|", color="crimson", ms=6, alpha=0.5,
            label="events")
    ax.axhline(lb, color="green", ls="--", lw=1,
               label=rf"$\bar\lambda={lb:.3f}\pm{1.96*lb_se:.3f}$")
    ax.set(xlabel="time (s)", ylabel=r"$\lambda(t)$ (events/s)", ylim=(-0.8, None),
           title="Fig 2. Hawkes intensity with delta-method 95% band [Part 2]")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, "fig02_intensity_band")
    return {"lam_bar": lb, "lam_bar_se": lb_se,
            "rel_band_median": float(np.median(
                (band.upper - band.lower) / np.maximum(band.intensity, 1e-9)))}


# --------------------------------------------------------------------------- #
# Part 3 -- uncertainty propagation (three beliefs through the MM)
# --------------------------------------------------------------------------- #
def part3_propagation(thb, ths, cov_buy, cov_sell, beliefs, n_paths, n_jobs) -> pd.DataFrame:
    """Run lower/central/upper Hawkes beliefs through the market maker."""
    specs = [StrategySpec("lower", "hawkes", belief="lower"),
             StrategySpec("central", "hawkes", belief="central"),
             StrategySpec("upper", "hawkes", belief="upper")]
    df = run_paired_paths(thb, ths, beliefs, cov_buy, cov_sell, AS_PARAMS, specs,
                          n_paths, impact=IMPACT, T=T_EPISODE, n_jobs=n_jobs)
    order = ["lower", "central", "upper"]
    metrics = ["avg_spread", "inv_var", "adverse_selection_ratio", "total_pnl"]
    labels = ["avg spread ($)", "inventory variance", "adverse-sel. ratio", "PnL ($)"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, m, lab in zip(axes, metrics, labels):
        means = [df[f"{m}_{b}"].mean() for b in order]
        ax.plot(order, means, "o-", lw=2, color="#1f77b4")
        ax.set(title=lab, xlabel="intensity belief")
    fig.suptitle("Fig 3. Uncertainty propagation: lower/central/upper beliefs "
                 "through the MM [Part 3]", fontweight="bold")
    _savefig(fig, "fig03_propagation")
    return df


# --------------------------------------------------------------------------- #
# Part 4 -- robust quote construction (static illustration)
# --------------------------------------------------------------------------- #
def part4_quotes(thb, ths, cov_buy, cov_sell) -> None:
    """Illustrate how the variance penalty widens the spread vs c and vs sd."""
    lbb, lbs = lambda_bar(thb), lambda_bar(ths)
    # sweep normalised intensity R and show robust half-spread for several c,
    # holding the delta-method sd at a representative burst level.
    R = np.linspace(0.5, 40, 300)
    sd_ref = 1.5   # representative intensity sd during activity
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for c in C_GRID:
        pol = RobustHawkesASPolicy(AS_PARAMS, lbb, lbs, c=c)
        half = [pol._half_micro(r) + pol._half_penalty(sd_ref, lbb, r) for r in R]
        axes[0].plot(R, half, lw=1.8, label=f"c={c}")
    axes[0].set(xlabel=r"normalised intensity $R=\lambda/\bar\lambda$",
                ylabel="ask half-spread ($)",
                title="Fig 4a. Variance-penalised half-spread vs $R$")
    axes[0].legend(fontsize=8)
    # penalty vs sd at fixed R
    sd = np.linspace(0, 4, 200)
    for c in C_GRID:
        pol = RobustHawkesASPolicy(AS_PARAMS, lbb, lbs, c=c)
        pen = [pol._half_penalty(x, lbb, 1.0) for x in sd]
        axes[1].plot(sd, pen, lw=1.8, label=f"c={c}")
    axes[1].set(xlabel=r"intensity sd $\sqrt{Var\lambda}$",
                ylabel="added half-spread ($)",
                title=r"Fig 4b. Penalty grows linearly in $\sqrt{Var\lambda}$ (at $R=1$)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Fig 4. Robust quote construction (Method B) [Part 4]",
                 fontweight="bold")
    _savefig(fig, "fig04_robust_quotes")


# --------------------------------------------------------------------------- #
# Part 5 + 6 -- robust backtest and significance
# --------------------------------------------------------------------------- #
def part5_backtest(thb, ths, cov_buy, cov_sell, beliefs, n_paths, n_jobs,
                   c_robust: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Synthetic paired backtest of Classical / Hawkes / Robust + LOBSTER replay."""
    specs = [StrategySpec("Classical", "poisson"),
             StrategySpec("Hawkes", "hawkes", belief="central"),
             StrategySpec("Robust", "robust", c=c_robust, belief="central")]
    df = run_paired_paths(thb, ths, beliefs, cov_buy, cov_sell, AS_PARAMS, specs,
                          n_paths, impact=IMPACT, T=T_EPISODE, n_jobs=n_jobs)

    names = [s.name for s in specs]
    keys = ["total_pnl", "sharpe", "inv_var", "max_drawdown", "avg_spread",
            "realized_spread", "adverse_selection_ratio", "fill_rate",
            "avg_quote_lifetime", "inventory_half_life"]
    summary = pd.DataFrame(
        {n: {k: df[f"{k}_{n}"].mean() for k in keys} for n in names})

    replay = _run_replay(thb, ths, cov_buy, cov_sell, beliefs, c_robust)
    _plot_backtest(df, names)
    df.to_csv(os.path.join(RESULTS, "backtest_paths.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS, "backtest_summary.csv"))
    return df, summary, replay


def _run_replay(thb, ths, cov_buy, cov_sell, beliefs, c_robust: float) -> dict:
    """Replay the actual LOBSTER stream for the three strategies."""
    from market_maker.hawkes_market_maker import HawkesASPolicy, PoissonASPolicy
    lob = load_lobster_events(DATA)
    pr = ASParameters(gamma=AS_PARAMS.gamma, sigma=AS_PARAMS.sigma,
                      kappa=AS_PARAMS.kappa, A=AS_PARAMS.A,
                      T=float(lob["time"].max()), q_max=AS_PARAMS.q_max)
    cb, cs = beliefs.buy["central"], beliefs.sell["central"]
    lbb, lbs = lambda_bar(cb), lambda_bar(cs)
    strategies = {
        "Classical": (PoissonASPolicy(pr), False),
        "Hawkes": (HawkesASPolicy(pr, lbb, lbs), False),
        "Robust": (RobustHawkesASPolicy(pr, lbb, lbs, c=c_robust), True),
    }
    out = {}
    for name, (pol, uv) in strategies.items():
        sr, sv = make_belief_engines(cb, cs, cov_buy, cov_sell)
        res = run_robust_backtest(lob, pol, sr, sv, pr, label=name,
                                  synthetic=False, use_variance=uv)
        out[name] = compute_metrics(res)
    return out


def _plot_backtest(df: pd.DataFrame, names: list[str]) -> None:
    metrics = ["total_pnl", "inv_var", "max_drawdown", "avg_spread",
               "adverse_selection_ratio", "fill_rate"]
    labels = ["Total PnL ($)", "Inventory variance", "Max drawdown ($)",
              "Avg spread ($)", "Adverse-selection ratio", "Fill rate (/s)"]
    colors = {"Classical": "#1f77b4", "Hawkes": "#ff7f0e", "Robust": "#2ca02c"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, m, lab in zip(axes.ravel(), metrics, labels):
        data = pd.DataFrame({n: df[f"{m}_{n}"] for n in names})
        sns.boxplot(data=data, ax=ax, showfliers=False,
                    palette=[colors[n] for n in names])
        ax.set(ylabel=lab)
    fig.suptitle("Fig 5. Robust backtest: Classical vs Hawkes vs Robust "
                 "(paired, synthetic) [Part 5]", fontweight="bold")
    _savefig(fig, "fig05_backtest")


def part6_significance(df: pd.DataFrame) -> dict:
    """Paired significance tests (Cohen's d) for Robust vs Hawkes and vs Classical."""
    comparisons = {"Robust_vs_Hawkes": ("Hawkes", "Robust"),
                   "Robust_vs_Classical": ("Classical", "Robust"),
                   "Hawkes_vs_Classical": ("Classical", "Hawkes")}
    tested = ["total_pnl", "sharpe", "inv_var", "max_drawdown",
              "adverse_selection_ratio"]
    out: dict[str, dict] = {}
    for cmp_name, (a, b) in comparisons.items():
        out[cmp_name] = {}
        for m in tested:
            res = st_tests.paired_comparison(df[f"{m}_{a}"].to_numpy(),
                                             df[f"{m}_{b}"].to_numpy())
            out[cmp_name][m] = res
    _plot_significance(out)
    return out


def _plot_significance(tests: dict) -> None:
    metrics = ["total_pnl", "sharpe", "inv_var", "max_drawdown",
               "adverse_selection_ratio"]
    cmp_name = "Robust_vs_Hawkes"
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ds = [tests[cmp_name][m].cohens_d for m in metrics]
    colors = ["green" if tests[cmp_name][m].significant else "gray" for m in metrics]
    ax.bar(metrics, ds, color=colors)
    ax.axhline(0, color="black", lw=1)
    for lvl, ls in [(0.2, ":"), (0.5, "--"), (0.8, "-.")]:
        ax.axhline(lvl, color="gray", ls=ls, lw=0.7)
        ax.axhline(-lvl, color="gray", ls=ls, lw=0.7)
    ax.set(ylabel="Cohen's d (Robust $-$ Hawkes)",
           title="Fig 6. Effect sizes, Robust vs Hawkes (green = significant) [Part 6]")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _savefig(fig, "fig06_significance")


# --------------------------------------------------------------------------- #
# Part 7 -- sensitivity
# --------------------------------------------------------------------------- #
def part7_sensitivity(thb, ths, cov_buy, cov_sell, n_paths, n_jobs) -> dict:
    """c x gamma and c x rho heatmaps + 3-D surface + response curve."""
    cs = [0.0, 0.5, 1.0, 2.0]
    gammas = [0.02, 0.05, 0.1, 0.2, 0.4]
    rhos = [0.2, 0.35, 0.47, 0.6, 0.75]
    cg = rs.sweep_c_gamma(cs, gammas, thb, ths, cov_buy, cov_sell,
                          n_paths=n_paths, base=AS_PARAMS, impact=IMPACT,
                          T=T_EPISODE, n_jobs=n_jobs)
    cr = rs.sweep_c_rho(cs, rhos, thb, ths, cov_buy, cov_sell, n_paths=n_paths,
                        base=AS_PARAMS, impact=IMPACT, T=T_EPISODE, n_jobs=n_jobs)
    curve = rs.response_curve(np.linspace(0, 2.5, 9), thb, ths, cov_buy, cov_sell,
                              n_paths=n_paths, base=AS_PARAMS, impact=IMPACT,
                              T=T_EPISODE, n_jobs=n_jobs)
    _plot_sensitivity(cg, cr, curve)
    curve.to_csv(os.path.join(RESULTS, "response_curve.csv"), index=False)
    return {"cg": cg, "cr": cr, "curve": curve}


def _plot_sensitivity(cg, cr, curve) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    sns.heatmap(cg["inv_var"], ax=axes[0], annot=True, fmt=".2f", cmap="magma_r",
                xticklabels=[f"{g:.2f}" for g in cg["gammas"]],
                yticklabels=[f"{c:.1f}" for c in cg["cs"]])
    axes[0].set(xlabel=r"$\gamma$", ylabel="robustness $c$",
                title="Fig 7a. Inventory variance ($c\\times\\gamma$)")
    sns.heatmap(cr["inv_var"], ax=axes[1], annot=True, fmt=".2f", cmap="magma_r",
                xticklabels=[f"{r:.2f}" for r in cr["rhos"]],
                yticklabels=[f"{c:.1f}" for c in cr["cs"]])
    axes[1].set(xlabel=r"branching ratio $\rho$", ylabel="robustness $c$",
                title="Fig 7b. Inventory variance ($c\\times\\rho$)")
    ax = axes[2]
    ax.plot(curve["c"], curve["total_pnl"], "o-", color="darkgreen", label="PnL ($)")
    ax.set(xlabel="robustness $c$", ylabel="PnL ($)",
           title="Fig 7c. PnL vs robustness $c$")
    ax2 = ax.twinx()
    ax2.plot(curve["c"], curve["inv_var"], "s--", color="navy", label="inv. var")
    ax2.set_ylabel("inventory variance", color="navy")
    fig.suptitle(r"Fig 7. Robust-MM sensitivity to $c,\gamma,\rho$ [Part 7]",
                 fontweight="bold")
    _savefig(fig, "fig07_sensitivity")

    # 3-D PnL surface over c x gamma
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    G, C = np.meshgrid(cg["gammas"], cg["cs"])
    ax.plot_surface(C, G, cg["pnl"], cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set(xlabel="c", ylabel=r"$\gamma$", zlabel="PnL ($)",
           title="Fig 8. PnL surface over $c\\times\\gamma$ [Part 7]")
    fig.savefig(os.path.join(PLOTS, "fig08_pnl_surface.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Part 8 -- HPC
# --------------------------------------------------------------------------- #
def part8_hpc(thb, ths, cov_buy, cov_sell, n_sims, n_jobs, c_robust) -> dict:
    """HPC uncertainty simulation: distributions under theta ~ N(theta_hat, Sigma)."""
    t0 = time.time()
    res = run_hpc(thb, ths, cov_buy, cov_sell, AS_PARAMS, n_sims=n_sims,
                  c=c_robust, impact=IMPACT, T=T_EPISODE, n_jobs=n_jobs,
                  batch=20000)
    secs = time.time() - t0
    _plot_hpc(res)
    summ = res.summary()
    with open(os.path.join(RESULTS, "hpc_summary.json"), "w") as f:
        json.dump({"n_success": res.n_success, "n_rejected_draws": res.n_rejected_draws,
                   "seconds": secs, "summary": summ}, f, indent=2)
    return {"result": res, "summary": summ, "seconds": secs}


def _plot_hpc(res) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("pnl", res.pnl, "Total PnL ($)", "#2ca02c"),
              ("inv_var", res.inv_var, "Inventory variance", "navy"),
              ("sharpe", res.sharpe, "Sharpe", "#1f77b4"),
              ("max_drawdown", res.max_drawdown, "Max drawdown ($)", "crimson")]
    for ax, (key, arr, lab, color) in zip(axes.ravel(), panels):
        a = arr[np.isfinite(arr)]
        ax.hist(a, bins=60, color=color, alpha=0.75, density=True)
        for q, ls in [(5, ":"), (50, "-"), (95, ":")]:
            ax.axvline(np.percentile(a, q), color="black", ls=ls, lw=1)
        ax.set(xlabel=lab, ylabel="density")
    fig.suptitle(f"Fig 9. HPC predictive distributions under "
                 rf"$\theta\sim N(\hat\theta,\Sigma)$ ({res.n_success:,} sims) [Part 8]",
                 fontweight="bold")
    _savefig(fig, "fig09_hpc_distributions")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(quick: bool = False) -> None:
    from analysis._robust_report import write_robust_report

    n_boot = 200 if quick else 1000
    n_paths = 40 if quick else 300
    n_sens = 20 if quick else 60
    n_hpc = 500 if quick else 5000
    n_jobs = 1 if quick else -1
    c_robust = 1.0

    streams = load_streams()
    thb, ths = streams["buy"].theta, streams["sell"].theta

    print(f"== Part 1: covariance (bootstrap B={n_boot}) ==")
    p1 = part1_covariance(streams, n_boot, n_jobs)
    cov_buy = p1["fisher"]["buy"].cov
    cov_sell = p1["fisher"]["sell"].cov
    print(p1["table"].to_string())

    print("== Part 2: delta-method band ==")
    p2 = part2_delta_band(streams, p1["fisher"])

    beliefs = SidedBeliefs.from_covariances(thb, cov_buy, ths, cov_sell)

    print(f"== Part 3: uncertainty propagation ({n_paths} paths) ==")
    p3 = part3_propagation(thb, ths, cov_buy, cov_sell, beliefs, n_paths, n_jobs)

    print("== Part 4: robust quote construction ==")
    part4_quotes(thb, ths, cov_buy, cov_sell)

    print(f"== Part 5: robust backtest ({n_paths} paths) ==")
    df, summary, replay = part5_backtest(thb, ths, cov_buy, cov_sell, beliefs,
                                         n_paths, n_jobs, c_robust)
    print(summary.round(4).to_string())

    print("== Part 6: significance ==")
    tests = part6_significance(df)

    print(f"== Part 7: sensitivity ({n_sens} paths/cell) ==")
    p7 = part7_sensitivity(thb, ths, cov_buy, cov_sell, n_sens, n_jobs)

    print(f"== Part 8: HPC ({n_hpc} sims) ==")
    p8 = part8_hpc(thb, ths, cov_buy, cov_sell, n_hpc, n_jobs, c_robust)
    print(json.dumps(p8["summary"], indent=0))

    print("== Part 9: report ==")
    write_robust_report(REPORT, streams, p1, p2, p3, df, summary, replay, tests,
                        p7, p8, {"n_boot": n_boot, "n_paths": n_paths,
                                 "n_sens": n_sens, "n_hpc": n_hpc,
                                 "c_robust": c_robust, "impact": IMPACT,
                                 "T": T_EPISODE, "as_params": AS_PARAMS})
    print(f"DONE. Report -> {os.path.relpath(REPORT, BASE)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
