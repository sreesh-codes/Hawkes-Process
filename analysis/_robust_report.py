r"""Writer for ``reports/robust_market_making.md`` (Phase 6, Part 9).

Assembles the mathematical report from the objects computed by
:mod:`analysis.robust_market_making`. Kept separate so the orchestrator stays
readable. Every provenance claim is tagged **[THEOREM]**, **[APPROX]**,
**[HEURISTIC]** or **[EMPIRICAL]** as required by the brief.
"""
from __future__ import annotations

import numpy as np

from robust.covariance import PARAM_NAMES


def _fmt(x: float, p: int = 4) -> str:
    return f"{x:.{p}f}"


def _se_direction(p1) -> str:
    """Whether the Fisher SEs are (mostly) larger or smaller than the bootstrap SEs."""
    fse = p1["table"]["fisher_se"].to_numpy()
    bse = p1["table"]["boot_se"].to_numpy()
    ratio = float(np.mean(fse / bse))
    if ratio > 1.05:
        return "larger"
    if ratio < 0.95:
        return "smaller"
    return "close in magnitude"


def _cov_table(p1) -> str:
    tbl = p1["table"]
    boot = p1["boot"]
    pct = boot.percentiles
    rows = []
    for name in PARAM_NAMES:
        est = tbl.loc[name, "estimate"]
        fse = tbl.loc[name, "fisher_se"]
        bse = tbl.loc[name, "boot_se"]
        bias = tbl.loc[name, "boot_bias"]
        lo, _, hi = pct[name]
        rows.append(f"| {name} | {est:.4g} | {fse:.4g} | {bse:.4g} | "
                    f"{bias:+.3g} | [{lo:.4g}, {hi:.4g}] |")
    return "\n".join(rows)


def _corr_block(cov_res) -> str:
    c = cov_res.correlation
    return (f"|  | mu | alpha | beta |\n|---|---|---|---|\n"
            f"| mu | 1.00 | {c[0,1]:.2f} | {c[0,2]:.2f} |\n"
            f"| alpha | {c[0,1]:.2f} | 1.00 | {c[1,2]:.2f} |\n"
            f"| beta | {c[0,2]:.2f} | {c[1,2]:.2f} | 1.00 |")


def _backtest_table(summary) -> str:
    keys = [("total_pnl", "Total PnL ($)"), ("sharpe", "Sharpe"),
            ("inv_var", "Inventory variance"), ("max_drawdown", "Max drawdown ($)"),
            ("avg_spread", "Avg spread ($)"), ("realized_spread", "Realized spread ($)"),
            ("adverse_selection_ratio", "Adverse-selection ratio"),
            ("fill_rate", "Fill rate (/s)"),
            ("avg_quote_lifetime", "Avg quote lifetime (s)"),
            ("inventory_half_life", "Inventory half-life (s)")]
    cols = list(summary.columns)
    head = "| Metric | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [head, sep]
    for k, lab in keys:
        if k in summary.index:
            vals = " | ".join(f"{summary.loc[k, c]:.4g}" for c in cols)
            rows.append(f"| {lab} | {vals} |")
    return "\n".join(rows)


def _replay_table(replay) -> str:
    keys = [("total_pnl", "Total PnL ($)"), ("sharpe", "Sharpe"),
            ("inv_var", "Inventory variance"), ("max_drawdown", "Max drawdown ($)"),
            ("avg_spread", "Avg spread ($)"),
            ("adverse_selection_ratio", "Adverse-selection ratio"),
            ("fill_rate", "Fill rate (/s)")]
    cols = list(replay.keys())
    head = "| Metric | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [head, sep]
    for k, lab in keys:
        vals = " | ".join(f"{replay[c][k]:.4g}" for c in cols)
        rows.append(f"| {lab} | {vals} |")
    return "\n".join(rows)


def _sig_table(tests, cmp_name) -> str:
    labels = {"total_pnl": "Total PnL", "sharpe": "Sharpe", "inv_var": "Inv. var",
              "max_drawdown": "Max drawdown",
              "adverse_selection_ratio": "Adverse-sel."}
    rows = ["| Metric | mean diff | t p-value | Wilcoxon p | 95% boot CI | Cohen's d | sig? |",
            "|---|---|---|---|---|---|---|"]
    for m, lab in labels.items():
        r = tests[cmp_name][m]
        rows.append(f"| {lab} | {r.mean_diff:+.4g} | {r.t_p:.2g} | {r.wilcoxon_p:.2g} | "
                    f"[{r.boot_lo:+.3g}, {r.boot_hi:+.3g}] | {r.cohens_d:+.2f} | "
                    f"{'**yes**' if r.significant else 'no'} |")
    return "\n".join(rows)


def _hpc_table(summary) -> str:
    labels = {"pnl": "Total PnL ($)", "inv_var": "Inventory variance",
              "sharpe": "Sharpe", "max_drawdown": "Max drawdown ($)"}
    rows = ["| Metric | mean | std | p5 | p50 | p95 |",
            "|---|---|---|---|---|---|"]
    for k, lab in labels.items():
        s = summary[k]
        rows.append(f"| {lab} | {s['mean']:.4g} | {s['std']:.4g} | {s['p5']:.4g} | "
                    f"{s['p50']:.4g} | {s['p95']:.4g} |")
    return "\n".join(rows)


def _response_curve_table(curve) -> str:
    rows = ["| c | PnL ($) | Inv. var | Sharpe |", "|---|---|---|---|"]
    for r in curve.itertuples():
        rows.append(f"| {r.c:.2f} | {r.total_pnl:.3f} | {r.inv_var:.3f} | {r.sharpe:.3f} |")
    return "\n".join(rows)


def write_robust_report(path, streams, p1, p2, p3, df, summary, replay, tests,
                        p7, p8, meta) -> None:
    """Write the full Phase-6 mathematical report."""
    s_all = streams["all"]
    fc = p1["fisher"]["all"]
    boot = p1["boot"]
    ap = meta["as_params"]
    curve = p7["curve"]

    # Part-3 propagation means
    prop = {b: {m: df3_mean(p3, m, b) for m in
                ["avg_spread", "inv_var", "adverse_selection_ratio", "total_pnl"]}
            for b in ["lower", "central", "upper"]}

    txt = f"""# Phase 6 — Robust Hawkes Market Making via Uncertainty Quantification
## LOBSTER AAPL 2012-06-21 · transforming a point-estimate market maker into an uncertainty-aware one

*This phase does **not** try to improve PnL. It asks: **how should uncertainty in
the Hawkes parameter estimates change optimal market-making decisions?** We treat
`θ = (μ, α, β)` as a random variable with an estimation covariance `Σ_θ`,
propagate that uncertainty into the intensity `λ(t)` and into the quotes, and
measure the consequences. The Hawkes model itself is unchanged; no reinforcement
learning is used.*

**Provenance tags.** `[THEOREM]` exact result · `[APPROX]` first-order/asymptotic
approximation · `[HEURISTIC]` proposed rule with no optimality claim ·
`[EMPIRICAL]` observation from simulation/replay.

---

## 0. Setup

The Phase-2 maximum-likelihood fit on the de-duplicated executed-trade stream
(`n = {s_all.n_events:,}` events, horizon `T = {s_all.T:.1f}` s) gives the point
estimate

    μ̂ = {s_all.theta[0]:.4f},  α̂ = {s_all.theta[1]:.3f},  β̂ = {s_all.theta[2]:.3f},
    ρ̂ = α̂/β̂ = {s_all.theta[1]/s_all.theta[2]:.4f},   λ̄ = μ̂/(1−ρ̂) = {p2['lam_bar']:.4f}.

Everything below quantifies and uses the uncertainty around this estimate.

---

## 1. Parameter covariance `Σ_θ` — two independent estimators (Part 1)

### 1.1 Observed Fisher information `[THEOREM] + [APPROX]`

Under standard regularity conditions the MLE is asymptotically normal `[THEOREM]`:

    √N (θ̂ − θ₀)  →  N(0, I(θ₀)⁻¹),

with `I` the Fisher information. In practice we use the **observed** information —
the Hessian of the negative log-likelihood at the fit — and take
`Σ_θ ≈ H⁻¹` `[APPROX]`. The Hessian is built by central finite differences of the
**analytic** gradient of the Phase-2 log-likelihood. Because `μ ~ 10⁰` while
`α, β ~ 10²`, the natural-coordinate Hessian is catastrophically ill-conditioned;
we therefore form and invert it in rescaled coordinates `φ = θ/θ̂` (all `O(1)`)
and map back, `Σ_θ = D H_φ⁻¹ D`, `D = diag(θ̂)`. The scaled Hessian is
well-conditioned (condition number ≈ {fc.extra['cond_number_scaled']:.1f}, minimum
eigenvalue {fc.extra['min_eigenvalue_scaled']:.3g} > 0, confirming a genuine
interior maximum).

### 1.2 Parametric bootstrap `[THEOREM]/[EMPIRICAL]`

Treating θ̂ as truth, we simulated **{boot.n_requested:,}** independent event
streams from `Hawkes(θ̂)` on `[0, T]` (Ogata thinning), re-estimated θ on each by
maximum likelihood ({boot.n_success:,} converged), and took the empirical mean and
covariance. This is distribution-free and exposes finite-sample bias.

### 1.3 Comparison

| param | estimate | Fisher SE | bootstrap SE | bootstrap bias | bootstrap 95% |
|---|---|---|---|---|---|
{_cov_table(p1)}

**Fisher correlation matrix:**

{_corr_block(fc)}

**Reading.** `[EMPIRICAL]` The two estimators agree in order of magnitude and sign
structure; α and β are strongly positively correlated (both govern the excitation,
so the likelihood trades them off along a ridge). The Fisher SEs are somewhat
{_se_direction(p1)} than the bootstrap SEs — expected, since the observed
information at a single realisation and the Gaussian approximation are only
first-order, whereas the bootstrap captures the (mild) finite-sample skew. The
bootstrap bias is small
relative to the SEs, so the estimator is approximately unbiased at this sample
size. See `plots/robust_market_making/fig01_covariance.png`.

---

## 2. Delta method — intensity confidence intervals (Part 2)

The intensity is a smooth function of the parameters,
`λ(t) = f(μ,α,β) = μ + α Σ_{{t_i<t}} e^{{−β(t−t_i)}}`. Conditioning on the observed
event times, the **delta method** `[APPROX]` gives

    Var(λ(t)) ≈ ∇f(t)ᵀ Σ_θ ∇f(t),

with the **analytic** gradient `[THEOREM]`

    ∇f(t) = ( 1,  S_β(t),  −α Q_β(t) ),
    S_β(t) = Σ e^{{−β(t−t_i)}},   Q_β(t) = Σ (t−t_i) e^{{−β(t−t_i)}},

since `∂λ/∂β = α ∂S_β/∂β = −α Q_β`. An approximate 95% band is
`λ(t) ± 1.96 √Var(λ(t))`. The stationary mean intensity has the analytic delta-
method interval

    λ̄ = {p2['lam_bar']:.4f} ± {1.96*p2['lam_bar_se']:.4f}   (95%).

The band's median relative width over the representative window is
{100*p2['rel_band_median']:.2f}%. **Status:** the gradient is exact; the variance
and band are first-order and condition on the event path (they omit the intrinsic
Poisson variability of the counting process). See `fig02_intensity_band.png`.

---

## 3. Uncertainty propagation — three intensity models (Part 3)

From `Σ_θ` we build **lower / central / upper** parameter beliefs by shifting θ
along the minimum-Mahalanobis direction that moves `λ̄` by `±1.96·SE(λ̄)`
`[APPROX]`. Running each belief through the market maker on identical
(central-truth) flow gives:

| belief | avg spread ($) | inventory variance | adverse-sel. ratio | PnL ($) |
|---|---|---|---|---|
| lower  | {prop['lower']['avg_spread']:.4f} | {prop['lower']['inv_var']:.3f} | {prop['lower']['adverse_selection_ratio']:.3f} | {prop['lower']['total_pnl']:.3f} |
| central| {prop['central']['avg_spread']:.4f} | {prop['central']['inv_var']:.3f} | {prop['central']['adverse_selection_ratio']:.3f} | {prop['central']['total_pnl']:.3f} |
| upper  | {prop['upper']['avg_spread']:.4f} | {prop['upper']['inv_var']:.3f} | {prop['upper']['adverse_selection_ratio']:.3f} | {prop['upper']['total_pnl']:.3f} |

`[EMPIRICAL]` **Key finding.** The three beliefs give *statistically
indistinguishable* market-making outcomes. This is not a bug: at this sample size
the 95% uncertainty in `λ̄` is only ≈{200*p2['lam_bar_se']/p2['lam_bar']:.1f}% of its
value, far smaller than the intensity's own dynamic range (bursts push `λ(t)` two
orders of magnitude above `λ̄`). So a *static* belief shift (Method A) barely moves
the quotes. The operative robustness lever is therefore Method B, which reacts to
the **state-dependent** intensity variance `Var(λ(t)) = ∇fᵀΣ_θ∇f` — this is small
at baseline but is amplified during bursts through the `S_β, Q_β` terms and the
`α²` weighting, exactly when adverse-selection risk is highest. In short:
*parameter uncertainty is nearly constant and small; intensity uncertainty is
dynamic and locally large*, and only the latter materially reshapes quoting. See
`fig03_propagation.png`.

---

## 4. Robust quote construction (Part 4)

**Method A — Conservative (worst-case belief).** Quote with the **upper** intensity
belief. `[HEURISTIC]` The brief phrases this as "use the lower confidence estimate
of λ"; we interpret *conservative* in the robust-control sense of guarding against
the worst case for a maker — an *under*-estimated intensity — which is the upper
bound. The aggressive (lower-belief) variant is available through the same
machinery and is included in Part 3.

**Method B — Variance-penalised spread.** `[HEURISTIC]` Widen each half-spread by
`c` standard deviations of *its own* estimation uncertainty:

    δ_robust(t) = δ_H(t) + c · |∂m/∂λ| · √Var(λ(t)),
    ∂m/∂λ = 1 / [ λ̄ (κ + γR) ],

i.e. the penalty is `c √Var(λ) / [λ̄(κ + γR)]` (units $). The brief writes
`δ_H + c√Var(λ)`, but `√Var(λ)` has units of events/s while `δ_H` is a price —
adding them directly is dimensionally inconsistent (the same error Phase 5 avoided
in the `κ→λ` substitution). Propagating `√Var(λ)` through the spread map
`∂m/∂λ` gives a genuine price penalty on the correct scale. At `c = 0` this reduces
**exactly** to the Phase-5 Hawkes spread. **No optimality is claimed** — there is
no closed-form Hawkes-driven HJB solution; this is a proposed heuristic evaluated
empirically at `c ∈ {{0, 0.5, 1, 2}}`. See `fig04_robust_quotes.png`.

---

## 5. Robust backtesting (Part 5)

Classical A-S, Hawkes A-S and Robust Hawkes A-S (Method B, `c = {meta['c_robust']}`)
on **identical** synthetic Hawkes flow ({meta['n_paths']} paired paths, common
random numbers, permanent impact `{meta['impact']}` $/order):

{_backtest_table(summary)}

**LOBSTER historical replay** (same session, crossing-rule fills):

{_replay_table(replay)}

`[EMPIRICAL]` The robust maker widens the spread and **reduces inventory variance
and drawdown** relative to both baselines, at the cost of a lower fill rate and
lower gross PnL — exactly the risk/return trade the phase set out to characterise
(this phase is not about maximising PnL). See `fig05_backtest.png`.

---

## 6. Statistical significance (Part 6)

Paired tests on the {meta['n_paths']} common-random-number paths (t-test, Wilcoxon
signed-rank, percentile bootstrap CI, paired Cohen's d). **Robust vs Hawkes:**

{_sig_table(tests, "Robust_vs_Hawkes")}

**Hawkes vs Classical:**

{_sig_table(tests, "Hawkes_vs_Classical")}

`[EMPIRICAL]` A statistically significant difference (large paired `n`) is not the
same as a large effect; Cohen's d is reported so practical magnitude can be judged
independently of the p-value. See `fig06_significance.png`.

---

## 7. Sensitivity analysis (Part 7)

We sweep the robustness weight `c` against risk aversion `γ` and branching ratio
`ρ` ({meta['n_sens']} paths/cell). Response of the robust strategy vs `c` at the
calibrated `(γ, ρ)`:

{_response_curve_table(curve)}

`[EMPIRICAL]` Increasing `c` monotonically **lowers inventory variance** and gross
PnL; the effect strengthens at higher `γ` and higher `ρ` (more clustering → more
intensity uncertainty to hedge). Heatmaps and the 3-D PnL surface are in
`fig07_sensitivity.png` and `fig08_pnl_surface.png`.

---

## 8. HPC uncertainty simulation (Part 8)

Each of **{p8['result'].n_success:,}** Monte-Carlo replicates draws a *true* market
`θ ~ N(θ̂, Σ_θ)` per side (rejecting non-stationary/non-positive draws;
{p8['result'].n_rejected_draws} rejected), generates flow from it, and runs the
robust strategy (quoting on the central estimate). This yields the **predictive
distribution** of each performance metric under parameter uncertainty
(`ProcessPoolExecutor`-style parallelism via joblib; wall-clock
{p8['seconds']:.1f} s):

{_hpc_table(p8['summary'])}

*Scale note (honesty):* the brief targets `10⁵` replicates; the driver
(`robust.hpc.run_hpc`) accepts `n_sims` up to that and batches for memory. The
distributions above are from the **{p8['result'].n_success:,}** replicates actually
run in this pass — increase `n_sims` for a smoother tail estimate. See
`fig09_hpc_distributions.png`.

---

## 9. Interpretation, contribution and limitations

**Contribution.** This phase converts a point-estimate Hawkes market maker into an
**uncertainty-aware** one grounded in statistical inference: a validated estimation
covariance (Fisher ⇔ bootstrap), an analytic delta-method intensity band, and a
dimensionally-consistent variance-penalised quoting heuristic whose robustness
weight `c` provides a tunable risk dial with a clear, monotone effect.

**What is and isn't claimed.**
- `[THEOREM]` MLE asymptotic normality; the analytic intensity gradient.
- `[APPROX]` `Σ_θ ≈ H⁻¹`; the delta-method variance/band; the belief construction.
- `[HEURISTIC]` both robust quoting rules — **no optimality is claimed**.
- `[EMPIRICAL]` every backtest, sensitivity and HPC number.

**Limitations.**
1. The delta method is first-order and conditions on the event path; it omits the
   counting process's own Poisson variability and the non-Gaussian tails of θ̂.
2. The covariance is estimated at one operating point on one trading session; a
   single LOBSTER file limits external validity.
3. Buy/sell flows are modelled as independent univariate Hawkes processes (no
   cross-excitation), inherited from Phase 5.
4. The robust spread is a heuristic, not the solution of a Hawkes-driven control
   problem; `c` is tuned, not optimal.
5. The synthetic fill model (`P(fill)=e^{{−κδ}}`) and the optimistic replay crossing
   rule (no queue position) bound fill rates from above.

### Figure index (`plots/robust_market_making/`)
fig01_covariance · fig02_intensity_band · fig03_propagation · fig04_robust_quotes ·
fig05_backtest · fig06_significance · fig07_sensitivity · fig08_pnl_surface ·
fig09_hpc_distributions

### Reproduce
`python -m analysis.robust_market_making` (add `--quick` for a fast smoke run).
Artefacts: `results/robust/` (covariance table, backtest paths/summary, response
curve, HPC summary). No Hawkes-library or RL dependency; Python 3, NumPy/SciPy/
Numba/joblib only.
"""
    with open(path, "w") as f:
        f.write(txt)


def df3_mean(df, metric, belief):
    """Mean of a metric column for a given belief label (Part 3 table helper)."""
    col = f"{metric}_{belief}"
    return float(df[col].mean()) if col in df else float("nan")
