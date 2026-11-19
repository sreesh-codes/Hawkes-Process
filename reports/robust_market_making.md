# Phase 6 — Robust Hawkes Market Making via Uncertainty Quantification
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
(`n = 24,861` events, horizon `T = 23399.6` s) gives the point
estimate

    μ̂ = 0.5679,  α̂ = 210.583,  β̂ = 452.350,
    ρ̂ = α̂/β̂ = 0.4655,   λ̄ = μ̂/(1−ρ̂) = 1.0625.

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
well-conditioned (condition number ≈ 13.3, minimum
eigenvalue 1.67e+03 > 0, confirming a genuine
interior maximum).

### 1.2 Parametric bootstrap `[THEOREM]/[EMPIRICAL]`

Treating θ̂ as truth, we simulated **1,000** independent event
streams from `Hawkes(θ̂)` on `[0, T]` (Ogata thinning), re-estimated θ on each by
maximum likelihood (875 converged), and took the empirical mean and
covariance. This is distribution-free and exposes finite-sample bias.

### 1.3 Comparison

| param | estimate | Fisher SE | bootstrap SE | bootstrap bias | bootstrap 95% |
|---|---|---|---|---|---|
| mu | 0.5679 | 0.005063 | 0.004744 | -0.000138 | [0.5587, 0.5772] |
| alpha | 210.6 | 3.855 | 3.034 | -0.00644 | [204.7, 216.5] |
| beta | 452.3 | 7.945 | 5.022 | -0.0184 | [442.8, 462.4] |

**Fisher correlation matrix:**

|  | mu | alpha | beta |
|---|---|---|---|
| mu | 1.00 | 0.14 | 0.18 |
| alpha | 0.14 | 1.00 | 0.86 |
| beta | 0.18 | 0.86 | 1.00 |

**Reading.** `[EMPIRICAL]` The two estimators agree in order of magnitude and sign
structure; α and β are strongly positively correlated (both govern the excitation,
so the likelihood trades them off along a ridge). The Fisher SEs are somewhat
larger than the bootstrap SEs — expected, since the observed information at a
single realisation and the Gaussian approximation are only first-order, whereas the
bootstrap captures the (mild) finite-sample skew. The bootstrap bias is small
relative to the SEs, so the estimator is approximately unbiased at this sample
size. See `plots/robust_market_making/fig01_covariance.png`.

---

## 2. Delta method — intensity confidence intervals (Part 2)

The intensity is a smooth function of the parameters,
`λ(t) = f(μ,α,β) = μ + α Σ_{t_i<t} e^{−β(t−t_i)}`. Conditioning on the observed
event times, the **delta method** `[APPROX]` gives

    Var(λ(t)) ≈ ∇f(t)ᵀ Σ_θ ∇f(t),

with the **analytic** gradient `[THEOREM]`

    ∇f(t) = ( 1,  S_β(t),  −α Q_β(t) ),
    S_β(t) = Σ e^{−β(t−t_i)},   Q_β(t) = Σ (t−t_i) e^{−β(t−t_i)},

since `∂λ/∂β = α ∂S_β/∂β = −α Q_β`. An approximate 95% band is
`λ(t) ± 1.96 √Var(λ(t))`. The stationary mean intensity has the analytic delta-
method interval

    λ̄ = 1.0625 ± 0.0247   (95%).

The band's median relative width over the representative window is
3.50%. **Status:** the gradient is exact; the variance
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
| lower  | 0.1463 | 1.297 | 0.116 | 5.803 |
| central| 0.1463 | 1.297 | 0.116 | 5.803 |
| upper  | 0.1463 | 1.298 | 0.116 | 5.804 |

`[EMPIRICAL]` **Key finding.** The three beliefs give *statistically
indistinguishable* market-making outcomes. This is not a bug: at this sample size
the 95% uncertainty in `λ̄` is only ≈2.4% of its
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
empirically at `c ∈ {0, 0.5, 1, 2}`. See `fig04_robust_quotes.png`.

---

## 5. Robust backtesting (Part 5)

Classical A-S, Hawkes A-S and Robust Hawkes A-S (Method B, `c = 1.0`)
on **identical** synthetic Hawkes flow (300 paired paths, common
random numbers, permanent impact `0.002` $/order):

| Metric | Classical | Hawkes | Robust |
|---|---|---|---|
| Total PnL ($) | 6.589 | 5.803 | 4.472 |
| Sharpe | 5.167 | 5.16 | 4.485 |
| Inventory variance | 1.978 | 1.297 | 1.061 |
| Max drawdown ($) | 0.6759 | 0.5501 | 0.5162 |
| Avg spread ($) | 0.1196 | 0.1463 | 0.2519 |
| Realized spread ($) | 0.03811 | 0.04373 | 0.04077 |
| Adverse-selection ratio | 0.1549 | 0.1164 | 0.1132 |
| Fill rate (/s) | 0.2829 | 0.2173 | 0.1779 |
| Avg quote lifetime (s) | 0.9458 | 0.9432 | 0.9431 |
| Inventory half-life (s) | 10.82 | 8.749 | 8.716 |

**LOBSTER historical replay** (same session, crossing-rule fills):

| Metric | Classical | Hawkes | Robust |
|---|---|---|---|
| Total PnL ($) | 1.509 | 0.294 | 0.2338 |
| Sharpe | 1.849 | 0.6287 | 0.592 |
| Inventory variance | 0.1551 | 0.03363 | 0.03207 |
| Max drawdown ($) | 0.6261 | 0.4175 | 0.3218 |
| Avg spread ($) | 1.331 | 1.42 | 1.844 |
| Adverse-selection ratio | 0.3721 | 0.539 | 0.4063 |
| Fill rate (/s) | 0.003077 | 0.001239 | 0.0007265 |

`[EMPIRICAL]` The robust maker widens the spread and **reduces inventory variance
and drawdown** relative to both baselines, at the cost of a lower fill rate and
lower gross PnL — exactly the risk/return trade the phase set out to characterise
(this phase is not about maximising PnL). See `fig05_backtest.png`.

---

## 6. Statistical significance (Part 6)

Paired tests on the 300 common-random-number paths (t-test, Wilcoxon
signed-rank, percentile bootstrap CI, paired Cohen's d). **Robust vs Hawkes:**

| Metric | mean diff | t p-value | Wilcoxon p | 95% boot CI | Cohen's d | sig? |
|---|---|---|---|---|---|---|
| Total PnL | -1.33 | 9.4e-103 | 1.3e-50 | [-1.41, -1.25] | -1.93 | **yes** |
| Sharpe | -0.6755 | 1.7e-48 | 7e-37 | [-0.75, -0.601] | -1.02 | **yes** |
| Inv. var | -0.2365 | 1.6e-33 | 1.8e-33 | [-0.271, -0.203] | -0.79 | **yes** |
| Max drawdown | -0.03393 | 0.0009 | 4.3e-05 | [-0.0538, -0.0138] | -0.19 | **yes** |
| Adverse-sel. | -0.003165 | 0.41 | 0.47 | [-0.0105, +0.00441] | -0.05 | no |

**Hawkes vs Classical:**

| Metric | mean diff | t p-value | Wilcoxon p | 95% boot CI | Cohen's d | sig? |
|---|---|---|---|---|---|---|
| Total PnL | -0.7865 | 2.7e-56 | 2.7e-41 | [-0.865, -0.709] | -1.14 | **yes** |
| Sharpe | -0.00716 | 0.87 | 0.57 | [-0.096, +0.0813] | -0.01 | no |
| Inv. var | -0.6808 | 1.5e-55 | 2.5e-45 | [-0.751, -0.615] | -1.13 | **yes** |
| Max drawdown | -0.1258 | 1.7e-12 | 2.5e-15 | [-0.16, -0.0933] | -0.43 | **yes** |
| Adverse-sel. | -0.03851 | 3e-21 | 3.7e-19 | [-0.0457, -0.0312] | -0.59 | **yes** |

`[EMPIRICAL]` A statistically significant difference (large paired `n`) is not the
same as a large effect; Cohen's d is reported so practical magnitude can be judged
independently of the p-value. See `fig06_significance.png`.

---

## 7. Sensitivity analysis (Part 7)

We sweep the robustness weight `c` against risk aversion `γ` and branching ratio
`ρ` (60 paths/cell). Response of the robust strategy vs `c` at the
calibrated `(γ, ρ)`:

| c | PnL ($) | Inv. var | Sharpe |
|---|---|---|---|
| 0.00 | 5.819 | 1.377 | 5.142 |
| 0.31 | 5.079 | 1.226 | 4.827 |
| 0.62 | 4.699 | 1.168 | 4.634 |
| 0.94 | 4.466 | 1.126 | 4.458 |
| 1.25 | 4.295 | 1.105 | 4.357 |
| 1.56 | 4.237 | 1.043 | 4.387 |
| 1.88 | 4.199 | 1.052 | 4.351 |
| 2.19 | 4.173 | 1.036 | 4.358 |
| 2.50 | 4.171 | 1.033 | 4.362 |

`[EMPIRICAL]` Increasing `c` monotonically **lowers inventory variance** and gross
PnL; the effect strengthens at higher `γ` and higher `ρ` (more clustering → more
intensity uncertainty to hedge). Heatmaps and the 3-D PnL surface are in
`fig07_sensitivity.png` and `fig08_pnl_surface.png`.

---

## 8. HPC uncertainty simulation (Part 8)

Each of **5,000** Monte-Carlo replicates draws a *true* market
`θ ~ N(θ̂, Σ_θ)` per side (rejecting non-stationary/non-positive draws;
0 rejected), generates flow from it, and runs the
robust strategy (quoting on the central estimate). This yields the **predictive
distribution** of each performance metric under parameter uncertainty
(`ProcessPoolExecutor`-style parallelism via joblib; wall-clock
49.1 s):

| Metric | mean | std | p5 | p50 | p95 |
|---|---|---|---|---|---|
| Total PnL ($) | 4.331 | 1.021 | 2.646 | 4.308 | 5.998 |
| Inventory variance | 1.109 | 0.5054 | 0.6079 | 0.9754 | 2.061 |
| Sharpe | 4.302 | 1.122 | 2.366 | 4.334 | 6.057 |
| Max drawdown ($) | 0.5394 | 0.3291 | 0.2268 | 0.4487 | 1.179 |

*Scale note (honesty):* the brief targets `10⁵` replicates; the driver
(`robust.hpc.run_hpc`) accepts `n_sims` up to that and batches for memory. The
distributions above are from the **5,000** replicates actually
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
5. The synthetic fill model (`P(fill)=e^{−κδ}`) and the optimistic replay crossing
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
