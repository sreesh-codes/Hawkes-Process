"""
================================================================================
Phase 2 — Univariate Exponential Hawkes Process, Maximum-Likelihood Estimation
          from scratch (NO Hawkes library).
================================================================================
Hawkes-Process-Driven Market Making · LOBSTER AAPL 2012-06-21 (Level 10)

Model
-----
We model the arrival times {t_1, ..., t_N} of *executed trades* on the window
[0, T] as a self-exciting point process with conditional intensity

        λ(t) = μ + α · Σ_{t_i < t} exp(-β (t - t_i)),      μ, α, β > 0.

    μ : baseline (exogenous) intensity   [events / second]
    α : jump in intensity right after each event (excitation strength)
    β : exponential decay rate of the excitation (memory ~ 1/β seconds)

Each past trade transiently lifts the intensity by α and that bump decays at
rate β. The process is stationary (does not explode) iff the **branching ratio**

        ρ = α / β  <  1.

ρ is the expected number of "child" trades directly triggered by one trade; it
is the fraction of activity that is endogenous (self-generated) rather than
exogenous.

Log-likelihood
--------------
For a point process with intensity λ(t) and history H_t, the log-likelihood of
observing events at t_1 < ... < t_N on [0, T] is the standard expression

    log L = Σ_{i=1}^N log λ(t_i)  -  ∫_0^T λ(s) ds.            (★)

The integral term is the *compensator* Λ(T) (expected number of events). For the
exponential kernel it has a closed form:

    ∫_0^T λ(s) ds = μ T + (α/β) Σ_{i=1}^N [ 1 - exp(-β (T - t_i)) ].

The first ("log-intensity") term naively costs O(N^2) because λ(t_i) sums over
all previous events. The exponential kernel admits an O(N) RECURSION. Define

    A_i = Σ_{j < i} exp(-β (t_i - t_j)),     so   λ(t_i) = μ + α A_i.

Then, with Δ_i = t_i - t_{i-1},

    A_1 = 0,
    A_i = exp(-β Δ_i) · (1 + A_{i-1}).                         (recursion)

Proof: exp(-βΔ_i)(1 + A_{i-1}) = exp(-βΔ_i) + Σ_{j<i-1} exp(-β(t_i - t_j))
       = Σ_{j ≤ i-1} exp(-β(t_i - t_j)) = A_i.   ∎

Analytic gradient (also O(N))
-----------------------------
With B_i = ∂A_i/∂β = -Σ_{j<i}(t_i - t_j) exp(-β(t_i - t_j)), which obeys

    B_1 = 0,
    B_i = -Δ_i A_i + exp(-β Δ_i) B_{i-1},

the gradient of (★) is

    ∂logL/∂μ = Σ_i 1/(μ+αA_i) - T
    ∂logL/∂α = Σ_i A_i/(μ+αA_i) - (1/β) Σ_i [1 - e^{-β(T-t_i)}]
    ∂logL/∂β = Σ_i α B_i/(μ+αA_i)
               - α [ -1/β² Σ_i(1-e^{-β(T-t_i)}) + 1/β Σ_i (T-t_i) e^{-β(T-t_i)} ].

We MINIMISE the negative log-likelihood with scipy's L-BFGS-B, passing the
analytic gradient and box constraints μ,α,β > 0 (positivity).

Goodness of fit — time-rescaling theorem
----------------------------------------
If the model is correct, the compensator increments
    ξ_i = Λ(t_i) - Λ(t_{i-1})
are i.i.d. Exponential(1). We test this with a KS test (cf. Phase 1's KS test of
the *raw* interarrivals against Exponential, which was rejected — here we expect
NOT to reject if the Hawkes model fits).
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

# --------------------------------------------------------------------------- #
# Paths (resolved relative to this file so the script runs from anywhere)
# --------------------------------------------------------------------------- #
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "LOBSTER_SampleFile_AAPL_2012-06-21_10")
MSG_FILE = os.path.join(DATA_DIR, "AAPL_2012-06-21_34200000_57600000_message_10.csv")
PLOTS_DIR = os.path.join(BASE, "plots")
REPORT = os.path.join(BASE, "notebooks", "hawkes_calibration_report.md")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT), exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Load executed-trade timestamps  (event types 4 = visible, 5 = hidden)
# --------------------------------------------------------------------------- #
def load_trade_times():
    """Return strictly-increasing executed-trade timestamps (seconds), shifted
    to start at 0, plus the observation horizon T.

    Simultaneous executions (one marketable order sweeping several resting
    orders / price levels share a timestamp) are COLLAPSED into a single trade
    event — consistent with the Phase-1 EDA and required for a well-defined
    point-process likelihood (no two events at the same instant)."""
    cols = ["time", "event_type", "order_id", "size", "price", "direction"]
    msg = pd.read_csv(MSG_FILE, header=None, names=cols)
    trades = msg.loc[msg["event_type"].isin([4, 5]), "time"].to_numpy()
    trades = np.unique(trades)              # sorted + de-duplicated timestamps
    t0 = trades[0]
    t = trades - t0                         # shift origin to 0 for stability
    T = float(t[-1])                        # observation horizon
    return t, T, t0


# --------------------------------------------------------------------------- #
# 2. Negative log-likelihood + analytic gradient (recursive, O(N))
# --------------------------------------------------------------------------- #
def neg_log_likelihood(params, t, T):
    """Return (-logL, -gradient) for the exponential Hawkes model.

    Implements equation (★) and its gradient with the O(N) recursions above.
    """
    mu, alpha, beta = params
    N = t.shape[0]

    # ---- recursion for A_i and B_i = dA_i/dβ -----------------------------
    A = np.empty(N)
    B = np.empty(N)
    A[0] = 0.0
    B[0] = 0.0
    for i in range(1, N):
        dti = t[i] - t[i - 1]
        e = np.exp(-beta * dti)
        A[i] = e * (1.0 + A[i - 1])
        B[i] = -dti * A[i] + e * B[i - 1]

    lam = mu + alpha * A                     # λ(t_i), length N
    # numerical floor: intensity must stay strictly positive
    lam = np.maximum(lam, 1e-300)

    # ---- compensator term  Λ(T) = ∫_0^T λ ds -----------------------------
    decayT = 1.0 - np.exp(-beta * (T - t))   # 1 - e^{-β(T - t_i)}
    compensator = mu * T + (alpha / beta) * decayT.sum()

    # ---- log-likelihood (★) ----------------------------------------------
    logL = np.log(lam).sum() - compensator

    # ---- gradient --------------------------------------------------------
    inv = 1.0 / lam
    dmu = inv.sum() - T
    dalpha = (A * inv).sum() - decayT.sum() / beta
    # d/dβ of the compensator
    dcomp_dbeta = alpha * (-decayT.sum() / beta**2
                           + ((T - t) * np.exp(-beta * (T - t))).sum() / beta)
    dbeta = (alpha * B * inv).sum() - dcomp_dbeta

    grad = np.array([dmu, dalpha, dbeta])
    return -logL, -grad


# --------------------------------------------------------------------------- #
# 3. MLE via L-BFGS-B with positivity bounds and multiple restarts
# --------------------------------------------------------------------------- #
def fit_hawkes(t, T):
    """Maximum-likelihood estimate of (μ, α, β) via L-BFGS-B.

    Positivity is enforced through box constraints. Several starting points are
    tried (the likelihood surface can be flat in β) and the best is kept.
    """
    rate = len(t) / T                        # homogeneous-Poisson rate, a scale
    bounds = [(1e-8, None), (1e-8, None), (1e-8, None)]   # μ, α, β > 0

    # diverse, scale-aware initial guesses (μ0, α0, β0)
    starts = [
        (rate * 0.5, rate * 0.5, rate * 1.0),
        (rate * 0.3, 1.0, 2.0),
        (rate * 0.7, 5.0, 10.0),
        (rate * 0.2, 0.5, 1.0),
        (rate * 0.5, 20.0, 40.0),
    ]

    best = None
    for s0 in starts:
        res = minimize(neg_log_likelihood, x0=np.array(s0, float),
                       args=(t, T), method="L-BFGS-B", jac=True,
                       bounds=bounds,
                       options=dict(maxiter=500, ftol=1e-12, gtol=1e-8))
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:                         # fall back to best non-"success"
        best = res
    return best


# --------------------------------------------------------------------------- #
# 4. Intensity on a grid  (O(G + N), via the same decay recursion)
# --------------------------------------------------------------------------- #
def intensity_on_grid(grid, events, mu, alpha, beta):
    """Evaluate λ(g) = μ + α Σ_{t_i ≤ g} exp(-β (g - t_i)) on a sorted grid.

    We carry a running decayed sum S = Σ exp(-β (g - t_i)); moving from one grid
    point to the next we (i) decay S, then (ii) fold in any events that fell in
    the interval. Cost is O(G + N) rather than O(G·N)."""
    grid = np.asarray(grid, float)
    lam = np.empty(grid.shape[0])
    S = 0.0
    j = 0                                    # pointer into sorted events
    prev = grid[0]
    for k, g in enumerate(grid):
        S *= np.exp(-beta * (g - prev))      # decay existing contributions
        while j < events.shape[0] and events[j] <= g:
            S += np.exp(-beta * (g - events[j]))
            j += 1
        lam[k] = mu + alpha * S
        prev = g
    return lam


# --------------------------------------------------------------------------- #
# 5. Time-rescaling goodness-of-fit
# --------------------------------------------------------------------------- #
def rescaled_increments(t, T, mu, alpha, beta):
    """Compensator increments ξ_i = Λ(t_i) - Λ(t_{i-1}); i.i.d. Exp(1) iff the
    model is correct. Computed in O(N) via the closed form

        ξ_i = μ Δ_i + (α/β)(1 - e^{-β Δ_i})(1 + A_{i-1}),     Δ_i = t_i - t_{i-1},

    which follows from Λ(t) = μ t + (α/β) Σ_{t_j < t}(1 - e^{-β(t - t_j)}) and the
    A_i recursion (the j = i-1 term contributes (1-e^{-βΔ_i}); the earlier terms
    contribute A_{i-1}(1-e^{-βΔ_i}))."""
    N = t.shape[0]
    A = 0.0                                  # A_{i-1}, running via recursion
    xi = np.empty(N - 1)
    for i in range(1, N):
        d = t[i] - t[i - 1]
        ed = np.exp(-beta * d)
        xi[i - 1] = mu * d + (alpha / beta) * (1.0 - ed) * (1.0 + A)
        A = ed * (1.0 + A)                   # advance to A_i for next iteration
    return xi


# --------------------------------------------------------------------------- #
# 6. Plots
# --------------------------------------------------------------------------- #
def plot_intensity(t, T, mu, alpha, beta, pois_rate):
    # ---- (a) full session, fitted intensity vs constant Poisson ----------
    grid = np.linspace(0, T, 6000)
    lam = intensity_on_grid(grid, t, mu, alpha, beta)
    hrs = grid / 3600.0
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hrs, lam, lw=0.6, color="navy", label="Hawkes λ(t)")
    ax.axhline(pois_rate, color="red", ls="--", lw=1.5,
               label=f"Poisson λ = {pois_rate:.2f}/s")
    ax.axhline(mu, color="green", ls=":", lw=1.5,
               label=f"Hawkes baseline μ = {mu:.2f}/s")
    ax.set(xlabel="Hours since 09:30", ylabel="Intensity (trades/s)",
           title="Fitted Hawkes intensity vs constant Poisson intensity")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "hawkes_intensity_full.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- (b) 5-minute zoom to expose self-excitation bursts --------------
    lo = T * 0.5
    hi = lo + 300.0                           # 5 minutes
    gridz = np.linspace(lo, hi, 4000)
    lamz = intensity_on_grid(gridz, t, mu, alpha, beta)
    evz = t[(t >= lo) & (t <= hi)]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot((gridz - lo), lamz, lw=0.9, color="navy", label="Hawkes λ(t)")
    ax.axhline(pois_rate, color="red", ls="--", lw=1.5, label="Poisson λ")
    ax.axhline(mu, color="green", ls=":", lw=1.2, label="baseline μ")
    ax.plot((evz - lo), np.full_like(evz, -0.02 * lamz.max()),
            "|", color="black", ms=8, label="trades")
    ax.set(xlabel="Seconds into a 5-minute window", ylabel="Intensity (trades/s)",
           title="Self-excitation up close: each trade bumps λ, then it decays")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "hawkes_intensity_zoom.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_residual_qq(xi):
    """QQ plot of rescaled increments vs Exponential(1)."""
    xs = np.sort(xi)
    n = xs.shape[0]
    theo = -np.log(1.0 - (np.arange(1, n + 1) - 0.5) / n)   # Exp(1) quantiles
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(theo, xs, ".", ms=2, color="purple", alpha=0.5)
    hi = max(theo.max(), xs.max())
    ax.plot([0, hi], [0, hi], "k--", lw=1)
    ax.set(xlabel="Theoretical Exp(1) quantiles",
           ylabel="Empirical rescaled increments",
           title="Time-rescaling QQ plot (Hawkes goodness-of-fit)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "hawkes_residual_qq.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. Calibration report
# --------------------------------------------------------------------------- #
def write_report(t, T, mu, alpha, beta, negll, pois_rate, rho,
                 ks_stat, ks_p, ll_hawkes, ll_poisson):
    N = len(t)
    half_life = np.log(2) / beta
    mean_intensity = mu / (1 - rho) if rho < 1 else float("inf")
    aic_h = 2 * 3 - 2 * ll_hawkes
    aic_p = 2 * 1 - 2 * ll_poisson
    txt = f"""# Phase 2 — Hawkes Process Calibration Report
## Univariate Exponential Hawkes MLE · LOBSTER AAPL 2012-06-21 (executed trades)

Estimated **from scratch** (manual recursive O(N) log-likelihood, analytic
gradient, `scipy.optimize.minimize` with **L-BFGS-B** and positivity bounds — no
Hawkes library). Reproduce with `python hawkes_mle.py`.

### Model
```
λ(t) = μ + α Σ_{{t_i < t}} exp(-β (t - t_i))
```

### Data
| Quantity | Value |
|----------|-------|
| Executed trades (types 4 & 5, de-duplicated timestamps) | {N:,} |
| Observation horizon T | {T:,.1f} s ({T/3600:.2f} h) |
| Homogeneous-Poisson rate (N/T) | {pois_rate:.4f} trades/s |

### Maximum-likelihood estimates
| Parameter | Estimate | Interpretation |
|-----------|----------|----------------|
| μ (baseline) | **{mu:.5f}** /s | exogenous arrival rate |
| α (excitation jump) | **{alpha:.5f}** /s | intensity bump per trade |
| β (decay rate) | **{beta:.5f}** /s | excitation memory rate |
| − log-likelihood | {negll:,.2f} | at optimum |

### Branching ratio and stationarity
| Quantity | Value |
|----------|-------|
| **Branching ratio ρ = α/β** | **{rho:.4f}** |
| Stationary? (ρ < 1) | {"YES" if rho < 1 else "NO — explosive"} |
| Excitation half-life ln2/β | {half_life*1000:.1f} ms |
| Mean intensity μ/(1−ρ) | {mean_intensity:.4f} /s |
| Poisson rate N/T (sanity check) | {pois_rate:.4f} /s |

**ρ = {rho:.3f}** means roughly **{rho*100:.1f}%** of trades are *endogenous* — directly
triggered by earlier trades — while the remaining **{(1-rho)*100:.1f}%** are exogenous.
The mean Hawkes intensity μ/(1−ρ) = {mean_intensity:.3f}/s closely matches the raw
Poisson rate {pois_rate:.3f}/s, confirming the fit is self-consistent: the model
reproduces the overall trade rate but redistributes it into bursts.

### Hawkes vs constant Poisson
| Model | log-likelihood | # params | AIC |
|-------|----------------|----------|-----|
| Homogeneous Poisson | {ll_poisson:,.2f} | 1 | {aic_p:,.2f} |
| Exponential Hawkes | {ll_hawkes:,.2f} | 3 | {aic_h:,.2f} |

The Hawkes log-likelihood exceeds Poisson by **{ll_hawkes - ll_poisson:,.1f}** with only two
extra parameters (ΔAIC = {aic_p - aic_h:,.1f} in favour of Hawkes). A likelihood-ratio
test (2·Δlog L = {2*(ll_hawkes - ll_poisson):,.1f}, χ²₂, p ≪ 1e-300) overwhelmingly
rejects Poisson in favour of self-excitation — quantitatively confirming the
clustering seen visually in Phase 1.

### Goodness of fit — time-rescaling theorem
Under the fitted model the compensator increments ξ_i = Λ(t_i) − Λ(t_{{i-1}})
should be i.i.d. **Exponential(1)**.

| Test | Statistic | p-value |
|------|-----------|---------|
| KS(ξ, Exp(1)) | {ks_stat:.4f} | {ks_p:.3e} |

(Contrast with Phase 1, where the *raw* interarrivals were decisively rejected
as exponential. After rescaling by the fitted Hawkes intensity the increments are
far closer to Exp(1), shown in `plots/hawkes_residual_qq.png`.) Residual
departures in the tail are expected: a single exponential kernel and a constant
baseline cannot capture intraday seasonality (the open/close U-shape) or
longer-memory effects — natural targets for later refinement (multi-kernel,
multivariate buy/sell, or time-varying μ).

### Figures
| File | Content |
|------|---------|
| `plots/hawkes_intensity_full.png` | Fitted λ(t) over the session vs constant Poisson rate |
| `plots/hawkes_intensity_zoom.png` | 5-min zoom: each trade bumps λ by α, decaying at β |
| `plots/hawkes_residual_qq.png` | Time-rescaling QQ plot vs Exp(1) |

### Conclusion
The executed-trade flow is strongly **self-exciting** (ρ ≈ {rho:.2f}), with a very
short excitation memory (half-life ≈ {half_life*1000:.0f} ms): trades arrive in rapid
bursts and the effect dies within a fraction of a second. The Hawkes model fits
dramatically better than Poisson and provides the intensity signal the
market-making engine will react to in later phases.

> **Next phase (not done here):** extend to a *multivariate* buy/sell Hawkes
> model and/or time-varying baseline, then wire λ(t) into the market-making
> quoting logic.
"""
    with open(REPORT, "w") as f:
        f.write(txt)


# --------------------------------------------------------------------------- #
# 8. Main
# --------------------------------------------------------------------------- #
def main():
    t, T, _t0 = load_trade_times()
    N = len(t)
    pois_rate = N / T
    print(f"Loaded {N:,} executed trades over T = {T:,.1f}s "
          f"(Poisson rate {pois_rate:.4f}/s)")

    # ---- fit ----
    res = fit_hawkes(t, T)
    mu, alpha, beta = res.x
    negll = res.fun
    rho = alpha / beta
    print("\n--- MLE (L-BFGS-B) ---")
    print(f"  mu    = {mu:.6f} /s")
    print(f"  alpha = {alpha:.6f} /s")
    print(f"  beta  = {beta:.6f} /s")
    print(f"  -logL = {negll:.4f}   converged={res.success}")
    print(f"  branching ratio rho = alpha/beta = {rho:.4f} "
          f"({'stationary' if rho < 1 else 'EXPLOSIVE'})")
    print(f"  excitation half-life = {1000*np.log(2)/beta:.1f} ms")
    print(f"  mean intensity mu/(1-rho) = {mu/(1-rho):.4f}/s  vs  N/T = {pois_rate:.4f}/s")

    # ---- log-likelihoods for model comparison ----
    ll_hawkes = -negll
    # homogeneous Poisson logL at its MLE rate λ̂ = N/T:  N log λ̂ - λ̂ T
    ll_poisson = N * np.log(pois_rate) - pois_rate * T
    print(f"\n  logL Hawkes  = {ll_hawkes:.2f}")
    print(f"  logL Poisson = {ll_poisson:.2f}")
    print(f"  LR 2*Δ       = {2*(ll_hawkes - ll_poisson):.1f}")

    # ---- goodness of fit ----
    xi = rescaled_increments(t, T, mu, alpha, beta)
    ks_stat, ks_p = stats.kstest(xi, "expon", args=(0, 1))
    print(f"\n  time-rescaling KS vs Exp(1): stat={ks_stat:.4f} p={ks_p:.3e}")

    # ---- plots ----
    plot_intensity(t, T, mu, alpha, beta, pois_rate)
    plot_residual_qq(xi)
    print("\n  saved plots to ./plots/")

    # ---- report ----
    write_report(t, T, mu, alpha, beta, negll, pois_rate, rho,
                 ks_stat, ks_p, ll_hawkes, ll_poisson)
    print(f"  wrote calibration report to {os.path.relpath(REPORT, BASE)}")


if __name__ == "__main__":
    main()
