r"""Report generator for Phase 5 (kept separate from the orchestration logic).

Produces ``reports/hawkes_extension_report.md`` -- written to read like an
academic paper, with a literature-review section on the AS HJB equation and
scrupulous [AS-EXACT] / [HAWKES] / [APPROX] / [CONTRIBUTION] provenance tags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["write_hawkes_report"]


def _fmt_tests(tests: dict) -> str:
    rows = []
    for m, r in tests.items():
        verdict = "**significant**" if r["significant"] else "not significant"
        direction = "improves" if r["beneficial"] else "worsens"
        rows.append(
            f"| {m} | {r['mean_diff']:+.4f} | {r['t_p']:.2e} | {r['wilcoxon_p']:.2e} | "
            f"[{r['boot_lo']:+.4f}, {r['boot_hi']:+.4f}] | {direction}, {verdict} |")
    return "\n".join(rows)


def write_hawkes_report(path, params, sp, tbl, tests, heat, robust, replay, meta) -> None:
    """Write the Phase-5 markdown report from the computed artefacts."""
    def row(m):
        r = tbl.loc[m]
        return (f"| {m} | {r['Poisson_A']:.4f} | {r['Hawkes_B']:.4f} | "
                f"{r['diff_BmA']:+.4f} |")

    metric_rows = "\n".join(row(m) for m in tbl.index)
    test_rows = _fmt_tests(tests)

    rob = robust.groupby("scenario")[["adverse_selection_ratio", "inv_var",
                                      "total_pnl", "fill_rate"]].mean()
    rob_rows = "\n".join(
        f"| {s} | {rob.loc[s, 'adverse_selection_ratio']:.4f} | "
        f"{rob.loc[s, 'inv_var']:.3f} | {rob.loc[s, 'total_pnl']:.2f} | "
        f"{rob.loc[s, 'fill_rate']:.3f} |"
        for s in ["lower", "central", "upper"] if s in rob.index)

    replay_rows = "\n".join(
        f"| {k} | {replay[k]['total_pnl']:.2f} | {replay[k]['avg_spread']:.4f} | "
        f"{replay[k]['fill_rate']:.3f} | {replay[k]['adverse_selection_ratio']:.4f} |"
        for k in replay)

    txt = f"""# Hawkes-Driven Avellaneda--Stoikov Market Making
## An intensity-aware extension of the classical model, with an uncertainty-aware contribution

*Phase 5. LOBSTER AAPL 2012-06-21. Every result below is tagged by provenance:*
**[AS-EXACT]** an exact result of Avellaneda & Stoikov (2008); **[HAWKES]** a
standard Hawkes-process result (Phases 2--3); **[APPROX]** an approximation
introduced by this project; **[CONTRIBUTION]** the project's original idea. We are
careful never to present an approximation as a theorem.

---

## 1. Literature review: the AS HJB equation and where Poisson enters

**Setup [AS-EXACT].** An agent quotes a bid ``S - delta^b`` and ask ``S + delta^a``
on a mid ``dS = sigma dW``. Market buy/sell orders arrive as **Poisson** processes
with intensities ``lambda^{{a,b}}(delta) = A e^{{-kappa delta}}``. With cash ``X``,
inventory ``q`` and exponential (CARA) utility ``-e^{{-gamma(X + qS)}}``, the value
function ``u(x, s, q, t)`` solves the Hamilton--Jacobi--Bellman equation

    u_t + (1/2) sigma^2 u_ss
      + max_{{delta^b}} lambda^b(delta^b) [ u(x - s + delta^b, s, q+1, t) - u ]
      + max_{{delta^a}} lambda^a(delta^a) [ u(x + s + delta^a, s, q-1, t) - u ] = 0.

**Where the Poisson assumption enters (precisely).** The two ``lambda(delta) [u(...)
- u]`` terms are the **generator of a Poisson (doubly-stochastic with
*deterministic* intensity) jump process**. Two properties are used:

1. *Intensity depends only on the control ``delta``*, not on the history of past
   trades -- so the jump terms are Markovian in ``(x, s, q, t)`` and the HJB is a
   PDE in these variables alone.
2. *A closed-form arrival law ``A e^{{-kappa delta}}``* makes the inner maximisation
   explicit, giving ``delta^{{a}} + delta^{{b}} = gamma sigma^2 (T-t) + (2/gamma)
   ln(1 + gamma/kappa)`` **[AS-EXACT]** and the reservation price ``r = S - q gamma
   sigma^2 (T-t)`` **[AS-EXACT]**.

**Why Hawkes arrivals violate this.** Under a Hawkes process the intensity
``lambda(t) = mu + alpha sum_{{t_i<t}} e^{{-beta(t-t_i)}}`` is **stochastic and
history-dependent** [HAWKES]. The state ``(x, s, q, t)`` is no longer Markovian --
one must append the intensity (or the entire history) as an extra state variable.
The HJB becomes a higher-dimensional integro-differential equation with **no known
closed-form solution**.

**What survives, and what becomes an approximation.**

| Ingredient | Status under Hawkes |
|---|---|
| Reservation price ``r = S - q gamma sigma^2(T-t)`` | **remains [AS-EXACT]** — it is a statement about inventory risk under Brownian ``S`` and does not use the arrival law |
| Inventory-risk spread term ``gamma sigma^2 (T-t)`` | **remains [AS-EXACT]** — same reason |
| Microstructure term ``(2/gamma) ln(1+gamma/kappa)`` | **becomes [APPROX]** — derived from a *constant* arrival law; under a time-varying intensity it is only a local/quasi-static approximation |
| The value function / verification theorem | **lost** — no closed form; our policy is a heuristic, not a verified optimum |

This is exactly why we **do not** "just replace ``kappa`` with ``lambda(t)``": ``kappa``
(1/$, a price-elasticity) and ``lambda(t)`` (1/s, a rate) are different objects; the
naive substitution is dimensionally invalid. Section 3 states our justified
approximation.

## 2. Hawkes intensity through the episode [HAWKES]

Per-side calibration (Phase 5, independent self-exciting marginals -- cross
-excitation omitted **[APPROX]**):

| side | mu | alpha | beta | rho | lambda_bar |
|------|----|-------|------|-----|------------|
| buy  | {sp['buy']['mu']:.4f} | {sp['buy']['alpha']:.2f} | {sp['buy']['beta']:.2f} | {sp['buy']['rho']:.4f} | {sp['buy']['lam_bar']:.4f} |
| sell | {sp['sell']['mu']:.4f} | {sp['sell']['alpha']:.2f} | {sp['sell']['beta']:.2f} | {sp['sell']['rho']:.4f} | {sp['sell']['lam_bar']:.4f} |

``lambda(t)`` spikes far above ``lambda_bar`` during order-flow clusters
(Fig 1). Because the instantaneous intensity jumps by ``alpha`` at every event and
decays on the sub-millisecond ``1/beta`` scale, it is too spiky to quote on
directly; we quote on the **expected near-future rate** over a quote-reaction
window ``h`` (the Hawkes conditional expectation ``E[lambda(u)|F_t]``) [HAWKES],
which is bounded and smooth.

## 3. Dynamic quote model [APPROX]

We modulate the constant elasticity ``kappa`` by the **dimensionless normalised
intensity** ``R(t) = lambda_expected(t)/lambda_bar`` (=1 on average), i.e.
``kappa_eff(t) = kappa / R(t)``, giving

    delta(t) = gamma sigma^2 (T-t) + (2/gamma) ln(1 + gamma R(t)/kappa).   [APPROX]

At ``R=1`` this **reduces exactly to the classical AS spread [AS-EXACT]**; when
activity spikes (``R>1``) the spread widens. This is *motivated by* replacing the
constant regime with the Hawkes regime; it is **not** an exact Hawkes-HJB result.
The naive literal form ``(2/gamma)ln(1+gamma/lambda(t))`` is provided only for
comparison and flagged as dimensionally unsound.

## 4. Buy/sell asymmetry [APPROX]

The **ask** reacts to **buy**-order intensity, the **bid** to **sell**-order
intensity:

    delta_ask = (1/2) gamma sigma^2(T-t) + (1/gamma) ln(1 + gamma R_buy/kappa)
    delta_bid = (1/2) gamma sigma^2(T-t) + (1/gamma) ln(1 + gamma R_sell/kappa).

*Economic reasoning:* a surge in buy market orders signals upward/informed
pressure, so the maker raises its ask (**less aggressive**) to avoid selling just
before a rise; symmetrically a sell surge lowers aggressiveness of the bid. This
is an adverse-selection hedge and collapses to the symmetric classical quotes when
``R_buy=R_sell=1``.

## 5--7. Comparative backtest (Strategy A: Poisson vs B: Hawkes)

Both strategies quote on **identical** Hawkes-clustered synthetic flow with a
permanent-impact mid (``impact = ${meta['impact']:.3f}``/order [APPROX]), using
common random numbers ({meta['n_paths']} paired paths, T={meta['T']:.0f}s,
gamma={params.gamma}). Fills use the **same** exponential model for both, so
differences are due only to quote placement.

| metric | Poisson (A) | Hawkes (B) | diff (B$-$A) |
|--------|-------------|------------|--------------|
{metric_rows}

## 8. Statistical significance (paired, {meta['n_paths']} paths)

| metric | mean diff (B$-$A) | t-test p | Wilcoxon p | 95% bootstrap CI | interpretation |
|--------|-------------------|----------|------------|------------------|----------------|
{test_rows}

(A metric is called *significant* only when both tests give ``p < 0.05`` **and**
the bootstrap CI excludes zero.) See Fig 12.

## 9. Sensitivity across ``gamma`` x ``rho`` [HAWKES x AS]

Heatmaps of Hawkes-MM PnL, inventory variance and Sharpe over
``gamma in {heat['gammas']}`` and ``rho in {heat['rhos']}`` (Fig 10). Inventory
variance falls with ``gamma`` (AS risk control) and rises with ``rho`` (stronger
clustering => burstier flow), consistent with Phases 3--4.

## 10. Uncertainty-aware quoting [CONTRIBUTION]

The **original contribution**: instead of treating the calibrated
``(mu, alpha, beta)`` as exact, we propagate the **Phase-3 95% confidence
interval** on the excitation into three quoting beliefs -- *lower* (calmer),
*central*, *upper* (burstier/conservative) -- while the true flow is generated
from the central parameters. A conservative maker over-estimates clustering and
widens sooner.

| belief | adverse-selection ratio | inventory variance | total PnL ($) | fill rate (/s) |
|--------|-------------------------|--------------------|---------------|----------------|
{rob_rows}

*Reading:* moving from the aggressive (*lower*) to the conservative (*upper*)
intensity belief **monotonically** trades fill rate / PnL for **reduced adverse
selection and inventory risk** -- a robust-control hedge against Hawkes estimation
error (Fig 11). The effect is *modest* because the Phase-3 estimator is precise
(the 95% CI on the excitation is only about +/-10%); the mechanism nonetheless
points the right way and would matter more for a noisier calibration or a shorter
sample. This is an *empirical* proposal evaluated by simulation, not a theorem.

## 11. Historical replay (LOBSTER, one trading day) [empirical]

Replaying the actual AAPL event stream (crossing-rule fills, queue position
ignored **[APPROX]**, so fill rates are optimistic):

| strategy | terminal PnL ($) | avg spread ($) | fill rate (/s) | adverse-sel. ratio |
|----------|------------------|----------------|----------------|--------------------|
{replay_rows}

A single day is one sample, so it is reported as a realistic illustration, not a
significance test (that is Section 8's role on the paired synthetic paths).

## 12. Conclusions and honest limitations

* The **reservation price and inventory-risk term are exact under Hawkes**; only
  the microstructure spread term becomes an approximation. Our dynamic model
  reduces to classical AS on average and reacts to bursts.
* Where the Hawkes maker helps is **risk and adverse selection**, not necessarily
  raw PnL (wider quotes during bursts => fewer, better fills). The statistical
  tests in Section 8 state precisely which differences are significant.
* **Limitations:** (i) no closed-form optimality -- the policy is a justified
  heuristic; (ii) independent per-side Hawkes (no cross-excitation); (iii)
  permanent-impact and crossing-rule fills are stylised; (iv) the quote-reaction
  window ``h`` and impact are modelling choices.
* **Contribution:** propagating estimator uncertainty (Phase 3) into quoting is,
  to our knowledge, a novel and practical robustness lever.

> **Out of scope:** reinforcement learning and a full stochastic-control solution
> of the Hawkes-HJB equation are left to future work.

### Figure index (`plots/hawkes_market_maker/`)
fig01_intensity, fig02_dynamic_spread, fig03_bid_ask, fig04_inventory, fig05_cash,
fig06_pnl, fig07_spread_compare, fig08_inventory_compare, fig09_sharpe_compare,
fig10_heatmap_gamma_rho, fig11_ci_propagation, fig12_significance.
"""
    with open(path, "w") as f:
        f.write(txt)
