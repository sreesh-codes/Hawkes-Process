# Phase 4 - Classical Avellaneda--Stoikov Market-Making Baseline
## LOBSTER AAPL 2012-06-21 | faithful reproduction of Avellaneda & Stoikov (2008)

*Implemented from scratch (`market_maker/`): mid-price dynamics, reservation
price, optimal spread, exponential Poisson execution, inventory/cash accounting
and an event-driven simulator. No Hawkes process and no RL appear in this phase.*

---

## 1. Model assumptions

1. **Mid-price** follows a driftless arithmetic Brownian motion, `dS = sigma dW`
   (eq. 1). No directional view; constant volatility. `sigma` is calibrated from
   data.
2. **Market-order arrivals** are **Poisson**, with intensity decaying
   exponentially in the quote's distance from the mid,
   `lambda(delta) = A exp(-kappa delta)` (eq. 2).
3. **Utility** is exponential (CARA), `U(x) = -exp(-gamma x)`; the agent maximises
   expected utility of terminal wealth over `[0, T]`.
4. **Unit order size**, a hard inventory cap `q_max`, and **terminal liquidation**
   of residual inventory at `T`.

**Calibrated inputs (from LOBSTER AAPL):** `sigma = 0.03477` $/sqrt(s),
`kappa = 23.96` 1/$, base intensity `A = 1.445` /s. Horizon
`T = 600` s, `q_max = 50`.

## 2. Derivation of the reservation price

With CARA utility the value function factorises as
`v = -exp(-gamma x) exp(-gamma theta(s,q,t))`, and the indifference price for one
unit of inventory gives the **reservation price**

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t).            (eq. 3)

A positive inventory (`q>0`) shifts the whole quote ladder **down** (the agent
prices risk more conservatively and is keener to sell); the shift is linear in
`q` and scales with risk aversion `gamma`, variance `sigma^2`, and the remaining
horizon `(T-t)`. As `t -> T` the adjustment vanishes -- there is no time left to
carry risk.

## 3. Derivation of the optimal spread

Maximising expected utility over the half-spreads, with the exponential arrival
law (eq. 2), yields the optimal **total** spread

    delta*(t) = gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/kappa).   (eq. 4)

* **Inventory-risk term** `gamma sigma^2 (T-t)`: widens with risk aversion,
  volatility and the time left -- the compensation demanded for holding risk.
* **Microstructure term** `(2/gamma) ln(1 + gamma/kappa)`: the monopoly mark-up
  set by the arrival elasticity `kappa`. As `gamma -> 0` it tends to the
  risk-neutral floor `2/kappa = 0.0835` $.

Quotes are posted symmetrically about the reservation price:
`bid = r - delta*/2`, `ask = r + delta*/2` (eq. 5).

## 4. Economic interpretation

The strategy couples two ideas: **earn the spread** by quoting both sides, while
**controlling inventory** by skewing quotes through the reservation price. When
the agent accumulates a long position it lowers both quotes, making sells more and
buys less likely, mean-reverting inventory toward zero. `gamma` is the dial: small
`gamma` => aggressive spread capture but volatile inventory; large `gamma` =>
tight inventory but wider spreads and fewer fills.

## 5. Sensitivity analysis (400 paths/gamma, T=600s)

| gamma | PnL mean | PnL std | inv. var | mean |inv| | avg spread | max drawdown |
|-------|----------|---------|----------|-----------|------------|--------------|
| 0.01 | 20.44 | 2.29 | 6.55 | 1.97 | 0.0871 | 1.15 |
| 0.05 | 19.33 | 1.42 | 1.84 | 0.93 | 0.1015 | 0.49 |
| 0.10 | 17.69 | 1.19 | 1.08 | 0.66 | 0.1196 | 0.35 |
| 0.20 | 14.30 | 1.05 | 0.61 | 0.44 | 0.1558 | 0.25 |
| 0.50 | 7.50 | 0.81 | 0.27 | 0.21 | 0.2642 | 0.19 |

As `gamma` rises from 0.01 to 0.5, the **inventory variance
collapses** (from 6.55 to 0.27) and the **average spread
widens**, while PnL trades off against its own volatility (a risk--return
frontier). See `plots/avellaneda_stoikov/sensitivity_gamma.png` and
`sensitivity_inventory_paths.png`.

## 6. Numerical verification

1. **Reservation price is linear in inventory.** Fitting `r` vs `q` gives slope
   **-7.251658e-02** vs the theoretical
   `-gamma sigma^2 (T-t)` = **-7.251658e-02**, with maximum
   deviation **3.41e-13** $ (machine precision) -- the
   relationship is exactly linear.
2. **Spread widens with `gamma`.**

   | gamma | spread at t=0 ($) | increasing? |
   |-------|-------------------|-------------|
| 0.01 | 0.0907 | yes |
| 0.05 | 0.1196 | yes |
| 0.10 | 0.1558 | yes |
| 0.20 | 0.2282 | yes |
| 0.50 | 0.4452 | yes |

3. **Inventory variance decreases with `gamma`.** The Monte-Carlo sweep above
   shows Var(inventory) falling monotonically as `gamma` grows -- stronger risk
   aversion buys tighter inventory control. Figure: `verification.png`.

## 7. Limitations -- and why Hawkes comes next

The model's tractability rests on the **Poisson arrival assumption** (independent
arrivals at a constant per-distance rate). But Phases 1--3 showed decisively that
real AAPL trade arrivals are **not** Poisson: they are over-dispersed
(CV approx 2.6), autocorrelated (Ljung--Box rejected), non-exponential (KS
rejected) and **self-exciting**, with branching ratio rho approx 0.47. Trades
cluster in bursts; a constant-intensity model systematically misprices the *timing
risk* of being adversely filled during a cascade and cannot react to bursts of
informed flow.

This is precisely the gap the **Hawkes extension (next phase)** fills: replacing
`lambda(delta) = A exp(-kappa delta)` with a self-exciting intensity
`lambda(t, delta)` that jumps after each trade lets the market maker widen quotes
exactly when execution risk spikes. The classical model implemented here is the
**correct, faithful baseline** against which that extension will be measured.

### Figure index (`plots/avellaneda_stoikov/`)
episode_quotes_and_price, episode_inventory, episode_cash, episode_pnl,
episode_spread, episode_quote_distances, episode_dashboard, sensitivity_gamma,
sensitivity_inventory_paths, verification.

> **Out of scope (next phases):** Hawkes-driven intensities and reinforcement
> learning.
