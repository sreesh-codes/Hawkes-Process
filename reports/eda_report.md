# Phase 1 — EDA of Order Arrivals
## Hawkes-Process-Driven Market Making · LOBSTER AAPL, 2012-06-21 (Level 10)

This report summarises the exploratory analysis of **executed-trade arrivals** in the AAPL
limit-order book for the full trading session (09:30–16:00). The accompanying notebook
[`eda.ipynb`](./eda.ipynb) reproduces every number and figure. Figures are saved in
[`../plots/`](../plots/).

> **Scope.** This is descriptive analysis only. No Hawkes model is calibrated here; that is the
> next phase.

---

## 1. Data

| File | Rows | Content |
|------|------|---------|
| `..._message_10.csv`   | 400,391 | One row per book-changing event (time, type, id, size, price, direction) |
| `..._orderbook_10.csv` | 400,391 | Book state (10 ask + 10 bid price/size pairs) after each event |

**Event-type breakdown (message file):**

| Type | Event | Count |
|------|-------|-------|
| 1 | New limit order | 191,015 |
| 2 | Partial cancellation | 3,260 |
| 3 | Deletion | 171,126 |
| 4 | **Visible execution (trade)** | 23,658 |
| 5 | **Hidden execution (trade)** | 11,332 |

We keep types **4 and 5** as trades: **34,990 executions**. Prices are LOBSTER's integer
(dollars × 10⁴) converted to dollars; `direction` is the resting limit-order side, from which the
aggressor side is inferred (hitting a sell limit ⇒ buyer-initiated, and vice-versa).

---

## 2. Interarrival statistics

A single marketable order that walks multiple price levels appears as several executions with an
**identical timestamp**; these are collapsed (only strictly positive gaps kept).

| Quantity | Value |
|----------|-------|
| Trade events | 34,990 |
| Simultaneous gaps dropped | 10,129 |
| Positive interarrivals | 24,860 |
| Mean interarrival μ | **0.941 s** |
| Std interarrival σ | 2.483 s |
| **Coefficient of variation CV = σ/μ** | **2.64** |
| Implied trade rate λ̂ = 1/μ | 1.06 trades/s (≈ 64/min) |

For a Poisson process CV = 1. The observed **CV ≈ 2.64** signals strong **overdispersion** — the
first quantitative sign of clustering.

---

## 3. Diagnostic figures

| Figure | File | What it shows |
|--------|------|---------------|
| Histogram vs Exp(λ) | `plots/hist_interarrival.png` | Large excess of short gaps + heavy tail vs the exponential fit |
| Empirical vs exponential CDF | `plots/ecdf_interarrival.png` | ECDF sits above the exponential at small τ |
| ACF of interarrivals | `plots/acf_interarrival.png` | Persistent positive autocorrelation (Poisson ⇒ ~0) |
| Rolling intensity | `plots/rolling_intensity.png` | Intraday U-shape + sharp bursts (Poisson ⇒ flat) |

---

## 4. Formal tests

**Ljung–Box** — H₀: interarrivals independent (required by Poisson).

| Lag | Statistic | p-value |
|-----|-----------|---------|
| 10 | 1702.4 | < 1e-300 |
| 20 | 2408.6 | < 1e-300 |
| 50 | 4344.6 | < 1e-300 |

→ Independence **decisively rejected** at every horizon.

**Kolmogorov–Smirnov** — H₀: τ ~ Exponential(λ̂).

| KS statistic | p-value |
|--------------|---------|
| 0.4927 | ≈ 0 |

→ Exponential interarrivals (hence Poisson) **decisively rejected**.

---

## 5. Conclusions

| Diagnostic | Poisson expectation | Observed | Verdict |
|---|---|---|---|
| CV | 1.0 | 2.64 | overdispersed → clustering |
| Histogram / ECDF | matches Exp(λ) | short-gap excess, heavy tail | not exponential |
| KS test | p large | p ≈ 0 | reject exponential |
| ACF | ~0 at all lags | persistent positive | reject independence |
| Ljung–Box | p large | p ≈ 0 | reject independence |
| Rolling intensity | flat λ | U-shape + spikes | non-stationary, bursty |

**Are arrivals Poisson?** **No.** Homogeneous Poisson is rejected on all three of its defining
properties — exponential interarrivals (KS), independence (Ljung–Box, ACF), and constant intensity
(rolling plot).

**Is there clustering?** **Yes.** Trades arrive in bursts: excess short interarrivals, a heavy tail,
slowly-decaying positive autocorrelation, and intensity spikes. Trades beget trades.

**Why a Hawkes process?** A Hawkes process models the conditional intensity as

```
λ(t) = μ + Σ_{t_i < t} α · exp(−β (t − t_i))
```

a baseline rate μ plus a **self-exciting** kernel, where each past trade transiently raises the
intensity of future ones. This reproduces precisely the empirical features above — overdispersion,
short-gap excess, positive interarrival autocorrelation, and intensity clustering — none of which
Poisson can capture. It is the natural arrival model for the downstream market-making engine.

**Next phase (not in this report):** calibrate the Hawkes intensity (estimate μ, α, β; consider a
multivariate buy/sell specification) and validate via the time-rescaling theorem.
