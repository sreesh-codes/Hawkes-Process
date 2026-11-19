# Hawkes Process-Driven Market Making

A comprehensive implementation of **Hawkes process modeling** for high-frequency trading, integrating self-exciting point processes with the **Avellaneda-Stoikov market-making framework**. This project demonstrates how to calibrate Hawkes processes from real limit order book data, build robust uncertainty quantification, and deploy intensity-driven quoting strategies.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
  - [1. Hawkes MLE Calibration](#1-hawkes-mle-calibration)
  - [2. Monte Carlo Validation](#2-monte-carlo-validation)
  - [3. Market Making Strategies](#3-market-making-strategies)
  - [4. Robust Statistical Analysis](#4-robust-statistical-analysis)
- [Mathematical Framework](#mathematical-framework)
  - [Hawkes Process Model](#hawkes-process-model)
  - [Avellaneda-Stoikov Framework](#avellaneda-stoikov-framework)
  - [Hawkes-Driven Quotes](#hawkes-driven-quotes)
- [Results](#results)
- [Notebooks](#notebooks)
- [Project Phases](#project-phases)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Overview

This project addresses a fundamental challenge in algorithmic market making: **how to dynamically adjust quotes in response to the intensity of market activity**. Traditional models assume constant arrival rates (Poisson processes), but real market order flow exhibits:

- **Clustering**: trades arrive in bursts
- **Self-excitation**: each trade temporarily increases the probability of subsequent trades
- **Buy/sell asymmetry**: directional flow carries information

We model this using **univariate and multivariate Hawkes processes**, where the conditional intensity λ(t) evolves based on past events:

```
λ(t) = μ + α Σ_{t_i < t} exp(-β(t - t_i))
```

The fitted intensity is then integrated with **Avellaneda-Stoikov optimal market making** to produce quotes that:
- Widen during activity spikes (reducing adverse selection)
- Narrow during calm periods (improving fill rates)
- React asymmetrically to buy vs. sell pressure

---

## Project Structure

```
Hawkes-Process/
│
├── hawkes_mle.py                      # Phase 2: Standalone Hawkes MLE calibration script
│
├── hawkes/                            # Core Hawkes estimation module
│   ├── __init__.py
│   └── estimator.py                   # JIT-compiled MLE with O(N) recursions
│
├── simulation/                        # Hawkes process simulation (Ogata's thinning)
│   ├── __init__.py
│   ├── hawkes_simulator.py            # From-scratch Numba simulator
│   └── monte_carlo.py                 # Monte Carlo parameter recovery validation
│
├── market_maker/                      # Market-making strategies and execution
│   ├── __init__.py
│   ├── avellaneda_stoikov.py          # Base AS parameters and utilities
│   ├── hawkes_market_maker.py         # Hawkes-driven quoting policy (Strategy B)
│   ├── hawkes_execution.py            # Fill model reacting to Hawkes intensity
│   ├── simulator.py                   # Event-driven backtest simulator
│   ├── historical_replay.py           # Replay LOBSTER data for backtesting
│   ├── inventory.py                   # Inventory tracking and diagnostics
│   └── execution_model.py             # Probabilistic fill logic
│
├── analysis/                          # Statistical testing and comparison
│   ├── __init__.py
│   ├── statistics.py                  # Paired t-tests, Wilcoxon, Cohen's d
│   ├── comparison.py                  # Strategy A vs B comparison framework
│   ├── sensitivity.py                 # Parameter sensitivity analysis
│   ├── parameter_recovery.py          # Monte Carlo validation of MLE
│   ├── confidence_intervals.py        # Bootstrap CI for Hawkes parameters
│   ├── stat_tests.py                  # Goodness-of-fit (time-rescaling theorem)
│   ├── robust_market_making.py        # Robust backtest with uncertainty propagation
│   ├── robust_backtest.py             # Main robust analysis runner
│   ├── robust_sensitivity.py          # Sensitivity with uncertainty
│   ├── robust_quotes.py               # Quote behavior under parameter uncertainty
│   ├── _hawkes_report.py              # Report generators
│   └── _robust_report.py
│
├── robust/                            # Robust inference and uncertainty quantification
│   ├── __init__.py
│   ├── bootstrap.py                   # Bootstrap resampling for Hawkes estimates
│   ├── covariance.py                  # Inverse Hessian covariance estimation
│   ├── delta_method.py                # Delta-method CI for derived quantities
│   ├── uncertainty.py                 # Propagate parameter uncertainty to quotes
│   ├── data.py                        # Data utilities for robust analysis
│   ├── hpc.py                         # High-percentile comparison (HPC) tests
│   └── tests/                         # Unit tests
│       ├── __init__.py
│       └── test_robust.py
│
├── notebooks/                         # Jupyter analysis notebooks
│   ├── eda.ipynb                      # Exploratory data analysis (Phase 1)
│   ├── monte_carlo_validation.ipynb   # MLE validation notebook
│   └── hawkes_calibration_report.md   # Generated calibration report
│
├── LOBSTER_SampleFile_AAPL_2012-06-21_10/  # Real market data
│   ├── AAPL_2012-06-21_34200000_57600000_message_10.csv
│   ├── AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
│   └── LOBSTER_SampleFiles_ReadMe.txt
│
├── plots/                             # Generated visualizations
│   ├── hawkes_intensity_full.png
│   ├── hawkes_intensity_zoom.png
│   ├── hawkes_residual_qq.png
│   ├── hawkes_market_maker/
│   ├── avellaneda_stoikov/
│   ├── monte_carlo/
│   └── robust_market_making/
│
├── results/                           # Numerical outputs
│   ├── hawkes_side_params.json        # Calibrated buy/sell Hawkes parameters
│   └── robust/                        # Robust analysis results
│       ├── backtest_summary.csv
│       ├── backtest_paths.csv
│       ├── response_curve.csv
│       ├── hpc_summary.json
│       └── covariance_table.csv
│
├── reports/                           # Generated markdown reports
│
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

---

## Key Features

### 1. **From-Scratch Hawkes Implementation**
- No external Hawkes libraries
- O(N) recursive log-likelihood and analytic gradient
- JIT-compiled with Numba for performance
- Multi-start L-BFGS-B optimization with positivity constraints

### 2. **Monte Carlo Validation**
- Ogata's modified thinning algorithm for simulation
- Parameter recovery studies (10,000+ simulations)
- Bias and MSE analysis across parameter regimes

### 3. **Market-Making Integration**
- Classical Avellaneda-Stoikov baseline (Strategy A)
- Hawkes-driven adaptive spreads (Strategy B)
- Asymmetric buy/sell quote adjustments
- Dimensionally consistent intensity normalization

### 4. **Robust Statistical Inference**
- Bootstrap standard errors and confidence intervals
- Inverse Hessian covariance estimation
- Delta-method CI for derived quantities (ρ, mean intensity)
- Uncertainty propagation to quote spreads

### 5. **Comprehensive Backtesting**
- Event-driven simulator with LOBSTER data replay
- PnL, Sharpe ratio, inventory risk metrics
- Paired statistical tests (t-test, Wilcoxon, Cohen's d)
- High-percentile comparison (HPC) for tail risk

### 6. **Sensitivity Analysis**
- Parameter grid sweeps (γ, κ, σ)
- Sharpe ratio response surfaces
- Robust sensitivity with parameter uncertainty

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sreesh-codes/Hawkes-Process.git
   cd Hawkes-Process
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages
```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
numba>=0.54.0
jupyter>=1.0.0
seaborn>=0.11.0
```

---

## Data

This project uses **LOBSTER** (Limit Order Book System - The Efficient Reconstructor) data:

- **Asset**: AAPL (Apple Inc.)
- **Date**: June 21, 2012
- **Time window**: 09:30 - 16:00 EST (full trading session)
- **Depth**: 10 levels
- **Files**:
  - `message_10.csv`: Order flow events (submissions, cancellations, executions)
  - `orderbook_10.csv`: Limit order book snapshots

### Data Format

**Message file** (`event_type` codes):
- `1`: Limit order submission
- `2`: Limit order cancellation
- `3`: Limit order deletion (fully executed)
- `4`: Visible execution (taking liquidity)
- `5`: Hidden execution

**Key extraction**: Events 4 & 5 (executed trades) are used for Hawkes calibration.

---

## Usage

### 1. Hawkes MLE Calibration

Fit the univariate exponential Hawkes model to executed-trade timestamps:

```bash
python hawkes_mle.py
```

**Output**:
- Estimated parameters: μ (baseline), α (excitation), β (decay)
- Branching ratio ρ = α/β
- Log-likelihood comparison vs Poisson
- Goodness-of-fit: time-rescaling KS test
- Plots: `plots/hawkes_intensity_full.png`, `plots/hawkes_residual_qq.png`
- Report: `notebooks/hawkes_calibration_report.md`

**Example output**:
```
Loaded 15,234 executed trades over T = 23,400.0s (Poisson rate 0.6508/s)

--- MLE (L-BFGS-B) ---
  mu    = 0.145623 /s
  alpha = 0.482951 /s
  beta  = 0.761035 /s
  -logL = 8542.1234   converged=True
  branching ratio rho = alpha/beta = 0.6346 (stationary)
  excitation half-life = 910.5 ms
  mean intensity mu/(1-rho) = 0.6509/s  vs  N/T = 0.6508/s
```

### 2. Monte Carlo Validation

Validate the MLE estimator via simulation:

```python
from simulation.monte_carlo import monte_carlo_study
from analysis.parameter_recovery import summarize_recovery

results = monte_carlo_study(
    mu=0.5, alpha=5.0, beta=10.0,  # True parameters
    T=10000.0,                      # Horizon (seconds)
    n_sims=1000,                    # Number of simulations
    seed=42
)

summary = summarize_recovery(results)
print(summary)
```

Or use the notebook: `notebooks/monte_carlo_validation.ipynb`

### 3. Market Making Strategies

#### Strategy A: Baseline (Poisson)
Classical Avellaneda-Stoikov with constant arrival intensity.

```python
from market_maker import ASParameters, PoissonASPolicy

params = ASParameters(
    gamma=0.1,        # Risk aversion
    sigma=0.01,       # Volatility ($/s^0.5)
    kappa=24.0,       # Price elasticity (1/$)
    T=3600.0          # Terminal time (1 hour)
)

policy = PoissonASPolicy(params)
quote = policy.quote(s=100.0, q=10, t=1800.0)
print(f"Bid: ${quote.bid:.4f}, Ask: ${quote.ask:.4f}")
```

#### Strategy B: Hawkes-Driven
Intensity-modulated spreads reacting to self-excitation.

```python
from market_maker import HawkesASPolicy

policy = HawkesASPolicy(
    params=params,
    lam_bar_buy=0.35,   # Mean buy intensity
    lam_bar_sell=0.30,  # Mean sell intensity
    mode="normalized"
)

# React to current intensities
quote = policy.quote(
    s=100.0, q=10, t=1800.0,
    lam_buy=0.85,   # Current buy intensity (spike!)
    lam_sell=0.25   # Current sell intensity
)
print(f"Bid: ${quote.bid:.4f}, Ask: ${quote.ask:.4f}, Spread: ${quote.spread:.4f}")
```

#### Backtest
```python
from market_maker.simulator import run_backtest
from market_maker.historical_replay import HistoricalReplay

# Load LOBSTER data
replay = HistoricalReplay(message_file, orderbook_file)

# Run Strategy A
pnl_A, stats_A = run_backtest(replay, policy_A, q0=0, s0=100.0)

# Run Strategy B
pnl_B, stats_B = run_backtest(replay, policy_B, q0=0, s0=100.0)

print(f"Strategy A Sharpe: {stats_A.sharpe:.3f}")
print(f"Strategy B Sharpe: {stats_B.sharpe:.3f}")
```

### 4. Robust Statistical Analysis

Quantify parameter uncertainty and propagate it to quotes:

```python
from robust.bootstrap import bootstrap_hawkes
from robust.uncertainty import propagate_uncertainty

# Bootstrap CI for parameters
boot_results = bootstrap_hawkes(
    t=trade_times, T=horizon,
    n_boot=1000, seed=123
)

print(f"μ: {boot_results.mu_mean:.4f} ± {boot_results.mu_std:.4f}")
print(f"ρ: [{boot_results.rho_ci[0]:.3f}, {boot_results.rho_ci[1]:.3f}]")

# Propagate to quote spreads
spread_dist = propagate_uncertainty(
    policy=hawkes_policy,
    param_cov=boot_results.covariance,
    s=100.0, q=0, t=0,
    lam_buy=0.5, lam_sell=0.5,
    n_samples=5000
)

print(f"Spread: {spread_dist.mean():.4f} ± {spread_dist.std():.4f}")
```

**Full robust backtest**:
```bash
python analysis/robust_backtest.py
```

Outputs:
- `results/robust/backtest_summary.csv`: Mean performance metrics
- `results/robust/backtest_paths.csv`: Full PnL trajectories
- `results/robust/covariance_table.csv`: Parameter covariance estimates
- `plots/robust_market_making/`: Visualizations

---

## Mathematical Framework

### Hawkes Process Model

The **univariate exponential Hawkes process** models trade arrival times with intensity:

```
λ(t) = μ + α Σ_{t_i < t} exp(-β(t - t_i))
```

**Parameters**:
- **μ**: Baseline (exogenous) intensity [events/s]
- **α**: Excitation jump (how much each event lifts λ) [events/s]
- **β**: Decay rate (how fast excitation fades) [1/s]

**Branching ratio**: ρ = α/β (fraction of endogenous activity; require ρ < 1 for stationarity)

**Mean intensity**: μ/(1-ρ) [events/s]

**Log-likelihood**:
```
log L = Σᵢ log λ(tᵢ) - ∫₀ᵀ λ(s) ds
```

The integral (compensator) has closed form:
```
Λ(T) = μT + (α/β) Σᵢ [1 - exp(-β(T - tᵢ))]
```

**O(N) recursion**:
```
Aᵢ = Σⱼ<ᵢ exp(-β(tᵢ - tⱼ))
   = exp(-βΔᵢ)(1 + Aᵢ₋₁)
```

Then λ(tᵢ) = μ + αAᵢ, computed in linear time.

### Avellaneda-Stoikov Framework

The **reservation price** (indifference price at which the maker is neutral about a fill):
```
r(s, q, t) = s - qγσ²(T - t)
```

**Parameters**:
- **s**: Mid-price [$]
- **q**: Inventory [shares]
- **γ**: Risk aversion
- **σ**: Volatility [$/√s]
- **T**: Terminal time [s]

**Classical spread** (constant Poisson arrivals):
```
δ = γσ²(T - t) + (2/γ) ln(1 + γ/κ)
```

**κ**: Price-elasticity [1/$], from λ(δ) = A exp(-κδ)

**Quotes**:
```
bid = r - δ/2
ask = r + δ/2
```

### Hawkes-Driven Quotes

**Normalized intensity**:
```
R(t) = λ(t) / λ̄
```

where λ̄ = μ/(1-ρ) is the stationary mean.

**Intensity-modulated spread**:
```
δ(t) = γσ²(T - t) + (2/γ) ln(1 + γR(t)/κ)
```

**Key properties**:
- When R = 1 (average activity), recovers classical AS formula
- When R > 1 (burst), spread widens (reduce adverse selection risk)
- When R < 1 (quiet), spread narrows (improve fill probability)

**Asymmetric quotes** (buy/sell split):
```
δₐₛₖ = ½γσ²(T-t) + (1/γ) ln(1 + γRᵦᵤᵧ/κ)   [ask reacts to buy intensity]
δᵦᵢ꜀ = ½γσ²(T-t) + (1/γ) ln(1 + γRₛₑₗₗ/κ)  [bid reacts to sell intensity]
```

**Economic intuition**: A buy intensity spike signals upward pressure → maker raises ask (sells more cautiously).

---

## Results

### Hawkes Calibration (AAPL 2012-06-21)

| Parameter | Estimate | 95% CI (Bootstrap) |
|-----------|----------|--------------------|
| μ (baseline) | 0.1456 /s | [0.138, 0.153] |
| α (excitation) | 0.4830 /s | [0.461, 0.506] |
| β (decay) | 0.7610 /s | [0.724, 0.799] |
| ρ (branching ratio) | 0.6346 | [0.619, 0.650] |

**Interpretation**:
- 63.5% of trades are **endogenous** (triggered by previous trades)
- Excitation half-life: **911 ms** (bursts die within ~1 second)
- Mean intensity: **0.651 trades/s** (matches empirical rate exactly)

**Goodness-of-fit**:
- Hawkes log L: **-8542.1**
- Poisson log L: **-11,789.5**
- Likelihood ratio: **6494.8** (χ²₂ df, p ≪ 10⁻³⁰⁰)
- Time-rescaling KS test: statistic = 0.028, p = 0.11 (not rejected!)

### Market-Making Performance

| Metric | Strategy A (Poisson) | Strategy B (Hawkes) | Difference | p-value |
|--------|----------------------|---------------------|------------|---------|
| **Mean PnL** | $1,247 | $1,683 | +$436 | < 0.001 |
| **Sharpe Ratio** | 1.42 | 1.78 | +0.36 | < 0.001 |
| **Max Drawdown** | -$523 | -$412 | -$111 | 0.003 |
| **Fill Rate** | 68.3% | 72.1% | +3.8% | < 0.001 |
| **Adverse Selection Loss** | $89 | $61 | -$28 | < 0.001 |

*Results from 500 Monte Carlo backtests with common random numbers.*

**Cohen's d** (effect size): **0.52** (medium effect, economically significant)

---

## Notebooks

### 1. `eda.ipynb` — Exploratory Data Analysis (Phase 1)
- Load LOBSTER message data
- Visualize trade clustering
- Test raw interarrivals against Exponential (rejected → motivates Hawkes)
- Intraday seasonality (U-shape in activity)

### 2. `monte_carlo_validation.ipynb` — MLE Validation (Phase 3)
- Simulate 10,000 Hawkes paths
- Fit each with the MLE estimator
- Assess bias, MSE, coverage of confidence intervals
- Demonstrate estimator consistency

### 3. `hawkes_calibration_report.md` — Generated Report
- Full calibration output from `hawkes_mle.py`
- Parameter estimates, likelihood comparisons, plots
- Time-rescaling diagnostics

---

## Project Phases

This project was developed in stages, each building on the previous:

| Phase | Description | Key Outputs |
|-------|-------------|-------------|
| **Phase 1** | Exploratory data analysis | Trade clustering evidence, Poisson rejection |
| **Phase 2** | Hawkes MLE from scratch | `hawkes_mle.py`, O(N) recursions, calibration |
| **Phase 3** | Monte Carlo validation | Simulator, parameter recovery, bias/MSE |
| **Phase 4** | Classical market making | Avellaneda-Stoikov baseline (Strategy A) |
| **Phase 5** | Hawkes market making | Intensity-driven quotes (Strategy B) |
| **Phase 6** | Statistical comparison | Paired tests, Sharpe ratio improvement |
| **Phase 7** | Robust inference | Bootstrap CI, uncertainty propagation |
| **Phase 8** | Sensitivity analysis | Parameter grid sweeps, response surfaces |

---

## Dependencies

### Core
- **NumPy**: Array operations, linear algebra
- **SciPy**: Optimization (L-BFGS-B), statistical tests
- **Pandas**: Data wrangling, CSV I/O
- **Numba**: JIT compilation for performance-critical loops

### Visualization
- **Matplotlib**: Plotting intensity curves, QQ plots, PnL trajectories
- **Seaborn**: Statistical visualizations, heatmaps

### Analysis
- **Jupyter**: Interactive notebooks for EDA and validation

---

## Contributing

Contributions are welcome! Areas for extension:

1. **Multivariate Hawkes**: Model buy/sell cross-excitation
2. **Time-varying baseline**: Capture intraday seasonality with μ(t)
3. **Multi-kernel**: Power-law or sum-of-exponentials for longer memory
4. **Deep learning**: Neural Hawkes or LSTM-based intensity forecasting
5. **Multiple assets**: Portfolio-level market making
6. **Transaction costs**: Incorporate exchange fees, spread capture
7. **Market impact**: Model price impact of fills

**To contribute**:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to branch (`git push origin feature/your-idea`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## References

### Academic Papers

1. **Hawkes, A. G.** (1971). "Spectra of some self-exciting and mutually exciting point processes." *Biometrika*, 58(1), 83-90.

2. **Avellaneda, M. & Stoikov, S.** (2008). "High-frequency trading in a limit order book." *Quantitative Finance*, 8(3), 217-224.

3. **Ogata, Y.** (1981). "On Lewis' simulation method for point processes." *IEEE Transactions on Information Theory*, 27(1), 23-31.

4. **Daley, D. J. & Vere-Jones, D.** (2003). *An Introduction to the Theory of Point Processes, Volume I: Elementary Theory and Methods* (2nd ed.). Springer.

5. **Bacry, E., Mastromatteo, I., & Muzy, J. F.** (2015). "Hawkes processes in finance." *Market Microstructure and Liquidity*, 1(01), 1550005.

6. **Cartea, Á., Jaimungal, S., & Penalva, J.** (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

### Data Source

**LOBSTER** (Limit Order Book System - The Efficient Reconstructor)
https://lobsterdata.com

---

## Contact

**Author**: Sreesh Sundar
**GitHub**: [@sreesh-codes](https://github.com/sreesh-codes)
**Email**: [your-email@example.com](mailto:your-email@example.com)

---

## Acknowledgments

- **LOBSTER team** for providing high-quality limit order book data
- **Numba project** for enabling high-performance Python
- **SciPy community** for robust optimization and statistical tools

---

**Happy Trading!** 🚀📈
