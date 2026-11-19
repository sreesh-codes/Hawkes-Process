r"""Robust backtesting: Classical vs Hawkes vs Robust-Hawkes market makers (Part 5).

A single event loop drives all three strategies on *identical* order flow
(common random numbers => a paired comparison), so any metric difference is due
solely to the quoting rule:

1. **Classical A-S**  -- :class:`PoissonASPolicy` (constant-:math:`\kappa` spread).
2. **Hawkes A-S**     -- :class:`HawkesASPolicy` (intensity-modulated spread,
   Phase 5).
3. **Robust Hawkes A-S** -- :class:`~robust.robust_quotes.RobustHawkesASPolicy`
   (Method B: the Hawkes spread plus the delta-method variance penalty).

The loop mirrors :func:`market_maker.historical_replay.run_backtest` exactly
(same mid-price diffusion + permanent impact, same exponential fill model for the
synthetic experiment and the crossing rule for the LOBSTER replay, same
mark-out), and additionally tracks the per-side delta-method intensity standard
deviation :math:`\sigma_\lambda(t)` via :class:`~robust.uncertainty.VarianceHawkes`
so the robust policy can widen under estimation uncertainty. The full metric
panel is computed by :func:`analysis.comparison.compute_metrics`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from market_maker import ASParameters
from market_maker.hawkes_execution import HawkesIntensity, SidedHawkes
from market_maker.hawkes_market_maker import HawkesASPolicy, PoissonASPolicy
from market_maker.historical_replay import (BacktestResult, _markout,
                                            generate_synthetic_events)
from market_maker.inventory import Portfolio
from robust.robust_quotes import RobustHawkesASPolicy
from robust.uncertainty import SidedVarianceHawkes, VarianceHawkes

__all__ = ["run_robust_backtest", "make_belief_engines", "StrategySpec",
           "run_paired_paths", "one_path_metrics"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class StrategySpec:
    r"""A named strategy: how to build its policy and whether it uses the penalty.

    Attributes
    ----------
    name : str
        Label (e.g. ``"Classical"``, ``"Hawkes"``, ``"Robust"``).
    kind : str
        ``"poisson"``, ``"hawkes"`` or ``"robust"``.
    c : float
        Robustness weight (only for ``kind == "robust"``).
    belief : str
        Which per-side belief the maker quotes on: ``"central"`` (default),
        ``"upper"`` (Method-A conservative) or ``"lower"`` (aggressive).
    """

    name: str
    kind: str
    c: float = 0.0
    belief: str = "central"


def make_belief_engines(theta_buy: FloatArray, theta_sell: FloatArray,
                        cov_buy: FloatArray, cov_sell: FloatArray,
                        ) -> tuple[SidedHawkes, SidedVarianceHawkes]:
    r"""Build the belief intensity engine and its variance companion.

    Parameters
    ----------
    theta_buy, theta_sell : numpy.ndarray
        Per-side belief parameters :math:`(\mu,\alpha,\beta)` the maker quotes on.
    cov_buy, cov_sell : numpy.ndarray
        Per-side parameter covariances for the delta-method variance.

    Returns
    -------
    tuple
        ``(sided_rate, sided_var)`` -- a :class:`SidedHawkes` (for the smoothed
        rate the policy reacts to) and a :class:`SidedVarianceHawkes` (for the
        online intensity standard deviation).
    """
    sided_rate = SidedHawkes(
        HawkesIntensity(*theta_buy), HawkesIntensity(*theta_sell))
    sided_var = SidedVarianceHawkes(
        VarianceHawkes(theta_buy[0], theta_buy[1], theta_buy[2], cov_buy),
        VarianceHawkes(theta_sell[0], theta_sell[1], theta_sell[2], cov_sell))
    return sided_rate, sided_var


def run_robust_backtest(
    events: pd.DataFrame,
    policy,
    sided_rate: SidedHawkes,
    sided_var: SidedVarianceHawkes,
    params: ASParameters,
    *,
    label: str,
    synthetic: bool,
    use_variance: bool,
    kappa_fill: float | None = None,
    impact: float = 0.0,
    sigma: float | None = None,
    s0: float = 585.0,
    seed: int = 0,
    markout_horizon: float = 5.0,
    quote_horizon: float = 1.0,
) -> BacktestResult:
    r"""Event-driven backtest with optional delta-method variance penalty.

    Parameters mirror :func:`market_maker.historical_replay.run_backtest`; the
    additions are:

    Parameters
    ----------
    sided_rate : SidedHawkes
        Belief intensity engine (drives the smoothed rate the policy reacts to).
    sided_var : SidedVarianceHawkes
        Belief variance engine (drives the per-side intensity standard deviation).
    use_variance : bool
        If ``True`` the policy is called with the extra ``sd_buy, sd_sell``
        arguments (Method B); otherwise the Phase-4/5 signature is used.

    Returns
    -------
    BacktestResult
        History, fills and terminal wealth (same schema as Phase 5).
    """
    sided_rate.reset(0.0)
    sided_var.reset(0.0)
    book = Portfolio(q_max=params.q_max, order_size=params.order_size)
    rng = np.random.default_rng(seed)
    kappa_fill = params.kappa if kappa_fill is None else kappa_fill
    sigma = params.sigma if sigma is None else sigma

    ev_t = events["time"].to_numpy()
    ev_side = events["side"].to_numpy()
    ev_mid = events["mid"].to_numpy() if "mid" in events else None
    ev_exec = events["exec_price"].to_numpy() if "exec_price" in events else None

    mid = s0
    prev_t = 0.0
    rows = []
    fills = []
    prev_bid = prev_ask = np.nan
    n_quote_changes = 0

    for i in range(len(ev_t)):
        t = float(ev_t[i])
        side = ev_side[i]
        if synthetic:
            dt = t - prev_t
            if dt > 0:
                mid += sigma * np.sqrt(dt) * rng.standard_normal()
        else:
            mid = float(ev_mid[i])

        lam_buy, lam_sell = sided_rate.intensities_at(t)                 # diagnostic
        rate_buy, rate_sell = sided_rate.expected_rates(t, quote_horizon)  # quoting
        _, sd_buy, _, sd_sell = sided_var.intensity_and_std(t)           # delta-method sd

        if use_variance:
            qte = policy.quote(mid, book.inventory, t, rate_buy, rate_sell,
                               sd_buy, sd_sell)
        else:
            qte = policy.quote(mid, book.inventory, t, rate_buy, rate_sell)

        if not np.isnan(prev_bid):
            if abs(qte.bid - prev_bid) + abs(qte.ask - prev_ask) > 1e-4:
                n_quote_changes += 1
        prev_bid, prev_ask = qte.bid, qte.ask

        if side == "buy":                       # buy MO lifts the ask
            delta_ask = max(qte.ask - mid, 0.0)
            if synthetic:
                hit = rng.random() < np.exp(-kappa_fill * delta_ask)
            else:
                hit = qte.ask <= float(ev_exec[i])
            if hit and book.can_sell():
                book.on_ask_fill(qte.ask)
                fills.append((t, "sell", qte.ask, mid, -1.0))
        else:                                   # sell MO hits the bid
            delta_bid = max(mid - qte.bid, 0.0)
            if synthetic:
                hit = rng.random() < np.exp(-kappa_fill * delta_bid)
            else:
                hit = qte.bid >= float(ev_exec[i])
            if hit and book.can_buy():
                book.on_bid_fill(qte.bid)
                fills.append((t, "buy", qte.bid, mid, 1.0))

        if synthetic and impact != 0.0:
            mid += impact if side == "buy" else -impact

        sided_rate.register(t, side)
        sided_var.register(t, side)

        rows.append({
            "time": t, "mid": mid, "bid": qte.bid, "ask": qte.ask,
            "spread": qte.spread, "reservation": qte.reservation,
            "lam_buy": lam_buy, "lam_sell": lam_sell,
            "sd_buy": sd_buy, "sd_sell": sd_sell,
            "inventory": book.inventory, "cash": book.cash,
            "wealth": book.mark_to_market(mid),
        })
        prev_t = t

    terminal_wealth = book.liquidate(mid)
    history = pd.DataFrame(rows)
    history.attrs["n_quote_changes"] = n_quote_changes
    history.attrs["horizon"] = float(ev_t[-1]) if len(ev_t) else params.T

    fills_df = pd.DataFrame(fills, columns=["time", "side", "price", "mid", "sign"])
    if len(fills_df):
        fills_df["markout"] = _markout(
            fills_df["time"].to_numpy(), fills_df["price"].to_numpy(),
            fills_df["sign"].to_numpy(), history["time"].to_numpy(),
            history["mid"].to_numpy(), markout_horizon)
    else:
        fills_df["markout"] = pd.Series(dtype=float)

    return BacktestResult(history=history, fills=fills_df,
                          terminal_wealth=terminal_wealth, label=label)


def _build_policy(spec: "StrategySpec", params: ASParameters,
                  lam_bar_buy: float, lam_bar_sell: float):
    """Instantiate the policy object for a strategy spec."""
    if spec.kind == "poisson":
        return PoissonASPolicy(params), False
    if spec.kind == "hawkes":
        return HawkesASPolicy(params, lam_bar_buy, lam_bar_sell), False
    if spec.kind == "robust":
        return RobustHawkesASPolicy(params, lam_bar_buy, lam_bar_sell, c=spec.c), True
    raise ValueError(f"unknown strategy kind {spec.kind!r}")  # pragma: no cover


def one_path_metrics(seed, true_buy, true_sell, sided_beliefs, cov_buy, cov_sell,
                     params, strategies, *, impact=0.002, T=600.0, s0=585.0):
    r"""Run every strategy on one shared flow realisation; return per-strategy metrics.

    Parameters
    ----------
    seed : int
        Path seed (drives both the Hawkes flow and the fill/mid RNG).
    true_buy, true_sell : numpy.ndarray
        *True* per-side parameters generating the order flow.
    sided_beliefs : robust.uncertainty.SidedBeliefs
        Per-side lower/central/upper beliefs the makers may quote on.
    cov_buy, cov_sell : numpy.ndarray
        Per-side parameter covariances for the variance penalty.
    params : ASParameters
        AS parameters.
    strategies : list[StrategySpec]
        Strategies to evaluate on this path.
    impact, T, s0 : float
        Permanent impact ($), horizon (s), initial mid ($).

    Returns
    -------
    dict[str, dict[str, float]]
        ``{strategy_name: metric_panel}``.
    """
    from analysis.comparison import compute_metrics
    from robust.delta_method import lambda_bar

    buy = HawkesIntensity(*true_buy)
    sell = HawkesIntensity(*true_sell)
    events = generate_synthetic_events(buy, sell, T, seed)
    out = {}
    for spec in strategies:
        bbuy = sided_beliefs.buy[spec.belief]
        bsell = sided_beliefs.sell[spec.belief]
        lbb, lbs = lambda_bar(bbuy), lambda_bar(bsell)
        policy, use_var = _build_policy(spec, params, lbb, lbs)
        sr, sv = make_belief_engines(bbuy, bsell, cov_buy, cov_sell)
        res = run_robust_backtest(events, policy, sr, sv, params, label=spec.name,
                                  synthetic=True, use_variance=use_var,
                                  impact=impact, s0=s0, seed=1000 + seed)
        out[spec.name] = compute_metrics(res)
    return out


def run_paired_paths(true_buy, true_sell, sided_beliefs, cov_buy, cov_sell,
                     params, strategies, n_paths, *, impact=0.002, T=600.0,
                     s0=585.0, n_jobs=-1):
    r"""Paired multi-path backtest: return a wide table of ``metric_<strategy>`` columns.

    Common random numbers across strategies (same ``seed`` per path) make this a
    low-variance paired comparison suitable for the significance tests.
    """
    from joblib import Parallel, delayed

    rows = Parallel(n_jobs=n_jobs)(
        delayed(one_path_metrics)(s, true_buy, true_sell, sided_beliefs, cov_buy,
                                  cov_sell, params, strategies, impact=impact,
                                  T=T, s0=s0)
        for s in range(n_paths))
    records = []
    for s, per in enumerate(rows):
        rec = {"seed": s}
        for name, metrics in per.items():
            rec.update({f"{k}_{name}": v for k, v in metrics.items()})
        records.append(rec)
    return pd.DataFrame(records)
