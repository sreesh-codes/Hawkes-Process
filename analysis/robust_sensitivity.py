r"""Sensitivity of the robust market maker to :math:`c`, :math:`\gamma`, :math:`\rho`.

Part 7 studies how the three knobs interact:

* :math:`c`      -- the robustness weight (variance-penalty strength);
* :math:`\gamma` -- the AS inventory risk aversion;
* :math:`\rho`   -- the Hawkes branching ratio (clustering strength).

:math:`\rho` is swept by rescaling the excitation :math:`\alpha=\rho\beta` on both
sides of the *true* flow (and of the maker's belief), holding :math:`\beta` fixed;
the per-side estimation covariance :math:`\Sigma_\theta` is held at its calibrated
value throughout (it encodes the *shape* of estimation uncertainty). For each grid
cell we run a small paired Monte-Carlo of the robust strategy and record the mean
PnL, inventory variance and Sharpe ratio.

Outputs: two heatmap panels (:math:`c\times\gamma` and :math:`c\times\rho`), a 3-D
PnL surface over :math:`c\times\gamma`, and response curves of each metric vs
:math:`c` at the calibrated operating point.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from market_maker import ASParameters
from robust.uncertainty import SidedBeliefs
from analysis.robust_backtest import StrategySpec, run_paired_paths

__all__ = ["scale_rho", "grid_cell", "sweep_c_gamma", "sweep_c_rho",
           "response_curve"]

FloatArray = npt.NDArray[np.float64]


def scale_rho(theta: FloatArray, rho: float) -> FloatArray:
    r"""Return a copy of ``theta`` with :math:`\alpha` set to :math:`\rho\beta`."""
    mu, _, beta = float(theta[0]), float(theta[1]), float(theta[2])
    return np.array([mu, rho * beta, beta])


def grid_cell(c: float, gamma: float, theta_buy: FloatArray,
              theta_sell: FloatArray, cov_buy: FloatArray, cov_sell: FloatArray,
              *, n_paths: int, base: ASParameters, impact: float, T: float,
              n_jobs: int = -1) -> dict[str, float]:
    r"""Mean robust-strategy metrics for one :math:`(c,\gamma)` (given true params).

    Returns
    -------
    dict
        ``{"total_pnl", "inv_var", "sharpe"}`` averaged over ``n_paths``.
    """
    params = ASParameters(gamma=gamma, sigma=base.sigma, kappa=base.kappa,
                          A=base.A, T=T, q_max=base.q_max)
    beliefs = SidedBeliefs.from_covariances(theta_buy, cov_buy, theta_sell, cov_sell)
    spec = [StrategySpec(name="Robust", kind="robust", c=c, belief="central")]
    df = run_paired_paths(theta_buy, theta_sell, beliefs, cov_buy, cov_sell,
                          params, spec, n_paths, impact=impact, T=T, n_jobs=n_jobs)
    return {
        "total_pnl": float(df["total_pnl_Robust"].mean()),
        "inv_var": float(df["inv_var_Robust"].mean()),
        "sharpe": float(df["sharpe_Robust"].mean()),
    }


def sweep_c_gamma(cs, gammas, theta_buy, theta_sell, cov_buy, cov_sell, *,
                  n_paths, base, impact, T, n_jobs=-1) -> dict[str, np.ndarray]:
    r"""Grid over :math:`c\times\gamma` at the calibrated :math:`\rho`.

    Returns arrays of shape ``(len(cs), len(gammas))`` for PnL, inventory
    variance and Sharpe, plus the axes.
    """
    pnl = np.zeros((len(cs), len(gammas)))
    ivar = np.zeros_like(pnl)
    shp = np.zeros_like(pnl)
    for i, c in enumerate(cs):
        for j, g in enumerate(gammas):
            m = grid_cell(c, g, theta_buy, theta_sell, cov_buy, cov_sell,
                          n_paths=n_paths, base=base, impact=impact, T=T,
                          n_jobs=n_jobs)
            pnl[i, j], ivar[i, j], shp[i, j] = m["total_pnl"], m["inv_var"], m["sharpe"]
    return {"cs": np.array(cs), "gammas": np.array(gammas), "pnl": pnl,
            "inv_var": ivar, "sharpe": shp}


def sweep_c_rho(cs, rhos, theta_buy, theta_sell, cov_buy, cov_sell, *,
                n_paths, base, impact, T, n_jobs=-1) -> dict[str, np.ndarray]:
    r"""Grid over :math:`c\times\rho` at the calibrated :math:`\gamma`.

    :math:`\rho` rescales :math:`\alpha=\rho\beta` on both sides of the true flow
    and the belief.
    """
    pnl = np.zeros((len(cs), len(rhos)))
    ivar = np.zeros_like(pnl)
    shp = np.zeros_like(pnl)
    for j, rho in enumerate(rhos):
        tb = scale_rho(theta_buy, rho)
        ts = scale_rho(theta_sell, rho)
        for i, c in enumerate(cs):
            m = grid_cell(c, base.gamma, tb, ts, cov_buy, cov_sell,
                          n_paths=n_paths, base=base, impact=impact, T=T,
                          n_jobs=n_jobs)
            pnl[i, j], ivar[i, j], shp[i, j] = m["total_pnl"], m["inv_var"], m["sharpe"]
    return {"cs": np.array(cs), "rhos": np.array(rhos), "pnl": pnl,
            "inv_var": ivar, "sharpe": shp}


def response_curve(cs, theta_buy, theta_sell, cov_buy, cov_sell, *, n_paths,
                   base, impact, T, n_jobs=-1) -> pd.DataFrame:
    r"""Metric-vs-:math:`c` response at the calibrated :math:`(\gamma,\rho)`.

    Returns a tidy DataFrame with columns ``c, total_pnl, inv_var, sharpe``.
    """
    rows = []
    for c in cs:
        m = grid_cell(c, base.gamma, theta_buy, theta_sell, cov_buy, cov_sell,
                      n_paths=n_paths, base=base, impact=impact, T=T, n_jobs=n_jobs)
        m["c"] = c
        rows.append(m)
    return pd.DataFrame(rows)[["c", "total_pnl", "inv_var", "sharpe"]]
