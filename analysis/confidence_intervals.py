"""Sampling-distribution statistics and confidence intervals for the estimator.

Given a table of Monte-Carlo estimates (one row per trial) and the known ground
truth, this module computes, per parameter, the quantities that describe an
estimator's quality:

* **Mean estimate** -- centre of the sampling distribution.
* **Bias** -- ``mean(estimate) - true`` (and relative bias).
* **Variance / standard deviation** -- spread of the estimator.
* **RMSE** -- ``sqrt(bias^2 + variance)``, the total error.
* **95% confidence interval** -- the empirical 2.5%/97.5% percentile interval of
  the sampling distribution (a Monte-Carlo, distribution-free CI).
* **95% CI of the mean** -- ``mean +/- 1.96 * SE`` with ``SE = std / sqrt(n)``,
  used to judge whether any bias is statistically significant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["PARAM_COLS", "summarise_parameter", "summary_table",
           "confidence_interval_table"]

# Mapping: ground-truth attribute -> estimate column in the results DataFrame.
PARAM_COLS: dict[str, str] = {
    "mu": "mu_hat",
    "alpha": "alpha_hat",
    "beta": "beta_hat",
    "rho": "rho_hat",
}


def summarise_parameter(estimates: np.ndarray, true_value: float) -> dict[str, float]:
    """Compute the full statistics block for one parameter.

    Parameters
    ----------
    estimates : numpy.ndarray
        Estimates of the parameter across trials (NaNs/inf are dropped).
    true_value : float
        Ground-truth value.

    Returns
    -------
    dict
        Keys: ``true, n, mean, bias, rel_bias_pct, variance, std, rmse,
        ci_low, ci_high, mean_se, mean_ci_low, mean_ci_high, bias_significant``.
    """
    x = np.asarray(estimates, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    mean = float(np.mean(x))
    bias = mean - true_value
    var = float(np.var(x, ddof=1))
    std = float(np.sqrt(var))
    rmse = float(np.sqrt(bias**2 + var))
    ci_low, ci_high = np.percentile(x, [2.5, 97.5])
    se = std / np.sqrt(n)
    mean_ci_low, mean_ci_high = mean - 1.96 * se, mean + 1.96 * se
    return {
        "true": true_value,
        "n": n,
        "mean": mean,
        "bias": bias,
        "rel_bias_pct": 100.0 * bias / true_value if true_value != 0 else np.nan,
        "variance": var,
        "std": std,
        "rmse": rmse,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "mean_se": se,
        "mean_ci_low": mean_ci_low,
        "mean_ci_high": mean_ci_high,
        # bias is "significant" if the true value lies outside the 95% CI of the
        # mean estimate, i.e. |bias| > 1.96 * SE.
        "bias_significant": bool(true_value < mean_ci_low or true_value > mean_ci_high),
    }


def summary_table(df: pd.DataFrame, true_params: dict[str, float],
                  *, only_success: bool = True) -> pd.DataFrame:
    """Build a per-parameter statistics table from Monte-Carlo results.

    Parameters
    ----------
    df : pandas.DataFrame
        Results with columns ``mu_hat, alpha_hat, beta_hat, rho_hat, success``.
    true_params : dict
        Ground truth keyed by ``mu, alpha, beta, rho``.
    only_success : bool, optional
        Restrict to converged trials (default ``True``).

    Returns
    -------
    pandas.DataFrame
        One row per parameter, indexed by parameter name.
    """
    data = df[df["success"]] if only_success else df
    rows = {}
    for name, col in PARAM_COLS.items():
        rows[name] = summarise_parameter(data[col].to_numpy(), true_params[name])
    return pd.DataFrame(rows).T


def confidence_interval_table(df: pd.DataFrame, true_params: dict[str, float],
                              *, only_success: bool = True) -> pd.DataFrame:
    """Compact 95% confidence-interval view (truth, mean, CI, coverage flag)."""
    full = summary_table(df, true_params, only_success=only_success)
    out = full[["true", "mean", "ci_low", "ci_high"]].copy()
    out["true_in_95ci"] = (out["true"] >= out["ci_low"]) & (out["true"] <= out["ci_high"])
    return out
