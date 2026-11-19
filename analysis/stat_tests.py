r"""Statistical significance of the Hawkes-vs-Poisson differences (Phase 5, Part 8).

For each metric we treat the ``n_paths`` paired differences ``B - A`` (same order
flow, common random numbers) and ask whether the mean difference is significantly
non-zero using three complementary procedures:

* **paired t-test** -- parametric, assumes approximately normal differences;
* **Wilcoxon signed-rank test** -- non-parametric, robust to non-normality;
* **bootstrap confidence interval** -- distribution-free 95% CI of the mean
  difference by resampling the paired differences.

A result is reported as *significant* when the t-test and Wilcoxon p-values are
below 0.05 and the bootstrap CI excludes zero.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["TESTED_METRICS", "bootstrap_ci", "paired_tests", "run_all_tests",
           "plot_significance"]

# metric -> (+1 if "larger is better", -1 if "smaller is better")
TESTED_METRICS = {
    "total_pnl": +1,
    "sharpe": +1,
    "inv_var": -1,
    "max_drawdown": -1,
}


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences.

    Parameters
    ----------
    diffs : numpy.ndarray
        Paired differences ``B - A``.
    n_boot : int, optional
        Number of bootstrap resamples.
    alpha : float, optional
        Significance level (0.05 => 95% CI).
    seed : int, optional
        RNG seed.

    Returns
    -------
    tuple of float
        Lower and upper CI bounds of the mean difference.
    """
    rng = np.random.default_rng(seed)
    n = diffs.shape[0]
    means = diffs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_tests(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired t-test, Wilcoxon test and bootstrap CI for ``b - a``.

    Returns
    -------
    dict
        ``mean_diff, t_stat, t_p, wilcoxon_stat, wilcoxon_p, boot_lo, boot_hi``.
    """
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[np.isfinite(d)]
    t_stat, t_p = stats.ttest_rel(b, a, nan_policy="omit")
    # Wilcoxon fails if all differences are zero; guard it.
    try:
        w_stat, w_p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        w_stat, w_p = np.nan, 1.0
    lo, hi = bootstrap_ci(d)
    return {"mean_diff": float(np.mean(d)), "t_stat": float(t_stat),
            "t_p": float(t_p), "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p), "boot_lo": lo, "boot_hi": hi}


def run_all_tests(df: pd.DataFrame) -> dict[str, dict]:
    """Run the paired tests for every metric in :data:`TESTED_METRICS`.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`analysis.comparison.run_comparison` (wide ``*_A/*_B``).

    Returns
    -------
    dict
        Per-metric test results, augmented with ``better`` (sign of a beneficial
        difference) and ``significant`` (bool).
    """
    out = {}
    for metric, direction in TESTED_METRICS.items():
        res = paired_tests(df[f"{metric}_A"].to_numpy(), df[f"{metric}_B"].to_numpy())
        significant = (res["t_p"] < 0.05 and res["wilcoxon_p"] < 0.05
                       and (res["boot_lo"] > 0 or res["boot_hi"] < 0))
        beneficial = np.sign(res["mean_diff"]) == direction
        res.update({"direction": direction, "significant": bool(significant),
                    "beneficial": bool(beneficial)})
        out[metric] = res
    return out


def plot_significance(df: pd.DataFrame, tests: dict[str, dict], out_dir: str) -> None:
    """Figure 12: standardised mean difference with bootstrap CIs per metric."""
    os.makedirs(out_dir, exist_ok=True)
    metrics = list(tests)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.2))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        a, b = df[f"{m}_A"].to_numpy(), df[f"{m}_B"].to_numpy()
        d = b - a
        # normalise the paired difference to units of its own std (effect size)
        sd = d.std() if d.std() > 0 else 1.0
        mean_es = d.mean() / sd
        lo, hi = tests[m]["boot_lo"] / sd, tests[m]["boot_hi"] / sd
        color = "green" if tests[m]["significant"] and tests[m]["beneficial"] else \
                ("crimson" if tests[m]["significant"] else "gray")
        ax.axhline(0, color="black", lw=1)
        ax.errorbar([0], [mean_es], yerr=[[mean_es - lo], [hi - mean_es]], fmt="o",
                    color=color, capsize=6, ms=10)
        star = "*" if tests[m]["significant"] else "n.s."
        ax.set_title(f"{m}\n(B$-$A) {star}")
        ax.set_xticks([])
        ax.set_ylabel("effect size (std units)")
    fig.suptitle("Fig 12. Statistical significance of Hawkes $-$ Poisson (95% bootstrap CI)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig12_significance.png"), bbox_inches="tight")
    plt.close(fig)
