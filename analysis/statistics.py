r"""Statistical significance of the robust-vs-baseline differences (Part 6).

For a paired metric (same order flow, common random numbers) with values
:math:`a_i` (baseline) and :math:`b_i` (robust), form the differences
:math:`d_i=b_i-a_i` and test whether their mean is significantly non-zero using
four complementary procedures:

* **paired t-test** -- parametric, assumes approximately normal differences;
* **Wilcoxon signed-rank test** -- non-parametric, robust to non-normality;
* **percentile bootstrap CI** -- distribution-free 95% CI of the mean difference;
* **Cohen's d (paired)** -- the standardised effect size
  :math:`\bar d / s_d`, reported alongside the p-values because with large
  :math:`n` a *statistically* significant difference can still be *practically*
  negligible.

The Cohen's d convention used here (mean difference over the standard deviation of
the differences) is the effect size appropriate to a paired design. Magnitude
guides (Cohen 1988): :math:`|d|<0.2` negligible, ``~0.2`` small, ``~0.5`` medium,
:math:`\ge 0.8` large.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["PairedTest", "cohens_d_paired", "bootstrap_ci", "paired_comparison"]


@dataclass(frozen=True)
class PairedTest:
    """Full significance summary for one paired difference ``b - a``.

    Attributes
    ----------
    mean_diff : float
        Mean paired difference :math:`\\bar d`.
    t_stat, t_p : float
        Paired t-test statistic and (two-sided) p-value.
    wilcoxon_stat, wilcoxon_p : float
        Wilcoxon signed-rank statistic and p-value.
    boot_lo, boot_hi : float
        95% percentile-bootstrap CI of the mean difference.
    cohens_d : float
        Paired Cohen's d effect size.
    n : int
        Number of finite paired observations.
    significant : bool
        ``True`` iff both p-values ``< 0.05`` and the bootstrap CI excludes 0.
    """

    mean_diff: float
    t_stat: float
    t_p: float
    wilcoxon_stat: float
    wilcoxon_p: float
    boot_lo: float
    boot_hi: float
    cohens_d: float
    n: int
    significant: bool


def cohens_d_paired(diffs: np.ndarray) -> float:
    r"""Paired Cohen's d: :math:`\bar d / s_d` (``0`` if the spread is zero)."""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    n = d.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    means = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_comparison(a: np.ndarray, b: np.ndarray, *, n_boot: int = 10000,
                      seed: int = 0) -> PairedTest:
    """Run all paired tests + Cohen's d for ``b - a``.

    Parameters
    ----------
    a, b : numpy.ndarray
        Paired baseline and treatment metric values (equal length).
    n_boot : int, optional
        Bootstrap resamples for the CI.
    seed : int, optional
        Bootstrap RNG seed.

    Returns
    -------
    PairedTest
        The full significance summary.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    d = b - a
    n = d.shape[0]

    t_stat, t_p = stats.ttest_rel(b, a)
    try:
        w_stat, w_p = stats.wilcoxon(d, zero_method="wilcox",
                                     alternative="two-sided")
    except ValueError:                       # all-zero differences
        w_stat, w_p = np.nan, 1.0
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    d_eff = cohens_d_paired(d)
    significant = bool(t_p < 0.05 and w_p < 0.05 and (lo > 0 or hi < 0))
    return PairedTest(
        mean_diff=float(d.mean()), t_stat=float(t_stat), t_p=float(t_p),
        wilcoxon_stat=float(w_stat), wilcoxon_p=float(w_p),
        boot_lo=lo, boot_hi=hi, cohens_d=d_eff, n=n, significant=significant,
    )
