r"""Estimation covariance of the Hawkes parameters :math:`\theta=(\mu,\alpha,\beta)`.

Two *independent* estimators of :math:`\Sigma_\theta` are provided so they can be
cross-checked (Phase 6, Part 1):

**Method 1 -- Observed Fisher information.**  [THEOREM + APPROX]
    Under standard regularity conditions the maximum-likelihood estimator is
    asymptotically normal,

    .. math::

        \sqrt{N}\,(\hat\theta-\theta_0)\;\xrightarrow{d}\;
        \mathcal N\!\big(0,\; \mathcal I(\theta_0)^{-1}\big),

    where :math:`\mathcal I` is the Fisher information. In practice we use the
    **observed** information -- the Hessian of the negative log-likelihood
    evaluated at the fit, :math:`H = \nabla^2(-\ell)(\hat\theta)` -- and take

    .. math::  \Sigma_\theta \approx H^{-1}.

    The asymptotic-normality *theorem* is exact; using the observed information
    at :math:`\hat\theta` in place of :math:`\mathcal I(\theta_0)` and inverting a
    finite-sample Hessian is the *approximation* (labelled [APPROX]).

    The Hessian is formed by central finite differences of the **analytic**
    gradient supplied by :func:`hawkes.estimator.neg_log_likelihood`, which is far
    more accurate than second differences of the objective. Because the three
    parameters differ by ~three orders of magnitude (:math:`\mu\sim0.6`,
    :math:`\alpha,\beta\sim10^2`), *relative* step sizes are used.

**Method 2 -- Parametric bootstrap.**  [THEOREM/EMPIRICAL]
    Simulate :math:`B` independent event streams from the fitted model on the same
    horizon, re-estimate :math:`\theta` on each, and take the empirical covariance
    of the :math:`B` estimates (see :mod:`robust.bootstrap`). This makes no
    Gaussian assumption and captures finite-sample skew, at the cost of :math:`B`
    full MLE refits.

A close agreement between the two is evidence that the asymptotic Gaussian
approximation is adequate at this sample size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hawkes.estimator import neg_log_likelihood

__all__ = [
    "CovarianceResult",
    "observed_information",
    "fisher_covariance",
    "correlation_from_covariance",
]

FloatArray = npt.NDArray[np.float64]
PARAM_NAMES = ("mu", "alpha", "beta")


@dataclass(frozen=True)
class CovarianceResult:
    r"""A parameter covariance estimate with derived quantities.

    Attributes
    ----------
    theta : numpy.ndarray
        The parameter vector :math:`(\mu, \alpha, \beta)` the covariance refers to.
    cov : numpy.ndarray
        The :math:`3\times3` covariance matrix :math:`\Sigma_\theta`.
    method : str
        ``"fisher"`` or ``"bootstrap"``.
    n_events : int
        Number of events the estimate is based on.
    extra : dict
        Method-specific diagnostics (e.g. the Hessian, bootstrap sample size).
    """

    theta: FloatArray
    cov: FloatArray
    method: str
    n_events: int
    extra: dict

    @property
    def std_errors(self) -> FloatArray:
        r"""Marginal standard errors :math:`\sqrt{\operatorname{diag}\Sigma}`."""
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, np.inf))

    @property
    def correlation(self) -> FloatArray:
        """Correlation matrix implied by the covariance."""
        return correlation_from_covariance(self.cov)

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-parameter ``{estimate, std_error, cv_percent}`` mapping."""
        se = self.std_errors
        out: dict[str, dict[str, float]] = {}
        for i, name in enumerate(PARAM_NAMES):
            est = float(self.theta[i])
            out[name] = {
                "estimate": est,
                "std_error": float(se[i]),
                "cv_percent": float(100.0 * se[i] / est) if est != 0 else np.nan,
            }
        return out


def correlation_from_covariance(cov: FloatArray) -> FloatArray:
    """Convert a covariance matrix to a correlation matrix (safe for zeros)."""
    d = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    denom = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def observed_information(
    theta: FloatArray,
    t: FloatArray,
    T: float,
    *,
    rel_step: float = 1e-3,
) -> FloatArray:
    r"""Observed Fisher information (Hessian of the negative log-likelihood).

    The Hessian is built column-by-column from central finite differences of the
    analytic gradient :math:`g(\theta)=\nabla(-\ell)(\theta)`:

    .. math::

        H_{:,j} \approx \frac{g(\theta + h_j e_j) - g(\theta - h_j e_j)}{2 h_j},
        \qquad h_j = \texttt{rel\_step}\cdot\max(|\theta_j|, 1).

    The result is symmetrised (:math:`H \leftarrow \tfrac12(H+H^\top)`) to remove
    finite-difference asymmetry.

    .. note::

       This returns the Hessian in the *natural* parameterisation. Because the
       three parameters span ~three decades, its condition number is enormous and
       a direct inverse is numerically unreliable;
       :func:`fisher_covariance` therefore inverts a **rescaled** Hessian (all
       parameters :math:`O(1)`) and transforms back. This function is exposed
       mainly for inspection.

    Parameters
    ----------
    theta : numpy.ndarray
        Parameter vector :math:`(\mu, \alpha, \beta)`, normally the MLE.
    t : numpy.ndarray
        Sorted, strictly increasing event times on ``[0, T]``.
    T : float
        Observation horizon.
    rel_step : float, optional
        Relative finite-difference step (default ``1e-3``).

    Returns
    -------
    numpy.ndarray
        The symmetric :math:`3\times3` observed-information (Hessian) matrix.
    """
    theta = np.asarray(theta, dtype=float)
    n = theta.shape[0]
    H = np.zeros((n, n))
    for j in range(n):
        h = rel_step * max(abs(theta[j]), 1.0)
        tp = theta.copy(); tp[j] += h
        tm = theta.copy(); tm[j] -= h
        _, gp = neg_log_likelihood(tp, t, T)
        _, gm = neg_log_likelihood(tm, t, T)
        H[:, j] = (gp - gm) / (2.0 * h)
    return 0.5 * (H + H.T)


def _scaled_hessian(
    theta: FloatArray, t: FloatArray, T: float, rel_step: float,
) -> tuple[FloatArray, FloatArray]:
    r"""Hessian in rescaled coordinates :math:`\varphi=\theta/\hat\theta`.

    With :math:`D=\operatorname{diag}(\hat\theta)` the chain rule gives
    :math:`H_\varphi = D\,H_\theta\,D`, whose entries are all :math:`O(N)` and
    whose condition number is many orders of magnitude smaller than
    :math:`H_\theta`'s -- so it can be inverted reliably. Returns
    ``(H_phi, D_diag)``.

    The scaled Hessian is formed directly (rather than via
    :func:`observed_information`) so that the finite-difference *and* the matrix
    entries live in the well-scaled coordinate system.
    """
    theta = np.asarray(theta, dtype=float)
    s = np.where(np.abs(theta) > 0, np.abs(theta), 1.0)  # scale = |theta_hat|

    def grad_phi(phi: FloatArray) -> FloatArray:
        # g_phi = s ⊙ g_theta  (chain rule, theta = s ⊙ phi)
        _, g = neg_log_likelihood(s * phi, t, T)
        return s * g

    n = theta.shape[0]
    H = np.zeros((n, n))
    phi0 = np.ones(n)
    for j in range(n):
        ep = phi0.copy(); ep[j] += rel_step
        em = phi0.copy(); em[j] -= rel_step
        H[:, j] = (grad_phi(ep) - grad_phi(em)) / (2.0 * rel_step)
    H = 0.5 * (H + H.T)
    return H, s


def fisher_covariance(
    theta: FloatArray,
    t: FloatArray,
    T: float,
    *,
    rel_step: float = 1e-3,
) -> CovarianceResult:
    r"""Estimate :math:`\Sigma_\theta \approx H^{-1}` via the observed information.

    To avoid the catastrophic ill-conditioning of the natural-coordinate Hessian
    (parameters spanning ~three decades), the Hessian is formed and inverted in
    the rescaled coordinates :math:`\varphi=\theta/\hat\theta` and mapped back:

    .. math::

        \Sigma_\theta = D\,H_\varphi^{-1}\,D, \qquad
        D = \operatorname{diag}(\hat\theta).

    Parameters
    ----------
    theta : numpy.ndarray
        Parameter vector :math:`(\mu, \alpha, \beta)`, normally the MLE.
    t : numpy.ndarray
        Sorted, strictly increasing event times on ``[0, T]``.
    T : float
        Observation horizon.
    rel_step : float, optional
        Relative finite-difference step for the (scaled) Hessian.

    Returns
    -------
    CovarianceResult
        With ``method="fisher"``; ``extra`` holds the scaled Hessian, the
        back-transformed natural Hessian, the (scaled) condition number and the
        minimum eigenvalue of the *scaled* Hessian -- a positive value confirms a
        genuine interior maximum of the likelihood.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the Hessian is singular.
    """
    theta = np.asarray(theta, dtype=float)
    H_phi, s = _scaled_hessian(theta, t, T, rel_step)
    cov_phi = np.linalg.inv(H_phi)
    cov_phi = 0.5 * (cov_phi + cov_phi.T)
    D = np.diag(s)
    cov = D @ cov_phi @ D                    # Sigma_theta = D Sigma_phi D
    cov = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(H_phi)
    extra = {
        "hessian_scaled": H_phi,
        "hessian": np.diag(1.0 / s) @ H_phi @ np.diag(1.0 / s),
        "cond_number_scaled": float(np.linalg.cond(H_phi)),
        "min_eigenvalue_scaled": float(eigvals.min()),
        "rel_step": rel_step,
    }
    return CovarianceResult(theta=theta, cov=cov, method="fisher",
                            n_events=int(t.shape[0]), extra=extra)
