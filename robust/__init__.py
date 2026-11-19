r"""Phase 6 -- Uncertainty-aware (robust) Hawkes market making.

This package turns the *point-estimate* Hawkes market maker of Phase 5 into an
**uncertainty-aware** one. The Phase-2 maximum-likelihood fit produced a single
parameter vector :math:`\hat\theta = (\hat\mu, \hat\alpha, \hat\beta)`; here we
treat :math:`\theta` as a random variable with an estimation covariance
:math:`\Sigma_\theta`, propagate that uncertainty into the conditional intensity
:math:`\lambda(t)`, and finally into the market maker's quotes.

Modules
-------
covariance
    Two independent estimators of :math:`\Sigma_\theta`: the **observed Fisher
    information** (inverse Hessian of the negative log-likelihood) and a
    **parametric bootstrap** empirical covariance.
bootstrap
    The parametric-bootstrap driver (simulate from :math:`\hat\theta`, refit,
    repeat) used by :mod:`robust.covariance`.
delta_method
    First-order (delta-method) propagation of :math:`\Sigma_\theta` through
    :math:`\lambda(t) = f(\mu, \alpha, \beta)` to a variance and confidence band
    for the intensity.
uncertainty
    Construction of *lower / central / upper* intensity beliefs and a stateful
    intensity-plus-variance evaluator used by the robust quoting policies.
robust_quotes
    Two robust quoting heuristics: (A) a conservative (worst-case) intensity
    belief and (B) a variance-penalised spread
    :math:`\delta_{\text{robust}} = \delta_H + c\sqrt{\operatorname{Var}\lambda}`.

Provenance tags used throughout the docstrings and the report, following the
Phase-4/5 convention:

    [THEOREM]      an exact mathematical result (Hawkes theory, MLE asymptotics).
    [APPROX]       a first-order / asymptotic approximation (e.g. the delta method).
    [HEURISTIC]    a proposed rule with no optimality claim (the robust spread).
    [EMPIRICAL]    an observation from simulation / historical replay.
"""
from __future__ import annotations

__all__ = [
    "covariance",
    "bootstrap",
    "delta_method",
    "uncertainty",
    "robust_quotes",
]
