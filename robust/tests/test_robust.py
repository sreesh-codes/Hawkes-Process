r"""Unit tests for the Phase-6 robust market-making package.

Run with ``python -m pytest robust/tests`` (or ``python -m robust.tests.test_robust``
for a dependency-free smoke run). The tests check the mathematical properties that
the report relies on, not just that the code executes:

* the observed-Fisher covariance is symmetric positive-definite and its scaled
  Hessian is well-conditioned;
* the parametric bootstrap recovers the true parameters (approximately unbiased);
* the delta-method intensity matches the reference evaluator and its gradient
  matches a finite-difference check;
* the belief construction moves ``lambda_bar`` by the intended amount and stays
  stationary;
* the variance-penalised policy reduces **exactly** to the Phase-5 Hawkes spread
  at ``c = 0`` and widens monotonically in ``c``.
"""
from __future__ import annotations

import numpy as np

from robust.data import load_streams
from robust.covariance import fisher_covariance
from robust.bootstrap import run_parametric_bootstrap
from robust.delta_method import (confidence_band, intensity_variance_on_grid,
                                 lambda_bar, s_and_q_on_grid)
from robust.uncertainty import build_beliefs, VarianceHawkes
from robust.robust_quotes import RobustHawkesASPolicy
from market_maker import ASParameters
from market_maker.hawkes_execution import intensity_on_grid
from market_maker.hawkes_market_maker import HawkesASPolicy


def _stream():
    return load_streams()["all"]


def test_fisher_spd_and_conditioned():
    s = _stream()
    fc = fisher_covariance(s.theta, s.t, s.T)
    # symmetric
    assert np.allclose(fc.cov, fc.cov.T, atol=1e-8)
    # positive definite
    assert np.all(np.linalg.eigvalsh(fc.cov) > 0)
    # scaled Hessian well-conditioned and a genuine maximum
    assert fc.extra["min_eigenvalue_scaled"] > 0
    assert fc.extra["cond_number_scaled"] < 1e4


def test_bootstrap_approx_unbiased():
    s = _stream()
    boot = run_parametric_bootstrap(s.theta, s.T, n_boot=60, n_starts=2)
    assert boot.n_success > 40
    # relative bias small on each parameter
    rel_bias = np.abs(boot.bias) / s.theta
    assert np.all(rel_bias < 0.1)


def test_delta_method_matches_reference():
    s = _stream()
    fc = fisher_covariance(s.theta, s.t, s.T)
    grid = np.linspace(0, 60, 1500)
    ev = s.t[s.t <= 60]
    lam, var = intensity_variance_on_grid(grid, ev, s.theta, fc.cov)
    lam_ref = intensity_on_grid(grid, ev, *s.theta)
    assert np.allclose(lam, lam_ref, atol=1e-9)
    assert np.all(var >= 0)


def test_intensity_gradient_beta_finite_difference():
    # d lambda / d beta = -alpha Q  (check at a point with excitation)
    s = _stream()
    mu, alpha, beta = s.theta
    grid = np.array([50.0])
    ev = s.t[s.t <= 50]
    S, Q = s_and_q_on_grid(grid, ev, beta)
    analytic = -alpha * Q[0]
    h = beta * 1e-5
    Sp, _ = s_and_q_on_grid(grid, ev, beta + h)
    Sm, _ = s_and_q_on_grid(grid, ev, beta - h)
    fd = alpha * (Sp[0] - Sm[0]) / (2 * h)
    assert abs(analytic - fd) < 1e-3 * max(abs(analytic), 1.0)


def test_beliefs_move_lambda_bar_and_stationary():
    s = _stream()
    fc = fisher_covariance(s.theta, s.t, s.T)
    beliefs = build_beliefs(s.theta, fc.cov, z=1.96)
    lb = {k: lambda_bar(v) for k, v in beliefs.items()}
    assert lb["lower"] < lb["central"] < lb["upper"]
    for th in beliefs.values():
        assert th[0] > 0 and th[1] > 0 and th[2] > 0
        assert th[1] / th[2] < 1.0            # stationary


def test_robust_reduces_to_hawkes_at_c0():
    params = ASParameters()
    lbb, lbs = 0.5, 0.55
    hawkes = HawkesASPolicy(params, lbb, lbs)
    robust0 = RobustHawkesASPolicy(params, lbb, lbs, c=0.0)
    qh = hawkes.quote(585.0, 3.0, 100.0, 1.2, 0.9)
    qr = robust0.quote(585.0, 3.0, 100.0, 1.2, 0.9, sd_buy=2.0, sd_sell=2.0)
    assert abs(qh.bid - qr.bid) < 1e-12
    assert abs(qh.ask - qr.ask) < 1e-12


def test_penalty_monotone_in_c():
    params = ASParameters()
    lbb, lbs = 0.5, 0.55
    spreads = []
    for c in [0.0, 0.5, 1.0, 2.0]:
        pol = RobustHawkesASPolicy(params, lbb, lbs, c=c)
        q = pol.quote(585.0, 0.0, 100.0, 1.2, 0.9, sd_buy=2.0, sd_sell=2.0)
        spreads.append(q.spread)
    assert all(x < y for x, y in zip(spreads, spreads[1:]))


def test_variance_hawkes_matches_delta_grid():
    # online VarianceHawkes std should match the grid delta-method std at a query
    s = _stream()
    fc = fisher_covariance(s.theta, s.t, s.T)
    ev = s.t[s.t <= 30]
    vh = VarianceHawkes(s.theta[0], s.theta[1], s.theta[2], fc.cov)
    for e in ev:
        vh.register_event(float(e))
    t_query = 30.0
    _, sd_online = vh.intensity_and_std(t_query)
    grid = np.array([t_query])
    _, var_grid = intensity_variance_on_grid(grid, ev, s.theta, fc.cov)
    assert abs(sd_online - np.sqrt(var_grid[0])) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
