"""Classical Avellaneda--Stoikov (2008) market-making framework (from scratch).

This package implements the *original* AS model -- arithmetic Brownian mid-price,
the inventory-aware reservation price, the optimal bid/ask spread, an exponential
(Poisson) execution model, inventory/cash accounting and an event-driven
simulator. No Hawkes process and no reinforcement learning appear here: this is
the deterministic-intensity baseline against which the later Hawkes extension is
compared.

Sub-modules
-----------
avellaneda_stoikov : reservation price, optimal spread, quote construction,
    and the data calibration of ``sigma`` and ``kappa``.
execution_model    : exponential market-order arrival / fill model.
inventory          : inventory, cash and mark-to-market accounting + risk limits.
simulator          : the event-driven simulation loop and episode plotting.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ASParameters", "DATA_CALIBRATED"]


@dataclass(frozen=True)
class ASParameters:
    """Parameters of the Avellaneda--Stoikov model.

    Attributes
    ----------
    gamma : float
        Inventory **risk aversion** (units 1/$). Larger ``gamma`` => the agent
        dislikes inventory more, skews quotes harder and widens the spread.
    sigma : float
        Mid-price **volatility** in $ per sqrt(second); ``dS = sigma dW``.
        Calibrated from LOBSTER data.
    kappa : float
        Order-arrival **decay** (units 1/$). Market-order arrival intensity falls
        off as ``exp(-kappa * delta)`` with distance ``delta`` from the mid; a
        large ``kappa`` means liquidity takers are reluctant to reach far, so
        posting deep earns few fills. Calibrated from LOBSTER data.
    A : float
        Base arrival **intensity** (events/second) at the mid (``delta = 0``); the
        scale of ``lambda(delta) = A exp(-kappa delta)``.
    T : float
        Trading **horizon** in seconds; the terminal time at which inventory is
        liquidated. ``(T - t)`` drives the inventory-risk urgency.
    q_max : int
        Maximum absolute inventory (risk limit, in shares).
    order_size : int
        Number of shares per fill (the original paper uses unit size).
    """

    gamma: float = 0.1
    sigma: float = 0.03477
    kappa: float = 23.96
    A: float = 1.445
    T: float = 600.0
    q_max: int = 50
    order_size: int = 1


# Defaults calibrated from LOBSTER AAPL 2012-06-21 (see avellaneda_stoikov.py:
# calibrate_sigma / calibrate_kappa). sigma in $/sqrt(s); kappa in 1/$.
DATA_CALIBRATED = ASParameters()
