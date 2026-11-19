r"""Reconstruct the exact event streams the Phase-2/5 Hawkes fits were built on.

The likelihood, the observed Fisher information and the delta-method bands all
need the *same* event times and horizon :math:`T` that produced the stored
maximum-likelihood estimates in ``results/hawkes_side_params.json``. Getting this
subtly wrong is fatal: if the origin shift or the horizon is defined
inconsistently, the last event can fall a fraction of a second **beyond** ``T``,
making :math:`T-t_i<0` and :math:`e^{-\beta(T-t_i)}` overflow -- the compensator
then explodes to :math:`\sim10^{52}` and the whole inference is nonsense.

The Phase-2 convention (``hawkes_mle.load_trade_times``) is reproduced exactly:

* executed trades are event types 4 (visible) and 5 (hidden);
* **simultaneous executions** (one marketable order sweeping several resting
  orders share a timestamp) are collapsed into a single event via
  :func:`numpy.unique`;
* times are shifted so the **first trade** sits at :math:`t_0=0`; and
* the horizon is :math:`T=t_{N-1}` (the last trade time).

For the per-side streams (buy-/sell-initiated market orders) the same rules are
applied to each aggressor stream separately, matching the per-side ``T`` values
stored in the calibration JSON.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["EventStream", "load_streams", "load_side_params"]

FloatArray = npt.NDArray[np.float64]

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PREFIX = os.path.join(
    BASE, "LOBSTER_SampleFile_AAPL_2012-06-21_10",
    "AAPL_2012-06-21_34200000_57600000_",
)
SIDE_PARAMS = os.path.join(BASE, "results", "hawkes_side_params.json")
_MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]


@dataclass(frozen=True)
class EventStream:
    r"""A single-side event stream with its fitted parameters.

    Attributes
    ----------
    name : str
        ``"all"``, ``"buy"`` or ``"sell"``.
    t : numpy.ndarray
        Sorted, strictly increasing event times on ``[0, T]`` (first event at 0).
    T : float
        Observation horizon ``t[-1]``.
    theta : numpy.ndarray
        Stored MLE :math:`(\hat\mu, \hat\alpha, \hat\beta)` for this stream.
    """

    name: str
    t: FloatArray
    T: float
    theta: FloatArray

    @property
    def n_events(self) -> int:
        return int(self.t.shape[0])


def load_side_params(path: str = SIDE_PARAMS) -> dict:
    """Load the Phase-5 per-side Hawkes calibration JSON."""
    with open(path) as f:
        return json.load(f)


def _prepare(times: FloatArray) -> tuple[FloatArray, float]:
    """Collapse ties, shift the first event to 0, return ``(t, T=t[-1])``."""
    t = np.unique(np.asarray(times, dtype=float))  # sort + de-duplicate ties
    t = t - t[0]
    return t, float(t[-1])


def load_streams(data_prefix: str = DATA_PREFIX,
                 params_path: str = SIDE_PARAMS) -> dict[str, EventStream]:
    r"""Reconstruct the ``all`` / ``buy`` / ``sell`` executed-trade streams.

    Parameters
    ----------
    data_prefix : str, optional
        LOBSTER file prefix (``..._message_10.csv`` / ``..._orderbook_10.csv``).
    params_path : str, optional
        Path to the per-side calibration JSON.

    Returns
    -------
    dict[str, EventStream]
        Keyed by ``"all"``, ``"buy"``, ``"sell"``. Each stream's ``(t, T)``
        matches the horizon used for the stored fit, so evaluating the Hawkes
        likelihood at ``theta`` is numerically well-posed.
    """
    sp = load_side_params(params_path)
    msg = pd.read_csv(data_prefix + "message_10.csv", header=None, names=_MSG_COLS)
    trades = msg.loc[msg["event_type"].isin([4, 5])]
    time = trades["time"].to_numpy()
    direction = trades["direction"].to_numpy()

    # LOBSTER 'direction' at a trade: -1 == buyer-initiated (buy market order),
    # +1 == seller-initiated (sell market order) -- matching load_lobster_events.
    masks = {
        "all": np.ones(time.shape[0], dtype=bool),
        "buy": direction == -1,
        "sell": direction == +1,
    }
    out: dict[str, EventStream] = {}
    for name, mask in masks.items():
        t, T = _prepare(time[mask])
        theta = np.array([sp[name]["mu"], sp[name]["alpha"], sp[name]["beta"]])
        out[name] = EventStream(name=name, t=t, T=T, theta=theta)
    return out
