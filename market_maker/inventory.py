r"""Inventory, cash and mark-to-market accounting with risk limits.

The market maker's wealth has two components: **cash** ``X`` and **inventory**
``q`` (signed shares) valued at the current mid. When the bid is hit the agent
*buys* one unit (``q`` up, cash down by the bid price); when the ask is lifted the
agent *sells* one unit (``q`` down, cash up by the ask price). The mark-to-market
(MtM) wealth is

    W(t) = X(t) + q(t) * S(t).                                           (7)

Risk control follows the original paper's spirit: a hard **maximum inventory**
``q_max`` (the agent stops quoting the side that would breach it), and a
**terminal liquidation** of the residual inventory at ``T``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Portfolio"]


@dataclass
class Portfolio:
    """Track inventory, cash and realised/MtM wealth with a hard inventory cap.

    Attributes
    ----------
    q_max : int
        Maximum absolute inventory (shares).
    order_size : int
        Shares per fill.
    inventory : int
        Current signed inventory (shares).
    cash : float
        Current cash ($).
    n_buys, n_sells : int
        Trade counts (for diagnostics).
    """

    q_max: int = 50
    order_size: int = 1
    inventory: int = 0
    cash: float = 0.0
    n_buys: int = field(default=0)
    n_sells: int = field(default=0)

    # -- risk limits ------------------------------------------------------- #
    def can_buy(self) -> bool:
        """Whether buying one unit keeps inventory within ``+q_max``."""
        return self.inventory + self.order_size <= self.q_max

    def can_sell(self) -> bool:
        """Whether selling one unit keeps inventory within ``-q_max``."""
        return self.inventory - self.order_size >= -self.q_max

    # -- fills ------------------------------------------------------------- #
    def on_bid_fill(self, bid_price: float) -> None:
        """Apply a bid fill: buy ``order_size`` shares at ``bid_price``."""
        self.inventory += self.order_size
        self.cash -= bid_price * self.order_size
        self.n_buys += 1

    def on_ask_fill(self, ask_price: float) -> None:
        """Apply an ask fill: sell ``order_size`` shares at ``ask_price``."""
        self.inventory -= self.order_size
        self.cash += ask_price * self.order_size
        self.n_sells += 1

    # -- valuation --------------------------------------------------------- #
    def mark_to_market(self, mid: float) -> float:
        """MtM wealth ``cash + inventory * mid``, eq. (7)."""
        return self.cash + self.inventory * mid

    def liquidate(self, mid: float) -> float:
        """Terminal liquidation: flatten inventory at ``mid``, return MtM wealth.

        The residual inventory is sold (or bought back) at the mid price -- a
        simplification of the paper's terminal condition that ignores the cost of
        crossing the spread on the final unwind.

        Parameters
        ----------
        mid : float
            Mid-price at the horizon ($).

        Returns
        -------
        float
            Final cash after liquidation (== MtM wealth before liquidation).
        """
        self.cash += self.inventory * mid
        if self.inventory > 0:
            self.n_sells += self.inventory
        elif self.inventory < 0:
            self.n_buys += -self.inventory
        self.inventory = 0
        return self.cash
