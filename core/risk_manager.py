"""
risk_manager.py
---------------
Margin safety, inventory caps, quote throttling, and supervisory position closure.
"""

import logging
import time
from .exchange_client import BinanceClient
from .position_tracker import PositionTracker
from .order_manager import OrderManager

log = logging.getLogger("engine")


class RiskManager:
    def __init__(self, client: BinanceClient, tracker: PositionTracker, leverage: int,
                 base_qty_usd: float, symbol: str, order_mgr: OrderManager):
        self.client = client
        self.tracker = tracker
        self.leverage = leverage
        self.base_qty_usd = base_qty_usd
        self.symbol = symbol
        self.order_mgr = order_mgr

        # Cooldown state for margin failures
        self.margin_failure_count = 0
        self.last_margin_failure_time = 0.0

    # ------------------------------------------------------------------
    # Margin checks
    # ------------------------------------------------------------------
    async def check_margin_sufficient(self, price: float, qty: float) -> bool:
        """Verify that placing an order of given size leaves enough free margin."""
        try:
            account = await self.client.get("/fapi/v2/account", {})
            if not account:
                return True  # proceed if check fails
            available = float(account.get("availableBalance", 0))
            required = (price * qty) / self.leverage
            return available >= required * 1.05  # 5% buffer
        except Exception as e:
            log.debug(f"Margin check failed: {e}")
            return True

    async def can_afford_new_quote(self, mid: float, max_position_usd: float, step_size: float) -> bool:
        """Global margin check before quoting both sides."""
        try:
            account = await self.client.get("/fapi/v2/account", {})
            available = float(account.get("availableBalance", 0))
            qty = self._compute_qty(mid, max_position_usd, step_size)
            new_margin = (mid * qty * 2) / self.leverage
            min_free_margin = 2.0  # require at least $2 free after placing orders
            return (available - new_margin) >= min_free_margin
        except Exception as e:
            log.debug(f"Global margin check failed: {e}")
            return True

    # ------------------------------------------------------------------
    # Inventory cap
    # ------------------------------------------------------------------
    def enforce_inventory_cap(self, pos_side: str, price: float, multiple: float, step_size: float) -> bool:
        """Return True if adding to this side is allowed."""
        if price <= 0:
            return True
        base_qty = self.base_qty_usd / price
        max_allowed = base_qty * multiple
        if pos_side == "LONG" and self.tracker.long_qty >= max_allowed:
            log.debug(f"Inventory cap: LONG {self.tracker.long_qty:.0f} >= {max_allowed:.0f}")
            return False
        if pos_side == "SHORT" and self.tracker.short_qty >= max_allowed:
            log.debug(f"Inventory cap: SHORT {self.tracker.short_qty:.0f} >= {max_allowed:.0f}")
            return False
        return True

    def _compute_qty(self, price: float, max_usd: float, step_size: float) -> float:
        """Helper to compute rounded quantity."""
        raw = self.base_qty_usd / price
        steps = int(raw / step_size)
        qty = steps * step_size
        max_qty = max_usd / price
        return min(qty, max_qty)

    # ------------------------------------------------------------------
    # Supervisory close (LLM action)
    # ------------------------------------------------------------------
    async def close_side(self, side: str, long_tp, short_tp, long_sl, short_sl):
        """Market‑close the given side and cancel its associated TP/SL orders."""
        pos_qty = self.tracker.long_qty if side == "LONG" else self.tracker.short_qty
        if pos_qty <= 0:
            log.info(f"No {side} position to close")
            return

        # Cancel TP/SL orders for this side
        if side == "LONG":
            if long_tp:
                await self.order_mgr.cancel_order(long_tp.order_id)
            if long_sl:
                await self.order_mgr.cancel_algo_order(long_sl.order_id)
        else:
            if short_tp:
                await self.order_mgr.cancel_order(short_tp.order_id)
            if short_sl:
                await self.order_mgr.cancel_algo_order(short_sl.order_id)

        order_side = "SELL" if side == "LONG" else "BUY"
        try:
            r = await self.order_mgr.place_market_order(order_side, side, pos_qty)
            log.info(f"LLM commanded: Closed {side} side ({pos_qty} qty). Response: {r}")
            self.tracker.reset_side(side)
        except Exception as e:
            log.error(f"Failed to close {side}: {e}")

    # ------------------------------------------------------------------
    # Cooldown helpers (used by strategy engine)
    # ------------------------------------------------------------------
    def record_margin_failure(self):
        self.margin_failure_count += 1
        self.last_margin_failure_time = time.time()

    def get_quote_interval(self, now: float) -> float:
        """Return dynamic quote interval based on recent margin pressure."""
        if self.margin_failure_count >= 3 and (now - self.last_margin_failure_time) < 60:
            return 15.0
        return 5.0