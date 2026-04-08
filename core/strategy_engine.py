"""
strategy_engine.py
------------------
Core market-making loop with Avellaneda-Stoikov inventory-aware pricing,
LLM supervisory actions, dynamic spread/skew/gamma, and exit management.
"""

import asyncio
import logging
import time
from .order_manager import OrderManager, TpOrder, SlOrder, ActiveOrder
from .position_tracker import PositionTracker
from .risk_manager import RiskManager
from .exchange_client import BinanceClient
from core.shared_state import SharedState

log = logging.getLogger("engine")

LEVERAGE = 10
COMMISSION_BPS = 9.0
TP_CHECK_INTERVAL = 10.0
MIN_PRICE_MOVE_BPS = 0.9
FALLBACK_SPREAD_BPS = 40.0      # increased to ensure profitability
FALLBACK_SL_BPS = 20.0
FALLBACK_TP_BPS = 45.0
FALLBACK_POSITION_MULTIPLE = 1.1
DEFAULT_GAMMA = 0.01            # risk-aversion coefficient


class StrategyEngine:
    def __init__(self, symbol: str, state: SharedState, qty_usd: float, client: BinanceClient,
                 order_mgr: OrderManager, tracker: PositionTracker, risk_mgr: RiskManager,
                 step_size: float):
        self.symbol = symbol
        self.state = state
        self.qty_usd = qty_usd
        self.client = client
        self.order_mgr = order_mgr
        self.tracker = tracker
        self.risk_mgr = risk_mgr
        self.step_size = step_size

        # Entry order state
        self._bid_order: ActiveOrder | None = None
        self._ask_order: ActiveOrder | None = None
        self._last_bid: float = 0.0
        self._last_ask: float = 0.0
        self._tick_count: int = 0
        self._consecutive_failures: int = 0
        self._last_quote_time: float = 0.0
        self._min_quote_interval: float = 5.0
        self._margin_failure_count: int = 0
        self._last_margin_failure_time: float = 0.0
        self._last_tp_check: float = 0.0

        # Exit order state
        self._long_tp: TpOrder | None = None
        self._short_tp: TpOrder | None = None
        self._long_sl: SlOrder | None = None
        self._short_sl: SlOrder | None = None

        # Runtime flag
        self._running = True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self):
        while self._running:
            market = self.state.get_market_snapshot()
            params = self.state.get_params_snapshot()

            if market.mid == 0:
                await asyncio.sleep(0.1)
                continue

            # ----- LLM Supervisory Actions -----
            action = getattr(params, 'action', 'CONTINUE')
            if action == "PAUSE":
                if self._bid_order or self._ask_order:
                    log.info("LLM commanded PAUSE – cancelling entry quotes")
                    await self._cancel_entry_orders()
                await asyncio.sleep(1.0)
                continue

            if action == "CLOSE_LONG":
                await self.risk_mgr.close_side("LONG", self._long_tp, self._short_tp,
                                               self._long_sl, self._short_sl)
                self._long_tp = self._long_sl = None
                await asyncio.sleep(1.0)
                continue

            if action == "CLOSE_SHORT":
                await self.risk_mgr.close_side("SHORT", self._long_tp, self._short_tp,
                                               self._long_sl, self._short_sl)
                self._short_tp = self._short_sl = None
                await asyncio.sleep(1.0)
                continue

            if action == "FLATTEN":
                await self.risk_mgr.close_side("LONG", self._long_tp, self._short_tp,
                                               self._long_sl, self._short_sl)
                await self.risk_mgr.close_side("SHORT", self._long_tp, self._short_tp,
                                               self._long_sl, self._short_sl)
                self._long_tp = self._long_sl = self._short_tp = self._short_sl = None
                await asyncio.sleep(1.0)
                continue

            # ----- Normal Market Making with AS Pricing -----
            await self._requote(market.mid, params)

            # ----- Periodic TP/SL Sync -----
            now = time.time()
            if now - self._last_tp_check >= TP_CHECK_INTERVAL:
                self._last_tp_check = now
                await self._sync_exits(params)

            self._tick_count += 1
            await asyncio.sleep(0.0)

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Avellaneda-Stoikov (AS) pricing model
    # ------------------------------------------------------------------

    def _compute_as_quotes(self, mid: float, spread_bps: float, gamma: float) -> tuple[float, float]:
        net_inventory = self.tracker.long_qty - self.tracker.short_qty
        reservation = mid - gamma * net_inventory
        half_spread = (spread_bps / 10_000) / 2 * mid

        bid = max(reservation - half_spread, mid * 0.9)   # 不低于 mid 的 90%
        ask = min(reservation + half_spread, mid * 1.1)   # 不高于 mid 的 110%

        if mid >= 1: precision = 4
        elif mid >= 0.1: precision = 5
        elif mid >= 0.01: precision = 6
        elif mid >= 0.001: precision = 7
        else: precision = 8

        return round(bid, precision), round(ask, precision)

    def _has_moved(self, new_bid: float, new_ask: float, mid: float) -> bool:
        if self._last_bid == 0:
            return True
        bid_move = abs(new_bid - self._last_bid) / mid * 10_000
        ask_move = abs(new_ask - self._last_ask) / mid * 10_000
        return max(bid_move, ask_move) >= MIN_PRICE_MOVE_BPS

    def _compute_qty(self, price: float, max_usd: float) -> float:
        raw = self.qty_usd / price
        steps = int(raw / self.step_size)
        qty = steps * self.step_size
        max_qty = max_usd / price
        return min(qty, max_qty)

    def _round_price(self, price: float, mid: float) -> float:
        if mid >= 1:
            return round(price, 4)
        elif mid >= 0.1:
            return round(price, 5)
        elif mid >= 0.01:
            return round(price, 6)
        elif mid >= 0.001:
            return round(price, 7)
        return round(price, 8)

    # ------------------------------------------------------------------
    # Quoting entry orders
    # ------------------------------------------------------------------
    async def _requote(self, mid: float, params):
        now = time.time()
        if now - self._last_quote_time < self._min_quote_interval:
            return
        self._last_quote_time = now

        if self._margin_failure_count >= 3 and (now - self._last_margin_failure_time) < 60:
            self._min_quote_interval = 15.0
        else:
            self._min_quote_interval = 5.0

        if not await self.risk_mgr.can_afford_new_quote(mid, params.max_position_usd, self.step_size):
            log.debug("Skipping quote – insufficient free margin")
            return

        # Get parameters from LLM with hard floors
        spread_bps = max(getattr(params, 'spread_bps', FALLBACK_SPREAD_BPS), 30.0)
        gamma = getattr(params, 'gamma', DEFAULT_GAMMA)

        # Compute inventory-aware quotes
        bid_price, ask_price = self._compute_as_quotes(mid, spread_bps, gamma)

        if not self._has_moved(bid_price, ask_price, mid):
            return

        qty = self._compute_qty(mid, params.max_position_usd)
        if qty <= 0:
            return

        await self._cancel_entry_orders()
        bid_ok = await self._place_entry_order("BUY", "LONG", bid_price, qty)
        ask_ok = await self._place_entry_order("SELL", "SHORT", ask_price, qty)

        if bid_ok and ask_ok:
            self._consecutive_failures = 0
            self._last_bid = bid_price
            self._last_ask = ask_price
            log.debug(
                f"Quoted BID={bid_price} ASK={ask_price} "
                f"spread={(ask_price-bid_price)/mid*10000:.1f}bps "
                f"net_inv={self.tracker.long_qty - self.tracker.short_qty:.0f} gamma={gamma:.3f}"
            )
        else:
            self._consecutive_failures += 1
            backoff = min(2 ** self._consecutive_failures, 60)
            log.warning(f"Order placement failed {self._consecutive_failures}x — backing off {backoff}s")
            await asyncio.sleep(backoff)

    async def _place_entry_order(self, side: str, pos_side: str, price: float, qty: float) -> bool:
        params = self.state.get_params_snapshot()
        position_multiple = getattr(params, 'position_multiple', FALLBACK_POSITION_MULTIPLE)

        if not self.risk_mgr.enforce_inventory_cap(pos_side, price, position_multiple, self.step_size):
            log.debug(f"Inventory cap reached for {pos_side}")
            return False

        if not await self.risk_mgr.check_margin_sufficient(price, qty):
            self._margin_failure_count += 1
            self._last_margin_failure_time = time.time()
            return False

        order = await self.order_mgr.place_entry_order(side, pos_side, price, qty)
        if order:
            if pos_side == "LONG":
                self.tracker.increment_long(qty)
            else:
                self.tracker.increment_short(qty)
            if side == "BUY":
                self._bid_order = order
            else:
                self._ask_order = order
            return True
        return False

    async def _cancel_entry_orders(self):
        for order in [self._bid_order, self._ask_order]:
            if order:
                await self.order_mgr.cancel_order(order.order_id)
        self._bid_order = self._ask_order = None

    # ------------------------------------------------------------------
    # Exit management (TP & SL)
    # ------------------------------------------------------------------
    async def _sync_exits(self, params):
        positions = await self.order_mgr.fetch_positions()
        if positions is None:
            return
        self.tracker.update_from_api(positions, self.symbol)

        tp_bps = max(getattr(params, 'tp_bps', FALLBACK_TP_BPS), 30.0)
        sl_bps = max(getattr(params, 'sl_bps', FALLBACK_SL_BPS), 15.0)

        await self._sync_long_exit(tp_bps, sl_bps)
        await self._sync_short_exit(tp_bps, sl_bps)

    async def _sync_long_exit(self, tp_bps: float, sl_bps: float):
        qty = self.tracker.long_qty
        entry = self.tracker.long_entry

        if qty > 0:
            desired_tp = self._round_price(entry * (1 + tp_bps / 10_000), entry)
            desired_sl = self._round_price(entry * (1 - sl_bps / 10_000), entry)
            desired_qty = int(qty / self.step_size) * self.step_size

            # TP
            if self._long_tp:
                if (abs(self._long_tp.entry_price - entry) > 1e-8 or
                    abs(self._long_tp.qty - desired_qty) > self.step_size):
                    await self.order_mgr.cancel_order(self._long_tp.order_id)
                    self._long_tp = None
                elif not await self.order_mgr.verify_order_exists(self._long_tp.order_id):
                    pnl = (self._long_tp.tp_price - entry) * qty
                    self.tracker.add_pnl(pnl)
                    log.info(f"LONG TP filled – P&L: {pnl:+.4f} USDT | Total: {self.tracker.realized_pnl:+.4f}")
                    if self._long_sl:
                        await self.order_mgr.cancel_algo_order(self._long_sl.order_id)
                        self._long_sl = None
                    self._long_tp = None

            if self._long_tp is None:
                tp = await self.order_mgr.place_tp_order("SELL", "LONG", desired_tp, desired_qty, entry)
                if tp:
                    self._long_tp = tp
                    log.info(f"TP placed: SELL/LONG qty={desired_qty} at {desired_tp} (entry={entry:.6f})")

            # SL
            if self._long_sl:
                if (abs(self._long_sl.entry_price - entry) > 1e-8 or
                    abs(self._long_sl.qty - desired_qty) > self.step_size):
                    await self.order_mgr.cancel_algo_order(self._long_sl.order_id)
                    self._long_sl = None

            if self._long_sl is None:
                sl = await self.order_mgr.place_sl_order("SELL", "LONG", desired_sl, desired_qty, entry)
                if sl:
                    self._long_sl = sl
                    log.info(f"SL placed: SELL/LONG qty={desired_qty} stopAt={desired_sl} (entry={entry:.6f})")
        else:
            if self._long_tp:
                await self.order_mgr.cancel_order(self._long_tp.order_id)
                self._long_tp = None
            if self._long_sl:
                await self.order_mgr.cancel_algo_order(self._long_sl.order_id)
                self._long_sl = None

    async def _sync_short_exit(self, tp_bps: float, sl_bps: float):
        qty = self.tracker.short_qty
        entry = self.tracker.short_entry

        if qty > 0:
            desired_tp = self._round_price(entry * (1 - tp_bps / 10_000), entry)
            desired_sl = self._round_price(entry * (1 + sl_bps / 10_000), entry)
            desired_qty = int(qty / self.step_size) * self.step_size

            # TP
            if self._short_tp:
                if (abs(self._short_tp.entry_price - entry) > 1e-8 or
                    abs(self._short_tp.qty - desired_qty) > self.step_size):
                    await self.order_mgr.cancel_order(self._short_tp.order_id)
                    self._short_tp = None
                elif not await self.order_mgr.verify_order_exists(self._short_tp.order_id):
                    pnl = (entry - self._short_tp.tp_price) * qty
                    self.tracker.add_pnl(pnl)
                    log.info(f"SHORT TP filled – P&L: {pnl:+.4f} USDT | Total: {self.tracker.realized_pnl:+.4f}")
                    if self._short_sl:
                        await self.order_mgr.cancel_algo_order(self._short_sl.order_id)
                        self._short_sl = None
                    self._short_tp = None

            if self._short_tp is None:
                tp = await self.order_mgr.place_tp_order("BUY", "SHORT", desired_tp, desired_qty, entry)
                if tp:
                    self._short_tp = tp
                    log.info(f"TP placed: BUY/SHORT qty={desired_qty} at {desired_tp} (entry={entry:.6f})")

            # SL
            if self._short_sl:
                if (abs(self._short_sl.entry_price - entry) > 1e-8 or
                    abs(self._short_sl.qty - desired_qty) > self.step_size):
                    await self.order_mgr.cancel_algo_order(self._short_sl.order_id)
                    self._short_sl = None

            if self._short_sl is None:
                sl = await self.order_mgr.place_sl_order("BUY", "SHORT", desired_sl, desired_qty, entry)
                if sl:
                    self._short_sl = sl
                    log.info(f"SL placed: BUY/SHORT qty={desired_qty} stopAt={desired_sl} (entry={entry:.6f})")
        else:
            if self._short_tp:
                await self.order_mgr.cancel_order(self._short_tp.order_id)
                self._short_tp = None
            if self._short_sl:
                await self.order_mgr.cancel_algo_order(self._short_sl.order_id)
                self._short_sl = None

    # ------------------------------------------------------------------
    # Public stats
    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict:
        return {
            "tick_count": self._tick_count,
            "last_bid": self._last_bid,
            "last_ask": self._last_ask,
            "realized_pnl": self.tracker.realized_pnl,
        }
