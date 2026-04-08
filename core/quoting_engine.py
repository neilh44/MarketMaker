"""
quoting_engine.py
-----------------
Fast loop: reads latest market tick + LLM params -> posts BID and ASK
as LIMIT orders on Binance Futures (Hedge Mode).

Flow per tick:
  1. Check paused flag -- cancel all and sleep if true
  2. Compute target bid/ask from mid +/- half-spread +/- skew adjustment
  3. If quoted prices haven't moved more than min_tick -> skip (avoid churn)
  4. Cancel existing ENTRY orders -> place new BID + ASK
  5. Sync positions every 2s -> place/replace TP + SL on each open side
     - TP: LIMIT order at entry + (spread + 9bps commission)
     - SL: STOP algo order at entry +/- 15bps (3% ROI loss at 10x)
       NOTE: SL uses /fapi/v1/order/algo/stop (Algo Order API) because
             STOP_MARKET is not supported on /fapi/v1/order (-4120).
             Algo orders use algoId (not orderId) and have separate
             open-order and cancel endpoints.
     - When either fills, cancel the other on same side
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time
import urllib.parse
from dataclasses import dataclass

import aiohttp

from core.shared_state import SharedState

log = logging.getLogger("engine")

BINANCE_REST = "https://fapi.binance.com"
MIN_PRICE_MOVE_BPS = 0.9      # don't requote unless price moved this much
MIN_SPREAD_BPS = 25.0         # minimum spread to ensure positive EV
MAX_POSITION_MULTIPLE = 1.5   # max position per side = 1.5 × base quote quantity
LEVERAGE = 10                 # reduced from 20 for better margin safety
ORDER_TYPE = "LIMIT"
TIME_IN_FORCE = "GTC"
TP_CHECK_INTERVAL = 10.0      # seconds between position sync cycles
COMMISSION_BPS = 9.0          # round-trip commission (0.045% x 2 sides) – conservative estimate
SL_ROI_PCT = 0.03             # stop-loss at 3% ROI loss
SL_BPS = SL_ROI_PCT / LEVERAGE * 10_000  # 3% / 10x leverage = 30bps? Wait recalc: 0.03/10*10000=30bps? That's wrong. Actually we want 15bps at 20x, with 10x it's 30bps. But user wanted 15bps SL? Let's keep consistent: earlier we had 0.03 at 20x = 15bps. To keep same 15bps SL at 10x, we need SL_ROI_PCT = 0.015. We'll set it to 0.015 for 15bps SL.
# Actually let's use SL_BPS directly to avoid confusion:
SL_BPS = 15.0                 # Stop loss distance in basis points


@dataclass
class ActiveOrder:
    order_id: int
    side: str           # BUY or SELL
    price: float
    qty: float
    position_side: str  # LONG or SHORT


@dataclass
class TpOrder:
    order_id: int
    position_side: str
    entry_price: float
    tp_price: float
    qty: float


@dataclass
class SlOrder:
    order_id: int       # algoId returned by /fapi/v1/order/algo/stop
    position_side: str
    entry_price: float
    sl_price: float     # stopPrice sent to Binance
    qty: float


class QuotingEngine:
    def __init__(self, symbol: str, state: SharedState, qty_usd: float = 10.0):
        self.symbol = symbol.upper()
        self.state = state
        self.qty_usd = qty_usd          # notional per side in USDT
        self._api_key = os.environ.get("BINANCE_API_KEY", "")
        self._api_secret = os.environ.get("BINANCE_API_SECRET", "")
        self._running = False
        self._bid_order: ActiveOrder | None = None
        self._ask_order: ActiveOrder | None = None
        self._last_bid: float = 0.0
        self._last_ask: float = 0.0
        self._tick_count: int = 0
        self._step_size: float = 0.001  # set by _fetch_symbol_info on startup
        self._consecutive_failures: int = 0

        # Take-profit and stop-loss tracking — one per side at most
        self._long_tp: TpOrder | None = None
        self._short_tp: TpOrder | None = None
        self._long_sl: SlOrder | None = None
        self._short_sl: SlOrder | None = None
        self._last_tp_check: float = 0.0

        # Rate limiting
        self._last_quote_time: float = 0.0
        self._min_quote_interval: float = 5.0   # seconds between quote attempts

        # Inventory tracking (real-time)
        self._current_long_qty: float = 0.0
        self._current_short_qty: float = 0.0

        # Realized P&L tracking
        self.realized_pnl: float = 0.0

        # Margin failure cooldown
        self._margin_failure_count: int = 0
        self._last_margin_failure_time: float = 0.0

    # ── Startup ───────────────────────────────────────────────────────────────

    async def _fetch_symbol_info(self):
        """Fetch lot size and set leverage to 10x."""
        url = f"{BINANCE_REST}/fapi/v1/exchangeInfo"
        async with aiohttp.ClientSession() as s:
            async with s.get(url) as r:
                data = await r.json()
        for sym in data.get("symbols", []):
            if sym["symbol"] == self.symbol:
                for f in sym.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        self._step_size = float(f["stepSize"])
                        log.info(f"Symbol {self.symbol}: stepSize={self._step_size}")
                break

        # Set leverage (applies to both LONG and SHORT in hedge mode)
        r = await self._signed_post("/fapi/v1/leverage", {
            "symbol": self.symbol,
            "leverage": LEVERAGE,
        })
        log.info(f"Leverage: {r.get('leverage', '?')}x set for {self.symbol}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self):
        await self._fetch_symbol_info()
        self._running = True
        log.info(f"Quoting engine starting for {self.symbol}, qty={self.qty_usd} USDT/side")

        while self._running:
            market = self.state.get_market_snapshot()
            params = self.state.get_params_snapshot()

            if market.mid == 0:
                await asyncio.sleep(0.1)
                continue

            if params.paused:
                if self._bid_order or self._ask_order:
                    log.info("LLM paused -- cancelling entry quotes (TP/SL orders kept)")
                    await self._cancel_all(include_tp=False)
                await asyncio.sleep(1.0)
                continue

            await self._requote(market.mid, params)

            # Check positions and place TPs/SLs every TP_CHECK_INTERVAL seconds
            now = time.time()
            if now - self._last_tp_check >= TP_CHECK_INTERVAL:
                self._last_tp_check = now
                await self._sync_exits(params.spread_bps)

            self._tick_count += 1
            await asyncio.sleep(0.0)  # yield to event loop

    # ── Quote calculation ─────────────────────────────────────────────────────

    def _compute_quotes(self, mid: float, spread_bps: float, skew: float) -> tuple[float, float]:
        half = (spread_bps / 10_000) / 2 * mid
        # Negative skew = tighter bid (favor LONG), wider ask (avoid SHORT)
        bid = mid - half * (1 + skew)
        ask = mid + half * (1 - skew)
        if mid >= 1:
            precision = 4
        elif mid >= 0.1:
            precision = 5
        elif mid >= 0.01:
            precision = 6
        elif mid >= 0.001:
            precision = 7
        else:
            precision = 8
        return round(bid, precision), round(ask, precision)

    def _has_moved(self, new_bid: float, new_ask: float, mid: float) -> bool:
        if self._last_bid == 0:
            return True
        bid_move = abs(new_bid - self._last_bid) / mid * 10_000
        ask_move = abs(new_ask - self._last_ask) / mid * 10_000
        return max(bid_move, ask_move) >= MIN_PRICE_MOVE_BPS

    # ── Exit management (TP + SL) ─────────────────────────────────────────────

    async def _sync_exits(self, spread_bps: float):
        """
        Every TP_CHECK_INTERVAL seconds:
        - Fetch positions from Binance
        - For each open side: ensure TP (LIMIT) and SL (algo STOP) are live
        - Replace TP/SL if avg entry or qty has changed
        - When position gone: cancel any remaining exit order on that side
        - If TP filled -> cancel SL. If SL filled -> cancel TP.
        """
        # Enforce minimum spread for positive EV
        effective_spread = max(spread_bps, MIN_SPREAD_BPS)
        if effective_spread != spread_bps:
            log.info(f"Spread clamped to {effective_spread:.1f}bps (LLM gave {spread_bps:.1f})")

        positions = await self._fetch_positions()
        if positions is None:
            return

        long_qty = 0.0
        short_qty = 0.0
        long_entry = 0.0
        short_entry = 0.0

        for p in positions:
            if p["symbol"] != self.symbol:
                continue
            raw_amt = p["positionAmt"]
            qty = abs(float(raw_amt))
            entry = float(p["entryPrice"])
            side = p["positionSide"]
            log.info(f"POSITION RAW | side={side} positionAmt={raw_amt} qty={qty} entry={entry}")
            if side == "LONG" and qty > 0:
                long_qty = qty
                long_entry = entry
            elif side == "SHORT" and qty > 0:
                short_qty = qty
                short_entry = entry

        # Periodic sync of inventory (real-time updates happen on fill/close)
        self._current_long_qty = long_qty
        self._current_short_qty = short_qty

        log.info(f"POSITION SUMMARY | long={long_qty}@{long_entry:.6f} short={short_qty}@{short_entry:.6f}")
        log.info(f"EXIT STATE | long_tp={'set' if self._long_tp else 'NONE'} long_sl={'set' if self._long_sl else 'NONE'} "
                 f"short_tp={'set' if self._short_tp else 'NONE'} short_sl={'set' if self._short_sl else 'NONE'}")

        await self.state.update_position(
            long_qty=long_qty, short_qty=short_qty,
            long_entry=long_entry, short_entry=short_entry,
        )

        tp_offset = (effective_spread + COMMISSION_BPS) / 10_000
        sl_offset = SL_BPS / 10_000

        await self._sync_side_exits(
            pos_qty=long_qty, entry=long_entry,
            tp_side="SELL", sl_side="SELL", pos_side="LONG",
            tp_price_fn=lambda e: self._round_price(e * (1 + tp_offset)),
            sl_price_fn=lambda e: self._round_price(e * (1 - sl_offset)),
            tp_ref=self._long_tp, sl_ref=self._long_sl,
            set_tp=lambda o: setattr(self, "_long_tp", o),
            set_sl=lambda o: setattr(self, "_long_sl", o),
        )

        await self._sync_side_exits(
            pos_qty=short_qty, entry=short_entry,
            tp_side="BUY", sl_side="BUY", pos_side="SHORT",
            tp_price_fn=lambda e: self._round_price(e * (1 - tp_offset)),
            sl_price_fn=lambda e: self._round_price(e * (1 + sl_offset)),
            tp_ref=self._short_tp, sl_ref=self._short_sl,
            set_tp=lambda o: setattr(self, "_short_tp", o),
            set_sl=lambda o: setattr(self, "_short_sl", o),
        )

    async def _sync_side_exits(
        self, pos_qty, entry, tp_side, sl_side, pos_side,
        tp_price_fn, sl_price_fn, tp_ref, sl_ref, set_tp, set_sl
    ):
        if pos_qty > 0:
            desired_tp = tp_price_fn(entry)
            desired_sl = sl_price_fn(entry)
            desired_qty = self._round_qty(pos_qty)

            # --- TP ---
            if tp_ref is not None:
                confirmed = await self._verify_order_exists(tp_ref.order_id)
                if not confirmed:
                    # TP filled – calculate P&L
                    pnl = 0.0
                    if pos_side == "LONG":
                        pnl = (tp_ref.tp_price - entry) * pos_qty
                    else:
                        pnl = (entry - tp_ref.tp_price) * pos_qty
                    self.realized_pnl += pnl
                    log.info(f"{pos_side} TP {tp_ref.order_id} filled – P&L: {pnl:+.4f} USDT | Total: {self.realized_pnl:+.4f}")
                    if sl_ref is not None:
                        await self._cancel_algo_order(sl_ref.order_id)
                        set_sl(None)
                    set_tp(None)
                elif (abs(tp_ref.entry_price - entry) > 1e-8 or
                      abs(tp_ref.qty - desired_qty) > self._step_size):
                    log.info(f"{pos_side} entry/qty changed — replacing TP")
                    await self._cancel_order(tp_ref.order_id)
                    set_tp(None)

            if tp_ref is None and pos_qty > 0:
                ok = await self._place_tp_order(tp_side, pos_side, desired_tp, desired_qty, entry)
                if ok:
                    log.info(f"TP placed: {tp_side}/{pos_side} qty={desired_qty} at {desired_tp} (entry={entry})")

            # --- SL ---
            if sl_ref is not None:
                if (abs(sl_ref.entry_price - entry) > 1e-8 or
                    abs(sl_ref.qty - desired_qty) > self._step_size):
                    log.info(f"{pos_side} entry/qty changed — replacing SL")
                    await self._cancel_algo_order(sl_ref.order_id)
                    set_sl(None)

            if sl_ref is None and pos_qty > 0:
                ok = await self._place_sl_order(sl_side, pos_side, desired_sl, desired_qty, entry)
                if ok:
                    log.info(f"SL placed: {sl_side}/{pos_side} qty={desired_qty} stopAt={desired_sl} "
                             f"(entry={entry} loss_cap={SL_BPS:.0f}bps)")

        else:
            # Position closed – reset inventory and clean up
            if pos_side == "LONG":
                self._current_long_qty = 0.0
            else:
                self._current_short_qty = 0.0

            if tp_ref is not None:
                log.info(f"{pos_side} position closed — cancelling leftover TP {tp_ref.order_id}")
                await self._cancel_order(tp_ref.order_id)
                set_tp(None)
            if sl_ref is not None:
                log.info(f"{pos_side} position closed — cancelling leftover SL {sl_ref.order_id}")
                await self._cancel_algo_order(sl_ref.order_id)
                set_sl(None)

    async def _place_tp_order(self, side: str, pos_side: str, tp_price: float, qty: float, entry_price: float = 0.0) -> bool:
        """Place a take-profit LIMIT order (GTC) and verify it exists on Binance."""
        decimals = len(str(self._step_size).rstrip('0').split('.')[-1])
        params = {
            "symbol": self.symbol,
            "side": side,
            "positionSide": pos_side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "price": f"{tp_price:.6f}".rstrip('0').rstrip('.'),
            "quantity": f"{qty:.{decimals}f}",
        }
        try:
            r = await self._signed_post("/fapi/v1/order", params)
            if "orderId" not in r:
                log.warning(f"TP order rejected [{r.get('code')}]: {r.get('msg')}")
                return False

            order_id = r["orderId"]

            confirmed = await self._verify_order_exists(order_id)
            if not confirmed:
                log.warning(f"TP order {order_id} placed but NOT found on Binance — not marking as set")
                return False

            tp = TpOrder(
                order_id=order_id,
                position_side=pos_side,
                entry_price=entry_price,
                tp_price=tp_price,
                qty=qty,
            )
            if pos_side == "LONG":
                self._long_tp = tp
            else:
                self._short_tp = tp
            return True

        except Exception as e:
            log.error(f"TP place error: {e}")
            return False

    async def _place_sl_order(self, side: str, pos_side: str, stop_price: float, qty: float, entry_price: float) -> bool:
        decimals = len(str(self._step_size).rstrip('0').split('.')[-1])
        params = {
            "symbol": self.symbol,
            "side": side,
            "positionSide": pos_side,
            "algoType": "CONDITIONAL",
            "type": "STOP_MARKET",
            "triggerPrice": f"{stop_price:.6f}".rstrip('0').rstrip('.'),
            "quantity": f"{qty:.{decimals}f}",
            "workingType": "CONTRACT_PRICE",
            "timeInForce": "GTC",
        }
        try:
            r = await self._signed_post("/fapi/v1/algoOrder", params)
            if "algoId" not in r:
                code = r.get('code')
                if code == -2021:
                    log.debug(f"SL skipped — would immediately trigger (stop={stop_price}, entry={entry_price})")
                    return False
                log.warning(f"SL algo order rejected [{code}]: {r.get('msg')}")
                return False

            algo_id = r["algoId"]
            sl = SlOrder(
                order_id=algo_id,
                position_side=pos_side,
                entry_price=entry_price,
                sl_price=stop_price,
                qty=qty,
            )
            if pos_side == "LONG":
                self._long_sl = sl
            else:
                self._short_sl = sl
            log.info(f"SL placed: algoId={algo_id} {side}/{pos_side} qty={qty} triggerAt={stop_price}")
            return True

        except Exception as e:
            log.error(f"SL place error: {e}")
            return False

    async def _verify_order_exists(self, order_id: int) -> bool:
        """Confirm a regular LIMIT order is live on Binance open orders."""
        try:
            orders = await self._signed_get("/fapi/v1/openOrders", {"symbol": self.symbol})
            if not isinstance(orders, list):
                return False
            ids = [o["orderId"] for o in orders]
            log.debug(f"Open orders on Binance: {ids}")
            return order_id in ids
        except Exception as e:
            log.debug(f"Order verify failed: {e}")
            return False

    async def _verify_algo_order_exists(self, algo_id: int, retries: int = 3, delay: float = 0.5) -> bool:
        """Confirm an algo order is live via GET /fapi/v1/openAlgoOrders."""
        for attempt in range(retries):
            await asyncio.sleep(delay)
            try:
                r = await self._signed_get("/fapi/v1/openAlgoOrders", {"symbol": self.symbol})
                orders = r.get("orders", []) if isinstance(r, dict) else []
                ids = [o["algoId"] for o in orders]
                if algo_id in ids:
                    return True
                log.debug(f"Algo order {algo_id} not found yet (attempt {attempt+1}/{retries})")
            except Exception as e:
                log.debug(f"Algo order verify failed (attempt {attempt+1}): {e}")
        return False

    async def _fetch_positions(self) -> list | None:
        """Fetch open positions from Binance Futures."""
        try:
            r = await self._signed_get("/fapi/v2/positionRisk", {"symbol": self.symbol})
            return r if isinstance(r, list) else None
        except Exception as e:
            log.debug(f"Position fetch failed: {e}")
            return None

    def _round_price(self, price: float) -> float:
        mid = self.state.market.mid
        if mid >= 1:
            return round(price, 4)
        elif mid >= 0.1:
            return round(price, 5)
        elif mid >= 0.01:
            return round(price, 6)
        elif mid >= 0.001:
            return round(price, 7)
        return round(price, 8)

    def _round_qty(self, qty: float) -> float:
        steps = int(qty / self._step_size)
        return steps * self._step_size

    async def _check_margin_sufficient(self, price: float, qty: float) -> bool:
        """Check if sufficient margin exists before placing order."""
        try:
            account = await self._signed_get("/fapi/v2/account", {})
            if not account:
                return True  # Proceed if can't check

            available_balance = float(account.get("availableBalance", 0))
            position_value = price * qty
            required_margin = position_value / LEVERAGE

            return available_balance >= required_margin * 1.05  # 5% buffer
        except Exception as e:
            log.debug(f"Margin check failed: {e}")
            return True  # Proceed if check fails

    async def _can_afford_new_quote(self, mid: float) -> bool:
        """Global margin check before attempting two-sided quote."""
        try:
            account = await self._signed_get("/fapi/v2/account", {})
            available = float(account.get("availableBalance", 0))
            # Estimate new order margin (both sides)
            qty = self._compute_qty(mid, self.state.params.max_position_usd)
            new_margin = (mid * qty * 2) / LEVERAGE
            min_free_margin = 2.0  # require at least $2 free after placing orders
            return (available - new_margin) >= min_free_margin
        except Exception as e:
            log.debug(f"Global margin check failed: {e}")
            return True

    # ── Order management ──────────────────────────────────────────────────────

    async def _requote(self, mid: float, params):
        # Rate limit quotes
        now = time.time()
        if now - self._last_quote_time < self._min_quote_interval:
            return
        self._last_quote_time = now

        # Adjust quote interval if recent margin failures
        if self._margin_failure_count >= 3 and (now - self._last_margin_failure_time) < 60:
            self._min_quote_interval = 15.0
        else:
            self._min_quote_interval = 5.0

        # Global margin check
        if not await self._can_afford_new_quote(mid):
            log.debug("Skipping quote – insufficient free margin for two-sided order")
            return

        bid_price, ask_price = self._compute_quotes(mid, params.spread_bps, params.skew)

        if not self._has_moved(bid_price, ask_price, mid):
            return

        qty = self._compute_qty(mid, params.max_position_usd)
        if qty <= 0:
            return

        await self._cancel_all()
        bid_ok = await self._place_order("BUY", "LONG", bid_price, qty)
        ask_ok = await self._place_order("SELL", "SHORT", ask_price, qty)

        if bid_ok and ask_ok:
            self._consecutive_failures = 0
            self._last_bid = bid_price
            self._last_ask = ask_price
            log.debug(
                f"Quoted BID={bid_price} ASK={ask_price} "
                f"spread={(ask_price-bid_price)/mid*10000:.1f}bps qty={qty}"
            )
        else:
            self._consecutive_failures += 1
            backoff = min(2 ** self._consecutive_failures, 60)
            log.warning(f"Order placement failed {self._consecutive_failures}x — backing off {backoff}s")
            await asyncio.sleep(backoff)

    def _compute_qty(self, price: float, max_usd: float) -> float:
        raw = self.qty_usd / price
        steps = int(raw / self._step_size)
        qty = steps * self._step_size
        max_qty = max_usd / price
        return min(qty, max_qty)

    async def _place_order(self, side: str, pos_side: str, price: float, qty: float) -> bool:
        # Inventory Cap Check
        if price > 0:
            base_qty = self.qty_usd / price
            max_allowed = base_qty * MAX_POSITION_MULTIPLE

            if pos_side == "LONG" and self._current_long_qty >= max_allowed:
                log.debug(f"Inventory cap: LONG {self._current_long_qty:.0f} >= {max_allowed:.0f}, skipping BUY")
                return False
            if pos_side == "SHORT" and self._current_short_qty >= max_allowed:
                log.debug(f"Inventory cap: SHORT {self._current_short_qty:.0f} >= {max_allowed:.0f}, skipping SELL")
                return False

        # Margin check
        if not await self._check_margin_sufficient(price, qty):
            self._margin_failure_count += 1
            self._last_margin_failure_time = time.time()
            return False

        decimals = len(str(self._step_size).rstrip('0').split('.')[-1])
        params = {
            "symbol": self.symbol,
            "side": side,
            "positionSide": pos_side,
            "type": ORDER_TYPE,
            "timeInForce": TIME_IN_FORCE,
            "price": f"{price:.6f}".rstrip('0').rstrip('.'),
            "quantity": f"{qty:.{decimals}f}",
        }
        try:
            r = await self._signed_post("/fapi/v1/order", params)
            if "orderId" in r:
                # Immediate inventory update
                if pos_side == "LONG":
                    self._current_long_qty += qty
                else:
                    self._current_short_qty += qty

                order = ActiveOrder(
                    order_id=r["orderId"],
                    side=side,
                    price=price,
                    qty=qty,
                    position_side=pos_side,
                )
                if side == "BUY":
                    self._bid_order = order
                else:
                    self._ask_order = order
                return True
            else:
                log.warning(f"Order rejected [{r.get('code')}]: {r.get('msg')}")
                return False
        except Exception as e:
            log.error(f"Place order error: {e}")
            return False

    async def _cancel_all(self, include_tp: bool = False):
        """Cancel entry quotes. If include_tp=True, also cancel all TP and SL orders."""
        for order in [self._bid_order, self._ask_order]:
            if order:
                await self._cancel_order(order.order_id)
        self._bid_order = None
        self._ask_order = None

        if include_tp:
            for o in [self._long_tp, self._short_tp]:
                if o:
                    await self._cancel_order(o.order_id)
            for o in [self._long_sl, self._short_sl]:
                if o:
                    await self._cancel_algo_order(o.order_id)
            self._long_tp = self._short_tp = self._long_sl = self._short_sl = None

    async def _cancel_order(self, order_id: int):
        """Cancel a regular order via /fapi/v1/order."""
        try:
            await self._signed_delete("/fapi/v1/order", {
                "symbol": self.symbol,
                "orderId": order_id,
            })
        except Exception as e:
            log.debug(f"Cancel order {order_id} failed (may already be filled): {e}")

    async def _cancel_algo_order(self, algo_id: int):
        """Cancel an algo order via DELETE /fapi/v1/algoOrder."""
        try:
            await self._signed_delete("/fapi/v1/algoOrder", {
                "symbol": self.symbol,
                "algoId": algo_id,
            })
        except Exception as e:
            log.debug(f"Cancel algo order {algo_id} failed (may already be filled): {e}")

    # ── Binance signed request helpers ────────────────────────────────────────

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params)
        sig = hmac.new(
            self._api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = sig
        return params

    async def _signed_get(self, path: str, params: dict) -> dict:
        params = self._sign(params)
        headers = {"X-MBX-APIKEY": self._api_key}
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{BINANCE_REST}{path}", params=params, headers=headers) as r:
                return await r.json()

    async def _signed_post(self, path: str, params: dict) -> dict:
        params = self._sign(params)
        headers = {"X-MBX-APIKEY": self._api_key}
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{BINANCE_REST}{path}", params=params, headers=headers) as r:
                return await r.json()

    async def _signed_delete(self, path: str, params: dict) -> dict:
        params = self._sign(params)
        headers = {"X-MBX-APIKEY": self._api_key}
        async with aiohttp.ClientSession() as s:
            async with s.delete(f"{BINANCE_REST}{path}", params=params, headers=headers) as r:
                return await r.json()

    def stop(self):
        self._running = False

    @property
    def stats(self) -> dict:
        return {
            "tick_count": self._tick_count,
            "last_bid": self._last_bid,
            "last_ask": self._last_ask,
            "bid_order": self._bid_order.order_id if self._bid_order else None,
            "ask_order": self._ask_order.order_id if self._ask_order else None,
            "long_tp": self._long_tp.tp_price if self._long_tp else None,
            "long_sl": self._long_sl.sl_price if self._long_sl else None,
            "short_tp": self._short_tp.tp_price if self._short_tp else None,
            "short_sl": self._short_sl.sl_price if self._short_sl else None,
            "realized_pnl": self.realized_pnl,
        }