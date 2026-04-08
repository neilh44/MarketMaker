"""
quoting_engine.py
-----------------
Orchestration layer that initializes all components and starts the strategy engine.
"""

import logging
import os
from .exchange_client import BinanceClient
from .order_manager import OrderManager
from .position_tracker import PositionTracker
from .risk_manager import RiskManager
from .strategy_engine import StrategyEngine
from core.shared_state import SharedState

log = logging.getLogger("engine")
LEVERAGE = 10


class QuotingEngine:
    def __init__(self, symbol: str, state: SharedState, qty_usd: float = 10.0):
        self.symbol = symbol.upper()
        self.state = state
        self.qty_usd = qty_usd
        self._api_key = os.environ.get("BINANCE_API_KEY", "")
        self._api_secret = os.environ.get("BINANCE_API_SECRET", "")
        self._step_size = 0.001
        self._running = False

        self.client = BinanceClient(self._api_key, self._api_secret)
        self.tracker = PositionTracker()
        self.order_mgr = OrderManager(self.symbol, self.client, self._step_size)
        # RiskManager expects exactly 6 arguments after self
        self.risk_mgr = RiskManager(
            self.client, self.tracker, LEVERAGE, qty_usd, self.symbol, self.order_mgr
        )
        self.engine = None

    async def _fetch_symbol_info(self):
        data = await self.client.get("/fapi/v1/exchangeInfo", {})
        for sym in data.get("symbols", []):
            if sym["symbol"] == self.symbol:
                for f in sym.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        self._step_size = float(f["stepSize"])
                        log.info(f"Symbol {self.symbol}: stepSize={self._step_size}")
                break
        await self.client.post("/fapi/v1/leverage", {"symbol": self.symbol, "leverage": LEVERAGE})
        await self.client.post("/fapi/v1/positionSide/dual", {"dualSidePosition": "true"})
        log.info("Hedge mode enabled")

    # Add this method inside the QuotingEngine class
    async def _cancel_all(self):
        """Cancel all open orders (emergency cleanup)."""
        if not self.engine:
            return
        # Cancel entry orders
        await self.engine._cancel_entry_orders()
        # Cancel TP/SL orders
        if self.engine._long_tp:
            await self.order_mgr.cancel_order(self.engine._long_tp.order_id)
        if self.engine._short_tp:
            await self.order_mgr.cancel_order(self.engine._short_tp.order_id)
        if self.engine._long_sl:
            await self.order_mgr.cancel_algo_order(self.engine._long_sl.order_id)
        if self.engine._short_sl:
            await self.order_mgr.cancel_algo_order(self.engine._short_sl.order_id)

    async def run(self):
        await self._fetch_symbol_info()
        self.order_mgr.step_size = self._step_size
        self.engine = StrategyEngine(
            self.symbol, self.state, self.qty_usd, self.client,
            self.order_mgr, self.tracker, self.risk_mgr, self._step_size
        )
        self._running = True
        log.info(f"Quoting engine starting for {self.symbol}, qty={self.qty_usd} USDT/side")
        await self.engine.run()

    def stop(self):
        self._running = False
        if self.engine:
            self.engine.stop()

    @property
    def stats(self) -> dict:
        if self.engine:
            return self.engine.stats
        return {}