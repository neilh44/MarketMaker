"""
feed_handler.py
---------------
Maintains a persistent WebSocket connection to Binance Futures.
Streams: bookTicker (best bid/ask) + miniTicker (24h stats).
Writes normalised data into SharedState every tick.
"""

import asyncio
import json
import logging
import math
import time
from collections import deque

import aiohttp

from core.shared_state import SharedState

log = logging.getLogger("feed")

BINANCE_WS_BASE = "wss://fstream.binance.com/stream"
RECONNECT_DELAY = 3       # seconds before reconnect on error
VOLATILITY_WINDOW = 60    # number of mid-prices used for rolling σ


class FeedHandler:
    def __init__(self, symbol: str, state: SharedState):
        self.symbol = symbol.upper()
        self.state = state
        self._mid_prices: deque[float] = deque(maxlen=VOLATILITY_WINDOW)
        self._running = False

    def _streams_url(self) -> str:
        sym = self.symbol.lower()
        return (
            f"{BINANCE_WS_BASE}?streams="
            f"{sym}@bookTicker/{sym}@miniTicker"
        )

    def _calc_volatility(self) -> float:
        """Rolling standard deviation of log-returns."""
        prices = list(self._mid_prices)
        if len(prices) < 10:
            return 0.0
        returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        n = len(returns)
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / n
        return math.sqrt(variance) * math.sqrt(525600)  # annualised

    async def run(self):
        self._running = True
        log.info(f"Feed handler starting for {self.symbol}")

        while self._running:
            try:
                await self._connect()
            except Exception as e:
                log.warning(f"Feed error: {e} — reconnecting in {RECONNECT_DELAY}s")
                await asyncio.sleep(RECONNECT_DELAY)

    async def _connect(self):
        url = self._streams_url()
        timeout = aiohttp.ClientTimeout(total=None, sock_read=30)
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url, timeout=timeout, heartbeat=20) as ws:
                log.info(f"WebSocket connected: {url}")
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle(json.loads(msg.data))
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                        log.warning(f"WS closed/error: {msg}")
                        break

    async def _handle(self, raw: dict):
        stream = raw.get("stream", "")
        data = raw.get("data", {})

        if "bookTicker" in stream:
            bid = float(data.get("b", 0))
            ask = float(data.get("a", 0))
            if bid <= 0 or ask <= 0:
                return
            mid = (bid + ask) / 2
            spread_bps = (ask - bid) / mid * 10_000
            self._mid_prices.append(mid)

            await self.state.update_market(
                symbol=self.symbol,
                bid=bid,
                ask=ask,
                mid=mid,
                spread_bps=round(spread_bps, 3),
                last_price=mid,
                volatility=self._calc_volatility(),
            )
            log.debug(f"Tick {self.symbol}: bid={bid} ask={ask} spread={spread_bps:.2f}bps")

        elif "miniTicker" in stream:
            await self.state.update_market(
                volume_24h=float(data.get("v", 0)),
                price_change_pct=float(data.get("P", 0)),
            )

    def stop(self):
        self._running = False
