"""
shared_state.py
---------------
Single source of truth for parameters the LLM advisor writes
and the quoting engine reads. Thread-safe via asyncio.Lock.
"""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class ParamsSnapshot:
    """Parameters updated by the LLM advisor every polling interval."""
    spread_bps: float = 25.0           # total spread in basis points
    skew: float = 0.0                  # -1.0 (favor SHORT) to +1.0 (favor LONG)
    paused: bool = False               # if True, quoting engine cancels all and waits
    regime: str = "neutral"            # e.g., "low_vol", "choppy", "trending"
    max_position_usd: float = 50.0     # max notional per side (fallback)
    
    # === NEW LLM-CONTROLLED FIELDS ===
    action: str = "CONTINUE"           # CONTINUE, PAUSE, CLOSE_LONG, CLOSE_SHORT, FLATTEN
    sl_bps: float = 25.0               # stop-loss distance in bps from entry
    tp_bps: float = 40.0               # take-profit distance in bps from entry
    position_multiple: float = 1.1     # max inventory per side as multiple of base qty
    
    updated_at: float = field(default_factory=time.time)
    reason: str = ""                   # LLM's last explanation


@dataclass
class MarketSnapshot:
    """Latest market data written by feed_handler, read by both engine and LLM."""
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    spread_bps: float = 0.0            # current exchange spread in bps
    last_price: float = 0.0
    volume_24h: float = 0.0
    price_change_pct: float = 0.0      # 1-min rolling change %
    volatility: float = 0.0            # recent σ of returns
    updated_at: float = field(default_factory=time.time)


@dataclass
class PositionState:
    """Live position and PnL tracking."""
    long_qty: float = 0.0
    short_qty: float = 0.0
    long_entry: float = 0.0
    short_entry: float = 0.0
    realised_pnl: float = 0.0
    unrealised_pnl: float = 0.0
    total_fills: int = 0
    wins: int = 0
    losses: int = 0


class SharedState:
    """
    Global shared state container.
    All writes go through async setters so consumers can await changes.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self.params = ParamsSnapshot()
        self.market = MarketSnapshot()
        self.position = PositionState()
        self._param_updated = asyncio.Event()

    async def update_params(self, **kwargs):
        """Update multiple param fields atomically."""
        async with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.params, k):
                    setattr(self.params, k, v)
            self.params.updated_at = time.time()
            self._param_updated.set()
            self._param_updated.clear()

    async def update_market(self, **kwargs):
        """Update multiple market fields atomically."""
        async with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.market, k):
                    setattr(self.market, k, v)
            self.market.updated_at = time.time()

    async def update_position(self, **kwargs):
        """Update multiple position fields atomically."""
        async with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.position, k):
                    setattr(self.position, k, v)

    def get_params_snapshot(self) -> ParamsSnapshot:
        """Return a copy of the current parameters."""
        return ParamsSnapshot(**self.params.__dict__)

    def get_market_snapshot(self) -> MarketSnapshot:
        """Return a copy of the current market data."""
        return MarketSnapshot(**self.market.__dict__)

    def get_position_summary(self):
        """Return (long_qty, short_qty, long_entry, short_entry) tuple."""
        return (
            self.position.long_qty,
            self.position.short_qty,
            self.position.long_entry,
            self.position.short_entry,
        )

    @property
    def realized_pnl(self) -> float:
        return self.position.realised_pnl