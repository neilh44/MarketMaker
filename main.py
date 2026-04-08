"""
main.py
-------
Entrypoint for the hybrid LLM market-making system.

Runs three coroutines concurrently:
  1. FeedHandler   — streams market data from Binance WS
  2. LLMAdvisor    — polls Claude every N seconds for spread/skew/pause params
  3. QuotingEngine — fast loop that posts BID+ASK using current params

Usage:
  python main.py --symbol BTCUSDT --qty 10 --interval 20

Environment variables required:
  ANTHROPIC_API_KEY
  BINANCE_API_KEY
  BINANCE_API_SECRET
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time

from core.shared_state import SharedState
from core.feed_handler import FeedHandler
from core.llm_advisor import LLMAdvisor
from core.quoting_engine import QuotingEngine

from dotenv import load_dotenv
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trader.log"),
    ],
)
log = logging.getLogger("main")


# ── Status printer ────────────────────────────────────────────────────────────

async def status_loop(state: SharedState, engine: QuotingEngine, advisor: LLMAdvisor):
    """Prints a one-line status summary every 5 seconds."""
    while True:
        await asyncio.sleep(5)
        m = state.get_market_snapshot()
        p = state.get_params_snapshot()
        pos = state.position
        log.info(
            f"STATUS | mid={m.mid:.4f} "
            f"spread={p.spread_bps:.1f}bps skew={p.skew:+.2f} "
            f"paused={p.paused} regime={p.regime} | "
            f"PnL R={pos.realised_pnl:+.4f} U={pos.unrealised_pnl:+.4f} "
            f"W/L={pos.wins}/{pos.losses} | "
            f"ticks={engine.stats['tick_count']} llm_calls={advisor.stats['call_count']}"
        )


# ── Graceful shutdown ─────────────────────────────────────────────────────────

def _setup_signals(loop: asyncio.AbstractEventLoop, tasks: list):
    # Windows does not support add_signal_handler — use KeyboardInterrupt instead
    if sys.platform == "win32":
        return

    def _shutdown(sig):
        log.info(f"Received {sig.name} — shutting down gracefully...")
        for t in tasks:
            t.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: _shutdown(s))


# ── Main ──────────────────────────────────────────────────────────────────────

async def amain(symbol: str, qty_usd: float, llm_interval: int):
    # Validate env vars
    missing = [k for k in ("GROQ_API_KEY", "BINANCE_API_KEY", "BINANCE_API_SECRET")
               if not os.environ.get(k)]
    if missing:
        log.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    log.info(f"Starting hybrid trader | symbol={symbol} qty={qty_usd}USDT llm_interval={llm_interval}s")

    state = SharedState()
    feed = FeedHandler(symbol=symbol, state=state)
    advisor = LLMAdvisor(state=state, interval=llm_interval)
    engine = QuotingEngine(symbol=symbol, state=state, qty_usd=qty_usd)

    loop = asyncio.get_event_loop()
    tasks = [
        asyncio.create_task(feed.run(),    name="feed"),
        asyncio.create_task(advisor.run(), name="llm_advisor"),
        asyncio.create_task(engine.run(),  name="quoting_engine"),
        asyncio.create_task(status_loop(state, engine, advisor), name="status"),
    ]
    _setup_signals(loop, tasks)

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        log.info("All tasks cancelled — exiting")
    finally:
        # Best-effort: cancel open orders before exit
        await engine._cancel_all()
        log.info("Cleanup done. Goodbye.")


def main():
    parser = argparse.ArgumentParser(description="Hybrid LLM market-making bot")
    parser.add_argument("--symbol",   default="BTCUSDT",  help="Binance Futures symbol")
    parser.add_argument("--qty",      type=float, default=10.0,
                        help="Notional per side in USDT (default 10)")
    parser.add_argument("--interval", type=int,   default=20,
                        help="LLM polling interval in seconds (default 20)")
    parser.add_argument("--debug",    action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        asyncio.run(amain(args.symbol, args.qty, args.interval))
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
