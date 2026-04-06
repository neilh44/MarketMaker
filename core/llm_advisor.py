"""
llm_advisor.py
--------------
Slow loop (default 20s) that calls the Groq API (OpenAI-compatible)
with a market snapshot and receives structured QuoteParams back as JSON.

Uses qwen/qwen3-32b via Groq's inference endpoint.
The model is instructed to output ONLY JSON — no prose, no markdown.
"""

import asyncio
import json
import logging
import os
import time

import aiohttp

from core.shared_state import SharedState

log = logging.getLogger("llm")

GROQ_API = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "qwen/qwen3-32b"
POLL_INTERVAL = 20      # seconds between LLM calls
MAX_TOKENS = 256

SYSTEM_PROMPT = """You are a market-making parameter advisor for a Binance Futures bot.
You receive a JSON market snapshot and must respond with ONLY a JSON object — no prose,
no markdown, no explanation, no <think> tags. Your output is parsed directly by code.

Output schema (all fields required):
{
  "spread_bps": <float, 10–50, total spread in basis points>,
  "skew": <float, -1.0 to 1.0, negative = lean short, positive = lean long>,
  "paused": <bool, true if conditions are dangerous>,
  "regime": <string, one of: "trending_up", "trending_down", "choppy", "low_vol", "news_spike">,
  "max_position_usd": <float, 10–200, max notional per side>,
  "reason": <string, max 80 chars, brief justification>
}

Decision rules:
- Widen spread (>15bps) when volatility is high or exchange spread is already wide
- Narrow spread (10 bps) during calm, low-vol sessions, never below 10bps
- Skew negative when recent price_change_pct is strongly positive (you'll get picked off on longs)
- Skew positive when price is falling and you want to accumulate longs cheap
- Pause when: volatility >0.8, price_change_pct >2% in 1 min, or spread_bps >30
- Never pause just because vol is moderate — that is normal market making conditions
- Commission is 0.045% per side per fill. Round-trip cost = 0.09% = 9bps.
- NEVER set spread_bps below 10. Minimum viable spread is 10bps.
- Target spread 12-15bps in normal conditions to ensure profit after fees.
- Only go to 10bps in very calm, low-vol sessions.
"""


class LLMAdvisor:
    def __init__(self, state: SharedState, poll_interval: int = POLL_INTERVAL):
        self.state = state
        self.poll_interval = poll_interval
        self._api_key = os.environ.get("GROQ_API_KEY", "")
        self._running = False
        self._call_count = 0
        self._last_error: str | None = None

    async def run(self):
        if not self._api_key:
            log.error("GROQ_API_KEY not set — LLM advisor disabled")
            return

        self._running = True
        log.info(f"LLM advisor starting (Groq / {MODEL}), polling every {self.poll_interval}s")

        while self._running:
            try:
                await self._advise()
            except Exception as e:
                self._last_error = str(e)
                log.warning(f"LLM call failed: {e}")
            await asyncio.sleep(self.poll_interval)

    async def _advise(self):
        market = self.state.get_market_snapshot()
        pos = self.state.position
        params = self.state.get_params_snapshot()

        if market.mid == 0:
            log.debug("Skipping LLM call — no market data yet")
            return

        snapshot = {
            "symbol": market.symbol,
            "mid_price": round(market.mid, 6),
            "exchange_spread_bps": round(market.spread_bps, 2),
            "volatility_annualised": round(market.volatility, 4),
            "price_change_pct_1m": round(market.price_change_pct, 4),
            "volume_24h": round(market.volume_24h, 2),
            "current_spread_bps": params.spread_bps,
            "current_skew": params.skew,
            "current_paused": params.paused,
            "long_qty": pos.long_qty,
            "short_qty": pos.short_qty,
            "realised_pnl": round(pos.realised_pnl, 4),
            "unrealised_pnl": round(pos.unrealised_pnl, 4),
            "wins": pos.wins,
            "losses": pos.losses,
        }

        user_msg = f"Market snapshot:\n{json.dumps(snapshot, indent=2)}\n\nAdvise new parameters."

        t0 = time.time()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {"type": "json_object"},
            "reasoning_effort": "none",  # disable Qwen3 <think> blocks — breaks JSON mode
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(GROQ_API, headers=headers, json=body) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Groq API {resp.status}: {text[:200]}")
                data = await resp.json()

        raw = data["choices"][0]["message"]["content"].strip()

        advice = json.loads(raw)
        latency_ms = int((time.time() - t0) * 1000)
        self._call_count += 1

        log.info(
            f"LLM #{self._call_count} ({latency_ms}ms) -> "
            f"spread={advice['spread_bps']}bps skew={advice['skew']} "
            f"paused={advice['paused']} regime={advice['regime']} | {advice['reason']}"
        )

        await self.state.update_params(
            spread_bps=float(advice["spread_bps"]),
            skew=float(advice["skew"]),
            paused=bool(advice["paused"]),
            regime=str(advice["regime"]),
            max_position_usd=float(advice.get("max_position_usd", 50.0)),
            reason=str(advice.get("reason", "")),
        )

    def stop(self):
        self._running = False

    @property
    def stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "last_error": self._last_error,
            "poll_interval": self.poll_interval,
        }