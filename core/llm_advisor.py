"""
llm_advisor.py
--------------
Polls LLM (Groq / qwen3-32b) every N seconds to obtain a market regime,
dynamic trading parameters, and supervisory actions.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Optional

import aiohttp

from core.shared_state import SharedState, ParamsSnapshot

log = logging.getLogger("llm")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "qwen/qwen3-32b"


class LLMAdvisor:
    def __init__(self, state: SharedState, interval: float = 10.0):
        self.state = state
        self.interval = interval
        self._api_key = os.environ.get("GROQ_API_KEY", "")
        self._running = False
        self._call_count = 0


    async def _call_llm(self, market_snapshot) -> Optional[dict]:
        """Send market context to Groq and return parsed JSON."""
        if not self._api_key:
            log.error("GROQ_API_KEY not set")
            return None

        # Gather current state
        long_qty, short_qty, long_entry, short_entry = self.state.get_position_summary()
        realized_pnl = self.state.realized_pnl

        mid = market_snapshot.mid
        price_change_pct = getattr(market_snapshot, 'price_change_pct', 0.0)
        volatility = getattr(market_snapshot, 'volatility', 0.0)
        exchange_spread = getattr(market_snapshot, 'spread_bps', 0.0)

        prompt = f"""
You are a quantitative trading strategist controlling a market-making bot on STOUSDT perpetual futures (Hedge Mode).
Current state:
- Mid price: {mid:.6f}
- 1m price change: {price_change_pct:.2f}%
- Volatility (ATR): {volatility:.2f}%
- Exchange spread: {exchange_spread:.1f} bps
- LONG position: {long_qty:.0f} @ {long_entry:.6f}
- SHORT position: {short_qty:.0f} @ {short_entry:.6f}
- Realized PnL: {realized_pnl:.4f} USDT

You must output a JSON object with the following fields:
- action: one of "CONTINUE", "PAUSE", "CLOSE_LONG", "CLOSE_SHORT", "FLATTEN"
- spread_bps: target bid-ask spread in basis points (only used if CONTINUE)
- skew: float between -1.0 and 1.0; negative favors LONG, positive favors SHORT
- sl_bps: stop-loss distance in basis points from entry
- tp_bps: take-profit distance in basis points from entry (overrides spread-based TP)
- position_multiple: max inventory per side as multiple of base quote (e.g., 1.1)
- reason: brief explanation of your decision

Example response:
{{"action": "CONTINUE", "spread_bps": 30, "skew": -0.2, "sl_bps": 20, "tp_bps": 45, "position_multiple": 1.2, "reason": "Low volatility, slight long bias"}}
"""

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a precise trading strategist. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 5000,  # Increased slightly to allow for reasoning + JSON
        }

        # Retry up to 2 times with increased timeout
        max_retries = 2
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    # Increased timeout: 45 seconds total
                    timeout = aiohttp.ClientTimeout(total=45)
                    async with session.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            log.error(f"Groq API error {resp.status}: {text[:200]}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2)
                                continue
                            return None

                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]

                        if not content or content.strip() == "":
                            log.error("LLM returned empty response")
                            return None

                        # Strip <think>...</think> blocks (DeepSeek reasoning)
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                        # Extract JSON from response
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content:
                            content = content.split("```")[1].split("```")[0].strip()

                        try:
                            return json.loads(content)
                        except json.JSONDecodeError as e:
                            log.error(f"LLM JSON decode failed: {e} | Raw: {content[:200]}")
                            return None

            except asyncio.TimeoutError:
                log.warning(f"LLM call timed out (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                log.error("LLM call timed out after all retries")
                return None
            except Exception as e:
                log.error(f"LLM call failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return None

        return None

    async def _update_params_from_llm(self):
        """Fetch LLM decision and update shared state."""
        market = self.state.get_market_snapshot()
        if market.mid == 0:
            log.debug("Skipping LLM call – no market data")
            return

        llm_response = await self._call_llm(market)
        if llm_response is None:
            # LLM failed – keep existing parameters unchanged
            log.debug("LLM call failed, retaining previous parameters")
            return

        try:
            # Parse with defaults for safety
            action = str(llm_response.get("action", "CONTINUE")).upper()
            if action not in ("CONTINUE", "PAUSE", "CLOSE_LONG", "CLOSE_SHORT", "FLATTEN"):
                log.warning(f"Invalid action '{action}', defaulting to CONTINUE")
                action = "CONTINUE"

            new_params = ParamsSnapshot(
                spread_bps=float(llm_response.get("spread_bps", 25.0)),
                skew=float(llm_response.get("skew", 0.0)),
                paused=(action == "PAUSE"),
                regime=llm_response.get("regime", "unknown"),
                max_position_usd=self.state.params.max_position_usd,
                action=action,
                sl_bps=float(llm_response.get("sl_bps", 25.0)),
                tp_bps=float(llm_response.get("tp_bps", 40.0)),
                position_multiple=float(llm_response.get("position_multiple", 1.1)),
            )
            self.state.update_params(**new_params.__dict__)
            self._call_count += 1

            log.info(
                f"LLM #{self._call_count} -> "
                f"action={action} spread={new_params.spread_bps:.1f}bps "
                f"skew={new_params.skew:+.2f} sl={new_params.sl_bps:.1f}bps "
                f"tp={new_params.tp_bps:.1f}bps mult={new_params.position_multiple:.2f} | "
                f"{llm_response.get('reason', '')}"
            )

        except (KeyError, ValueError, TypeError) as e:
            log.error(f"Failed to parse LLM response: {e} | Raw: {llm_response}")

    async def run(self):
        """Main polling loop."""
        self._running = True
        log.info(f"LLM advisor starting (Groq / {MODEL_NAME}), polling every {self.interval}s")

        while self._running:
            await self._update_params_from_llm()
            await asyncio.sleep(self.interval)

    @property
    def stats(self) -> dict:
        return {
            "call_count": self._call_count,
        }

    def stop(self):
        self._running = False