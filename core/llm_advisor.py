"""
llm_advisor.py
--------------
Polls LLM (Groq / qwen3-32b) every N seconds to obtain dynamic trading parameters
and supervisory actions. Includes robust JSON parsing for truncated responses.
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


def _repair_truncated_json(content: str) -> Optional[dict]:
    """Attempt to repair truncated or malformed JSON."""
    if not content or content.strip() == "":
        return None

    # Strip <think>...</think> blocks
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Extract from markdown blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    # Find JSON object boundaries
    start = content.find('{')
    end = content.rfind('}')
    if start == -1:
        return None
    if end == -1:
        end = len(content)
    content = content[start:end+1]

    # Attempt to close unclosed strings and braces
    # Count quotes to detect unclosed strings
    in_string = False
    escaped = False
    for i, ch in enumerate(content):
        if ch == '"' and not escaped:
            in_string = not in_string
        elif ch == '\\' and not escaped:
            escaped = True
            continue
        escaped = False

    # If string is unclosed, try to close it
    if in_string:
        content += '"'

    # Close unclosed braces
    open_braces = content.count('{') - content.count('}')
    if open_braces > 0:
        content += '}' * open_braces
    open_brackets = content.count('[') - content.count(']')
    if open_brackets > 0:
        content += ']' * open_brackets

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


class LLMAdvisor:
    def __init__(self, state: SharedState, interval: float = 10.0):
        self.state = state
        self.interval = interval
        self._api_key = os.environ.get("GROQ_API_KEY", "")
        self._running = False
        self._call_count = 0

    async def _call_llm(self, market_snapshot) -> Optional[dict]:
        """Send market context and state vector to Groq and return parsed JSON."""
        if not self._api_key:
            log.error("GROQ_API_KEY not set")
            return None

        long_qty, short_qty, long_entry, short_entry = self.state.get_position_summary()
        realized_pnl = self.state.realized_pnl

        mid = market_snapshot.mid
        price_change_pct = getattr(market_snapshot, 'price_change_pct', 0.0)
        volatility = getattr(market_snapshot, 'volatility', 0.0)
        exchange_spread = getattr(market_snapshot, 'spread_bps', 0.0)

        prompt = f"""
You are a quantitative trading strategist controlling a market-making bot on STOUSDT perpetual futures (Hedge Mode).

**CRITICAL PROFITABILITY CONSTRAINTS – YOU MUST OBEY:**
- Round-trip commission is 9 basis points (0.09%). Your NET profit per round-trip = spread_bps - 9.
- To be profitable, you MUST set spread_bps to at least 30 bps, preferably 40–50 bps.
- stop-loss distance (sl_bps) MUST be at least 15 bps to avoid "Order would immediately trigger" errors.
- take-profit distance (tp_bps) should be 1.5× to 2× the sl_bps (e.g., sl=20, tp=35–45).
- position_multiple should stay at 1.0–1.2.
- gamma (risk-aversion) typical range: 0.005–0.05.

**Current State:**
- Mid price: {mid:.6f}
- 1m price change: {price_change_pct:.2f}%
- Volatility: {volatility:.2f}%
- Exchange spread: {exchange_spread:.1f} bps
- LONG: {long_qty:.0f} @ {long_entry:.6f}
- SHORT: {short_qty:.0f} @ {short_entry:.6f}
- Realized PnL: {realized_pnl:.4f} USDT

Respond with ONLY valid JSON:
{{"action": "CONTINUE", "spread_bps": 45, "skew": 0.0, "gamma": 0.02, "sl_bps": 20, "tp_bps": 40, "position_multiple": 1.1, "reason": "..."}}
"""

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a precise trading strategist. Output only valid JSON, no markdown, no explanations outside the JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 400,  # Increased further
        }

        max_retries = 2
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
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

                        parsed = _repair_truncated_json(content)
                        if parsed is not None:
                            return parsed

                        log.error(f"LLM JSON decode failed. Raw: {content[:300]}")
                        return None

            except asyncio.TimeoutError:
                log.warning(f"LLM call timed out (attempt {attempt+1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return None
            except Exception as e:
                log.error(f"LLM call failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return None

        return None

    async def _update_params_from_llm(self):
        market = self.state.get_market_snapshot()
        if market.mid == 0:
            return

        llm_response = await self._call_llm(market)
        if llm_response is None:
            return

        try:
            action = str(llm_response.get("action", "CONTINUE")).upper()
            if action not in ("CONTINUE", "PAUSE", "CLOSE_LONG", "CLOSE_SHORT", "FLATTEN"):
                action = "CONTINUE"

            new_params = ParamsSnapshot(
                spread_bps=float(llm_response.get("spread_bps", 40.0)),
                skew=float(llm_response.get("skew", 0.0)),
                paused=(action == "PAUSE"),
                regime=llm_response.get("regime", "unknown"),
                max_position_usd=self.state.params.max_position_usd,
                action=action,
                sl_bps=float(llm_response.get("sl_bps", 20.0)),
                tp_bps=float(llm_response.get("tp_bps", 40.0)),
                position_multiple=float(llm_response.get("position_multiple", 1.1)),
                gamma=float(llm_response.get("gamma", 0.01)),
            )

            await self.state.update_params(**new_params.__dict__)
            self._call_count += 1

            log.info(
                f"LLM #{self._call_count} -> "
                f"action={action} spread={new_params.spread_bps:.1f}bps "
                f"gamma={new_params.gamma:.3f} sl={new_params.sl_bps:.1f}bps "
                f"tp={new_params.tp_bps:.1f}bps mult={new_params.position_multiple:.2f} | "
                f"{llm_response.get('reason', '')}"
            )

        except (KeyError, ValueError, TypeError) as e:
            log.error(f"Failed to parse LLM response: {e}")

    async def run(self):
        self._running = True
        log.info(f"LLM advisor starting (Groq / {MODEL_NAME}), polling every {self.interval}s")

        while self._running:
            await self._update_params_from_llm()
            await asyncio.sleep(self.interval)

    @property
    def stats(self) -> dict:
        return {"call_count": self._call_count}

    def stop(self):
        self._running = False
