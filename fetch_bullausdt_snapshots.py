#!/usr/bin/env python3
"""
fetch_bullausdt_snapshots.py
----------------------------
Fetch 1-second order book snapshots for BULLAUSDT and save as JSONL.
"""

import asyncio
import json
import time
import os
import sys
from datetime import datetime

# Add project root to path so we can import core modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.exchange_client import BinanceClient


async def fetch_snapshots(symbol="BULLAUSDT", duration_seconds=600, interval=1.0):
    """
    Fetch order book depth and ticker every `interval` seconds for `duration_seconds`.
    Saves each snapshot as a JSON line.
    """
    # No auth needed for public depth/ticker endpoints
    client = BinanceClient(api_key="", api_secret="")
    start_time = time.time()
    snapshots = []
    fetch_count = 0

    print(f"Starting data collection for {symbol}...")
    print(f"Duration: {duration_seconds}s, Interval: {interval}s")
    print("-" * 50)

    while time.time() - start_time < duration_seconds:
        try:
            # Fetch order book (top 10 levels)
            ob = await client.get("/fapi/v1/depth", {"symbol": symbol, "limit": 10})
            # Fetch 24hr ticker for volume and change %
            ticker = await client.get("/fapi/v1/ticker/24hr", {"symbol": symbol})

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "mid": (float(ob["bids"][0][0]) + float(ob["asks"][0][0])) / 2,
                "best_bid": float(ob["bids"][0][0]),
                "best_ask": float(ob["asks"][0][0]),
                "bids": ob["bids"][:10],   # list of [price, qty]
                "asks": ob["asks"][:10],
                "volume_24h": float(ticker["volume"]),
                "price_change_pct": float(ticker["priceChangePercent"]),
            }
            snapshots.append(snapshot)
            fetch_count += 1

            if fetch_count % 10 == 0:
                print(f"Fetched {fetch_count} snapshots, latest mid={snapshot['mid']:.6f}")

            await asyncio.sleep(interval)
        except Exception as e:
            print(f"Error fetching snapshot: {e}")
            await asyncio.sleep(interval)

    # Save to file
    filename = f"bullausdt_snapshots_{int(start_time)}.jsonl"
    with open(filename, "w") as f:
        for s in snapshots:
            f.write(json.dumps(s) + "\n")

    print(f"\n✅ Saved {len(snapshots)} snapshots to {filename}")
    return filename


if __name__ == "__main__":
    # Fetch 10 minutes of data at 1-second intervals
    asyncio.run(fetch_snapshots(symbol="BULLAUSDT", duration_seconds=600, interval=1.0))
