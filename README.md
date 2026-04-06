# Hybrid LLM Market-Making Bot

A two-layer trading system:
- **Fast path** — asyncio quoting engine posts BID+ASK on every Binance Futures tick
- **Slow path** — Claude (via Anthropic API) advises spread/skew/pause parameters every 20 seconds

## Architecture

```
Binance WS (bookTicker + miniTicker)
        │
   feed_handler.py           ← normalises ticks into SharedState
        │                ╲
quoting_engine.py       llm_advisor.py
(every tick)            (every 20s → Claude API)
        │                ╱
   shared_state.py      ← spread_bps, skew, paused, regime
        │
Binance REST (LIMIT post-only orders)
```

## Setup

```bash
# 1. Clone / unzip the project
cd hybrid_trader

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env with your keys, then:
export $(cat .env | xargs)
```

## Binance requirements

- Futures account with **Hedge Mode** enabled (Portfolio > Position Mode)
- API key with Futures trading permissions
- Sufficient margin for your chosen `--qty`

## Running

```bash
# Paper-safe: start with small qty and watch logs
python main.py --symbol BTCUSDT --qty 10 --interval 20

# With debug logging
python main.py --symbol ETHUSDT --qty 5 --interval 30 --debug

# Custom symbol
python main.py --symbol SOLUSDT --qty 20 --interval 15
```

## Parameters

| Argument | Default | Description |
|---|---|---|
| `--symbol` | `BTCUSDT` | Binance Futures symbol |
| `--qty` | `10.0` | Notional per side in USDT |
| `--interval` | `20` | Seconds between LLM calls |
| `--debug` | off | Verbose logging |

## What Claude outputs (every interval)

```json
{
  "spread_bps": 8.0,
  "skew": -0.3,
  "paused": false,
  "regime": "choppy",
  "max_position_usd": 50.0,
  "reason": "Moderate vol, slight downtrend — lean short"
}
```

The quoting engine immediately picks up these parameters on the next tick.

## Files

```
hybrid_trader/
├── main.py                  ← entrypoint
├── requirements.txt
├── .env.example
├── core/
│   ├── shared_state.py      ← dataclasses + asyncio-safe state
│   ├── feed_handler.py      ← Binance WS client
│   ├── llm_advisor.py       ← Claude API slow loop
│   └── quoting_engine.py    ← fast BID/ASK posting loop
└── logs/
    └── trader.log           ← auto-created on first run
```

## Safety notes

- Orders are placed as `GTX` (post-only / maker-only) — they are cancelled if they would immediately fill as taker
- The LLM can set `paused: true` to cancel all quotes and halt trading
- `--qty 10` means max ~10 USDT notional per side — very small for testing
- Always test on Binance Futures **Testnet** first: change `BINANCE_REST` and `BINANCE_WS_BASE` in the source to the testnet URLs

## Testnet URLs (for safe testing)

In `core/feed_handler.py`:
```python
BINANCE_WS_BASE = "wss://stream.binancefuture.com/stream"
```
In `core/quoting_engine.py`:
```python
BINANCE_REST = "https://testnet.binancefuture.com"
```
