# ── Edit your keys below, then just run: .\run.ps1 ──────────────────────────
$env:GROQ_API_KEY     = "gsk_YOUR_GROQ_KEY_HERE"
$env:BINANCE_API_KEY  = "YOUR_BINANCE_API_KEY_HERE"
$env:BINANCE_API_SECRET = "YOUR_BINANCE_API_SECRET_HERE"

# ── Bot settings ─────────────────────────────────────────────────────────────
$SYMBOL   = "STOUSDT"
$QTY      = 7
$INTERVAL = 20

python main.py --symbol $SYMBOL --qty $QTY --interval $INTERVAL
