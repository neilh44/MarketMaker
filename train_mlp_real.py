#!/usr/bin/env python3
"""
train_mlp_real.py - Train pure Python MLP on Binance snapshot data
Usage: python train_mlp_real.py <snapshots_file.jsonl>
"""

import json
import math
import sys
import random
from pure_mlp import PurePythonMLP

def load_snapshots(filename):
    snapshots = []
    with open(filename, "r") as f:
        for line in f:
            s = json.loads(line.strip())
            # Convert timestamp to integer (Unix ms) if it's a string
            if isinstance(s["timestamp"], str):
                # If it's ISO format, parse it; otherwise assume integer string
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(s["timestamp"].replace('Z', '+00:00'))
                    s["timestamp"] = int(dt.timestamp() * 1000)
                except:
                    s["timestamp"] = int(s["timestamp"])
            snapshots.append(s)
    return snapshots

def compute_features(snapshots, lookback_seconds=60, future_steps=5):
    """
    Extract feature vectors and labels.
    Returns generator of (features, label).
    """
    for i in range(lookback_seconds, len(snapshots) - future_steps):
        window = snapshots[i-lookback_seconds:i+1]
        features = []

        # Price and volume statistics over the window
        mids = [s["mid"] for s in window]
        mid_mean = sum(mids) / len(mids)
        mid_std = (sum((m - mid_mean)**2 for m in mids) / len(mids)) ** 0.5

        # Latest snapshot
        s = window[-1]
        mid = s["mid"]

        # 1. Normalized mid price (z-score over window)
        norm_mid = (mid - mid_mean) / (mid_std + 1e-8)

        # 2. Spread in bps
        spread_bps = (s["best_ask"] - s["best_bid"]) / mid * 10000

        # 3. Order book imbalance (OBI)
        bid_vol = sum(float(b[1]) for b in s["bids"])
        ask_vol = sum(float(a[1]) for a in s["asks"])
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)

        # 4. Volatility (standard deviation of returns in window)
        returns = [(mids[j] - mids[j-1]) / mids[j-1] for j in range(1, len(mids))]
        volatility = (sum(r*r for r in returns) / len(returns)) ** 0.5 if returns else 0.0

        # 5. Recent price change (last 5 seconds vs window average)
        if len(window) >= 5:
            recent_mid = sum(snapshots[i-5:i+1][k]["mid"] for k in range(5)) / 5
            price_momentum = (mid - recent_mid) / recent_mid
        else:
            price_momentum = 0.0

        # 6. Volume ratio (bid/ask)
        vol_ratio = bid_vol / (ask_vol + 1e-8)

        # 7. Spread relative to volatility (risk-adjusted)
        spread_vol_ratio = spread_bps / (volatility * 10000 + 1e-8)

        features = [norm_mid, spread_bps, obi, volatility, price_momentum, vol_ratio, spread_vol_ratio]

        # Label: future price direction
        future_mid = snapshots[i + future_steps]["mid"]
        label = 1 if future_mid > mid else 0

        yield features, label

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_mlp_real.py <snapshots_file.jsonl>")
        sys.exit(1)

    filename = sys.argv[1]
    print(f"Loading {filename}...")
    snapshots = load_snapshots(filename)
    print(f"Loaded {len(snapshots)} snapshots.")

    # Filter out snapshots with zero mid price
    snapshots = [s for s in snapshots if s.get("mid", 0) > 0]
    print(f"After filtering: {len(snapshots)} valid snapshots.")

    X, y = [], []
    for feat, label in compute_features(snapshots, lookback_seconds=60, future_steps=5):
        X.append(feat)
        y.append(label)

    print(f"Extracted {len(X)} samples, each with {len(X[0])} features.")

    # Check class balance
    pos = sum(y)
    neg = len(y) - pos
    print(f"Class balance: UP={pos} ({pos/len(y)*100:.1f}%), DOWN={neg} ({neg/len(y)*100:.1f}%)")

    if len(X) < 50:
        print("Not enough samples. Collect more data (at least 5-10 minutes).")
        sys.exit(1)

    # Shuffle and split
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    X, y = list(X), list(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    mlp = PurePythonMLP(input_size=len(X[0]), hidden_sizes=(32, 16), output_size=1)
    mlp.fit(X_train, y_train, epochs=10, lr=0.005)

    correct = 0
    for x, true_y in zip(X_test, y_test):
        pred = mlp.predict(x)
        if (pred > 0.5) == true_y:
            correct += 1
    acc = correct / len(X_test)
    print(f"\nTest Accuracy: {acc:.4f}")

    # Baseline: predict majority class
    majority = max(pos, neg) / len(y)
    print(f"Baseline (predict majority): {majority:.4f}")

    if acc > majority + 0.02:
        print("✅ Features show predictive power!")
    else:
        print("⚠️ Accuracy not significantly better than baseline.")
        print("   Suggestions: collect more data, try different lookback, or engineer new features.")

if __name__ == "__main__":
    main()
