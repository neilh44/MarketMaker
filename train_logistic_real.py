#!/usr/bin/env python3
"""
Train logistic regression on BULLAUSDT snapshot data with robust error handling.
"""

import json
import math
import sys
import random
import os

# Check if logistic_regression.py exists
try:
    from logistic_regression import LogisticRegression
except ImportError:
    print("ERROR: logistic_regression.py not found in current directory.")
    sys.exit(1)

def load_snapshots(filename):
    if not os.path.exists(filename):
        print(f"ERROR: File '{filename}' not found.")
        sys.exit(1)
    snapshots = []
    with open(filename, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue
            # Handle timestamp
            ts = s.get("timestamp")
            if ts is None:
                s["timestamp"] = 0
            elif isinstance(ts, str):
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    s["timestamp"] = int(dt.timestamp() * 1000)
                except:
                    try:
                        s["timestamp"] = int(ts)
                    except:
                        s["timestamp"] = 0
            # Ensure required fields exist
            if "mid" not in s or s["mid"] <= 0:
                continue
            if "best_bid" not in s or "best_ask" not in s:
                continue
            if "bids" not in s or "asks" not in s:
                continue
            snapshots.append(s)
    return snapshots

def compute_features(snapshots, lookback_seconds=60, future_steps=5):
    total = len(snapshots)
    print(f"Computing features from {total} snapshots...")
    generated = 0
    for i in range(lookback_seconds, total - future_steps):
        window = snapshots[i-lookback_seconds:i+1]
        s = window[-1]
        mid = s["mid"]
        if mid <= 0:
            continue
        # Compute mids list
        mids = [w["mid"] for w in window if w.get("mid", 0) > 0]
        if len(mids) < 2:
            continue
        mid_mean = sum(mids) / len(mids)
        mid_std = (sum((m - mid_mean)**2 for m in mids) / len(mids)) ** 0.5
        norm_mid = (mid - mid_mean) / (mid_std + 1e-8)

        # Spread
        spread_bps = (s["best_ask"] - s["best_bid"]) / mid * 10000

        # OBI
        bid_vol = sum(float(b[1]) for b in s["bids"]) if s["bids"] else 0
        ask_vol = sum(float(a[1]) for a in s["asks"]) if s["asks"] else 0
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)

        # Volatility
        returns = [(mids[j] - mids[j-1]) / mids[j-1] for j in range(1, len(mids))]
        volatility = (sum(r*r for r in returns) / len(returns)) ** 0.5 if returns else 0.0

        # Momentum
        if len(window) >= 5:
            recent = sum(snapshots[i-5:i+1][k]["mid"] for k in range(5)) / 5
            momentum = (mid - recent) / recent
        else:
            momentum = 0.0

        # Volume ratio
        vol_ratio = bid_vol / (ask_vol + 1e-8)

        # Risk-adjusted spread
        spread_vol_ratio = spread_bps / (volatility * 10000 + 1e-8)

        features = [norm_mid, spread_bps, obi, volatility, momentum, vol_ratio, spread_vol_ratio]

        future_mid = snapshots[i + future_steps]["mid"]
        label = 1 if future_mid > mid else 0

        yield features, label
        generated += 1
    print(f"Generated {generated} feature samples.")

def feature_correlation(X, y, feature_names):
    n_features = len(X[0])
    n_samples = len(X)
    print("\nFeature correlations with label:")
    for j in range(n_features):
        feat_vals = [xi[j] for xi in X]
        mean_f = sum(feat_vals) / n_samples
        mean_y = sum(y) / n_samples
        num = sum((feat_vals[i] - mean_f) * (y[i] - mean_y) for i in range(n_samples))
        den = (sum((f - mean_f)**2 for f in feat_vals) * sum((yi - mean_y)**2 for yi in y)) ** 0.5
        corr = num / (den + 1e-8)
        print(f"  {j}: {feature_names[j]:20s} corr = {corr:+.4f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_logistic_real.py <snapshots_file.jsonl>")
        sys.exit(1)

    filename = sys.argv[1]
    print(f"Loading {filename}...")
    snapshots = load_snapshots(filename)
    print(f"Loaded {len(snapshots)} valid snapshots.")

    if len(snapshots) < 65:  # need at least lookback + future_steps
        print(f"ERROR: Need at least 65 snapshots, got {len(snapshots)}. Collect more data.")
        sys.exit(1)

    X, y = [], []
    for feat, label in compute_features(snapshots, lookback_seconds=60, future_steps=5):
        X.append(feat)
        y.append(label)

    print(f"Extracted {len(X)} samples, each with {len(X[0])} features.")

    if len(X) == 0:
        print("ERROR: No samples generated. Check feature extraction logic.")
        sys.exit(1)

    pos = sum(y)
    neg = len(y) - pos
    print(f"Class balance: UP={pos} ({pos/len(y)*100:.1f}%), DOWN={neg} ({neg/len(y)*100:.1f}%)")

    feature_names = ["norm_mid", "spread_bps", "obi", "volatility", "momentum", "vol_ratio", "spread_vol_ratio"]
    feature_correlation(X, y, feature_names)

    # Shuffle and split
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    X, y = list(X), list(y)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = LogisticRegression(input_size=len(X[0]), lr=0.005)
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=True)

    correct = 0
    for x, true_y in zip(X_test, y_test):
        pred = model.predict(x)
        if pred == true_y:
            correct += 1
    acc = correct / len(X_test)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Baseline (predict majority): {max(pos, neg)/len(y):.4f}")

    print("\nLearned weights:")
    for j, name in enumerate(feature_names):
        print(f"  {name:20s}: {model.weights[j]:+.4f}")

if __name__ == "__main__":
    main()
