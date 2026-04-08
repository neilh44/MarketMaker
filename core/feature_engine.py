# core/feature_engine.py (new file)
import numpy as np

class FeatureEngine:
    def __init__(self, order_book_depth: int = 10):
        self.depth = order_book_depth

    def compute_state_vector(self, order_book: dict, mid: float, volatility: float,
                             long_qty: float, short_qty: float) -> np.ndarray:
        """
        Returns a normalized feature vector representing the current market state.
        Features:
          - Normalized bid/ask prices and volumes (top depth levels)
          - Mid price (log)
          - Spread in bps
          - Order book imbalance (OBI)
          - Volatility (ATR)
          - 1-minute price change (%)
          - Normalized inventory (long - short) / base_qty
          - Unrealized PnL (clipped)
        """
        features = []

        # 1. Order book snapshot (top 'depth' levels)
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        for i in range(self.depth):
            bid_price = bids[i][0] if i < len(bids) else 0.0
            bid_vol = bids[i][1] if i < len(bids) else 0.0
            ask_price = asks[i][0] if i < len(asks) else 0.0
            ask_vol = asks[i][1] if i < len(asks) else 0.0
            features.extend([bid_price / mid, bid_vol, ask_price / mid, ask_vol])

        # 2. Mid price (log)
        features.append(np.log(mid))

        # 3. Spread in bps
        best_bid = bids[0][0] if bids else mid
        best_ask = asks[0][0] if asks else mid
        spread_bps = (best_ask - best_bid) / mid * 10000
        features.append(spread_bps)

        # 4. Order book imbalance (OBI) - simple version
        total_bid_vol = sum(b[1] for b in bids[:self.depth])
        total_ask_vol = sum(a[1] for a in asks[:self.depth])
        obi = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-8)
        features.append(obi)

        # 5. Volatility (ATR)
        features.append(volatility)

        # 6. 1-minute price change (%)
        # (You'll need to track this in feed_handler; assume it's available)
        price_change_pct = getattr(self, 'price_change_1m', 0.0)
        features.append(price_change_pct)

        # 7. Normalized inventory
        net_inv = long_qty - short_qty
        # Normalize by base quote quantity (you can pass this from engine)
        base_qty = 40  # example; adjust to your actual base qty
        features.append(net_inv / base_qty)

        # 8. Unrealized PnL (clipped)
        unrealized_pnl = 0.0  # get from tracker
        features.append(np.clip(unrealized_pnl, -10, 10))

        return np.array(features, dtype=np.float32)
