class PositionTracker:
    def __init__(self):
        self.long_qty = 0.0
        self.short_qty = 0.0
        self.long_entry = 0.0
        self.short_entry = 0.0
        self.realized_pnl = 0.0

    def update_from_api(self, positions: list, symbol: str):
        self.long_qty = self.short_qty = 0.0
        for p in positions:
            if p["symbol"] != symbol:
                continue
            qty = abs(float(p["positionAmt"]))
            entry = float(p["entryPrice"])
            if p["positionSide"] == "LONG" and qty > 0:
                self.long_qty = qty
                self.long_entry = entry
            elif p["positionSide"] == "SHORT" and qty > 0:
                self.short_qty = qty
                self.short_entry = entry

    def increment_long(self, qty: float):
        self.long_qty += qty

    def increment_short(self, qty: float):
        self.short_qty += qty

    def reset_side(self, side: str):
        if side == "LONG":
            self.long_qty = 0.0
            self.long_entry = 0.0
        else:
            self.short_qty = 0.0
            self.short_entry = 0.0

    def add_pnl(self, amount: float):
        self.realized_pnl += amount