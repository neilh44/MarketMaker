import logging
from dataclasses import dataclass
from .exchange_client import BinanceClient

log = logging.getLogger("engine")

@dataclass
class ActiveOrder:
    order_id: int
    side: str
    price: float
    qty: float
    position_side: str

@dataclass
class TpOrder:
    order_id: int
    position_side: str
    entry_price: float
    tp_price: float
    qty: float

@dataclass
class SlOrder:
    order_id: int          # algoId
    position_side: str
    entry_price: float
    sl_price: float
    qty: float

class OrderManager:
    def __init__(self, symbol: str, client: BinanceClient, step_size: float):
        self.symbol = symbol
        self.client = client
        self.step_size = step_size

    # Add this method inside the OrderManager class
    async def fetch_positions(self) -> list:
        """Fetch open positions from Binance Futures."""
        return await self.client.get("/fapi/v2/positionRisk", {"symbol": self.symbol})

    # --- TP order (LIMIT) ---
    async def place_tp_order(self, side: str, pos_side: str, tp_price: float, qty: float,
                             entry_price: float = 0.0) -> TpOrder | None:
        decimals = len(str(self.step_size).rstrip('0').split('.')[-1])
        params = {
            "symbol": self.symbol,
            "side": side,
            "positionSide": pos_side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "price": f"{tp_price:.6f}".rstrip('0').rstrip('.'),
            "quantity": f"{qty:.{decimals}f}",
        }
        r = await self.client.post("/fapi/v1/order", params)
        if "orderId" not in r:
            log.warning(f"TP order rejected [{r.get('code')}]: {r.get('msg')}")
            return None
        order_id = r["orderId"]
        if not await self.verify_order_exists(order_id):
            log.warning(f"TP order {order_id} placed but not found")
            return None
        return TpOrder(order_id=order_id, position_side=pos_side, entry_price=entry_price,
                       tp_price=tp_price, qty=qty)

    # --- SL order (STOP‑LIMIT via Algo API) ---

    async def place_sl_order(self, side: str, pos_side: str, stop_price: float, qty: float,
                         entry_price: float, limit_offset_bps: float = 5.0) -> SlOrder | None:
        decimals = len(str(self.step_size).rstrip('0').split('.')[-1])
        offset = limit_offset_bps / 10_000

        if pos_side == "LONG":
            # For LONG close: limit price slightly below stop
            limit_price = stop_price * (1 - offset)
        else:
            # For SHORT close: limit price slightly above stop
            limit_price = stop_price * (1 + offset)

        # Round to correct tick size (0.00001 for STOUSDT)
        limit_price = round(limit_price, 5)
        stop_price = round(stop_price, 5)

        params = {
            "symbol": self.symbol,
            "side": side,
            "positionSide": pos_side,
            "algoType": "CONDITIONAL",
            "type": "STOP",
            "quantity": f"{qty:.{decimals}f}",
            "price": f"{limit_price:.5f}".rstrip('0').rstrip('.'),
            "triggerPrice": f"{stop_price:.5f}".rstrip('0').rstrip('.'),
            "workingType": "CONTRACT_PRICE",
            "timeInForce": "GTC",
        }
    
        r = await self.client.post("/fapi/v1/algoOrder", params)
        if "algoId" not in r:
            log.warning(f"SL algo order rejected [{r.get('code')}]: {r.get('msg')}")
            return None
        algo_id = r["algoId"]
        return SlOrder(order_id=algo_id, position_side=pos_side, entry_price=entry_price,
                       sl_price=stop_price, qty=qty)

    # --- Entry order (BID/ASK) ---
    async def place_entry_order(self, side: str, pos_side: str, price: float, qty: float) -> ActiveOrder | None:
        decimals = len(str(self.step_size).rstrip('0').split('.')[-1])
        params = {
            "symbol": self.symbol,
            "side": side,
            "positionSide": pos_side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "price": f"{price:.6f}".rstrip('0').rstrip('.'),
            "quantity": f"{qty:.{decimals}f}",
        }
        r = await self.client.post("/fapi/v1/order", params)
        if "orderId" not in r:
            log.warning(f"Entry order rejected [{r.get('code')}]: {r.get('msg')}")
            return None
        return ActiveOrder(order_id=r["orderId"], side=side, price=price, qty=qty, position_side=pos_side)

    async def cancel_order(self, order_id: int):
        await self.client.delete("/fapi/v1/order", {"symbol": self.symbol, "orderId": order_id})

    async def cancel_algo_order(self, algo_id: int):
        await self.client.delete("/fapi/v1/algoOrder", {"symbol": self.symbol, "algoId": algo_id})

    async def verify_order_exists(self, order_id: int) -> bool:
        orders = await self.client.get("/fapi/v1/openOrders", {"symbol": self.symbol})
        if not isinstance(orders, list):
            return False
        return order_id in [o["orderId"] for o in orders]