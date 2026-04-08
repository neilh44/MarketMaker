import asyncio
import hashlib
import hmac
import logging
import os
import time
import urllib.parse
import aiohttp

log = logging.getLogger("engine")

class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://fapi.binance.com"):
        self._api_key = api_key
        self._api_secret = api_secret
        self.base_url = base_url

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params)
        sig = hmac.new(self._api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    async def _request(self, method: str, path: str, params: dict) -> dict:
        params = self._sign(params.copy())
        headers = {"X-MBX-APIKEY": self._api_key}
        url = f"{self.base_url}{path}"
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url, params=params, headers=headers) as resp:
                    return await resp.json()
            elif method == "POST":
                async with session.post(url, params=params, headers=headers) as resp:
                    return await resp.json()
            elif method == "DELETE":
                async with session.delete(url, params=params, headers=headers) as resp:
                    return await resp.json()

    async def get(self, path: str, params: dict) -> dict:
        return await self._request("GET", path, params)

    async def post(self, path: str, params: dict) -> dict:
        return await self._request("POST", path, params)

    async def delete(self, path: str, params: dict) -> dict:
        return await self._request("DELETE", path, params)