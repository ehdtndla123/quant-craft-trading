import asyncio
import aiohttp
import ssl
import certifi
from .base_collector import BaseCollector

class BinanceFundingRate(BaseCollector):
    def __init__(self, symbol, producer, topic):
        super().__init__(symbol, producer, topic)
        self.base_url = "https://fapi.binance.com"
        self.is_running = True
        self.session = None

    async def create_session(self):
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context))

    async def fetch_funding_rate(self):
        if not self.session:
            await self.create_session()

        endpoint = f"{self.base_url}/fapi/v1/fundingRate"
        params = {
            "symbol": self.symbol.replace("/", ""),
            "limit": 1
        }
        async with self.session.get(endpoint, params=params) as response:
            data = await response.json()
            return data[0] if isinstance(data, list) and data else None

    async def start(self):
        print(f"Starting Binance funding rate collector for {self.symbol}")
        try:
            while self.is_running:
                try:
                    data = await self.fetch_funding_rate()
                    if data:
                        self.producer.send(self.topic, {
                            'exchange': 'binance',
                            'symbol': self.symbol,
                            'type': 'funding_rate',
                            'data': data
                        })
                    print(data)
                except Exception as e:
                    print(f"Error fetching funding rate for {self.symbol}: {e}")
                await asyncio.sleep(300)  # 5 minutes
        finally:
            if self.session:
                await self.session.close()

    async def stop(self):
        self.is_running = False
        if self.session:
            await self.session.close()