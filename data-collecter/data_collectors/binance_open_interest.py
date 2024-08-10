import asyncio
import aiohttp
import ssl
import certifi
from .base_collector import BaseCollector


class BinanceOpenInterest(BaseCollector):
    def __init__(self, symbol, producer, topic):
        super().__init__(symbol, producer, topic)
        self.base_url = "https://fapi.binance.com"
        self.is_running = True
        self.session = None

    async def create_session(self):
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context))

    async def fetch_open_interest(self):
        if not self.session:
            await self.create_session()

        endpoint = f"{self.base_url}/fapi/v1/openInterest"
        params = {
            "symbol": self.symbol.replace("/", ""),
        }
        async with self.session.get(endpoint, params=params) as response:
            data = await response.json()
            return data

    async def start(self):
        print(f"Starting Binance open interest collector for {self.symbol}")
        try:
            while self.is_running:
                try:
                    data = await self.fetch_open_interest()
                    if data:
                        self.producer.send(self.topic, {
                            'exchange': 'binance',
                            'symbol': self.symbol,
                            'type': 'open_interest',
                            'data': data
                        })
                except Exception as e:
                    print(f"Error fetching open interest for {self.symbol}: {e}")
                await asyncio.sleep(60)
        finally:
            if self.session:
                await self.session.close()

    async def stop(self):
        self.is_running = False
        if self.session:
            await self.session.close()
