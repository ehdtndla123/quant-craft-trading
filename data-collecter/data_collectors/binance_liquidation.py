import asyncio
import websockets
import json
import ssl
import certifi
from .base_collector import BaseCollector


class BinanceLiquidation(BaseCollector):
    def __init__(self, symbol, producer, topic):
        super().__init__(symbol, producer, topic)
        self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol.lower().replace('/', '')}@forceOrder"
        self.ws = None
        self.is_running = False
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    async def start(self):
        self.is_running = True
        while self.is_running:
            try:
                async with websockets.connect(self.ws_url, ssl=self.ssl_context) as websocket:
                    self.ws = websocket
                    print(f"Connected to Binance WebSocket for {self.symbol} liquidation data")
                    while self.is_running:
                        message = await websocket.recv()
                        await self._process_message(message)
            except websockets.ConnectionClosed:
                print(f"WebSocket connection closed for {self.symbol}.")
            except Exception as e:
                print(f"Error in Binance liquidation WebSocket for {self.symbol}: {e}")

            if self.is_running:
                await asyncio.sleep(5)

    async def _process_message(self, message):
        data = json.loads(message)
        self.producer.send(self.topic, {
            'exchange': 'binance',
            'symbol': self.symbol,
            'type': 'liquidation',
            'data': data
        })

    async def stop(self):
        self.is_running = False
        if self.ws:
            await self.ws.close()
        print(f"Stopped Binance liquidation data collection for {self.symbol}")