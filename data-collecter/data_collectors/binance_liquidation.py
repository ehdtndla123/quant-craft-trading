from .base_collector import BaseCollector
import websockets
import json
from influxdb_client import Point, WritePrecision


class BinanceLiquidation(BaseCollector):
    def __init__(self, symbol, kafka_bootstrap_servers, topic, db_manager):
        super().__init__(symbol, kafka_bootstrap_servers, topic, db_manager)
        self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol.lower().replace('/', '')}@forceOrder"
        self.ws = None

    async def fetch_data(self):
        if not self.ws:
            self.ws = await websockets.connect(self.ws_url, ssl=self.ssl_context)
            self.logger.info(f"Binance Websocket 연결 완료 : {self.symbol} liquidation data")
        return await self.ws.recv()

    async def process_data(self, message):
        data = json.loads(message)

        kafka_data = {
            'exchange': 'binance',
            'symbol': self.symbol,
            'type': 'liquidation',
            'data': {
                "order_price": float(data['o']['p']),
                "average_price": float(data['o']['ap']),
                "quantity": float(data['o']['q']),
                "side": data['o']['S']
            }
        }
        await self.send_to_kafka(kafka_data)

        point = Point("liquidation") \
            .tag("exchange", "binance") \
            .tag("symbol", self.symbol) \
            .field("order_price", float(data['o']['p'])) \
            .field("average_price", float(data['o']['ap'])) \
            .field("quantity", float(data['o']['q'])) \
            .field("side", data['o']['S']) \
            .time(data['o']['T'], WritePrecision.MS)

        await self.store_to_influxdb(point)

    def get_interval(self):
        return 0

    async def stop(self):
        if self.ws:
            await self.ws.close()
        await super().stop()
        self.logger.info(f"Binance Websocket 연결 종료 : {self.symbol} liquidation data")
