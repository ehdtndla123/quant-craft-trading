import websockets
import json
from .base_collector import BaseCollector
from influxdb_client import Point, WritePrecision


class OkxLiquidation(BaseCollector):
    def __init__(self, symbol, kafka_bootstrap_servers, topic, db_manager=None):
        super().__init__(symbol, kafka_bootstrap_servers, topic, db_manager)
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        self.ws = None
        self.inst_types = ["SWAP", "FUTURES"]
        # self.symbols = [symbol.upper() for symbol in symbols] if symbols else []

    async def fetch_data(self):
        if not self.ws:
            self.ws = await websockets.connect(self.ws_url, ssl=self.ssl_context)
            await self._subscribe(self.ws)
            self.logger.info(f"Okx Websocket 연결 완료 : {self.symbol} liquidation data")
        return await self.ws.recv()

    async def process_data(self, message):
        data = json.loads(message)
        if 'data' in data:
            for liquidation in data['data']:
                inst_id = liquidation.get('instId', '')
                symbol = inst_id.split('-')[0]
                if symbol.strip() == self.symbol.strip():
                # if not self.symbols or symbol in self.symbols:
                    details = liquidation['details'][0]
                    kafka_data = {
                        'exchange': 'okx',
                        'symbol': symbol,
                        'type': 'liquidation',
                        'data': {
                            "price": details['bkPx'],
                            "quantity": details['sz'],
                            "side": details['side']
                        }
                    }
                    await self.send_to_kafka(kafka_data)

                    point = Point("liquidation") \
                        .tag("exchange", "okx") \
                        .tag("symbol", liquidation.get('instFamily', '').replace('-', '/')) \
                        .field("price", float(details['bkPx'])) \
                        .field("quantity", float(details['sz'])) \
                        .field("side", details['side']) \
                        .time(int(details['ts']), WritePrecision.MS)

                    await self.store_to_influxdb(point)

    async def _subscribe(self, websocket):
        subscribe_message = {
            "op": "subscribe",
            "args": [
                {
                    "channel": "liquidation-orders",
                    "instType": inst_type
                } for inst_type in self.inst_types
            ]
        }
        await websocket.send(json.dumps(subscribe_message))

    def get_interval(self):
        return 0

    async def stop(self):
        if self.ws:
            await self.ws.close()
        await super().stop()
        self.logger.info(f"Okx Websocket 연결 종료 : {self.symbol} liquidation data")

