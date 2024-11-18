from .base_collector import BaseCollector
import websockets
import json
from influxdb_client import Point, WritePrecision
import asyncio


class BinanceLiquidation(BaseCollector):
    def __init__(self, symbol, kafka_bootstrap_servers, topic, db_manager):
        super().__init__(symbol, kafka_bootstrap_servers, topic, db_manager)
        self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol.lower().replace('/', '')}@forceOrder"
        self.ws = None
        self.max_retries = 5
        self.retry_delay = 5

    async def fetch_data(self):
        while True:
            try:
                if not self.ws:
                    self.logger.info("WebSocket 연결 시작...")
                    self.ws = await websockets.connect(self.ws_url, ssl=self.ssl_context)
                    self.logger.info(f"Binance Websocket 연결 완료 : {self.symbol} liquidation data")

                message = await self.ws.recv()
                return message

            except websockets.exceptions.ConnectionClosed as e:
                self.logger.error(f"WebSocket 연결이 닫혔습니다. 코드: {e.code}, 이유: {e.reason}")
                await self.reset_connection()
            except Exception as e:
                print(e)
                self.logger.error(f"예기치 않은 오류 발생: {str(e)}")
                await self.reset_connection()

    async def reset_connection(self):
        self.logger.info("WebSocket 연결 재설정 중...")
        if self.ws:
            await self.ws.close()
        self.ws = None
        await asyncio.sleep(self.retry_delay)

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
        print(kafka_data)
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
        self.is_running = False
        if self.ws:
            await self.ws.close()
        await super().stop()
        self.logger.info(f"Binance Websocket 연결 종료 : {self.symbol} liquidation data")

    async def run(self):
        await self.initialize()
        self.is_running = True
        while self.is_running:
            try:
                data = await self.fetch_data()
                if data:
                    await self.process_data(data)
            except Exception as e:
                self.logger.error(f"데이터 수집 중 오류 발생: {e}")
            await asyncio.sleep(self.get_interval())