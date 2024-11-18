import websockets
import json
from .base_collector import BaseCollector
from influxdb_client import Point, WritePrecision
import asyncio


class OkxLiquidation(BaseCollector):
    def __init__(self, symbol, kafka_bootstrap_servers, topic, db_manager=None):
        super().__init__(symbol, kafka_bootstrap_servers, topic, db_manager)
        self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        self.ws = None
        self.inst_types = ["SWAP", "FUTURES"]
        self.max_retries = 5
        self.retry_delay = 5

    async def fetch_data(self):
        while True:
            try:
                if not self.ws:
                    self.logger.info("WebSocket 연결 시작...")
                    self.ws = await websockets.connect(self.ws_url, ssl=self.ssl_context)
                    await self._subscribe(self.ws)
                    self.logger.info(f"Okx Websocket 연결 완료 : {self.symbol} liquidation data")

                message = await self.ws.recv()
                return message

            except websockets.exceptions.ConnectionClosed as e:
                self.logger.error(f"WebSocket 연결이 닫혔습니다. 코드: {e.code}, 이유: {e.reason}")
                await self.reset_connection()
            except Exception as e:
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
        if 'data' in data:
            for liquidation in data['data']:
                inst_id = liquidation.get('instId', '')
                symbol = inst_id.split('-')[0]
                if symbol.strip() == self.symbol.strip():
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
                    print(kafka_data)
                    await self.send_to_kafka(kafka_data)

                    point = Point("liquidation") \
                        .tag("exchange", "okx") \
                        .tag("symbol", liquidation.get('instFamily', '').replace('-', '/')) \
                        .field("price", float(details['bkPx'])) \
                        .field("quantity", float(details['sz'])) \
                        .field("side", details['side']) \
                        .time(int(details['ts']), WritePrecision.MS)

                    await self.store_to_influxdb(point)
        elif 'event' in data:
            if data['event'] == 'subscribe':
                self.logger.info(f"채널 구독 성공: {data.get('arg', {}).get('channel')}")
            elif data['event'] == 'error':
                self.logger.error(f"구독 오류 발생: {data.get('msg', '')}")

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
        self.is_running = False
        if self.ws:
            await self.ws.close()
        await super().stop()
        self.logger.info(f"Okx Websocket 연결 종료 : {self.symbol} liquidation data")

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