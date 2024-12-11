import asyncio
from .base_collector import BaseCollector
from influxdb_client import Point, WritePrecision


class GenericExchangeMarketData(BaseCollector):
    def __init__(self, exchange, exchange_name, symbol, kafka_bootstrap_servers, topic, db_manager=None):
        super().__init__(symbol, kafka_bootstrap_servers, topic, db_manager)
        self.exchange = exchange
        self.exchange_name = exchange_name

    async def fetch_data(self):
        try:
            ohlcv = await self.exchange.watch_ohlcv(self.symbol, '1m')
            return {
                'ohlcv': ohlcv
            }
        except Exception as e:
            self.logger.error(f"Error fetching data for {self.exchange_name} {self.symbol}: {e}")
            return None

    async def process_data(self, message):
        try:
            ohlcv = message['ohlcv'][0]
            kafka_data = {
                'exchange': self.exchange_name,
                'symbol': self.symbol,
                'type': 'ohlcv',
                'timestamp': ohlcv[0],
                'open': ohlcv[1],
                'high': ohlcv[2],
                'low': ohlcv[3],
                'close': ohlcv[4],
                'volume': ohlcv[5]
            }
            await self.send_to_kafka(kafka_data)

            if self.db_manager:
                point = Point("ohlcv") \
                    .tag("exchange", self.exchange_name) \
                    .tag("symbol", self.symbol) \
                    .field("open", float(ohlcv[1])) \
                    .field("high", float(ohlcv[2])) \
                    .field("low", float(ohlcv[3])) \
                    .field("close", float(ohlcv[4])) \
                    .field("volume", float(ohlcv[5])) \
                    .time(ohlcv[0], WritePrecision.MS)
                await self.store_to_influxdb(point)

        except Exception as e:
            self.logger.error(f"Error in process_data: {e}", exc_info=True)
            raise

    def get_interval(self):
        return 0

    async def stop(self):
        self.is_running = False
        await super().stop()
        self.logger.info(f"Generic Exchange 연결 종료 : {self.symbol}")

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