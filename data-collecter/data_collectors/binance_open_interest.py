from influxdb_client import Point, WritePrecision
from .base_collector import RESTAPICollector


class BinanceOpenInterest(RESTAPICollector):
    def __init__(self, symbol, kafka_bootstrap_servers, topic, db_manager):
        super().__init__(symbol, kafka_bootstrap_servers, topic, db_manager)
        self.base_url = "https://fapi.binance.com"

    async def fetch_data(self):
        endpoint = f"{self.base_url}/fapi/v1/openInterest"
        params = {
            "symbol": self.symbol.replace("/", ""),
        }
        async with self.session.get(endpoint, params=params) as response:
            data = await response.json()
            return data

    async def process_data(self, data):
        kafka_data = {
            'exchange': 'binance',
            'symbol': self.symbol,
            'type': 'open_interest',
            'data': data
        }
        await self.send_to_kafka(kafka_data)

        point = Point("open_interest") \
            .tag("exchange", "binance") \
            .tag("symbol", self.symbol) \
            .field("open_interest", float(data['openInterest'])) \
            .time(int(data['time']), WritePrecision.MS)
        await self.store_to_influxdb(point)


    def get_interval(self):
        return 60
