from .base_collector import RESTAPICollector
from influxdb_client import Point, WritePrecision


class BinanceFundingRate(RESTAPICollector):
    def __init__(self, symbol, kafka_bootstrap_servers, topic, db_manager):
        super().__init__(symbol, kafka_bootstrap_servers, topic, db_manager)
        self.base_url = "https://fapi.binance.com"

    async def fetch_data(self):
        endpoint = f"{self.base_url}/fapi/v1/fundingRate"
        params = {
            "symbol": self.symbol.replace("/", ""),
            "limit": 1
        }
        async with self.session.get(endpoint, params=params) as response:
            data = await response.json()
            return data[0] if isinstance(data, list) and data else None

    async def process_data(self, data):
        kafka_data = {
            'exchange': 'binance',
            'symbol': self.symbol,
            'type': 'funding_rate',
            'data': data
        }
        await self.send_to_kafka(kafka_data)

        point = Point("funding_rate") \
            .tag("exchange", "binance") \
            .tag("symbol", self.symbol) \
            .field("funding_rate", float(data['fundingRate'])) \
            .time(int(data['fundingTime']), WritePrecision.MS)

        await self.store_to_influxdb(point)

    def get_interval(self):
        return 300
