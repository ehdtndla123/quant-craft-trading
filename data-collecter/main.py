import asyncio
import yaml
from kafka import KafkaProducer
import json
import ccxt.pro as ccxtpro
from data_collectors.binance_liquidation import BinanceLiquidation
from data_collectors.binance_open_interest import BinanceOpenInterest
from data_collectors.binance_funding_rate import BinanceFundingRate
from data_collectors.generic_exchange_market_data import GenericExchangeMarketData


class DataCollector:
    def __init__(self, config):
        self.config = config
        self.producer = KafkaProducer(
            bootstrap_servers=config['kafka']['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        # 추후에 압축 및 직렬화 방식 비교후 지정.
        # batch 설정 필요
        # acks 설정 필요

        self.exchanges = {}
        self.topics = config['kafka']['topics']
        self.tasks = []
        self.collectors = []

    async def initialize_exchanges(self):
        for exchange_config in self.config['exchanges']:
            exchange_name = exchange_config['name']
            exchange_class = getattr(ccxtpro, exchange_name)
            self.exchanges[exchange_name] = exchange_class()

    async def collect_data(self):
        for exchange_config in self.config['exchanges']:
            exchange_name = exchange_config['name']
            exchange = self.exchanges[exchange_name]
            for symbol in exchange_config['symbols']:

                collector = GenericExchangeMarketData(
                    exchange,
                    exchange_name,
                    symbol,
                    self.producer,
                    self.topics
                )
                self.collectors.append(collector)
                self.tasks.append(asyncio.create_task(collector.start()))

                for data_type in exchange_config['data_types']:
                    if data_type == 'liquidation' and exchange_name == 'binance':
                        collector = BinanceLiquidation(symbol, self.producer, self.topics['liquidation'])
                        self.collectors.append(collector)
                        self.tasks.append(asyncio.create_task(collector.start()))
                    elif data_type == 'open_interest' and exchange_name == 'binance':
                        collector = BinanceOpenInterest(symbol, self.producer, self.topics['open_interest'])
                        self.collectors.append(collector)
                        self.tasks.append(asyncio.create_task(collector.start()))
                    elif data_type == 'funding_rate' and exchange_name == 'binance':
                        collector = BinanceFundingRate(symbol, self.producer, self.topics['funding_rate'])
                        self.collectors.append(collector)
                        self.tasks.append(asyncio.create_task(collector.start()))

        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def run(self):
        await self.initialize_exchanges()
        await self.collect_data()

    async def close(self):
        for collector in self.collectors:
            await collector.stop()
        for task in self.tasks:
            task.cancel()
        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        for exchange in self.exchanges.values():
            await exchange.close()
        self.producer.close()


async def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    collector = DataCollector(config)
    try:
        await collector.run()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())
