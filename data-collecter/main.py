import asyncio
import yaml
from aiokafka import AIOKafkaProducer
import json
import ccxt.pro as ccxtpro
import importlib
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import ASYNCHRONOUS
import datetime
from aiohttp import web

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)


class CollectorStatus:
    def __init__(self, collector, exchange, symbol, data_type):
        self.collector = collector
        self.exchange = exchange
        self.symbol = symbol
        self.data_type = data_type
        self.last_update = datetime.datetime.now()

    def to_dict(self):
        return {
            "거래소": self.exchange,
            "심볼": self.symbol,
            "데이터 유형": self.data_type,
            "마지막 업데이트": self.last_update.isoformat(),
        }


class DBManager:
    def __init__(self, config):
        self.client = InfluxDBClient(
            url=config['url'],
            token=config['token'],
            org=config['org']
        )
        self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)
        self.bucket = config['bucket']

    def close(self):
        self.client.close()


class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, data_collector):
        self.data_collector = data_collector

    def on_modified(self, event):
        if event.src_path.endswith('config.yaml'):
            logging.info("설정 파일 변경 감지. 설정을 다시 로드합니다.")
            self.data_collector.schedule_config_reload()


class DataCollector:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        self.producer = None
        self.exchanges = {}
        self.topics = self.config['kafka']['topics']
        self.tasks = []
        self.collectors = {}
        self.db_manager = DBManager(self.config['influxdb'])
        self.reload_event = asyncio.Event()
        self.collector_statuses = {}
        self.app = web.Application()
        self.app.router.add_get('/status', self.get_status)
        self.runner = None

    def load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    async def reload_config(self):
        new_config = self.load_config()
        logging.info("리로딩!")
        await self.update_collectors(new_config)

    def schedule_config_reload(self):
        self.reload_event.set()

    async def config_reload_handler(self):
        while True:
            await self.reload_event.wait()
            self.reload_event.clear()
            await self.reload_config()

    async def update_collectors(self, new_config):
        new_collectors = set()
        for exchange_config in new_config['exchanges']:
            exchange_name = exchange_config['name']
            for symbol in exchange_config['symbols']:
                for data_type in exchange_config['data_types']:
                    collector_key = (exchange_name, symbol, data_type)
                    new_collectors.add(collector_key)
                    if collector_key not in self.collectors:
                        await self.add_collector(exchange_name, symbol, data_type)

        for collector_key in list(self.collectors.keys()):
            if collector_key not in new_collectors:
                await self.remove_collector(collector_key)

    async def add_collector(self, exchange_name, symbol, data_type):
        collector_class = self.get_collector_class(exchange_name, data_type)
        if collector_class:
            collector = collector_class(symbol, self.config['kafka']['bootstrap_servers'],
                                        self.topics[data_type], self.db_manager)
            self.collectors[(exchange_name, symbol, data_type)] = collector
            self.collector_statuses[(exchange_name, symbol, data_type)] = CollectorStatus(collector, exchange_name,
                                                                                          symbol, data_type)
            self.tasks.append(asyncio.create_task(collector.run()))
            logging.info(f"Collector 추가: {exchange_name} - {symbol} - {data_type}")

    async def remove_collector(self, collector_key):
        collector = self.collectors.pop(collector_key)
        await collector.stop()
        task = next(t for t in self.tasks if t.get_coro().__name__ == collector.run.__name__)
        task.cancel()
        self.tasks.remove(task)
        self.collector_statuses.pop(collector_key)
        logging.info(f"Collector 제거: {collector_key}")

    def get_collector_class(self, exchange_name, data_type):
        module_name = f"data_collectors.{exchange_name}_{data_type}"
        class_name = f"{exchange_name.capitalize()}{data_type.replace('_', ' ').title().replace(' ', '')}"
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            logging.error(f"Collector를 찾을 수 없습니다: {module_name}.{class_name}")
            return None

    async def initialize_producer(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        await self.producer.start()

    async def initialize_exchanges(self):
        for exchange_config in self.config['exchanges']:
            exchange_name = exchange_config['name']
            exchange_class = getattr(ccxtpro, exchange_name)
            self.exchanges[exchange_name] = exchange_class()

    async def get_status(self, request):
        statuses = [status.to_dict() for status in self.collector_statuses.values()]
        return web.json_response(statuses)

    async def start_http_server(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, 'localhost', 8080)
        await site.start()

    async def collect_data(self):
        await self.update_collectors(self.config)
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def run(self):
        await self.initialize_producer()
        await self.initialize_exchanges()
        self.tasks.append(asyncio.create_task(self.config_reload_handler()))
        await self.start_http_server()

        # 설정 파일 변경 감지를 위한 watchdog 설정
        event_handler = ConfigChangeHandler(self)
        observer = Observer()
        observer.schedule(event_handler, path='.', recursive=False)
        observer.start()

        try:
            await self.collect_data()
        finally:
            observer.stop()
            observer.join()
            if self.runner:
                await self.runner.cleanup()

    async def close(self):
        for collector in self.collectors.values():
            await collector.stop()
        for task in self.tasks:
            task.cancel()
        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        for exchange in self.exchanges.values():
            await exchange.close()
        await self.producer.stop()
        self.db_manager.close()
        logging.info("모든 리소스가 정상적으로 종료되었습니다.")


async def main():
    collector = DataCollector('config.yaml')
    try:
        logging.info("데이터 수집을 시작합니다.")
        await collector.run()
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())
