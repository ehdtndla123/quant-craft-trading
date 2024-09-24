from abc import ABC, abstractmethod
import asyncio
import logging
from aiokafka import AIOKafkaProducer
import json
import ssl
import certifi
import aiohttp


class BaseCollector(ABC):
    def __init__(self, symbol, kafka_bootstrap_servers, topic, db_manager=None):
        self.symbol = symbol  # 배열 또는 단일 심볼
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.topic = topic
        self.db_manager = db_manager
        self.producer = None
        self.is_running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ssl_context = None

    async def initialize(self):
        self.producer = AIOKafkaProducer(bootstrap_servers=self.kafka_bootstrap_servers)
        await self.producer.start()
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    @abstractmethod
    async def fetch_data(self):
        # 데이터 가져오기
        pass

    async def send_to_kafka(self, data):
        try:
            await self.producer.send_and_wait(self.topic, json.dumps(data).encode())
        except Exception as e:
            self.logger.error(f"Kafka로 데이터 전송 실패: {e}")

    async def store_to_influxdb(self, point):
        if self.db_manager:
            try:
                self.db_manager.write_api.write(bucket=self.db_manager.bucket, record=point)
            except Exception as e:
                self.logger.error(f"InfluxDB에 데이터 저장 실패: {e}")

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

    async def stop(self):
        self.is_running = False
        if self.producer:
            await self.producer.stop()

    @abstractmethod
    async def process_data(self, data):
        # 데이터 처리
        pass

    @abstractmethod
    def get_interval(self):
        # 몇분 주기로 받을지
        pass


class RESTAPICollector(BaseCollector):
    def __init__(self, symbol, kafka_bootstrap_servers, topic, db_manager=None):
        super().__init__(symbol, kafka_bootstrap_servers, topic, db_manager)
        self.session = None

    async def initialize(self):
        await super().initialize()
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ssl_context))

    async def stop(self):
        await super().stop()
        if self.session:
            await self.session.close()
