from abc import ABC, abstractmethod

class BaseCollector(ABC):
    def __init__(self, symbol, producer, topic):
        self.symbol = symbol
        self.producer = producer
        self.topic = topic

    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass