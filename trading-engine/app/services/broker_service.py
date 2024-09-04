import json
from typing import Optional, Dict, Any
from uuid import uuid4
from kafka import KafkaProducer
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.models import TradingBot
from app.model.broker_interface import IBroker
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class BrokerService(IBroker):
    def __init__(self, trading_bot: TradingBot, symbol: str, exchange: str,
                 leverage: float, exclusive_orders: bool, hedge_mode: bool):
        self.db: Session = SessionLocal()
        self._trading_bot_id = trading_bot.id
        self._symbol = symbol
        self._exchange = exchange
        self._leverage = leverage
        self._exclusive_orders = exclusive_orders
        self._hedge_mode = hedge_mode
        self.kafka_producer = self._create_kafka_producer()

    def _create_kafka_producer(self) -> KafkaProducer:
        return KafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def new_order(self, size: float, limit: Optional[float] = None,
                  stop: Optional[float] = None, sl: Optional[float] = None,
                  tp: Optional[float] = None, *args, **kwargs) -> Dict[str, Any]:
        order_id = str(uuid4())
        order_data = self._create_order_data(order_id, size, limit, stop, sl, tp)
        self._send_order_to_kafka(order_data)
        return {"status": "submitted", "order_id": order_id}

    def _create_order_data(self, order_id: str, size: float, limit: Optional[float],
                           stop: Optional[float], sl: Optional[float], tp: Optional[float]
                           ) -> Dict[str, Any]:
        return {
            "order_id": order_id,
            "trading_bot_id": self._trading_bot_id,
            "symbol": self._symbol,
            "exchange": self._exchange,
            "size": size,
            "limit": limit,
            "stop": stop,
            "sl": sl,
            "tp": tp,
            "leverage": self._leverage,
            "exclusive_orders": self._exclusive_orders,
            "hedge_mode": self._hedge_mode,
        }

    def _send_order_to_kafka(self, order_data: Dict[str, Any]) -> None:
        try:
            self.kafka_producer.send(settings.KAFKA_ORDERS_TOPIC, order_data)
            self.kafka_producer.flush()
            logger.info(f"Order {order_data['order_id']} sent to Kafka!")
        except Exception as e:
            # 나중에 에러 처리 제대로 해야함.
            logger.error(f"Failed to send order {order_data['order_id']} to Kafka: {str(e)}")

    def __del__(self):
        if hasattr(self, 'kafka_producer'):
            self.kafka_producer.close()
