from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd

class IBroker(ABC):
    @abstractmethod
    def __init__(self, db, cash: float, commission: float, leverage: float,
                 trade_on_close: bool, hedging: bool, exclusive_orders: bool):
        pass

    @abstractmethod
    def new_order(self, size: float, limit: Optional[float] = None,
                  stop: Optional[float] = None, sl: Optional[float] = None,
                  tp: Optional[float] = None, *, trade=None):
        """새로운 주문 생성"""
        pass

    @property
    @abstractmethod
    def last_price(self) -> float:
        """마지막(현재) 종가"""
        pass

    @property
    @abstractmethod
    def equity(self) -> float:
        """현재 계정 자산"""
        pass

    @property
    @abstractmethod
    def margin_available(self) -> float:
        """사용 가능한 마진"""
        pass

    @abstractmethod
    def next(self):
        """다음 단계로 진행"""
        pass

    @abstractmethod
    def process_orders(self):
        """주문 처리"""
        pass
