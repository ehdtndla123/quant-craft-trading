from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd

class IBroker(ABC):

    def new_order(self, size: float, limit: Optional[float] = None,
                  stop: Optional[float] = None, sl: Optional[float] = None,
                  tp: Optional[float] = None, *, trade=None):
        """새로운 주문 생성"""
        pass

    @property
    def last_price(self) -> float:
        """마지막(현재) 종가"""
        pass

    @property
    def equity(self) -> float:
        """현재 계정 자산"""
        pass

    @property
    def margin_available(self) -> float:
        """사용 가능한 마진"""
        pass

    def next(self):
        """다음 단계로 진행"""
        pass

    def process_orders(self):
        """주문 처리"""
        pass

    def update_data(self, new_data: float):
        """거래 업데이트"""
        pass