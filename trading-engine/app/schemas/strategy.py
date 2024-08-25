from pydantic import BaseModel
from datetime import datetime


class StrategyBase(BaseModel):
    name: str
    description: str
    leverage: float
    exclusive_orders: bool = False
    hedge_mode: bool = False
    timeframe: str
    symbol: str
    exchange: str


class StrategyCreate(StrategyBase):
    pass


class StrategyUpdate(StrategyBase):
    pass


class StrategyResponse(StrategyBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
