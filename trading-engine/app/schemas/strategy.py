from pydantic import BaseModel
from datetime import datetime


class StrategyBase(BaseModel):
    name: str
    description: str
    leverage: float
    trade_on_close: bool = False
    hedge_mode: bool = False
    exclusive_mode: bool = False
    timeframe: str
    symbol: str
    exchange: str
    commission: float


class StrategyCreate(StrategyBase):
    pass


class StrategyUpdate(StrategyBase):
    pass


class StrategyResponse(StrategyBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
