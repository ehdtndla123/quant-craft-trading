from pydantic import BaseModel
from datetime import datetime


class TradingBotBase(BaseModel):
    bot_id: int
    strategy_id: int


class TradingBotCreate(TradingBotBase):
    pass


class TradingBotResponse(TradingBotBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
