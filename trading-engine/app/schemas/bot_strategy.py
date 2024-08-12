from pydantic import BaseModel
from datetime import datetime


class BotStrategyBase(BaseModel):
    bot_id: int
    strategy_id: int


class BotStrategyCreate(BotStrategyBase):
    pass


class BotStrategyInDB(BotStrategyBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
