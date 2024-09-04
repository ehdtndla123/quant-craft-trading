from pydantic import BaseModel
from app.db.models import TradingBotStatus


class TradingBotBase(BaseModel):
    name: str
    dry_run: bool
    cash: float
    user_id: int
    exchange_api_key_id: int
    strategy_id: int


class TradingBotCreate(TradingBotBase):
    status: TradingBotStatus
    pass


class TradingBotResponse(TradingBotBase):
    id: int

    class Config:
        from_attributes = True
