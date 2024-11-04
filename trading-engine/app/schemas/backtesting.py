from datetime import datetime
from pydantic import BaseModel


class BacktestingBase(BaseModel):
    user_id: int
    strategy_id: int
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    trades: str
    equity_curve: str


class BacktestingCreate(BacktestingBase):
    pass


class BacktestingResponse(BacktestingCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class BacktestRunRequest(BaseModel):
    user_id: int
    start_date: str
    end_date: str
    strategy_id: int
    cash: float
    commission: float
