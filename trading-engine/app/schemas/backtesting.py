from pydantic import BaseModel
from typing import Dict, List, Union
from datetime import datetime


class BacktestingBase(BaseModel):
    strategy_id: int
    parameters: Dict
    results: Dict
    trades: List[Dict]
    equity_curve: List[Dict]


class BacktestingCreate(BaseModel):
    strategy_id: int
    parameters: Dict
    results: Dict
    trades: List[Dict]
    equity_curve: List[Dict]


class BacktestingResponse(BacktestingCreate):
    strategy_name: str
    parameters: Dict
    results: Dict
    trades: List[Dict]
    equity_curve: List[Dict]
    created_at: datetime

    class Config:
        orm_mode = True


class BacktestRunRequest(BaseModel):
    start_date: str
    end_date: str
    strategy_id: int
    cash: float
