from sqlalchemy.orm import Session
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Union
from app.db.models import Backtesting, Strategy
from app.model.backtest.Backtest import Backtest
from app.schemas.backtesting import BacktestingCreate
from app.repositories import backtesting_repository
import importlib


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        return super().default(obj)


def run(db: Session, data: str or pd.DataFrame, strategy: Strategy, cash: float, start_date: str, end_date: str) -> Backtesting:

    # 데이터가 문자열이면 CSV 파일 경로로, DataFrame으로 로드
    if isinstance(data, str):
        data = pd.read_csv(data, index_col=0, parse_dates=True)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be either a DataFrame or a path to a CSV file.")

    # 전략 클래스 동적 로드
    strategy_module = importlib.import_module(f"app.strategy.strategy")
    StrategyClass = getattr(strategy_module, strategy.name)

    # Backtest 실행
    bt = Backtest(data, StrategyClass, cash=cash, commission=strategy.commission)
    stats = bt.run()

    backtesting_data = BacktestingCreate(
        strategy_id=strategy.id,
        parameters={"cash": cash, "start_date": start_date, "end_date": end_date},
        results={k: v for k, v in stats.items() if not k.startswith('_') and not callable(v)},
        trades=stats._trades.to_dict('records'),
        equity_curve=stats._equity_curve.to_dict('records')
    )

    return backtesting_repository.create_backtesting(db, backtesting_data.dict())


def get_backtesting(db: Session, backtesting_id: int) -> Backtesting:
    return db.query(Backtesting).filter(Backtesting.id == backtesting_id).first()


def get_backtestings(db: Session, skip: int = 0, limit: int = 100) -> List[Backtesting]:
    return db.query(Backtesting).offset(skip).limit(limit).all()


def update(db: Session, backtesting_id: int, backtesting_data: Dict) -> Backtesting:
    db_backtesting = get_backtesting(db, backtesting_id)
    if db_backtesting:
        for key, value in backtesting_data.items():
            if key in ['parameters', 'results', 'trades', 'equity_curve']:
                setattr(db_backtesting, key, json.dumps(value, cls=CustomJSONEncoder))
            else:
                setattr(db_backtesting, key, value)
        db.commit()
        db.refresh(db_backtesting)
    return db_backtesting


def delete(db: Session, backtesting_id: int) -> bool:
    db_backtesting = get_backtesting(db, backtesting_id)
    if db_backtesting:
        db.delete(db_backtesting)
        db.commit()
        return True
    return False
