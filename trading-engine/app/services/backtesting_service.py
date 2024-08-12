from app.model.backtest.Backtest import Backtest
import importlib
import pandas as pd
import json
from app.db.models import BacktestResult
from sqlalchemy.orm import Session
from datetime import datetime, date


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        return super().default(obj)


def run_backtest(
        data: str or pd.DataFrame,
        strategy_name: str,
        commission: float,
        cash: float,
        exclusive_orders: bool,
        **kwargs
):
    # 데이터가 문자열이면 CSV 파일 경로로, DataFrame으로 로드
    if isinstance(data, str):
        data = pd.read_csv(data, index_col=0, parse_dates=True)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be either a DataFrame or a path to a CSV file.")

    # 동적으로 전략 클래스 불러오기
    strategy_module = importlib.import_module(f"app.strategy.strategy")
    StrategyClass = getattr(strategy_module, strategy_name)

    bt = Backtest(data, StrategyClass, commission=commission, cash=cash,
                  exclusive_orders=exclusive_orders)
    stats = bt.run()
    bt.plot()

    return stats


def save_backtest_result(db: Session, stats, strategy_name: str, parameters: dict):
    # _Stats 객체에서 필요한 정보 추출
    results = {k: v for k, v in stats.items() if not k.startswith('_') and not callable(v)}

    # DataFrame을 리스트로 변환
    trades = stats._trades.to_dict('records')
    equity_curve = stats._equity_curve.to_dict('records')

    # BacktestResult 객체 생성 및 저장
    backtest_result = BacktestResult(
        strategy_name=strategy_name,
        parameters=json.dumps(parameters, cls=CustomJSONEncoder),
        results=json.dumps(results, cls=CustomJSONEncoder),
        trades=json.dumps(trades, cls=CustomJSONEncoder),
        equity_curve=json.dumps(equity_curve, cls=CustomJSONEncoder)
    )

    db.add(backtest_result)
    db.commit()
    db.refresh(backtest_result)

    return backtest_result


def run_and_save_backtest(
        db: Session,
        data: str or pd.DataFrame,
        strategy_name: str,
        commission: float,
        cash: float,
        exclusive_orders: bool,
        **kwargs
):
    stats = run_backtest(data, strategy_name, commission, cash, exclusive_orders, **kwargs)

    parameters = {
        "commission": commission,
        "cash": cash,
        "exclusive_orders": exclusive_orders,
        **kwargs
    }

    saved_result = save_backtest_result(db, stats, strategy_name, parameters)

    return saved_result


def create_backtest_result(db: Session, backtest_data: dict) -> BacktestResult:
    new_backtest = BacktestResult(
        strategy_name=backtest_data['strategy_name'],
        parameters=json.dumps(backtest_data['parameters']),
        results=json.dumps(backtest_data['results']),
        trades=json.dumps(backtest_data['trades'].to_dict('records')),
        equity_curve=json.dumps(backtest_data['equity_curve'].to_dict('records'))
    )
    db.add(new_backtest)
    db.commit()
    db.refresh(new_backtest)
    return new_backtest


def get_backtest_result(db: Session, result_id: int) -> BacktestResult:
    return db.query(BacktestResult).filter(BacktestResult.id == result_id).first()


def get_backtest_results(db: Session, skip: int = 0, limit: int = 100):
    return db.query(BacktestResult).offset(skip).limit(limit).all()


def get_backtest_result_data(db: Session, result_id: int) -> dict:
    result = get_backtest_result(db, result_id)
    if result:
        return json.loads(json.dumps({
            'results': json.loads(result.results),
            'trades': pd.DataFrame(json.loads(result.trades)).to_dict(orient='records'),
            'equity_curve': pd.DataFrame(json.loads(result.equity_curve)).to_dict(orient='records')
        }, cls=CustomJSONEncoder))
    return None


def update_backtest_result(db: Session, result_id: int, backtest_data: dict) -> BacktestResult:
    db_result = get_backtest_result(db, result_id)
    if db_result:
        db_result.strategy_name = backtest_data.get('strategy_name', db_result.strategy_name)
        db_result.parameters = json.dumps(backtest_data.get('parameters', json.loads(db_result.parameters)))
        db_result.results = json.dumps(backtest_data.get('results', json.loads(db_result.results)))
        db_result.trades = json.dumps(
            backtest_data.get('trades', pd.DataFrame(json.loads(db_result.trades))).to_dict('records'))
        db_result.equity_curve = json.dumps(
            backtest_data.get('equity_curve', pd.DataFrame(json.loads(db_result.equity_curve))).to_dict('records'))
        db.commit()
        db.refresh(db_result)
    return db_result


def delete_backtest_result(db: Session, result_id: int) -> bool:
    db_result = get_backtest_result(db, result_id)
    if db_result:
        db.delete(db_result)
        db.commit()
        return True
    return False
