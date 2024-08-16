from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.backtesting import BacktestRunRequest, BacktestingResponse
from app.services import backtesting_service
from app.services.data_loader_service import DataLoaderService
from app.services import strategy_service
from typing import List

router = APIRouter()


@router.get("/backtestings/{backtesting_id}", response_model=BacktestingResponse)
def read_backtesting(backtesting_id: int, db: Session = Depends(get_db)):
    backtesting = backtesting_service.get_backtesting(db, backtesting_id)
    if backtesting is None:
        raise HTTPException(status_code=404, detail="Backtesting result not found")
    return backtesting


@router.get("/backtestings", response_model=List[BacktestingResponse])
def read_backtestings(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return backtesting_service.get_backtestings(db, skip=skip, limit=limit)


# 논블로킹으로 해야함. 블로킹으로 하면 다른 요청을 처리하지 못함
@router.post("/backtestings/run", response_model=BacktestingResponse)
def run_backtest(request: BacktestRunRequest, db: Session = Depends(get_db)):
    strategy = strategy_service.get_strategy(db, request.strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")

    data = DataLoaderService.load_data_from_ccxt(strategy.exchange, strategy.symbol, strategy.timeframe,
                                                 request.start_date, request.end_date)

    try:
        result = backtesting_service.run(
            db,
            data,
            strategy,
            request.cash,
            request.start_date,
            request.end_date
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while running the backtest: {str(e)}")
