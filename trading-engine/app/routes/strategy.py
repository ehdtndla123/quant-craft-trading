from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.strategy import StrategyCreate, StrategyUpdate, StrategyResponse
from app.services import strategy_service

router = APIRouter()


@router.post("/strategies", response_model=StrategyResponse)
def create_strategy(strategy_data: StrategyCreate, db: Session = Depends(get_db)):
    return strategy_service.create_strategy(db, strategy_data)


@router.get("/strategies/{strategy_id}", response_model=StrategyResponse)
def read_strategy(strategy_id: int, db: Session = Depends(get_db)):
    strategy = strategy_service.get_strategy(db, strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strategy


@router.get("/strategies", response_model=list[StrategyResponse])
def read_strategies(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return strategy_service.get_strategies(db, skip=skip, limit=limit)


@router.put("/strategies/{strategy_id}", response_model=StrategyResponse)
def update_strategy(strategy_id: int, strategy_data: StrategyUpdate, db: Session = Depends(get_db)):
    updated_strategy = strategy_service.update_strategy(db, strategy_id, strategy_data)
    if updated_strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return updated_strategy


@router.delete("/strategies/{strategy_id}", response_model=bool)
def delete_strategy(strategy_id: int, db: Session = Depends(get_db)):
    result = strategy_service.delete_strategy(db, strategy_id)
    if not result:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return result
