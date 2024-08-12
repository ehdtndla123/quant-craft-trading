from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.strategy import StrategyCreate, StrategyUpdate, StrategyInDB
from app.services import strategy_service

router = APIRouter()


@router.post("/strategies", response_model=StrategyInDB)
def create_strategy(strategy: StrategyCreate, db: Session = Depends(get_db)):
    return strategy_service.create_strategy(db, strategy)


@router.get("/strategies/{strategy_id}", response_model=StrategyInDB)
def read_strategy(strategy_id: int, db: Session = Depends(get_db)):
    db_strategy = strategy_service.get_strategy(db, strategy_id)
    if db_strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return db_strategy


@router.get("/strategies", response_model=list[StrategyInDB])
def read_strategies(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    strategies = strategy_service.get_strategies(db, skip=skip, limit=limit)
    return strategies


@router.put("/strategies/{strategy_id}", response_model=StrategyInDB)
def update_strategy(strategy_id: int, strategy: StrategyUpdate, db: Session = Depends(get_db)):
    db_strategy = strategy_service.update_strategy(db, strategy_id, strategy)
    if db_strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return db_strategy


@router.delete("/strategies/{strategy_id}", response_model=bool)
def delete_strategy(strategy_id: int, db: Session = Depends(get_db)):
    result = strategy_service.delete_strategy(db, strategy_id)
    if not result:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return result
