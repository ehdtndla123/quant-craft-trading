from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.bot_strategy import BotStrategyCreate, BotStrategyInDB
from app.services import bot_strategy_service

router = APIRouter()


@router.post("/bot-strategies", response_model=BotStrategyInDB)
def create_bot_strategy(bot_strategy: BotStrategyCreate, db: Session = Depends(get_db)):
    return bot_strategy_service.create_bot_strategy(db, bot_strategy)


@router.get("/bot-strategies/{bot_strategy_id}", response_model=BotStrategyInDB)
def read_bot_strategy(bot_strategy_id: int, db: Session = Depends(get_db)):
    db_bot_strategy = bot_strategy_service.get_bot_strategy(db, bot_strategy_id)
    if db_bot_strategy is None:
        raise HTTPException(status_code=404, detail="Bot strategy not found")
    return db_bot_strategy


@router.get("/bot-strategies", response_model=list[BotStrategyInDB])
def read_bot_strategies(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    bot_strategies = bot_strategy_service.get_bot_strategies(db, skip=skip, limit=limit)
    return bot_strategies


@router.delete("/bot-strategies/{bot_strategy_id}", response_model=bool)
def delete_bot_strategy(bot_strategy_id: int, db: Session = Depends(get_db)):
    result = bot_strategy_service.delete_bot_strategy(db, bot_strategy_id)
    if not result:
        raise HTTPException(status_code=404, detail="Bot strategy not found")
    return result
