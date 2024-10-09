from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.trading_bot import TradingBotCreate, TradingBotResponse
from app.services import trading_bot_service

router = APIRouter()


@router.post("/trading-bots", response_model=TradingBotResponse)
def create_trading_bot(trading_bot_data: TradingBotCreate, db: Session = Depends(get_db)):
    return trading_bot_service.create_trading_bot(db, trading_bot_data)


@router.get("/trading-bots/{trading_bot_id}", response_model=TradingBotResponse)
def read_trading_bot(trading_bot_id: int, db: Session = Depends(get_db)):
    trading_bot = trading_bot_service.get_trading_bot(db, trading_bot_id)
    if trading_bot is None:
        raise HTTPException(status_code=404, detail="TradingBot not found")
    return trading_bot


@router.get("/trading-bots", response_model=list[TradingBotResponse])
def read_trading_bots(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return trading_bot_service.get_trading_bots(db, skip=skip, limit=limit)


@router.delete("/trading-bots/{trading_bot_id}", response_model=bool)
def delete_trading_bot(trading_bot_id: int, db: Session = Depends(get_db)):
    result = trading_bot_service.delete_trading_bot(db, trading_bot_id)
    if not result:
        raise HTTPException(status_code=404, detail="TradingBot not found")
    return result
