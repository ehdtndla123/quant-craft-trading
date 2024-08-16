from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.trading_bot import TradingBotCreate, TradingBotResponse
from app.services import trading_bot_service


router = APIRouter()


@router.post("/trading-bots", response_model=TradingBotResponse)
def create_and_start_trading_bot(
        trading_bot_data: TradingBotCreate,
        db: Session = Depends(get_db)
):
    new_trading_bot = trading_bot_service.create_trading_bot(db, trading_bot_data)

    return TradingBotResponse(
        id=new_trading_bot.id,
        bot_id=new_trading_bot.bot_id,
        strategy_id=new_trading_bot.strategy_id,
    )


@router.get("/trading-bots/{trading_bot_id}", response_model=TradingBotResponse)
def read_trading_bot(trading_bot_id: int, db: Session = Depends(get_db)):
    trading_bot = trading_bot_service.get_trading_bot(db, trading_bot_id)
    if trading_bot is None:
        raise HTTPException(status_code=404, detail="TradingBot not found")
    return TradingBotResponse(
        id=trading_bot.id,
        bot_id=trading_bot.bot_id,
        strategy_id=trading_bot.strategy_id,
    )


@router.get("/trading-bots", response_model=list[TradingBotResponse])
def read_trading_bots(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    trading_bots = trading_bot_service.get_trading_bots(db, skip=skip, limit=limit)
    return [
        TradingBotResponse(
            id=trading_bot.id,
            bot_id=trading_bot.bot_id,
            strategy_id=trading_bot.strategy_id,
        )
        for trading_bot in trading_bots
    ]


@router.delete("/trading-bots/{trading_bot_id}", response_model=bool)
def delete_trading_bot(trading_bot_id: int, db: Session = Depends(get_db)):
    trading_bot = trading_bot_service.delete_trading_bot(db, trading_bot_id)
    if trading_bot is None:
        raise HTTPException(status_code=404, detail="TradingBot not found")
    return TradingBotResponse(
        id=trading_bot.id,
        bot_id=trading_bot.bot_id,
        strategy_id=trading_bot.strategy_id,
    )
