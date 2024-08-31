from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.bot import BotCreate, BotResponse
from app.schemas.trading_bot import TradingBotCreate, TradingBotResponse
from app.schemas.strategy import StrategyCreate, StrategyResponse
import app.services.bot_service as bot_service
import app.services.strategy_service as strategy_service
import app.services.trading_bot_service as trading_bot_service

router = APIRouter()


@router.post("/example/trading-bot", response_model=TradingBotResponse)
def create_example_trading_bot(db: Session = Depends(get_db)):
    # 봇 생성
    bot_create = BotCreate(
        dry_run=True,
        name="Dongsoo",
        cash=1000000.0,
    )
    new_bot = bot_service.create_bot(db, bot_create)

    # 전략 생성
    strategy_create = StrategyCreate(
        name="MyStrategy",
        description="A simple trading strategy",
        leverage=1.0,
        exclusive_order=False,
        hedge_mode=True,
        timeframe="1m",
        symbol="BTC/USDT",
        exchange="simulated"
    )
    new_strategy = strategy_service.create_strategy(db, strategy_create)

    # 트레이딩 봇 생성
    trading_bot_create = TradingBotCreate(
        bot_id=new_bot.id,
        strategy_id=new_strategy.id
    )
    new_trading_bot = trading_bot_service.create_trading_bot(db, trading_bot_create)

    # 트레이딩 봇 정보 반환
    return trading_bot_service.get_trading_bot_with_relations(db, new_trading_bot.id)
