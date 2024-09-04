from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.trading_bot import TradingBotCreate, TradingBotResponse
from app.schemas.strategy import StrategyCreate
import app.services.strategy_service as strategy_service
import app.services.trading_bot_service as trading_bot_service
from app.db.models import User, ExchangeApiKey, TradingBotStatus

router = APIRouter()

@router.post("/example/trading-bot", response_model=TradingBotResponse)
def create_example_trading_bot(db: Session = Depends(get_db)):
    # User 생성
    new_user = User(username="example_user")
    db.add(new_user)
    db.flush()

    # ExchangeApiKey 생성
    new_exchange_api_key = ExchangeApiKey(
        exchange_name="Binance",
        api_key="example_api_key",
        secret_key="example_secret_key",
        user_id=new_user.id
    )
    db.add(new_exchange_api_key)
    db.flush()

    # 전략 생성
    strategy_create = StrategyCreate(
        name="MyStrategy",
        description="A simple trading strategy",
        leverage=1.0,
        exclusive_orders=False,
        hedge_mode=True,
        timeframe="1m",
        symbol="BTC/USDT",
        exchange="simulated"
    )
    new_strategy = strategy_service.create_strategy(db, strategy_create)

    # 트레이딩 봇 생성
    trading_bot_create = TradingBotCreate(
        name="Dongsoo",
        dry_run=True,
        cash=1000000.0,
        user_id=new_user.id,
        exchange_api_key_id=new_exchange_api_key.id,
        strategy_id=new_strategy.id,
        status=TradingBotStatus.PENDING
    )
    new_trading_bot = trading_bot_service.create_trading_bot(db, trading_bot_create)

    db.commit()
    return trading_bot_service.get_trading_bot_with_relations(db, new_trading_bot.id)