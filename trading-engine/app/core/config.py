# app/core/config.py

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 데이터베이스 설정
    DATABASE_URL: str = "sqlite:///./test.db"

    # 거래 설정
    INITIAL_CASH: float = 1000000.0
    COMMISSION: float = 0.001  # 0.1%
    MARGIN: float = 0.1  # 10% 마진

    # 기타 설정
    TRADE_ON_CLOSE: bool = False
    HEDGING: bool = False
    EXCLUSIVE_ORDERS: bool = False

    # 거래 봇 설정
    EXCHANGE_NAME: str = "binance"
    SYMBOL: str = "BTC/USDT"
    TIMEFRAME: str = "1m"
    STRATEGY_NAME: str = "MyStrategy"

    class Config:
        env_file = ".env"


settings = Settings()