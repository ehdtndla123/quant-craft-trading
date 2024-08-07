import asyncio
from app.services.bot_service import TradingBot
from app.core.config import settings
from app.db.database import initialize_database

async def main():
    print("Starting database initialization...")  # 로그 추가
    initialize_database()
    print("Database initialization completed")  # 로그 추가
    bot = TradingBot(
        exchange_name=settings.EXCHANGE_NAME,
        symbol=settings.SYMBOL,
        timeframe=settings.TIMEFRAME,
        strategy_name=settings.STRATEGY_NAME
    )

    try:
        await bot.run()
    except KeyboardInterrupt:
        print("\nStopping the bot...")
    finally:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())