import asyncio
from app.service.TradingBot import TradingBot

async def main():

    bot = TradingBot('binance', 'BTC/USDT', '1m', "MyStrategy", 100000, 0.002)

    try:
        await bot.run()
    except KeyboardInterrupt:
        await bot.stop()


if __name__ == '__main__':
    asyncio.run(main())