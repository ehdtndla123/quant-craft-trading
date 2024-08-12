import asyncio
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from starlette.responses import JSONResponse
from typing import Dict, Any
from app.services import backtesting_service
from app.services.trading_engine import TradingEngine
from app.db.database import initialize_database, SessionLocal
import app.services.trade_service as trade_service
import app.services.order_service as order_service
import app.services.bot_service as bot_service
import app.services.strategy_service as strategy_service
import app.services.bot_strategy_service as bot_strategy_service
import numpy as np
import os

app = FastAPI()

# 프로젝트 루트 디렉토리 경로를 가져옵니다.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# static 파일 설정
static_dir = os.path.join(project_root, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# templates 설정
templates = Jinja2Templates(directory=os.path.join(project_root, "templates"))


class TradingEngineManager:
    def __init__(self):
        self.engine = None

    async def initialize(self):
        example_bot = create_example_bot_strategy()
        self.engine = TradingEngine(example_bot)
        asyncio.create_task(self.engine.run())

    async def shutdown(self):
        if self.engine:
            await self.engine.stop()


trading_engine_manager = TradingEngineManager()


def create_example_bot_strategy():
    db = SessionLocal()
    try:
        # 봇 생성
        bot_data = {
            "dry_run": True,
            "name": "Dongsoo",
            "cash": 1000000.0,
        }
        new_bot = bot_service.create_bot(db, bot_data)
        print(f"새로운 봇이 생성되었습니다. ID: {new_bot.id}")
        print(f"봇 이름: {new_bot.name}")
        print(f"초기 자금: {new_bot.cash}")

        # 전략 생성
        strategy_data = {
            "name": "MyStrategy",
            "description": "A simple trading strategy",
            "leverage": 1.0,
            "trade_on_close": False,
            "hedge_mode": True,
            "exclusive_mode": True,
            "timeframe": "1m",
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "commission": 0.001
        }
        new_strategy = strategy_service.create_strategy(db, strategy_data)
        print(f"새로운 전략이 생성되었습니다. ID: {new_strategy.id}")
        print(f"전략 이름: {new_strategy.name}")
        print(f"거래소: {new_strategy.exchange}")
        print(f"심볼: {new_strategy.symbol}")

        new_bot_strategy = bot_strategy_service.create_bot_strategy(db, new_bot.id, new_strategy.id)

        return bot_strategy_service.get_bot_strategy_with_relations(db, new_bot_strategy.id)
    finally:
        db.close()


@app.on_event("startup")
async def startup_event():
    print("Starting database initialization...")
    initialize_database()
    print("Database initialization completed")
    await trading_engine_manager.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    print("Stopping the bot...")
    await trading_engine_manager.shutdown()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_trading_engine():
    if trading_engine_manager.engine is None:
        raise RuntimeError("TradingEngine is not initialized")
    return trading_engine_manager.engine


async def get_real_data(engine: TradingEngine):
    db = next(get_db())

    cash = engine.broker_service.cash
    equity = engine.broker_service.equityList
    position = 0

    orders = order_service.get_open_orders(db)
    open_trades = trade_service.get_open_trades(db)
    closed_trades = trade_service.get_closed_trades(db)
    all_trades = open_trades + closed_trades

    trades_df = pd.DataFrame({
        'Size': [t.size for t in all_trades],
        'EntryPrice': [t.entry_price for t in all_trades],
        'ExitPrice': [t.exit_price if t.exit_price is not None else engine.broker_service.last_price for t in
                      all_trades],
        'PnL': [trade_service.calculate_pl(t, engine.broker_service.last_price) for t in all_trades],
        'ReturnPct': [trade_service.calculate_pl_pct(t, engine.broker_service.last_price) for t in all_trades],
        'EntryTime': [pd.to_datetime(t.entry_time) for t in all_trades],
        'ExitTime': [pd.to_datetime(t.exit_time) if t.exit_time is not None else pd.NaT for t in all_trades],
    })

    return cash, position, equity, orders, trades_df


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/data")
async def get_data(engine: TradingEngine = Depends(get_trading_engine)):
    cash, position, equity_list, orders, trades_df = await get_real_data(engine)

    def clean_float(value):
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)

    return {
        'cash': clean_float(cash),
        'position': position,
        'equity': [clean_float(eq) for eq in equity_list],
        'orders': [
            {
                "size": order.size,
                "limit_price": clean_float(order.limit_price),
                "stop_price": clean_float(order.stop_price),
                "sl_price": clean_float(order.sl_price),
                "tp_price": clean_float(order.tp_price),
                "is_long": order.size > 0,
                "is_short": order.size < 0,
                "is_contingent": order.is_contingent
            } for order in orders
        ],
        'trades': [
            {
                'Size': trade['Size'],
                'EntryPrice': clean_float(trade['EntryPrice']),
                'ExitPrice': clean_float(trade['ExitPrice']),
                'PnL': clean_float(trade['PnL']),
                'ReturnPct': clean_float(trade['ReturnPct']),
                'EntryTime': trade['EntryTime'].isoformat() if pd.notna(trade['EntryTime']) else None,
                'ExitTime': trade['ExitTime'].isoformat() if pd.notna(trade['ExitTime']) else None,
            } for trade in trades_df.to_dict(orient='records')
        ],
        'trade_stats': {
            'total_trades': len(trades_df),
            'profitable_trades': int(sum(trades_df['PnL'] > 0)),
            'loss_making_trades': int(sum(trades_df['PnL'] < 0)),
            'total_pnl': clean_float(trades_df['PnL'].sum()),
            'avg_pnl_per_trade': clean_float(trades_df['PnL'].mean()),
            'avg_return_pct': clean_float(trades_df['ReturnPct'].mean()),
            'best_trade': clean_float(trades_df['PnL'].max()),
            'worst_trade': clean_float(trades_df['PnL'].min()),
        }
    }


@app.get("/backtest/{result_id}", response_model=Dict[str, Any])
async def get_backtest_data(result_id: int):
    result_data = backtesting_service.get_backtest_result_data(next(get_db()),result_id)
    if result_data is None:
        raise HTTPException(status_code=404, detail="Backtest result not found")
    return JSONResponse(content=result_data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
