# from fastapi import FastAPI
# from app.controller import trading_controller
#
# app = FastAPI()
#
# app.include_router(trading_controller.router, prefix="/trading", tags=["trading"])
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from app.service.backtest import BacktestManager
BacktestManager.run_backtest(
    exchange_name="binance",
    symbol="BTC/USDT",
    timeframe="1d",
    start_time="2019-01-01",
    end_time="2024-06-21",
    timezone="Asia/Seoul",
    strategy_name="MyStrategy",
    commission=0.002,
    cash=100000,
    exclusive_orders=True
)

# bt.plot(
#     plot_width=None,  # 기본값 유지
#     plot_equity=True,
#     plot_return=True,
#     plot_pl=True,
#     plot_volume=True,
#     plot_drawdown=True,
#     smooth_equity=True,
#     relative_equity=True,
#     superimpose=True,
#     resample=True,
#     reverse_indicators=True,
#     show_legend=True,
# )