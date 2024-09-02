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


from app.ML.backtest_train import BacktestManager
BacktestManager.run_backtest(
    exchange_name="binance",
    symbol="BTC/USDT",
    timeframe="1m",
    start_time="2019-01-01",
    # start_time="2024-07-27",
    end_time="2024-07-30",
    timezone="Asia/Seoul",
    strategy_name="DRLStrategy",
    commission=0.002,
    cash=100000,
    exclusive_orders=True,
    # margin=0.2
    margin=0.008
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