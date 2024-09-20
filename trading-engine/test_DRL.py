from app.ML.backtest_test import BacktestManager

BacktestManager.run_backtest(
    exchange_name="binance",
    symbol="BTC/USDT",
    timeframe="1m",
    start_time="2019-01-01",
    # start_time="2024-07-27",
    end_time="2024-07-30",
    timezone="Asia/Seoul",
    strategy_name="DRLStrategyTest",
    commission=0.002,
    cash=100000,
    exclusive_orders=True,
    # margin=0.2
    margin=0.008
)