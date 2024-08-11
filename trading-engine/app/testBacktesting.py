from app.service.backtest import BacktestManager
from app.service.data_loader import DataLoader

data = DataLoader.load_data_from_ccxt("binance", "BTC/USDT", "1h", "2024-08-01", "2024-08-10", "UTC")
print(data)

BacktestManager.run_backtest(data, "MyStrategy", 0.0005, 1000000, True)
