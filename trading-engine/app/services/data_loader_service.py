import pandas as pd
import ccxt
import os


class DataLoaderService:
    @staticmethod
    def load_data_from_ccxt(exchange_name: str, symbol: str, timeframe: str, start_time: str, end_time: str,
                            timezone: str = "UTC") -> pd.DataFrame:
        exchange = getattr(ccxt, exchange_name)()

        start_timestamp = int(pd.Timestamp(start_time, tz='UTC').timestamp() * 1000)
        end_timestamp = int(pd.Timestamp(end_time, tz='UTC').timestamp() * 1000)

        all_ohlcv = []
        current_timestamp = start_timestamp

        while current_timestamp < end_timestamp:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_timestamp,
                limit=1000  # 고정된 limit 값
            )

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            current_timestamp = ohlcv[-1][0] + 1  # 마지막 캔들의 다음 타임스탬프

        df = pd.DataFrame(all_ohlcv, columns=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])

        pd_ts = pd.to_datetime(df['datetime'], utc=True, unit='ms')

        if timezone != "UTC":
            pd_ts = pd_ts.dt.tz_convert(timezone)

        pd_ts = pd_ts.dt.tz_localize(None)

        df.set_index(pd_ts, inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # end_time 이후의 데이터 제거
        df = df[df.index <= pd.Timestamp(end_time)]

        return df

    @staticmethod
    def save_data_from_ccxt(exchange_name: str, symbol: str, timeframe: str, start_time: str, end_time: str,
                            timezone: str = "UTC", filename: str = None, directory: str = "data"):
        df = DataLoaderService.load_data_from_ccxt(exchange_name, symbol, timeframe, start_time, end_time, timezone)

        if filename is None:
            filename = f"{exchange_name}_{symbol.replace('/', '_')}_{timeframe}_{start_time}_{end_time}.csv"

        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, filename)

        df.to_csv(file_path)
        print(f"Data saved to {file_path}")

        return file_path
