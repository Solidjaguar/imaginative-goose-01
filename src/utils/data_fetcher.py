import yfinance as yf
from loguru import logger
import pandas as pd

class DataFetcher:
    @staticmethod
    def fetch_gold_data(start_date, end_date):
        try:
            gold = yf.download("GC=F", start=start_date, end=end_date)
            usd_index = yf.download("DX-Y.NYB", start=start_date, end=end_date)["Close"]
            oil = yf.download("CL=F", start=start_date, end=end_date)["Close"]
            sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
            
            df = pd.DataFrame({
                'Gold_Price': gold['Close'],
                'USD_Index': usd_index,
                'Oil_Price': oil,
                'SP500': sp500
            })
            
            return df.dropna()
        except Exception as e:
            logger.error(f"Error fetching gold data: {str(e)}")
            raise

def fetch_all_data(start_date='2010-01-01', end_date=None):
    data_fetcher = DataFetcher()
    return data_fetcher.fetch_gold_data(start_date, end_date)