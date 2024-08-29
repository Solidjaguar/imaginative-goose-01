import yfinance as yf
from loguru import logger
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

class DataFetcher:
    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_data(ticker, start_date, end_date):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            return data['Close']
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise

    @staticmethod
    def fetch_gold_data(start_date, end_date):
        try:
            gold = DataFetcher.fetch_data("GC=F", start_date, end_date)
            usd_index = DataFetcher.fetch_data("DX-Y.NYB", start_date, end_date)
            oil = DataFetcher.fetch_data("CL=F", start_date, end_date)
            sp500 = DataFetcher.fetch_data("^GSPC", start_date, end_date)
            treasury_yield = DataFetcher.fetch_data("^TNX", start_date, end_date)
            vix = DataFetcher.fetch_data("^VIX", start_date, end_date)
            
            df = pd.DataFrame({
                'Gold_Price': gold,
                'USD_Index': usd_index,
                'Oil_Price': oil,
                'SP500': sp500,
                'Treasury_Yield': treasury_yield,
                'VIX': vix
            })
            
            return df.dropna()
        except Exception as e:
            logger.error(f"Error fetching gold data: {str(e)}")
            raise

def fetch_all_data(start_date='2010-01-01', end_date=None):
    data_fetcher = DataFetcher()
    return data_fetcher.fetch_gold_data(start_date, end_date)