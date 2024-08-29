import pandas as pd
import yfinance as yf
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_data(symbol, start_date, end_date):
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Close']

def fetch_all_data(start_date, end_date):
    gold = fetch_data("GC=F", start_date, end_date)
    usd_index = fetch_data("DX-Y.NYB", start_date, end_date)
    oil = fetch_data("CL=F", start_date, end_date)
    sp500 = fetch_data("^GSPC", start_date, end_date)
    treasury_yield = fetch_data("^TNX", start_date, end_date)
    vix = fetch_data("^VIX", start_date, end_date)

    df = pd.concat([gold, usd_index, oil, sp500, treasury_yield, vix], axis=1)
    df.columns = ['Gold_Price', 'USD_Index', 'Oil_Price', 'SP500', 'Treasury_Yield', 'VIX']
    df.dropna(inplace=True)

    return df