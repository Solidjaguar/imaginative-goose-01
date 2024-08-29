import yfinance as yf

def fetch_all_data(start_date, end_date):
    gold = yf.Ticker("GC=F")
    data = gold.history(start=start_date, end=end_date)
    return data