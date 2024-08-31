# Script for loading and saving data from Yahoo Finance locally.

import yfinance as yf 
import pandas as pd

# all the tickers
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOG", 
    "INTC", "NVDA", "META", "CSCO", 
    "TSLA", "ORCL", "IBM", "CRM", 
    "TSM", "ADBE", "QCOM", "TCEHY", 
    "AVGO", "BABA"
]

# fixed dates
start_date = "2015-04-30"
end_date = "2024-04-30"

if __name__ == "__main__":
    # download all data
    for i in range(len(TICKERS)):
        tk = TICKERS[i]
        data = yf.download(
            tk, start=start_date, end=end_date
        )
        # check if NaN values, forward fill
        if data.isnull().values.any():
            print("> {} found null. Forward filled. ".format(tk))
            data = data.ffill()
        # save data
        path = "./data/{}_{}_TO_{}.csv".format(tk, start_date, end_date)
        data.to_csv(path)
        print("> {} saved. size = {}".format(tk, data.shape))
        
    # check data format
    for tk in TICKERS:
        try:
            path = "./data/{}_{}_TO_{}.csv".format(tk, start_date, end_date)
            data = pd.read_csv(path)
        except:
            print("{} in the range {} to {} does not exist locally. ".format(tk, start_date, end_date))
        assert not data.isnull().values.any()

    # ensure all indices match 
    tmp = None
    for tk in TICKERS:
        path = "./data/{}_{}_TO_{}.csv".format(tk, start_date, end_date)
        prev = tmp
        tmp = pd.read_csv(path).Date
        if prev is not None:
            assert (tmp == prev).all()