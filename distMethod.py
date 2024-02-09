import yfinance as yf
import pandas as pd
import numpy as np

# Taking all the tickers in S&P 500 from the financial industry
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
financial_tickers = tickers.loc[tickers['GICS Sector'] == 'Financials']['Symbol']
tickers_list = ["".join(ticker) for ticker in financial_tickers]

# Downloading data for all tickers
stocks = yf.download(tickers_list[:5])
close = stocks.loc[:,"Close"].dropna()
print(close.head())

