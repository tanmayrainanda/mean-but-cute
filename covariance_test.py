import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt

# Alpha Vantage API key
api_key = 'E1HSVZYAYTJIZURA'

# Function to get stock data
def get_stock_data(symbol, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    return data['4. close']

# List of stocks in portfolio
stocks = ['AAPL', 'MSFT', 'GOOGL']

# Get stock data
stock_data = {}
for stock in stocks:
    stock_data[stock] = get_stock_data(stock, api_key)
    print(stock_data[stock])