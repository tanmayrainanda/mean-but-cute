import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt

# Alpha Vantage API key
api_key = '0QCKUKJ70VNZ6MKO'

# Function to get stock data
def get_stock_data(symbol, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    return data['4. close']

# List of stocks in portfolio with number of shares
stocks = {'AAPL': 100, 'MSFT': 100, 'GOOGL': 100}

#original buy price
buy_price = {'AAPL': 100, 'MSFT': 200, 'GOOGL': 50}

#value of portfolio at time of purchase
buy_value = {}
for stock in stocks:
    buy_value[stock] = buy_price[stock] * stocks[stock]

print ("Buy Value", buy_value)

#current price and value of portfolio
current_price = {}
current_value = {}
for stock in stocks:
    current_price[stock] = get_stock_data(stock, api_key)[0]
    current_value[stock] = current_price[stock] * stocks[stock]

#print ("Current Price of each share", current_price)
print ("Current Value of Portfolio", current_value)

#mean return value of each stock for the period of the last 2 months
mean_return = {}
for stock in stocks:
    mean_return[stock] = get_stock_data(stock, api_key).pct_change().mean()

print ("Mean Return", mean_return)