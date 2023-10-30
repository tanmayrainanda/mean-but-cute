import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Function to get stock data
def get_stock_data(symbol):
    data = yf.download(symbol, start='2021-01-01')
    return data['Close']

df = pd.DataFrame()
# List of stocks in portfolio with number of shares
stocks = {'AAPL': 1, 'MSFT': 1, 'GOOGL': 1}

#original buy price
buy_price = {'AAPL': 100, 'MSFT': 200, 'GOOGL': 100}

#value of portfolio at time of purchase
buy_value = {}
for stock in stocks:
    buy_value[stock] = buy_price[stock] * stocks[stock]

print ("Buy Value", buy_value)

#current price and value of portfolio
current_price = {}
current_value = {}

#add closing price of stocks for the last 2 months to dataframe
for stock in stocks:
    df[stock] = get_stock_data(stock)
    current_price[stock] = df[stock][0]
    current_value[stock] = current_price[stock] * stocks[stock]

#print ("Current Price of each share", current_price)
print ("Current Value of Portfolio", current_value)

#mean return value of each stock for the period of the last 2 months
mean_return = {}
for stock in stocks:
    mean_return[stock] = get_stock_data(stock).mean()

print ("Mean Return", mean_return)

#Calculate the value of return that each stock has given in 2 month intervals from jan 1st 2021 to current date
for stock in stocks:
    df[stock + 'Return'] = df[stock]-buy_price[stock]

print(df)

#Calculate the covariance of the returns