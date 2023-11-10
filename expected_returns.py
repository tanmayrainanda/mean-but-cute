import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt.discrete_allocation import DiscreteAllocation

# Function to get stock data
def get_stock_data(symbol):
    data = yf.download(symbol, start='2021-01-01')
    return data['Close']

df = pd.DataFrame()

stocks = {'UNH': 1, 'TSLA': 1, 'GOOGL': 1}
buy_price = {'UNH': 100, 'TSLA': 200, 'GOOGL': 100}
buy_value = {stock: buy_price[stock] * stocks[stock] for stock in stocks}

print("Buy Value", buy_value)

current_price = {}
current_value = {}

for stock in stocks:
    df[stock] = get_stock_data(stock)
    current_price[stock] = df[stock][0]
    current_value[stock] = current_price[stock] * stocks[stock]
print("Current Value of Portfolio", current_value)

# mean return value of each stock for the period of the last 2 months
mean_return = {stock: get_stock_data(stock).mean() for stock in stocks}
print("Mean Return", mean_return)

# Calculate the value of return that each stock has given in 2-month intervals from Jan 1st, 2021, to the current date
for stock in stocks:
    df[stock + 'Return'] = df[stock] - buy_price[stock]

cov_matrix = df.pct_change().cov()
sns.heatmap(cov_matrix)
plt.show()

# Calculate the expected return of the portfolio
expected_return = sum(mean_return[stock] * stocks[stock] for stock in stocks)
print("Expected Return of Portfolio", expected_return)
mu = expected_returns.capm_return(df)
mu.plot.barh(figsize=(10, 6))

# calculate efficient frontier
ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, 1))
weights = ef.max_sharpe()
print(weights)
cleaned_weights = ef.clean_weights()
print(dict(cleaned_weights))

ef.portfolio_performance(verbose=True)

# Convert current_value to a pandas Series and drop any NaN values
current_value_series = pd.Series(current_value).dropna()

da = DiscreteAllocation(weights, current_value_series, total_portfolio_value=10000)