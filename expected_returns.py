import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import datetime
from pypfopt import plotting

start_date = datetime.datetime(2021,4,1)
end_date = datetime.datetime(2023,11,10)

# Function to get stock data
def get_stock_data(symbol):
    data = yf.download(symbol, start=start_date, end=end_date)
    data = data['Adj Close']
    return data

ticker_list = ['UNH', 'TSLA', 'GOOGL']
portfolio = get_stock_data(ticker_list)
portfolio.to_csv("portfolio.csv",index=True)
portfolio = pd.read_csv("portfolio.csv",parse_dates=True,index_col="Date")

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

sample_cov = risk_models.sample_cov(portfolio, frequency=252)
S = risk_models.CovarianceShrinkage(portfolio).ledoit_wolf()

# Calculate the expected return of the portfolio
expected_return = sum(mean_return[stock] * stocks[stock] for stock in stocks)
print("Expected Return of Portfolio", expected_return)
mu = expected_returns.capm_return(portfolio)
mu.plot.barh(figsize=(10, 6))

# calculate efficient frontier
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(dict(cleaned_weights))

ef.portfolio_performance(verbose=True)

# Convert current_value to a pandas Series and drop any NaN values
current_value_series = pd.Series(current_value).dropna()

da = DiscreteAllocation(weights, current_value_series, total_portfolio_value=10000)

allocation, leftover = da.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))

n_samples = 10000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
sharpes = rets / stds

print("Sample portfolio returns:", rets)
print("Sample portfolio volatilities:", stds)

# Plot efficient frontier with Monte Carlo sim
ef = EfficientFrontier(mu, S)

fig, ax = plt.subplots(figsize= (10,10))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

