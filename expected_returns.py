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

start_date = datetime.datetime(2023, 12, 1)
end_date = datetime.datetime(2023, 12, 13)

# Function to get stock data
def get_stock_data(symbol):
    data = yf.download(symbol, start=start_date, end=end_date)
    data = data['Adj Close']
    return data

ticker_list = [ 'AMZN', 'JPM', 'JNJ','PG','CVX','WMT','NVDA','MMM','UNH','TSLA','HD','KO']
portfolio = get_stock_data(ticker_list)
portfolio.to_csv("portfolio.csv", index=True)
portfolio = pd.read_csv("portfolio.csv", parse_dates=True, index_col="Date")

df = pd.DataFrame()

# Initial portfolio composition
initial_weights = {'AMZN': 0.08, 'JPM': 0.08, 'JNJ': 0.08, 'PG': 0.08, 'CVX': 0.08, 'WMT': 0.08, 'NVDA': 0.08, 'MMM': 0.08, 'UNH': 0.08, 'TSLA': 0.08, 'HD': 0.08, 'KO': 0.08}
budget = 30000  # Specify your budget
#print todays share price of each stock
print("Todays share price of each stock:")
print(portfolio.iloc[0])
# Calculate the initial number of shares based on the initial weights and budget
initial_allocation = {stock: round(initial_weights[stock] * budget / get_stock_data(stock)[0], 0) for stock in initial_weights}

print("Initial Allocation:", initial_allocation)

for stock in initial_allocation:
    df[stock] = get_stock_data(stock)

# Calculate current portfolio value
current_value = {stock: df[stock][0] * initial_allocation[stock] for stock in initial_allocation}
print("Current Value of Portfolio:", current_value)

# Calculate mean return value of each stock for the period of the last 2 months
mean_return = {stock: get_stock_data(stock).mean() for stock in initial_allocation}
print("Mean Return:", mean_return)

# Calculate the value of return that each stock has given in 2-month intervals from Jan 1st, 2021, to the current date
for stock in initial_allocation:
    df[stock + 'Return'] = df[stock] - df[stock][0]

# cov_matrix = df.pct_change().cov()
# sns.heatmap(cov_matrix)
# plt.show()

sample_cov = risk_models.sample_cov(portfolio, frequency=252)
S = risk_models.CovarianceShrinkage(portfolio).ledoit_wolf()
plotting.plot_covariance(S);

# Calculate the expected return of the portfolio
expected_return = sum(mean_return[stock] * initial_allocation[stock] for stock in initial_allocation)
print("Expected Return of Portfolio:", expected_return)
mu = expected_returns.capm_return(portfolio)
mu.plot.barh(figsize=(10, 6))

# Calculate efficient frontier
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(dict(cleaned_weights))

ef.portfolio_performance(verbose=True)

# Get the latest prices for each stock
latest_prices = get_latest_prices(portfolio)

# Convert the current allocation to units of stocks based on the latest prices
current_allocation_units = {stock: current_value[stock] / latest_prices[stock] for stock in current_value}

# Convert current_allocation_units to a pandas Series and drop any NaN values
current_allocation_series = pd.Series(current_allocation_units).dropna()

# Calculate the discrete allocation based on the latest prices and total_portfolio_value
da = DiscreteAllocation(weights, current_allocation_series, total_portfolio_value=budget)

#print continuous allocation
print("Continuous allocation:", weights)

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

fig, ax = plt.subplots(figsize=(10, 10))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find and plot the tangency portfolio
ef2 = EfficientFrontier(mu, S)
ef2.max_sharpe()
ret_tangent, std_tangent, _ = ef2.portfolio_performance()

# Plot random portfolios
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
ax.scatter(std_tangent, ret_tangent, c='red', marker='X', s=150, label='Max Sharpe')

im = ax.scatter(stds, rets, c=sharpes, marker=".", cmap="viridis_r")

# Format
cb = fig.colorbar(im, ax=ax)
cb.set_label("Sharpe ratio")
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.show()
