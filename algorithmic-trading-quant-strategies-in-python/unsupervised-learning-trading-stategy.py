#!/usr/bin/env python
# coding: utf-8

# ## Unsupervised Learning Trading Strategy
# 
# * Download/load SP500 stocks prices data.
# * Calculate different features and indicators on each stock.
# * Aggregate on monthly level and filter top 150 most liquid stocks.
# * Calculate Monthly Returns for different time horizons.
# * Download Fama-French Factors and Calculate Rolling Factor Betas.
# * For each month fit a K-means Clustering Algorithm to group similar assets based on their features.
# * For each month select assets based on the cluster and form a portfolio based on Efficient Frontier max sharpe ratio optimization.
# * Visualize portfolio returns and compare to SP500 returns.
# 
# #### All packages needed
# 
# * pandas, numpy, matplotlib, statsmodels, pandas_datareader, datetime, yfinance, sklearn, PyPortfolioOpt


# ### 1. Download/Load SP500 stocks prices data.


from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
# sp500["Symbol"] = sp500["Symbol"].str.replace('.', '-')
# symbols_list = sp500["Symbol"].unique().tolist()
# symbols_list


# end_date = "2023-09-27"
# start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)
# start_date, end_date


# df = yf.download(tickers=symbols_list,
#                 start=start_date,
#                 end=end_date)


# df.to_pickle('df.pkl')
df = pd.read_pickle('df.pkl')

df = df.stack()


df.columns = df.columns.str.lower()


# ### 2. Calculate features and technical indicators for each stock.
# 
# * Garman-Klass Volatility
# * RSI
# * Bollinger Bands
# * ATR
# * MACD
# * Dollar Volume

# Garman-Klass Volatility = ((ln(High) - ln(Low))^2 / 2 ) - (2ln(2) - 1)(ln(Adj Close) - ln(Open))^2

df['garman_klass_vol'] = (np.log(df["high"]) - np.log(df["low"]))**2/2 - (2*np.log(2)-1)*(np.log(df["adj close"]) - np.log(df["open"]))**2
df


df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))


df.xs('AAPL', level=1)['rsi'].plot()


df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])


def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                         low=stock_data['low'],
                         close=stock_data['close'],
                         length=14)
    return atr.sub(atr.mean()).div(atr.std())

df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)


def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:, 0]
    return macd.sub(macd.mean()).div(macd.std())

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

# ### 3. Aggregate to monthly level and filter top 150 most liquid stocks for each month.
# 
# * To reduce training time and experiment with features and strategies, we convert the business-daily data to month-end frequency.


last_cols = [c for c in df.columns.unique(0) if c not in 
                 ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]



data = (pd.concat([df.unstack('Ticker')['dollar_volume'].resample('M').mean().stack('Ticker').to_frame('dollar_volume'),
           df.unstack()[last_cols].resample('M').last().stack('Ticker')], axis=1)).dropna()


# Calculate 5-year rolling average of dollar volume for each stocks before  filtering

data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('Ticker').rolling(5*12, min_periods=12).mean().stack())

data['dollar_vol_rank'] = data.groupby('Date')['dollar_volume'].rank(ascending=False)

data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)


#### 4. Calculate Monthly Returns for different time horizons as features.

# * To capture time series dynamics that reflect, for example, momentum patterns, we compute historical returns using the method .pct_change(lag), that is, returns over various monthly periods as identified by lags. 


g = df.xs('AAPL', level=1)

lags = [1, 2, 3, 6, 9, 12]

outlier_cutoff = 0.005

for lag in lags:
    g[f'return_{lag}m'] = (g['adj close'].pct_change(lag).
                                       pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                             upper=x.quantile(1-outlier_cutoff)))
                                       .add(1)
                                       .pow(1/lag)
                                       .sub(1))


def calculate_returns(df):

    lags = [1, 2, 3, 6, 9, 12]

    outlier_cutoff = 0.005

    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close'].pct_change(lag).
                                           pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                                 upper=x.quantile(1-outlier_cutoff)))
                                           .add(1)
                                           .pow(1/lag)
                                           .sub(1))
    
    return df


data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

"""
## 5. Download Fama French factors and calculate rolling factor betas

* We will introduce the Fama-French data to estimate the exposure of assets to common
risk factors using linear regression.
* The five Fama-French factors, namely market risk, size, value, operating profitability, 
and investment have been shown empirically to explain asset returns and are commonly used
to assess the risk/return profile of portfolios. Hence, it is natural to include past
factor exposures as financial features in models.
* We can assess the historical factor returns using the pandas-datareader and estimate
the historical exposures using the RollingOLS rolling linear regression.   
"""

factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                             'famafrench',
                             start='2010')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp()

factor_data = factor_data.resample('M').last().div(100)


factor_data = factor_data.join(data['return_1m']).sort_index()

# checking the join
factor_data.xs('MSFT', level=1).head()

## Filter out stocks with less than 10 months of data
observations = factor_data.groupby(level=1).size()

valid_stocks = observations[observations >= 10]

factor_data = factor_data[factor_data.index.get_level_values('Ticker').isin(valid_stocks.index)]


# Calculate the Rolling Factor Betas.
betas = (factor_data.groupby(level=1,
                    group_keys=False)
            .apply(lambda x:
                    RollingOLS(endog=x['return_1m'],
                               exog=sm.add_constant(x.drop('return_1m', axis=1)),
                               window=min(24, x.shape[0]),
                               min_nobs=len(x.columns)+1)
                    .fit(params_only=True)
                    .params
                    .drop('const', axis=1)))

data = data.join(betas.groupby(level=1).shift())

factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

data.loc[:, factors] = data.groupby('Ticker', 
                                    group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

data = data.dropna()
data = data.drop('adj close', axis=1)

data.info()


# At this point we have to decide on what ML model and approach to use for 
# predictions etc. 

### 6. For each month fit a K Means Clustering Algorithm to group similar assets 
# based on their features. 

## K Means Clustering
# * You may want to initialize predefined centroids for each cluster based on your research
# * For visualization purpose of this tutorial we will initially rely on the 
# 'k-means++' initialization.
# * Then we will predefine our centroids for each cluster.

from sklearn.cluster import KMeans

## Apply pre defined centroids
target_rsi_values = [30, 45, 55, 70]

# data = data.drop('cluster', axis=1)

initial_centroids = np.zeros((len(target_rsi_values), 18))
initial_centroids[:, 1] = target_rsi_values

def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init=initial_centroids).fit(df).labels_
    return df

data = data.dropna().groupby('Date', group_keys=False).apply(get_clusters)

def plot_clusters(data):
    
    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]
    
    plt.scatter(cluster_0.iloc[:,5], cluster_0.iloc[:, 1], color='red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,5], cluster_1.iloc[:, 1], color='green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,5], cluster_2.iloc[:, 1], color='blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,5], cluster_3.iloc[:, 1], color='black', label='cluster 3')
    
    plt.legend()
    plt.show()
    return

plt.style.use('ggplot')

for i in data.index.get_level_values('Date').unique().tolist():
    g = data.xs(i, level=0)
    plt.title(f"Date {i}")
    plot_clusters(g)    

"""
## 7. For Each month select assets based on the cluster and form a portfolio based on
Efficient Frontier max sharpe ratio optimization

* First we will filter only stocks corresponding to the cluster we choose based on our
hypothesis.

* Momentum is persistent and my idea would be that stocks clustered around RSI70 centroid
should continue to outperform in the following month - thus I would select stocks 
corresponding to cluster 3
"""

filtered_df = data[data["cluster"]==1].copy()

filtered_df = filtered_df.reset_index(level=1)

filtered_df.index = filtered_df.index + pd.DateOffset(1)

filtered_df = filtered_df.reset_index().set_index(['Date', 'Ticker'])

dates = filtered_df.index.get_level_values('Date').unique().tolist()

fixed_dates = {}

for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
    
"""
## Define Portfolio Optimization Function

* We will define a function which optimizes portfolio weights using PyPortfolioOpt 
package and EfficientFrontier optimizer to maximize the sharpe ratio. 

* To optimize the weights of a given portfolio we would need to supply last 1 year prices
to the function. 

* Apply single stock weight bounds constraint for diversification (minimum half of 
 equally weight and maximum 10% of portfolio).
"""

from pypfopt.efficient_frontier import EfficientFrontier 
from pypfopt import risk_models
from pypfopt import expected_returns

def optimize_weights(prices, lower_bound=0):
    
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)

    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    
    ef = EfficientFrontier(expected_returns = returns,
                           cov_matrix = cov,
                           weight_bounds=(lower_bound, .1),
                           solver='SCS') # code forchecking the available solvers
                           # import cvxpy as cp
                           # cp.installed_solvers()
    
    weights = ef.max_sharpe()
    
    return ef.clean_weights()


## Download Fresh Daily Prices Data only for short listed stocks.

stocks = data.index.get_level_values('Ticker').unique().tolist()

new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('Date').unique()[0] - pd.DateOffset(months=12),
                     end=data.index.get_level_values('Date').unique()[-1])

"""
# * Calculate daily returns for each stock which could land up in our portfolio.
# * Then loop over each month start, select the stocks for the month and calculate their
weights for the next month.
# * If the maximum sharpe ratio optimization fails for a given month, apply equally
weighted weights.
# * Calculate each day portfolio return.
"""

returns_dataframe = np.log(new_df['Adj Close']).diff()

portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():
    try:
        end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        # print(start_date)
        # print(end_date)
    
        cols = fixed_dates[start_date]
        optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
        optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")            
    
    
        optimization_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][fixed_dates['2017-11-01']]    
        
        success = False
        try:
            weights = optimize_weights(prices=optimization_df, 
                                   lower_bound=round(1/(len(optimization_df.columns)*2), 3))
        
            weights = pd.DataFrame(weights, index=pd.Series(0))
            success = True
        except:
            print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights')
        
        if success == False:
            weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                         index=optimization_df.columns.tolist(),
                         columns=pd.Series(0)).T
            
        temp_df = returns_dataframe[start_date:end_date]
        
        temp_df = pd.merge(temp_df.stack().to_frame('return').reset_index(level=0), 
                weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                   left_index=True,
                   right_index=True)\
                .reset_index().set_index(['Date', 'Ticker']).stack().unstack()
                
        temp_df['weighted_return'] = temp_df['return']*temp_df['weight']
        
        temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')
        
        portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

    except Exception as e: 
        print(e)


potfolio_df = portfolio_df.drop_duplicates()

portfolio_df.plot()


"""
## 8. Visualize Portfolio Returns and compare to SP500 returns.
"""

spy = yf.download(tickers="SPY",
                  start='2015-01-01',
                  end=dt.date.today())

spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close': 'SPY Buy&Hold'}, axis=1)

portfolio_df = pd.merge(portfolio_df, 
                        spy_ret,
                        left_index=True,
                        right_index=True)


import matplotlib.ticker as mtick
plt.style.use('ggplot')

portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()) - 1

portfolio_cumulative_return[:'2023-09-29'].plot(figsize=(16,6))
plt.title("Unsupervised Learning Trading Strategy Returns Over Time")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.ylabel('Return')
plt.show()







