"""
Twitter Sentiment Investment Strategy 

1. Load Twitter Sentiment Data

* Load the twitter sentiment dataset, set the index, calculcate engagement ratio and filter
out stocks with no significant twitter activity.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import *
import yfinance as yf
import matplotlib.ticker as mtick
import os
plt.style.use('ggplot')


data_folder = '/home/aditya/Desktop/ds-projects/algorithmic-trading-quant-strategies-in-python/'

sentiment_df = pd.read_csv(os.path.join(data_folder, 'sentiment_data.csv'))

sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

sentiment_df = sentiment_df.set_index(['date', 'symbol'])

sentiment_df['engagement_ratio'] = sentiment_df['twitterComments']/sentiment_df['twitterLikes']

sentiment_df = sentiment_df[(sentiment_df['twitterLikes']>20)&(sentiment_df['twitterComments']>10)]


"""
2. Aggregate Monthly and Calculate Average Sentiment for the month

* Aggregate on a monthly level and calculate average monthly metric, for the one we choose.
"""

aggregated_df = (sentiment_df.reset_index('symbol').groupby([pd.Grouper(freq='M'), 'symbol'])\
 [['engagement_ratio']].mean())

aggregated_df['rank'] = (aggregated_df.groupby(level=0)['engagement_ratio']
                         .transform(lambda x: x.rank(ascending=False)))


"""
3. Select Top 5 Stocks based on their cross-sectional ranking for each month

* Select top 5 stocks by rank for each month and fix the date to start at the beginning
of next month.
"""

filtered_df = aggregated_df[aggregated_df['rank']<6].copy()
filtered_df = filtered_df.reset_index(level=1)

filtered_df.index = filtered_df.index + pd.DateOffset(1)

filtered_df = filtered_df.reset_index().set_index(['date', 'symbol'])


"""
4. Extract the stocks to form portfolios with at the start of each new month

* Create a dictionary containing start of month and corresponded selected stocks.
"""

dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}

for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
    

"""
5. Download fresh stock prices for only selected/shortlisted stocks
"""

stocks_list = sentiment_df.index.get_level_values('symbol').unique().tolist()

prices_df = yf.download(tickers=stocks_list,
                        start='2021-01-01',
                        end='2023-03-01')


"""
6. Calculate Portfolio Returns with Monthly Rebalancing
"""

returns_df = np.log(prices_df['Adj Close']).diff().dropna(how='all')

portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():
    end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
    cols = fixed_dates[start_date]
    temp_df = returns_df[start_date:end_date][cols].mean(axis=1).to_frame('portfolio_return')
    portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)


"""
7. Download NASDAQ/QQQ prices and calculate returns to compare to our strategy    
"""
qqq_df= yf.download(tickers='QQQ',
                    start='2021-01-01',
                    end='2023-03-01')

qqq_ret = np.log(qqq_df['Adj Close']).diff().to_frame('nasdaq_return')

portfolio_df = pd.merge(portfolio_df, 
                      qqq_ret,
                      left_index=True,
                      right_index=True)

portfolios_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()).sub(1)

portfolios_cumulative_return.plot(figsize=(16, 6))
plt.title("Twitter Engagement Ratio Strategy Return Over Time")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.ylabel('Return')
plt.show()





















