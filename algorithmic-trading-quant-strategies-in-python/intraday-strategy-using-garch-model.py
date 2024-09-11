
"""
IntraDay Strategy Using GARCH Model

* Using simulated daily data and intraday 5-min data. 
* Load daily and 5 min data.
* Define function to fit GARCH model on the daily data and predict 1-day ahead volatility 
in a rolling window.
* Calculate prediction premium and form a daily signal from it. 
* Merge with intraday data and calculate intraday indicators to form the intraday signal.
* Generate the position entry and hold until the end of the day.
* Calculate final strategy returns. 
"""

"""
1. Load Simulated Daily and Simulated 5-minute daata.
* We are loading both datasets, set the indexes and calculate daily log returns. 
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from arch import arch_model
import pandas_ta
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
pd.set_option('display.max_columns', None)

data_folder = "/home/aditya/Desktop/ds-projects/finance-projects/algorithmic-trading-quant-strategies-in-python"

daily_df = pd.read_csv(os.path.join(data_folder, 'simulated_daily_data.csv'))
daily_df['Date'] = pd.to_datetime(daily_df['Date'])
daily_df = daily_df.set_index('Date')
daily_df.drop(daily_df.columns[-1], axis=1, inplace=True)
daily_df['log_ret'] = np.log(daily_df['Adj Close']).diff()

intraday_5min_df = pd.read_csv(os.path.join(data_folder, 'simulated_5min_data.csv'))
intraday_5min_df.drop('Unnamed: 6', axis=1, inplace=True)
intraday_5min_df['datetime'] = pd.to_datetime(intraday_5min_df['datetime'])
intraday_5min_df['date'] = intraday_5min_df['datetime'].dt.date
intraday_5min_df = intraday_5min_df.set_index('datetime')
intraday_5min_df['datetime'] = pd.to_datetime(intraday_5min_df['datetime'])
intraday_5min_df['date'] = pd.to_datetime(intraday_5min_df['date'])

"""
2. Define function to fit GARCH model and predict 1-day ahead volatility in a rolling window.

* We are first calculating the 6-month rolling variance and then we are creating a function
in a 6 month rolling window to fit a garch model and predict the next day variance.
"""

daily_df['variance'] = daily_df['log_ret'].rolling(180).var()

daily_df['variance'].plot()

daily_df = daily_df['2020-01-01':]

def predict_volatility(x):
    best_model = arch_model(y=x,
                            p=1,
                            q=3).fit(update_freq=5, disp='off')
    variance_forecast = best_model.forecast(horizon=1).variance.iloc[-1, 0]
    print(x.index[-1])
    return variance_forecast

daily_df['predictions'] = daily_df['log_ret'].rolling(180).apply(lambda x: predict_volatility(x))

daily_df[['variance', 'predictions']].plot()


"""
3. Calculate prediction premium and form a daily signal from it.

* We are calculating the prediction premium. And calculate its 6-month rolling standard deviation.
* From this we are creating this daily signal.
"""

daily_df['prediction_premium'] = (daily_df['predictions'] - daily_df['variance'])/daily_df['variance']

daily_df['premium_std'] = daily_df['prediction_premium'].rolling(180).std()

daily_df['prediction_premium'].plot()
daily_df['premium_std'].plot()

daily_df['signal_daily'] = daily_df.apply(lambda x: 1 if (x['prediction_premium']>x['premium_std']*1.5)
                                          else (-1 if (x['prediction_premium']<x['premium_std']*-1.5) 
                                                else np.nan),
                                          axis=1)

daily_df['signal_daily'].plot(kind='hist')
plt.show()


"""
4. Merge with Intraday data and calculate intraday indicators to form the intraday signal.

* Calculate all intraday indicators and intraday signal.
"""
final_df = pd.merge(intraday_5min_df.reset_index(), 
         daily_df[['signal_daily']].reset_index(),
             left_on = 'date',
             right_on='Date').set_index('datetime')

final_df = final_df.drop(['date', 'Date'], axis=1)

final_df['rsi'] = pandas_ta.rsi(close=final_df["close"],
                                length = 20)

final_df['lband'] = pandas_ta.bbands(close=final_df['close'],
                                     length=20).iloc[:, 0]

final_df['uband'] = pandas_ta.bbands(close=final_df['close'],
                                     length=20).iloc[:, 2]

final_df['signal_intraday'] = final_df.apply(lambda x: 1 if (x['rsi']>70)&
                                             (x['close']>x['uband'])
                                             else (-1 if (x['rsi']>30)&
                                             (x['close']<x['lband']) else np.nan),
                                             axis=1)


"""
5. Generate the position entry and hold untile the end of the day.
"""

final_df['return_sign'] = final_df.apply(lambda x: -1 if (x['signal_daily']==1)
                                         &(x['signal_intraday']==1)
                                         else (1 if (x['signal_daily']==-1)
                                               & (x['signal_intraday']==-1) else np.nan),
                                               axis=1)

final_df['return_sign'] = final_df.groupby(pd.Grouper(freq='D'))['return_sign']\
    .transform(lambda x: x.ffill())

final_df['return'] = final_df['close'].pct_change()

final_df['forward_return'] = final_df['return'].shift(-1)

final_df['strategy_return'] = final_df['forward_return']*final_df['return_sign']

daily_return_df = final_df.groupby(pd.Grouper(freq='D'))[['strategy_return']].sum()


"""
6. Calculate final strategy returns.
"""

strategy_cumulative_return = np.exp(np.log1p(daily_return_df).cumsum()).sub(1)

strategy_cumulative_return.plot(figsize=(16, 6))
plt.title('Intraday Strategy Returns')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.ylabel('Return')
plt.show()
