# %% ## Import functions
import pyreadr
import linearmodels
import pandas as pd
from collections import OrderedDict
from Functions import *
from datetime import timedelta, datetime, date
import numpy as np
import time as tm
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import math
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from linearmodels import PanelOLS
import warnings
from linearmodels.panel import compare
import seaborn as sns
#%%
# Load most recent data
with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_halfhourly_20240605.pkl",
    "rb",
) as f:
    etf_sel_halfhourly = pickle.load(f)

with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_daily_20240605.pkl",
    "rb",
) as f:
    etf_sel_daily = pickle.load(f)

df_buysell = pd.read_csv(r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\ETF data\BuySell_Imbalance_data.csv")
# Define mappings
etf_dict, event_dict, suffix_list, interval_mapping_dict, event_after_release_dict, code_dictionary = define_mappings()
# %%
# Take out only SPY data

df_spy = etf_sel_halfhourly['SPY']
df_buysell_spy = df_buysell[df_buysell['SYM_ROOT'] == 'SPY'].reset_index(drop=True)

#%%
df_agg = etf_sel_halfhourly['AGG']
df_buysell_agg = df_buysell[df_buysell['SYM_ROOT'] == 'AGG'].reset_index(drop=True)
#%%
# Get data from 2014-07-01, before that the buy sell volume data seems wrong
df_buysell_spy = df_buysell_spy[df_buysell_spy['DATE'] >= '2014-07-01'].reset_index(drop=True)


#%%

df_buysell_agg = add_daily_sums(df_agg, 'DATE', 'Volume', df_buysell_agg, 'DATE', 'total_interval_vol')


# %%
# Add close-to-close and open-to-close returns to the daily dataframe
df_buysell_spy = calculate_daily_return(df_spy, 'DATE', 'TIME', 'PRICE', df_buysell_spy, 'DATE')
#%%
df_buysell_agg = calculate_daily_return(df_agg, 'DATE', 'TIME', 'PRICE', df_buysell_agg, 'DATE')
# %%
# Add column of buy-sell imbalance following Diether et al. (2009)
df_buysell_spy['Buy_Sell_Imb'] = (df_buysell_spy['BuyVol_LR'] - df_buysell_spy['SellVol_LR']) / df_buysell_spy['total_vol']
#%%
df_buysell_agg['Buy_Sell_Imb'] = (df_buysell_agg['BuyVol_LR'] - df_buysell_agg['SellVol_LR']) / df_buysell_agg['total_vol']
# Add a lag
df_buysell_spy = add_lag(df_buysell_spy, 'Buy_Sell_Imb', 1)
#%%
df_buysell_spy = add_lag(df_buysell_spy, 'Return_close_close', 1)
df_buysell_spy = add_lag(df_buysell_spy, 'Return_open_close', 1)

#%%
# Options:
#  'Volume_scaled_rolling_window_interval_250days',
#  'Short_scaled_rolling_window_interval_250days',
#  'Short_Ratio_scaled_rolling_window_interval_250days',
#  'Volume_scaled_rolling_window_interval_20days',
#  'Short_scaled_rolling_window_interval_20days',
#  'Short_Ratio_scaled_rolling_window_interval_20days',
# 'abn_short_absolute_scaled_rolling_window_interval_250days',
#  'abn_short_absolute_scaled_rolling_window_interval_60days',
#  'abn_short_absolute_scaled_rolling_window_interval_20days',
# 'abn_short_absolute',
# 'abn_short_ratio'
df_buysell_spy = add_daily_sums(df_spy, 'DATE', 'Volume_scaled_rolling_window_interval_250days', df_buysell_spy, 'DATE', 'total_scaled_volume_250days')
df_buysell_spy = add_daily_sums(df_spy, 'DATE', 'Short_scaled_rolling_window_interval_250days', df_buysell_spy, 'DATE', 'total_scaled_shortvolume_250days')
df_buysell_spy = add_daily_sums(df_spy, 'DATE', 'Volume_scaled_rolling_window_interval_20days', df_buysell_spy, 'DATE', 'total_scaled_volume_20days')
df_buysell_spy = add_daily_sums(df_spy, 'DATE', 'Short_scaled_rolling_window_interval_20days', df_buysell_spy, 'DATE', 'total_scaled_shortvolume_20days')
df_buysell_spy = add_daily_sums(df_spy, 'DATE', 'abn_short_absolute_scaled_rolling_window_interval_250days', df_buysell_spy, 'DATE', 'total_scaled_abnshort_250days')
df_buysell_spy = add_daily_sums(df_spy, 'DATE', 'abn_short_absolute_scaled_rolling_window_interval_20days', df_buysell_spy, 'DATE', 'total_scaled_abnshort_20days')
df_buysell_spy = add_daily_sums(df_spy, 'DATE', 'abn_short_absolute', df_buysell_spy, 'DATE', 'total_abnshort')
df_buysell_spy = add_daily_sums(df_spy, 'DATE', 'abn_short_ratio', df_buysell_spy, 'DATE', 'total_abnshort_ratio')
#%%
# Add columns with total volume and total short volume, and short ratio 
df_buysell_spy = add_daily_sums(df_spy, 'DATE', 'Volume', df_buysell_spy, 'DATE', 'total_interval_vol')
df_buysell_spy = add_daily_sums(df_spy, 'DATE', 'Short', df_buysell_spy, 'DATE', 'total_short_vol')
df_buysell_spy['short_ratio'] = df_buysell_spy['total_short_vol'] / df_buysell_spy['total_interval_vol']

#%%
# Add a column with the difference in buy-sell imbalance and a lag
df_buysell_spy['Buy_Sell_Imb_diff'] = df_buysell_spy['Buy_Sell_Imb'] - df_buysell_spy['Buy_Sell_Imb_lag1']
df_buysell_spy = add_lag(df_buysell_spy, 'Buy_Sell_Imb_diff', 1)

#%%
# Add a column with a different measure to compare short volume and normal volume
df_buysell_spy['short_ratio_alt'] = (df_buysell_spy['total_interval_vol'] - df_buysell_spy['total_short_vol'])/(df_buysell_spy['total_interval_vol'] + df_buysell_spy['total_short_vol'])

#%%
df_buysell_spy['Buy_Sell_Imb_alt'] = df_buysell_spy['SellVol_LR'] / df_buysell_spy['BuyVol_LR']


#%%
df_buysell_spy = add_lag(df_buysell_spy,  'short_ratio', 1)
# df_buysell_spy['BuyVol_LR'] - df_buysell_spy['SellVol_LR']
#%% 
# Do some regressions

dep_var_list = [
   'Return_close_close',
#    'Return_open_close',
    # 'Buy_Sell_Imb',
    # 'total_interval_vol',
    # 'SellVol_LR',
    # 'BuyVol_LR',
]

indep_var_list = [
#  'Buy_Sell_Imb',
 'Buy_Sell_Imb_lag1',
#  'Return_close_close_lag1',
#  'Return_open_close_lag1',
    # 'total_interval_vol',
    # 'total_short_vol',
    # 'short_ratio',
    # 'short_ratio_alt',
    # 'Return_close_close',
    'Return_open_close',
    # 'Buy_Sell_Imb_diff',
    # 'Buy_Sell_Imb_diff_lag1',
    # 'total_scaled_volume_250days',
    # 'total_scaled_shortvolume_250days',
    # 'total_scaled_volume_20days',
    # 'total_scaled_shortvolume_20days',
    # 'total_scaled_abnshort_250days',
    # 'total_scaled_abnshort_20days',
    # 'total_abnshort',
    # 'total_abnshort_ratio',
    # 'Buy_Sell_Imb_alt'
    

]

covariance_type = 'HAC'

# Do regression with HAC standard errors, max leg is set to number of observations to the power 1/4
result = do_regression(
            df_buysell_spy, indep_var_list, dep_var_list, covariance_type, max_lag=7
        )

print(result[dep_var_list[0]].summary())

result_dict = {}
result_dict[dep_var_list[0]] = result

#%%
latex_table = get_latex_table(result_dict, dep_var_list, indep_var_list)

print(latex_table[0])

#%%
plot_time_series(df_buysell_agg, 'DATE', ['total_interval_vol'])

###


#%%
class ETF_Product:
    start_date = '2014-01-01'
    def __init__(self, ticker: str, interval_df, daily_df, interval_date_col:str='DATE', interval_time_col:str='TIME', interval_price_col:str='PRICE', daily_date_col:str='DATE'):

        self.ticker = ticker
        self.interval_df = interval_df
        self.daily_df = daily_df
        self.interval_date_col = interval_date_col
        self.daily_date_col = daily_date_col
        self.interval_time_col = interval_time_col
        self.interval_price_col = interval_price_col


    

    def set_daily_startdate(self, date):
        self.daily_df = self.daily_df[self.daily_df[self.daily_date_col] >= date].reset_index(drop=True)

    def add_total_col(self, target_col, output_col_name):
        import pandas as pd

        self.interval_df[self.interval_date_col] = pd.to_datetime(self.interval_df[self.interval_date_col])
        self.daily_df[self.daily_date_col] = pd.to_datetime(self.daily_df[self.daily_date_col] )

        daily_sums = self.interval_df.groupby(self.interval_date_col)[target_col].sum().reset_index()
        daily_sums.columns = [self.daily_date_col, output_col_name]

        result_df = pd.merge(self.daily_df, daily_sums, left_on=self.daily_date_col, right_on=self.daily_date_col, how='left')

        self.daily_df = result_df

    
    def add_returns(self):
        import pandas as pd

        self.interval_df[self.interval_date_col] = pd.to_datetime(self.interval_df[self.interval_date_col])
        self.daily_df[self.daily_date_col] = pd.to_datetime(self.daily_df[self.daily_date_col] )

        interval_df_16 = self.interval_df[self.interval_df[self.interval_time_col] == '16:00:00']
        interval_df_0930 = self.interval_df[self.interval_df[self.interval_time_col] == '09:30:00']

        # Merge the filtered interval DataFrame with the daily DataFrame on the date column
        merged_df = pd.merge(self.daily_df, interval_df_16[[self.interval_date_col, self.interval_price_col]], 
                         left_on=self.daily_date_col, right_on=self.interval_date_col, how='left', suffixes=('', '_16'))
        merged_df = pd.merge(merged_df, interval_df_0930[[self.interval_date_col, self.interval_price_col]], 
                         left_on=self.daily_date_col, right_on=self.interval_date_col, how='left', suffixes=('', '_0930'))
        
        merged_df = merged_df.sort_values(by=daily_date_col)
        
        # Calculate the return values
        merged_df['Return_close_close'] = (merged_df[self.interval_price_col] / merged_df[self.interval_price_col].shift(1)) - 1
        merged_df['Return_open_close'] = (merged_df[self.interval_price_col] / merged_df[self.interval_price_col + '_0930']) - 1

        self.daily_df = merged_df

#%%
class ETF_Product:
    start_date = '2014-01-01'

    def __init__(self, ticker: str, interval_dict, daily_df_full, interval_date_col: str = 'DATE', interval_time_col: str = 'TIME', interval_price_col: str = 'PRICE', daily_date_col: str = 'DATE'):
        self.ticker = ticker
        self.interval_df = interval_dict[self.ticker]
        self.daily_df = daily_df_full[daily_df_full['SYM_ROOT'] == self.ticker].reset_index(drop=True)
        self.interval_date_col = interval_date_col
        self.daily_date_col = daily_date_col
        self.interval_time_col = interval_time_col
        self.interval_price_col = interval_price_col

        self.daily_df['Buy_Sell_Imb'] = (self.daily_df['BuyVol_LR'] - self.daily_df['SellVol_LR']) / self.daily_df['total_vol']

        # Ensure date columns are in datetime format
        self.interval_df[self.interval_date_col] = pd.to_datetime(self.interval_df[self.interval_date_col])
        self.daily_df[self.daily_date_col] = pd.to_datetime(self.daily_df[self.daily_date_col])

    def set_daily_startdate(self, date):
        """
        Filter the daily dataframe to include data from a specific start date.
        """
        self.daily_df = self.daily_df[self.daily_df[self.daily_date_col] >= date].reset_index(drop=True)

    def add_total_col(self, target_col, output_col_name):
        """
        Add a column to the daily dataframe with the total sum of a target column from interval data.
        """
        # Calculate daily sums from interval data
        daily_sums = self.interval_df.groupby(self.interval_date_col)[target_col].sum().reset_index()
        daily_sums.columns = [self.daily_date_col, output_col_name]

        # Merge daily sums into the daily dataframe
        self.daily_df = pd.merge(self.daily_df, daily_sums, left_on=self.daily_date_col, right_on=self.daily_date_col, how='left')

    def add_returns(self):
        """
        Calculate and add return columns to the daily dataframe.
        """
        # Filter interval data for specific times
        interval_df_16 = self.interval_df[self.interval_df[self.interval_time_col] == '16:00:00']
        interval_df_0930 = self.interval_df[self.interval_df[self.interval_time_col] == '09:30:00']

        # Merge filtered interval data with daily data
        merged_df = pd.merge(self.daily_df, interval_df_16[[self.interval_date_col, self.interval_price_col]],
                             left_on=self.daily_date_col, right_on=self.interval_date_col, how='left', suffixes=('', '_16'))
        merged_df = pd.merge(merged_df, interval_df_0930[[self.interval_date_col, self.interval_price_col]],
                             left_on=self.daily_date_col, right_on=self.interval_date_col, how='left', suffixes=('', '_0930'))

        # Ensure the dataframe is sorted by date
        merged_df = merged_df.sort_values(by=self.daily_date_col)

        # Calculate the return values
        merged_df['Return_close_close'] = (merged_df[self.interval_price_col] / merged_df[self.interval_price_col].shift(1)) - 1
        merged_df['Return_open_close'] = (merged_df[self.interval_price_col] / merged_df[self.interval_price_col + '_0930']) - 1

        self.daily_df = merged_df


# %%

SPY = ETF_Product("SPY", etf_sel_halfhourly, df_buysell)

#%%
SPY.set_daily_startdate('2014-07-01')
SPY.add_returns()

#%%
total_cols = [
    'Short',
    'Volume',
]

for col in total_cols:
    SPY.add_total_col(col, f'Total_{col}')
# %%

SPY.daily_df['Short_Ratio'] = SPY.daily_df['Total_Short'] / SPY.daily_df['Total_Volume']
# %%
