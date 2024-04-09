#%%
## Import functions
import pyreadr
import pandas as pd
from collections import OrderedDict
from Functions import (
    split_df_on_symbol,
    merge_df_on_vol_columns,
    merge_df_on_price_rows,
    fill_missing_intervals,
    intraday_plot,
)
from datetime import datetime, timedelta
import numpy as np
import time as tm
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#%%
## Load data pre-processed through Data file

# Load data with intervals on the rows
with open(r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\etf_merged_30min_daily_dict.pkl", "rb") as f:
    etf_merged_30min_daily_dict = pickle.load(f)

# Load data with intervals in the columns
with open(r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\etf_merged_30min_halfhourly_dict.pkl", "rb") as f:
    etf_merged_30min_halfhourly_dict = pickle.load(f)


#%%
## Look at average volume of different ETFs
average_volumes= {}
average_short_volumes = {}
average_short_ratio = {}

for key, df in etf_merged_30min_halfhourly_dict.items():
    average_vol_dollar = df['Volume_dollar'].mean()
    average_short_dollar = df['Short_dollar'].mean()
    average_sr = df['Short_Ratio'].mean()
    average_volumes[key] = average_vol_dollar
    average_short_volumes[key] = average_short_dollar
    average_short_ratio[key] = average_sr
 
for key, value in average_volumes.items():
    formatted_value = '{:,.2f}'.format(value)
    print(f"Average 30-minute volume for {key}: {formatted_value}")

for key, value in average_short_volumes.items():
    formatted_value = '{:,.2f}'.format(value)
    print(f"Average 30-minute volume for {key}: {formatted_value}")

for key, value in average_short_ratio.items():
    formatted_value = '{:,.2f}'.format(value)
    print(f"Average 30-minute volume for {key}: {formatted_value}")

#%%
## Make selection of ETFs to be investigated for preliminary analysis

included_etfs = ['AGG', 'HYG', 'IEF', 'LQD', 'SPY', 'SHY', 'TLT']

etf_sel_daily = {key: etf_merged_30min_daily_dict[key] for key in included_etfs if key in etf_merged_30min_daily_dict}
etf_sel_halfhourly = {key: etf_merged_30min_halfhourly_dict[key] for key in included_etfs if key in etf_merged_30min_halfhourly_dict}

#%%
df = etf_sel_halfhourly['HYG']
start_date = '2014-07-09' 
end_date = '2014-07-10'
title = 'Test'
y_1 = 'RETURN'
y_2 = 'Short_Ratio'


test_fig = intraday_plot(df, 'DT', 'DATE', start_date, end_date, title, y_1, "Price", 'Price', y_2, 'Short Ratio', 'Short Ratio')
test_fig.show()
# %%
