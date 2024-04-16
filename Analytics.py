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
# Get rid of short ratio in 09:30:00 column and of infinite values (temporary fix for infinite values until discussed further)

for key in etf_merged_30min_halfhourly_dict.keys():
    etf_merged_30min_halfhourly_dict[key].loc[etf_merged_30min_halfhourly_dict[key]['TIME'] == '09:30:00', 'Short_Ratio'] = pd.NA
    etf_merged_30min_halfhourly_dict[key] = etf_merged_30min_halfhourly_dict[key].replace([np.inf, -np.inf], np.nan)






#%%
## Look at average volume of different ETFs
average_volumes= {}
average_short_volumes = {}
average_short_ratio = {}

for key, df in etf_merged_30min_halfhourly_dict.items():
    average_vol_dollar = df['Volume'].mean()
    average_short_dollar = df['Short'].mean()
    average_sr = df['Short_Ratio'].mean()
    average_volumes[key] = average_vol_dollar
    average_short_volumes[key] = average_short_dollar
    average_short_ratio[key] = average_sr
 
for key, value in average_volumes.items():
    formatted_value = '{:,.2f}'.format(value)
    print(f"Average 30-minute volume for {key}: {formatted_value}")

for key, value in average_short_volumes.items():
    formatted_value = '{:,.2f}'.format(value)
    print(f"Average 30-minute short volume for {key}: {formatted_value}")

for key, value in average_short_ratio.items():
    formatted_value = '{:,.2f}'.format(value)
    print(f"Average 30-minute short ratio for {key}: {formatted_value}")

#%%
## Make selection of ETFs to be investigated for preliminary analysis

included_etfs = ['AGG', 'HYG', 'IEF', 'LQD', 'SPY', 'SHY', 'TLT']

etf_sel_daily = {key: etf_merged_30min_daily_dict[key] for key in included_etfs if key in etf_merged_30min_daily_dict}
etf_sel_halfhourly = {key: etf_merged_30min_halfhourly_dict[key] for key in included_etfs if key in etf_merged_30min_halfhourly_dict}

#%%
df = etf_sel_halfhourly['TLT']
start_date = '2020-01-01' 
end_date = '2022-12-31'
title = 'Test'
y_1 = 'PRICE'
y_2 = 'Short_Ratio'


test_fig = intraday_plot(df, 'DT', 'DATE', start_date, end_date, title, y_1, "Price", 'Price', y_2, 'Short Ratio', 'Short Ratio')
test_fig.show()
# %%


for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key]['YEAR'] = etf_sel_halfhourly[key]['DT'].dt.year

# %%
grouped = etf_sel_halfhourly['SPY'].groupby(['YEAR', pd.Grouper(key='DT', freq='30T')])['Short'].mean().reset_index()
#%%
years = grouped['YEAR'].unique()
for year in years:
    year_data = grouped[grouped['YEAR'] == year]
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.bar(year_data['DT'].dt.strftime('%H:%M'), year_data['Short'])
    plt.title(f'Average Short Volume per 30-minute Interval - {year}')
    plt.xlabel('Time')
    plt.ylabel('Average Short Volume')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()
# %%

for key in etf_sel_halfhourly.keys():
    grouped = etf_sel_halfhourly[key].groupby(pd.Grouper(key='DT', freq='30T'))['Short'].mean().reset_index()

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.bar(grouped['DT'].dt.strftime('%H:%M'), grouped['Short'])
    plt.title(f'Average Short Volume per 30-minute Interval - {key}')
    plt.xlabel('Time')
    plt.ylabel('Average Short Volume')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()
# %%
for key in etf_sel_halfhourly.keys():

    df_grouped = etf_sel_halfhourly[key].groupby(etf_sel_halfhourly[key]['TIME'])['Short'].mean().reset_index()

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df_grouped['TIME'], df_grouped['Short'], color='skyblue')
    plt.xlabel('Time')
    plt.ylabel(f'Average Short Volume {key}')
    plt.title(f'Average Short Volume per half hour - {key}')
    plt.xticks(rotation=45)
    plt.show()

# %%
for key in etf_sel_halfhourly.keys():

    df_grouped = etf_sel_halfhourly[key].groupby(etf_sel_halfhourly[key]['TIME'])['Volume'].mean().reset_index()

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df_grouped['TIME'], df_grouped['Volume'], color='skyblue')
    plt.xlabel('Time')
    plt.ylabel(f'Average Volume {key}')
    plt.title(f'Average Volume per half hour - {key}')
    plt.xticks(rotation=45)
    plt.show()

# %%
for key in etf_sel_halfhourly.keys():

    df_grouped = etf_sel_halfhourly[key].groupby(etf_sel_halfhourly[key]['TIME'])['Short_Ratio'].mean().reset_index()

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df_grouped['TIME'], df_grouped['Short_Ratio'], color='skyblue')
    plt.xlabel('Time')
    plt.ylabel(f'Average Short_Ratio Volume {key}')
    plt.title(f'Average Short_Ratio Volume per half hour - {key}')
    plt.xticks(rotation=45)
    plt.show()
