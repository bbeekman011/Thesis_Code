# %%
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
    add_daily_cols,
)
from datetime import datetime, timedelta
import numpy as np
import time as tm
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# %%
## Load data pre-processed through Data file

# Load data with intervals on the rows
with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\etf_merged_30min_daily_dict.pkl",
    "rb",
) as f:
    etf_merged_30min_daily_dict = pickle.load(f)

# Load data with intervals in the columns
with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\etf_merged_30min_halfhourly_dict.pkl",
    "rb",
) as f:
    etf_merged_30min_halfhourly_dict = pickle.load(f)


# %%
# Get rid of short ratio in 09:30:00 column and of infinite values (temporary fix for infinite values until discussed further)

for key in etf_merged_30min_halfhourly_dict.keys():
    etf_merged_30min_halfhourly_dict[key].loc[
        etf_merged_30min_halfhourly_dict[key]["TIME"] == "09:30:00", "Short_Ratio"
    ] = pd.NA
    etf_merged_30min_halfhourly_dict[key] = etf_merged_30min_halfhourly_dict[
        key
    ].replace([np.inf, -np.inf], np.nan)


# %%
## Make selection of ETFs to be investigated for preliminary analysis

included_etfs = ["AGG", "HYG", "IEF", "LQD", "SPY", "SHY", "TLT"]

etf_sel_daily = {
    key: etf_merged_30min_daily_dict[key]
    for key in included_etfs
    if key in etf_merged_30min_daily_dict
}
etf_sel_halfhourly = {
    key: etf_merged_30min_halfhourly_dict[key]
    for key in included_etfs
    if key in etf_merged_30min_halfhourly_dict
}

#%%


# %%
## Make time series plot for specific period and variables
ticker = "HYG"
df = etf_sel_halfhourly[ticker]
start_date = "2022-03-14"
end_date = "2022-03-14"
title = ticker
y_1 = "Volume"
y_2 = "PRICE"
vert_line = "2022-03-14 14:00:00"


test_fig = intraday_plot(
    df,
    "DT",
    "DATE",
    start_date,
    end_date,
    title,
    y_1,
    y_1,
    y_1,
    y_2,
    y_2,
    y_2,
    vert_line,
)
test_fig.show()


# %%
## Get years and weekdays column in dataframes

for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key]["YEAR"] = etf_sel_halfhourly[key]["DT"].dt.year
    etf_sel_halfhourly[key]["Weekday"] = etf_sel_halfhourly[key]["DT"].dt.day_name()
# %%
## Get plots per year
grouped = (
    etf_sel_halfhourly["SPY"]
    .groupby(["YEAR", pd.Grouper(key="DT", freq="30T")])["Short"]
    .mean()
    .reset_index()
)

years = grouped["YEAR"].unique()
for year in years:
    year_data = grouped[grouped["YEAR"] == year]
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.bar(year_data["DT"].dt.strftime("%H:%M"), year_data["Short"])
    plt.title(f"Average Short Volume per 30-minute Interval - {year}")
    plt.xlabel("Time")
    plt.ylabel("Average Short Volume")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

# %%
## Plot short volume for selected tickers
for key in etf_sel_halfhourly.keys():

    df_grouped = (
        etf_sel_halfhourly[key]
        .groupby(etf_sel_halfhourly[key]["TIME"])["Short"]
        .mean()
        .reset_index()
    )

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df_grouped["TIME"], df_grouped["Short"], color="skyblue")
    plt.xlabel("Time")
    plt.ylabel(f"Average Short Volume {key}")
    plt.title(f"Average Short Volume per half hour - {key}")
    plt.xticks(rotation=45)
    plt.show()

# %%
## Plot volume for selected tickers
for key in etf_sel_halfhourly.keys():

    df_grouped = (
        etf_sel_halfhourly[key]
        .groupby(etf_sel_halfhourly[key]["TIME"])["Volume"]
        .mean()
        .reset_index()
    )

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df_grouped["TIME"], df_grouped["Volume"], color="skyblue")
    plt.xlabel("Time")
    plt.ylabel(f"Average Volume {key}")
    plt.title(f"Average Volume per half hour - {key}")
    plt.xticks(rotation=45)
    plt.show()

# %%
## Plot short ratio for selected tickers
for key in etf_sel_halfhourly.keys():

    df_grouped = (
        etf_sel_halfhourly[key]
        .groupby(etf_sel_halfhourly[key]["TIME"])["Short_Ratio"]
        .mean()
        .reset_index()
    )

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df_grouped["TIME"], df_grouped["Short_Ratio"], color="skyblue")
    plt.xlabel("Time")
    plt.ylabel(f"Average Short_Ratio Volume {key}")
    plt.title(f"Average Short_Ratio Volume per half hour - {key}")
    plt.xticks(rotation=45)
    plt.show()

# %%
ticker = "LQD"
## Get average volume per weekday
df = etf_sel_halfhourly[ticker]
# Define the order of weekdays
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Calculate daily averages
daily_avg = df.groupby("Weekday")["Short"].mean().reindex(weekday_order)
plt.bar(daily_avg.index, daily_avg.values)
plt.xlabel("Day")
plt.ylabel("Average Short Volume")
plt.title(f"Average Short volume per day - {ticker}")
plt.show()

df = etf_sel_halfhourly[ticker]
## Get average volume on specific weekdays
start_time = pd.Timestamp("09:30:00")
end_time = pd.Timestamp("16:00:00")
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Generate intervals programmatically
interval_length = pd.Timedelta(minutes=30)
intervals = pd.date_range(
    start=start_time, end=end_time, freq=interval_length
).strftime("%H:%M:%S")


# Function to calculate average value for each interval for all weekdays
def interval_avg_for_all_weekdays(df):
    avg_per_interval = {}
    for interval in intervals:
        avg_per_interval[interval] = (
            df[df["TIME"] == interval]
            .groupby("Weekday")["Short"]
            .mean()
            .reindex(weekday_order)
        )
    return avg_per_interval


# Calculate average value for each interval for all weekdays
avg_per_interval = interval_avg_for_all_weekdays(df)

# Plotting the bar plot
fig, ax = plt.subplots(figsize=(12, 6))

# Define colors for each weekday
colors = ["blue", "orange", "green", "red", "purple"]

# Plot each interval for all weekdays separately
for i, interval in enumerate(intervals):
    # Get average values for the interval
    avg_values = avg_per_interval[interval]
    # Plot each weekday as a separate bar with a different color
    for j, (day, avg) in enumerate(avg_values.items()):
        ax.bar(i + j * 0.1, avg, width=0.1, label=day, color=colors[j])

ax.set_xlabel("Interval")
ax.set_ylabel("Short Volume")
ax.set_title(f"Average Short Volume per day - {ticker}")

# Create a legend with each weekday appearing only once
handles, labels = ax.get_legend_handles_labels()
unique_labels = []
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)
ax.legend(
    unique_handles,
    unique_labels,
    title="Weekday",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

# Set x-axis ticks and labels
ax.set_xticks(range(len(intervals)))
ax.set_xticklabels(intervals, rotation=45)

plt.tight_layout()
plt.show()

# %%
## Linear regression to test intraday momentum effect

x = etf_sel_daily["SPY"]["Return_FH"].values.reshape(-1, 1)
y = etf_sel_daily["SPY"]["Return_LH"].values

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
intercept = model.intercept_
coeff = model.coef_


print(f"R-squared: {r_sq}, intercept: {intercept}, coeff: {coeff}")

# %%
# Code for testing with regressions
# included_etfs = ['AGG', 'HYG', 'IEF', 'LQD', 'SPY', 'SHY', 'TLT']
ticker = "SHY"

# Remove rows with nan or inf values

etf_sel_daily[ticker] = etf_sel_daily[ticker].dropna()
etf_sel_daily[ticker] = (
    etf_sel_daily[ticker].replace([np.inf, -np.inf], np.nan).dropna()
)


# x = etf_sel_daily[ticker][["Short_Ratio_FH", "Short_Ratio_10first"]].values
x = etf_sel_daily[ticker]["Short_Ratio_SLH"].values.reshape(-1, 1)
y = etf_sel_daily[ticker]["Return_LH"].values

x = sm.add_constant(x)

model = sm.OLS(y, x)

results = model.fit()

print(results.summary())

# %%


# Define the function to calculate rolling window average for trading days, including the day for which it is calculated
def rolling_avg_trading_days(series, window_size):
    trading_days = series.index.dayofweek < 5  # Filter only weekdays
    return series[trading_days].rolling(window=window_size, min_periods=1).mean()


## Get 5, 10 and 20 trading day rolling-windows of short volume

for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key].set_index("DT", inplace=True)

    rolling_avg_5 = (
        etf_sel_halfhourly[key]
        .groupby(etf_sel_halfhourly[key].index.time)["Short"]
        .apply(rolling_avg_trading_days, window_size=5)
    )
    rolling_avg_10 = (
        etf_sel_halfhourly[key]
        .groupby(etf_sel_halfhourly[key].index.time)["Short"]
        .apply(rolling_avg_trading_days, window_size=10)
    )
    rolling_avg_20 = (
        etf_sel_halfhourly[key]
        .groupby(etf_sel_halfhourly[key].index.time)["Short"]
        .apply(rolling_avg_trading_days, window_size=20)
    )

    rolling_avg_5 = rolling_avg_5.reset_index(level=0, drop=True)
    rolling_avg_10 = rolling_avg_10.reset_index(level=0, drop=True)
    rolling_avg_20 = rolling_avg_20.reset_index(level=0, drop=True)

    etf_sel_halfhourly[key]["Week_Avg_Short"] = rolling_avg_5
    etf_sel_halfhourly[key]["Two_Week_Avg_Short"] = rolling_avg_10
    etf_sel_halfhourly[key]["Month_Avg_Short"] = rolling_avg_20

    etf_sel_halfhourly[key].reset_index(inplace=True)

# %%
## Get deviations from averages
for key in etf_sel_halfhourly.keys():

    etf_sel_halfhourly[key]["Abn_Short_Week"] = (
        etf_sel_halfhourly[key]["Short"] - etf_sel_halfhourly[key]["Week_Avg_Short"]
    )
    etf_sel_halfhourly[key]["Abn_Short_Two_Week"] = (
        etf_sel_halfhourly[key]["Short"] - etf_sel_halfhourly[key]["Two_Week_Avg_Short"]
    )
    etf_sel_halfhourly[key]["Abn_Short_Month"] = (
        etf_sel_halfhourly[key]["Short"] - etf_sel_halfhourly[key]["Month_Avg_Short"]
    )

    # %%
etf_sel_halfhourly["AGG"]
column_name = "Short_Ratio"
plt.figure(figsize=(10, 6))
plt.hist(etf_sel_halfhourly["AGG"][column_name], density=True, bins=30, alpha=0.7)
plt.title("Density Plot of {}".format(column_name))
plt.xlabel(column_name)
plt.ylabel("Density")
plt.show()
# %%


# %%


def add_daily_cols(df, suffix_list, func, input_col1, input_col2, new_col):

    output_df = df.copy()
    for suffix in suffix_list:

        input_col_full_1 = f"{input_col1}_{suffix}"
        input_col_full_2 = f"{input_col2}_{suffix}"

        new_col_full = f"{new_col}_{suffix}"

        output_df[new_col_full] = func(
            output_df[input_col_full_1], output_df[input_col_full_2]
        )

    return output_df


# %%
suffix_list = [
    "FH",
    "10first",
    "10second",
    "11first",
    "11second",
    "12first",
    "12second",
    "13first",
    "13second",
    "14first",
    "14second",
    "SLH",
    "LH",
]


def short_ratio(value1, value2):
    return value1 / value2


# %%
## Add short ratio to the 'daily' dataframes
for key in etf_sel_daily.keys():
    etf_sel_daily[key] = add_daily_cols(
        etf_sel_daily[key], suffix_list, short_ratio, "Short", "Volume", "Short_Ratio"
    )
# %%
for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key]['Short_Ratio_fdiff'] = etf_sel_halfhourly[key]['Short_Ratio'].diff()