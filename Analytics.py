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
    get_eventday_plots,
    short_ratio,
    add_daily_cols,
    rolling_avg_trading_days,
    add_rolling_window_average_col,
    intraday_barplot,
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
import os

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

# Load event data
df_events = pd.read_excel(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Relevant_event_data.xlsx"
)


etf_dict = {
    "AGG": "iShares Core U.S. Aggregate Bond ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "DIA": "SPDR Dow Jones Industrial Average ETF Trust",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "EFA": "iShares MSCI EAFE ETF",
    "FXI": "iShares China Large-Cap ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "IWM": "iShares Russell 2000 ETF",
    "IYR": "iShares U.S. Real Estate ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "QQQ": "Invesco QQQ Trust Series I (Nasdaq)",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "SPY": "SPDR S&P 500 ETF Trust",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "USHY": "iShares Broad USD High Yield Corporate Bond ETF",
    "VCIT": "Vanguard Intermediate-Term Corporate Bond ETF",
    "VCSH": "Vanguard Short-Term Corporate Bond ETF",
    "VWO": "Vanguard FTSE Emerging Markets ETF",
    "XLB": "Materials Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLK": "Technology Select Sector SPDR Fund",
    "XLP": "Consumer Staples Select Sector SPDR Fund",
    "XLRE": "Real Estate Select Sector SPDR Fund",
    "XLU": "Utilities Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
    "XLY": "Consumer Discretionary Select Sector SPDR Fund",
}


## List of column suffixes for the 'daily' dataframe
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
## Get event_df in correct time range
## Get event_df in correct time range
start_date = '2014-01-01'
end_date = '2022-12-31'

df_events = df_events[(df_events['DATE'] >= start_date) & (df_events['DATE']<= end_date)]

## Bit gimmicky way to get 'lagged' dummy variables in a bit
df_events["DATE"] = pd.to_datetime(df_events["DATE"])
df_events["DATE_LAG"] = df_events["DATE"] - pd.Timedelta(days=1)

# Back to string
df_events["DATE"] = df_events["DATE"].dt.strftime("%Y-%m-%d")
df_events["DATE_LAG"] = df_events["DATE_LAG"].dt.strftime("%Y-%m-%d")

df_events.reset_index(drop=True, inplace=True)


#%%
# selected_events = [
#     'ISM Manufacturing',
#     'FOMC Rate Decision (Upper Bound)',
#     'Change in Nonfarm Payrolls',
#     'CPI YoY',
#     'GDP Annualized QoQ',
#     'Industrial Production MoM',
#     'Personal Income',
#     'Housing Starts',
#     'PPI Ex Food and Energy MoM',
# ]
for key in etf_merged_30min_halfhourly_dict.keys():
    event_map = {}
    
    # Populate the event map with date-event pairs
    for index, row in df_events.iterrows():
        date = row["DATE"]
        event = row["Event"]
        if date in event_map:
            event_map[date].append(event)
        else:
            event_map[date] = [event]

    df1 = etf_merged_30min_halfhourly_dict[key].copy()

    # Create a function to check if an event exists on a given date
    def check_event(date, event_list):
        if date in event_map and any(event in event_map[date] for event in event_list):
            return 1
        return 0

    # Apply the check_event function to create dummy variables for each event
    df1["ISM"] = df1["DATE"].apply(lambda x: check_event(x, ["ISM Manufacturing"]))
    df1["FOMC"] = df1["DATE"].apply(lambda x: check_event(x, ["FOMC Rate Decision (Upper Bound)"]))
    df1["NFP"] = df1["DATE"].apply(lambda x: check_event(x, ["Change in Nonfarm Payrolls"]))
    df1["CPI"] = df1["DATE"].apply(lambda x: check_event(x, ["CPI YoY"]))
    df1["GDP"] = df1["DATE"].apply(lambda x: check_event(x, ["GDP Annualized QoQ"]))
    df1["IP"] = df1["DATE"].apply(lambda x: check_event(x, ["Industrial Production MoM"]))
    df1["PI"] = df1["DATE"].apply(lambda x: check_event(x, ["Personal Income"]))
    df1["HST"] = df1["DATE"].apply(lambda x: check_event(x, ["Housing Starts"]))
    df1["PPI"] = df1["DATE"].apply(lambda x: check_event(x, ["PPI Ex Food and Energy MoM"]))

    # Create a combined 'EVENT' column
    df1["EVENT"] = df1[["ISM", "FOMC", "NFP", "CPI", "GDP", "IP", "PI", "HST", "PPI"]].max(axis=1)

    etf_merged_30min_halfhourly_dict[key] = df1


#%%
## Add the lagged dummies

## Change the date column of the event dataframe
df_events["temp"] = df_events["DATE"]
df_events["DATE"] = df_events["DATE_LAG"]



for key in etf_merged_30min_halfhourly_dict.keys():
    event_map = {}
    
    # Populate the event map with date-event pairs
    for index, row in df_events.iterrows():
        date = row["DATE"]
        event = row["Event"]
        if date in event_map:
            event_map[date].append(event)
        else:
            event_map[date] = [event]

    df1 = etf_merged_30min_halfhourly_dict[key].copy()

    # Create a function to check if an event exists on a given date
    def check_event(date, event_list):
        if date in event_map and any(event in event_map[date] for event in event_list):
            return 1
        return 0

    # Apply the check_event function to create dummy variables for each event
    df1["ISM_lag"] = df1["DATE"].apply(lambda x: check_event(x, ["ISM Manufacturing"]))
    df1["FOMC_lag"] = df1["DATE"].apply(lambda x: check_event(x, ["FOMC Rate Decision (Upper Bound)"]))
    df1["NFP_lag"] = df1["DATE"].apply(lambda x: check_event(x, ["Change in Nonfarm Payrolls"]))
    df1["CPI_lag"] = df1["DATE"].apply(lambda x: check_event(x, ["CPI YoY"]))
    df1["GDP_lag"] = df1["DATE"].apply(lambda x: check_event(x, ["GDP Annualized QoQ"]))
    df1["IP_lag"] = df1["DATE"].apply(lambda x: check_event(x, ["Industrial Production MoM"]))
    df1["PI_lag"] = df1["DATE"].apply(lambda x: check_event(x, ["Personal Income"]))
    df1["HST_lag"] = df1["DATE"].apply(lambda x: check_event(x, ["Housing Starts"]))
    df1["PPI_lag"] = df1["DATE"].apply(lambda x: check_event(x, ["PPI Ex Food and Energy MoM"]))

    # Create a combined 'EVENT' column
    df1["EVENT_lag"] = df1[["ISM_lag", "FOMC_lag", "NFP_lag", "CPI_lag", "GDP_lag", "IP_lag", "PI_lag", "HST_lag", "PPI_lag"]].max(axis=1)

    etf_merged_30min_halfhourly_dict[key] = df1

df_events["DATE"] = df_events["temp"]
df_events = df_events.drop(columns=["DATE_LAG", "temp"])

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


# %%
## Add short ratio to the 'daily' dataframes
for key in etf_sel_daily.keys():
    etf_sel_daily[key] = add_daily_cols(
        etf_sel_daily[key], suffix_list, short_ratio, "Short", "Volume", "Short_Ratio"
    )

# %%
## Get event counts

abbr_list = [
    "ISM",
    "FOMC",
    "NFP",
    "CPI",
    "GDP",
    "IP",
    "PI",
    "HST",
    "PPI",
    "EVENT",
    "ISM_lag",
    "FOMC_lag",
    "NFP_lag",
    "CPI_lag",
    "GDP_lag",
    "IP_lag",
    "PI_lag",
    "HST_lag",
    "PPI_lag",
    "EVENT_lag",
]

for abbr in abbr_list:
    event_count = (
        etf_sel_halfhourly["AGG"].groupby(etf_sel_halfhourly["AGG"]["DATE"])[abbr].max()
    )
    num_events = (event_count > 0).sum()
    print(f"{abbr}: {num_events}")

# %%
# Get barplots for average intraday short volume for event- and non-event days
# In the sample, no two selected events happen on the same day, so a zero in any of the event columns coincides with a zero in 'EVENT'

ticker = "TLT"  # Choose from "AGG", "HYG", "IEF", "LQD", "SPY", "SHY", "TLT"
metric = "Abn_Short_20day"  # Choose from "Short", "Short_dollar", "Volume", "Volume_dollar", "Short_Ratio", "RETURN"
event = "PI"  # Choose from "ISM", "FOMC", "NFP", "CPI", "GDP", "IP", "PI", "HST", "PPI", "EVENT" or any of the lags, e.g. ISM_lag
start_date = "2014-01-01"
end_date = "2022-12-31"
non_event_def = True  # Set to True if non-event is defined as no events at all, set to False if non-event is defined as no other event of that specific event (so other events are counted as non-event)

intraday_barplot(
    etf_sel_halfhourly, ticker, metric, start_date, end_date, event, non_event_def
)


# %%
## Make time series plot for specific period and variables
ticker = "IEF"
df = etf_sel_halfhourly[ticker]
start_date = "2022-04-01"
end_date = "2022-04-05"
title = ticker
y_1 = "Short_Ratio"
y_2 = "PRICE"
vert_line = "2022-04-02 14:00:00"


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
## Loop through tickers, event dates

# ## Events corresponding to some FOMC meetings, see specifics in the word file
# event_dt_list = [
#     "2020-03-03 14:00:00",
#     "2022-01-26 14:00:00",
#     "2022-03-16 14:00:00",
#     "2022-05-04 14:00:00",
#     "2022-06-15 14:00:00",
#     "2022-07-27 14:00:00",
#     "2022-09-21 14:00:00",
#     "2022-11-02 14:00:00",
#     "2022-12-14 14:00:00",
# ]


## Events correspdoning to some CPI announcements, see specifics in the word file
event_dt_list = [
    "2021-05-12 08:30:00",
    "2021-06-10 08:30:00",
    "2021-07-13 08:30:00",
    "2021-08-11 08:30:00",
    "2021-09-14 08:30:00",
    "2021-10-13 08:30:00",
    "2021-11-10 08:30:00",
    "2021-12-10 08:30:00",
    "2022-01-12 08:30:00",
    "2022-02-10 08:30:00",
    "2022-03-10 08:30:00",
    "2022-04-12 08:30:00",
    "2022-05-11 08:30:00",
    "2022-06-10 08:30:00",
    "2022-07-13 08:30:00",
    "2022-08-10 08:30:00",
    "2022-09-13 08:30:00",
    "2022-10-13 08:30:00",
    "2022-11-10 08:30:00",
    "2022-12-13 08:30:00",
]


col_list = ["Short_Ratio", "Short", "Volume"]
y_2 = "PRICE"


## Get plots of the selected etfs for pre-defined day range and for selected variables in col_list

## Specify parent directory
parent_dir = (
    r"C:/Users/ROB7831/OneDrive - Robeco Nederland B.V/Documents/Thesis/Plots/CPI"
)

## Save a lot of plots
get_eventday_plots(
    etf_dict,
    etf_sel_halfhourly,
    included_etfs,
    event_dt_list,
    col_list,
    y_2,
    parent_dir,
    3,
)

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
## Add new column of average short_ratio

col_list_ave = [
    "Volume_FH",
    "Volume_10first",
    "Volume_10second",
    "Volume_11first",
    "Volume_11second",
    "Volume_12first",
    "Volume_12second",
    "Volume_13first",
    "Volume_13second",
    "Volume_14first",
    "Volume_14second",
    "Volume_SLH",
]

for key in etf_sel_daily.keys():
    etf_sel_daily[key]["Volume_FDNLH"] = etf_sel_daily[key][col_list_ave].mean(axis=1)

# %%
# Code for testing with regressions
# # included_etfs = ['AGG', 'HYG', 'IEF', 'LQD', 'SPY', 'SHY', 'TLT']
# ticker = "SHY"

# # Remove rows with nan or inf values

# etf_sel_daily[ticker] = etf_sel_daily[ticker].dropna()
# etf_sel_daily[ticker] = (
#     etf_sel_daily[ticker].replace([np.inf, -np.inf], np.nan).dropna()
# )


# ## Define regression variables
# x = etf_sel_daily[ticker][["Short_Ratio_FDNLH", "Short_FDNLH"]].values
# # x = etf_sel_daily[ticker]["Short_FDNLH"].values.reshape(-1, 1)
# y = etf_sel_daily[ticker]["Return_LH"].values

# x = sm.add_constant(x)

# model = sm.OLS(y, x)

# results = model.fit()


# print(results.summary())
# %%

## Get 5, 10 and 20 and 100 trading day rolling-windows of short volume

for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key] = add_rolling_window_average_col(
        etf_sel_halfhourly[key], "Short", 5, "DT"
    )
    etf_sel_halfhourly[key] = add_rolling_window_average_col(
        etf_sel_halfhourly[key], "Short", 10, "DT"
    )
    etf_sel_halfhourly[key] = add_rolling_window_average_col(
        etf_sel_halfhourly[key], "Short", 20, "DT"
    )
    etf_sel_halfhourly[key] = add_rolling_window_average_col(
        etf_sel_halfhourly[key], "Short", 100, "DT"
    )


# %%
## Get deviations from averages

def add_abn_col(df_in, var, window_size, new_col_name):

    df = df_in.copy()

    df[f'{new_col_name}_{var}_{window_size}day'] = df[var] - df[f'{var}_Average_{window_size}day']

    return df

#%%

var = 'Short'
new_col_name = 'Abn'


for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key] = add_abn_col(etf_sel_halfhourly[key], var, 5, new_col_name)
    etf_sel_halfhourly[key] = add_abn_col(etf_sel_halfhourly[key], var, 10, new_col_name)
    etf_sel_halfhourly[key] = add_abn_col(etf_sel_halfhourly[key], var, 20, new_col_name)
    etf_sel_halfhourly[key] = add_abn_col(etf_sel_halfhourly[key], var, 100, new_col_name)


#%%

for key in etf_sel_halfhourly.keys():

    etf_sel_halfhourly[key]["Abn_Short_5day"] = (
        etf_sel_halfhourly[key]["Short"] - etf_sel_halfhourly[key]["Week_Avg_5day"]
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
