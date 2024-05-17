# %% ## Import functions
import pyreadr
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
import os
import math
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

# %% ## Load data pre-processed through Data file (no additional analysis or variables present)


# # Load data with intervals on the rows
# with open(
#     r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\etf_merged_30min_daily_dict.pkl",
#     "rb",
# ) as f:
#     etf_merged_30min_daily_dict = pickle.load(f)

# # Load data with intervals in the columns
# with open(
#     r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\etf_merged_30min_halfhourly_dict.pkl",
#     "rb",
# ) as f:
#     etf_merged_30min_halfhourly_dict = pickle.load(f)

# # Load event data
# df_events = pd.read_excel(
#     r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Relevant_event_data.xlsx"
# )


# Load data with most variables in there

# Data per 10-05-2024
# with open(
#     r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_halfhourly_20240510.pkl",
#     "rb",
# ) as f:
#     etf_sel_halfhourly = pickle.load(f)

# with open(
#     r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_daily_20240510.pkl",
#     "rb",
# ) as f:
#     etf_sel_daily = pickle.load(f)

# # Data per 13-05-2024
# with open(
#     r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_halfhourly_20240513.pkl",
#     "rb",
# ) as f:
#     etf_sel_halfhourly = pickle.load(f)

# with open(
#     r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_daily_20240513.pkl",
#     "rb",
# ) as f:
#     etf_sel_daily = pickle.load(f)

# # Data per 14-05-2024, including different future cumulative return specifications
# with open(
#     r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_halfhourly_20240514.pkl",
#     "rb",
# ) as f:
#     etf_sel_halfhourly = pickle.load(f)

# with open(
#     r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_daily_20240514.pkl",
#     "rb",
# ) as f:
#     etf_sel_daily = pickle.load(f)

# Data per 15-05-2024, including different future cumulative return specifications
with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_halfhourly_20240515.pkl",
    "rb",
) as f:
    etf_sel_halfhourly = pickle.load(f)

with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_daily_20240515.pkl",
    "rb",
) as f:
    etf_sel_daily = pickle.load(f)



# %% # Define some dictionaries for mapping
# Make selection of ETFs to investigate

# included_etfs = ["AGG", "HYG", "IEF", "LQD", "SPY", "SHY", "TLT"]

# etf_sel_daily = {
#     key: etf_merged_30min_daily_dict[key]
#     for key in included_etfs
#     if key in etf_merged_30min_daily_dict
# }
# etf_sel_halfhourly = {
#     key: etf_merged_30min_halfhourly_dict[key]
#     for key in included_etfs
#     if key in etf_merged_30min_halfhourly_dict
# }
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

## Dictionary linking event abbreviations to their relevant description in Bloomberg
event_dict = {
    "ISM": "ISM Manufacturing",
    "FOMC": "FOMC Rate Decision (Upper Bound)",
    "NFP": "Change in Nonfarm Payrolls",
    "CPI": "CPI YoY",
    "GDP": "GDP Annualized QoQ",
    "IP": "Industrial Production MoM",
    "PI": "Personal Income",
    "HST": "Housing Starts",
    "PPI": "PPI Ex Food and Energy MoM",
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

interval_mapping_dict = {
    "09:30:00": "09first",
    "10:00:00": "FH",
    "10:30:00": "10first",
    "11:00:00": "10second",
    "11:30:00": "11first",
    "12:00:00": "11second",
    "12:30:00": "12first",
    "13:00:00": "12second",
    "13:30:00": "13first",
    "14:00:00": "13second",
    "14:30:00": "14first",
    "15:00:00": "14second",
    "15:30:00": "SLH",
    "16:00:00": "LH",
}

event_after_release_dict = {
    "ISM": "10:30:00",
    "FOMC": "14:30:00",
    "NFP": "09:30:00",
    "CPI": "09:30:00",
    "GDP": "09:30:00",
    "IP": "09:30:00",
    "PI": "09:30:00",
    "HST": "09:30:00",
    "PPI": "09:30:00",
}

intervals = etf_sel_halfhourly["AGG"]["TIME"].unique()
# %% # Get rid of short ratio in 09:30:00 column and of infinite values (temporary fix for infinite values until discussed further)


for key in etf_merged_30min_halfhourly_dict.keys():
    etf_merged_30min_halfhourly_dict[key].loc[
        etf_merged_30min_halfhourly_dict[key]["TIME"] == "09:30:00", "Short_Ratio"
    ] = pd.NA
    etf_merged_30min_halfhourly_dict[key] = etf_merged_30min_halfhourly_dict[
        key
    ].replace([np.inf, -np.inf], np.nan)

## Get event_df in correct time range
start_date = "2014-01-01"
end_date = "2022-12-31"

df_events = event_date_transformation(df_events, start_date, end_date)


# %% ## Add dummy variables for different events, see events specified in event_dict

etf_sel_halfhourly = add_event_dummies(etf_sel_halfhourly, df_events, event_dict, 0)
etf_sel_daily = add_event_dummies(etf_sel_daily, df_events, event_dict, 0)

## Add lagged dummies
etf_sel_halfhourly = add_event_dummies(etf_sel_halfhourly, df_events, event_dict, 1)
etf_sel_daily = add_event_dummies(etf_sel_daily, df_events, event_dict, 1)

# %% # Add surprise dummies to half-hourly dict

# Specify events for which surprise dummies are added
event_list = ["FOMC", "ISM"]


# etf_sel_halfhourly = add_surprise_dummies(etf_sel_halfhourly, event_dict, df_events, event_list, 'absolute')
etf_sel_halfhourly = add_surprise_dummies(
    etf_sel_halfhourly, event_dict, df_events, event_list, "1_stdev"
)
etf_sel_halfhourly = add_surprise_dummies(
    etf_sel_halfhourly, event_dict, df_events, event_list, "2_stdev"
)
etf_sel_halfhourly = add_surprise_dummies(
    etf_sel_halfhourly, event_dict, df_events, event_list, "marketfh"
)
# etf_sel_halfhourly = add_surprise_dummies(etf_sel_halfhourly, event_dict, df_events, event_list, '0.0025_marketfh')
etf_sel_halfhourly = add_surprise_dummies(
    etf_sel_halfhourly, event_dict, df_events, event_list, "marketrod"
)
# etf_sel_halfhourly = add_surprise_dummies(etf_sel_halfhourly, event_dict, df_events, event_list, '0.0025_marketrod')
# etf_sel_halfhourly = add_surprise_dummies(etf_sel_halfhourly, event_dict, df_events, event_list, '0.005_marketfh')
# etf_sel_halfhourly = add_surprise_dummies(etf_sel_halfhourly, event_dict, df_events, event_list, '0.005_marketrod')
# etf_sel_halfhourly = add_surprise_dummies(etf_sel_halfhourly, event_dict, df_events, event_list, '0.001_marketfh')
# %% # Add surprise dummies to daily dict

# etf_sel_daily = add_surprise_dummies(
#     etf_sel_daily, event_dict, df_events, event_list, "absolute"
# )
etf_sel_daily = add_surprise_dummies(
    etf_sel_daily, event_dict, df_events, event_list, "1_stdev"
)
etf_sel_daily = add_surprise_dummies(
    etf_sel_daily, event_dict, df_events, event_list, "2_stdev"
)
etf_sel_daily = add_surprise_dummies(
    etf_sel_daily, event_dict, df_events, event_list, "marketfh"
)
# etf_sel_daily = add_surprise_dummies(
#     etf_sel_daily, event_dict, df_events, event_list, "0.005_marketfh"
# )
etf_sel_daily = add_surprise_dummies(
    etf_sel_daily, event_dict, df_events, event_list, "marketrod"
)
# etf_sel_daily = add_surprise_dummies(
#     etf_sel_daily, event_dict, df_events, event_list, "0.005_marketrod"
# )


# %% ## Add short ratio to the 'daily' dataframes

for key in etf_sel_daily.keys():
    etf_sel_daily[key] = add_daily_cols(
        etf_sel_daily[key], suffix_list, short_ratio, "Short", "Volume", "Short_Ratio"
    )

# %% Add lagged variables
# Add 1-interval return lags
for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key] = add_lag(etf_sel_halfhourly[key], "RETURN", 1)
# Add interval dummies to the halfhourly dataframe

intervals = etf_sel_halfhourly["AGG"]["TIME"].unique()

for key in etf_sel_halfhourly.keys():
    for interval in intervals:
        etf_sel_halfhourly[key][f"d_{interval}"] = (
            etf_sel_halfhourly[key]["TIME"] == interval
        ).astype(int)

# Add event-interval interaction dummies
for key in etf_sel_halfhourly.keys():
    for interval in intervals:
        for event in ["FOMC", "ISM"]:
            etf_sel_halfhourly[key][f"d_{event}_{interval}"] = (
                etf_sel_halfhourly[key][f"d_{interval}"]
                * etf_sel_halfhourly[key][event]
            )

# %% abn_short lag
for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key] = add_lag(etf_sel_halfhourly[key], "abn_short_scaled", 15)

# %% # Add interval-lagreturn interaction

for key in etf_sel_halfhourly.keys():
    for interval in intervals:
        etf_sel_halfhourly[key][f"i_lagReturn_{interval}"] = (
            etf_sel_halfhourly[key][f"d_{interval}"]
            * etf_sel_halfhourly[key]["RETURN_lag1"]
        )

# %% # Get cumulative returns column

for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key]["cum_ret_factor"] = etf_sel_halfhourly[key]["RETURN"] + 1
    etf_sel_halfhourly[key]["cum_ret"] = (
        etf_sel_halfhourly[key].groupby("DATE")["cum_ret_factor"].cumprod() - 1
    )
    etf_sel_halfhourly[key].drop(columns=["cum_ret_factor"], inplace=True)
    # Add lag
    etf_sel_halfhourly[key] = add_lag(etf_sel_halfhourly[key], "cum_ret", 1)


# %% Add realized variance


for key in etf_sel_halfhourly.keys():

    df = etf_sel_halfhourly[key].copy()
    df["sq_ret"] = df["RETURN"] ** 2
    date_list = df["DATE"].unique()

    for day in date_list:

        date_index = np.where(date_list == day)[0][0]
        if date_index == 0:
            df.loc[df["DATE"] == day, "prev_day_rv"] = 0
        else:
            prev_day = date_list[date_index - 1]
            sumsq_ret = df[df["DATE"] == prev_day]["sq_ret"].sum()
            df.loc[df["DATE"] == day, "prev_day_rv"] = np.sqrt(sumsq_ret)

    etf_sel_halfhourly[key] = df

# %% # Add average realized variance
for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key] = add_rolling_window_average_col(
        etf_sel_halfhourly[key], "prev_day_rv", 5, "DT"
    )
# %% ## Add scaled Volume and Short

vars_scale = [
    # "Volume",
    # "Short",
    # "Volume_dollar",
    # "Short_dollar",
    # "RETURN",
    # "cum_ret",
    # "cum_ret_lag1",
    # "PRICE/NAV",
    # "PRICE/NAV_lag",
    # "prev_day_rv",
    # "prev_day_rv_Average_5day",
    # "RETURN",
    "Short_Ratio"
]

for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key] = scale_vars_exp_window(
        etf_sel_halfhourly[key], vars_scale, StandardScaler(), 100
    )

# %% Add scaled NAV ratios
vars_scale = ["PRICE/NAV", "PRICE/NAV_lag14"]

for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key] = scale_vars_exp_window(
        etf_sel_halfhourly[key], vars_scale, StandardScaler(), 100
    )
# %% ## Do augmented Dickey-Fuller

for key in etf_sel_halfhourly.keys():

    X = etf_sel_halfhourly[key]["vol_index"].values
    result = adfuller(X)
    print(f"ADF Statistic {key}: %f" % result[0])
    print(f"p-value {key}: %f" % result[1])

# %% ## Get event counts


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
    "ISM_lag1",
    "FOMC_lag1",
    "NFP_lag1",
    "CPI_lag1",
    "GDP_lag1",
    "IP_lag1",
    "PI_lag1",
    "HST_lag1",
    "PPI_lag1",
    "EVENT_lag1",
]

for abbr in abbr_list:
    event_count = (
        etf_sel_halfhourly["AGG"].groupby(etf_sel_halfhourly["AGG"]["DATE"])[abbr].max()
    )
    num_events = (event_count > 0).sum()
    print(f"{abbr}: {num_events}")

# %% Add percentage price nav ratio
for key in etf_sel_halfhourly.keys():
    df = etf_sel_halfhourly[key].copy()

    price_16_df = df[df["TIME"] == "16:00:00"][["DATE", "PRICE"]]

    merged = pd.merge(df, price_16_df, on="DATE", suffixes=("", "_16"))

    merged["PRICE/NAV_pct"] = (
        (merged["PRICE_16"] - merged["NAV"]) / merged["NAV"]
    ) * 100
    merged = merged.drop(columns=["PRICE_16"])
    etf_sel_halfhourly[key] = merged
# Add lags
for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key] = add_lag(etf_sel_halfhourly[key], "PRICE/NAV_pct", 14)
# %% # Get barplots for average intraday short volume for event- and non-event days

# In the sample, no two selected events happen on the same day, so a zero in any of the event columns coincides with a zero in 'EVENT'

ticker = "HYG"  # Choose from "AGG", "HYG", "IEF", "LQD", "SPY", "SHY", "TLT"
metric = "abn_short_scaled"  # Choose from "Short", "Short_dollar", "Volume", "Volume_dollar", "Short_Ratio", "RETURN"
event = "FOMC"  # Choose from "ISM", "FOMC", "NFP", "CPI", "GDP", "IP", "PI", "HST", "PPI", "EVENT" or any of the lags, e.g. ISM_lag
start_date = "2014-01-01"
end_date = "2022-12-31"
non_event_def = True  # Set to True if non-event is defined as no events at all, set to False if non-event is defined as no other event of that specific event (so other events are counted as non-event)
lag_bar = False  # Set to True if you want to show a bar with the lagged metric, does not work in combination with surprise_split
lag_num = 1  # Specify number of lags, make sure that the columns with these lag dummies are actually added to the dataframes.
surprise_split = True  # Set to True if the plot should show the negative and positive surprises separately, surprise type is defined in surprise_col
surprise_col = "surprise_2_stdev"  # Choose from options which are added above, e.g. 'absolute', '1_stdev' etc. for measures based on
# The analyst surprise, and 'marketfh', '0.001_marketfh', 'marketrod' etc. for market based surprise measure
intraday_barplot(
    etf_sel_halfhourly,
    ticker,
    metric,
    start_date,
    end_date,
    event,
    non_event_def=non_event_def,
    lag_bar=lag_bar,
    lag_num=lag_num,
    surprise_split=surprise_split,
    surprise_col=surprise_col,
)

plt.show()


# %% ## Create grid of plots and save to computer

start_date = "2014-01-01"
end_date = "2022-12-31"
non_event_def = True  # Set to True if non-event is defined as no events at all, set to False if non-event is defined as no other event of that specific event (so other events are counted as non-event)
lag_bar = False  # Set to True if you want to show a bar with the lagged metric, does not work in combination with surprise_split
lag_num = 1  # Specify number of lags, make sure that the columns with these lag dummies are actually added to the dataframes.
surprise_split = True  # Set to True if the plot should show the negative and positive surprises separately, surprise type is defined in surprise_col
surprise_col = "surprise_2_stdev"  # Choose from options which are added above, e.g. 'absolute', '1_stdev' etc. for measures based on
# The analyst surprise, and 'marketfh', '0.001_marketfh', 'marketrod' etc. for market based surprise measure


preset = "Treasuries"  # Choose from 'Treasuries', 'SPY_FI', and 'Credits'

if preset == "Treasuries":
    ticker_list = ["SHY", "IEF", "TLT"]
elif preset == "SPY_FI":
    ticker_list = ["SPY", "AGG"]
elif preset == "Credits":
    ticker_list = ["HYG", "LQD"]

metric_list = ["Short", "abn_short_scaled", "RETURN"]
event_list = ["ISM"]
plot_title = f"{preset}_{event_list[0]}_abnormal_2stdev"
image = create_grid_barplots(
    plot_title,
    etf_sel_halfhourly,
    ticker_list,
    metric_list,
    event_list,
    start_date,
    end_date,
    non_event_def,
    lag_bar,
    lag_num,
    surprise_split,
    surprise_col,
)


# %%  ## Get 5, 10 and 20 and 100 trading day rolling-windows of short volume


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


# %% ## Get deviations from averages


def add_abn_col(df_in, var, window_size, new_col_name):

    df = df_in.copy()

    df[f"{new_col_name}_{var}_{window_size}day"] = (
        df[var] - df[f"{var}_Average_{window_size}day"]
    )

    return df


var = "Short"
new_col_name = "Abn"


for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key] = add_abn_col(etf_sel_halfhourly[key], var, 5, new_col_name)
    etf_sel_halfhourly[key] = add_abn_col(
        etf_sel_halfhourly[key], var, 10, new_col_name
    )
    etf_sel_halfhourly[key] = add_abn_col(
        etf_sel_halfhourly[key], var, 20, new_col_name
    )
    etf_sel_halfhourly[key] = add_abn_col(
        etf_sel_halfhourly[key], var, 100, new_col_name
    )

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

# %% # Plot volume for different surprise types over time (evaluate development over time)
df = etf_sel_halfhourly["AGG"].copy()

pos_surprise_dates = df[df["0.25percentile_pos_dummy"] == 1]["DATE"]
neg_surprise_dates = df[df["0.25percentile_neg_dummy"] == -1]["DATE"]
# neu_surprise_dates = df[df["FOMC_surprise_0.001_marketfh"] == 0]["DATE"].unique()
df["DATE"] = pd.to_datetime(df["DATE"])

pos_dates = [datetime.strptime(date, "%Y-%m-%d") for date in pos_surprise_dates]
neg_dates = [datetime.strptime(date, "%Y-%m-%d") for date in neg_surprise_dates]
# neu_dates = [datetime.strptime(date, "%Y-%m-%d") for date in neu_surprise_dates]


def total_volume_in_month(date):
    month_start = date.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    df_month = df[(df["DATE"] >= month_start) & (df["DATE"] <= month_end)]
    return df_month["Short"].sum()


start_date = datetime(2014, 1, 1)
end_date = datetime(2022, 12, 31)
date_range = [
    start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)
]
# Calculate total volume in the month for each specific date
total_volume_pos = [total_volume_in_month(date) for date in pos_dates]
total_volume_neg = [total_volume_in_month(date) for date in neg_dates]
total_volume_neu = [total_volume_in_month(date) for date in neu_dates]


# Plot
plt.figure(figsize=(10, 5))
plt.plot(pos_dates, total_volume_pos, marker="o", linestyle="", color="green")
plt.plot(neg_dates, total_volume_neg, marker="o", linestyle="", color="red")
plt.plot(neu_dates, total_volume_neu, marker="o", linestyle="", color="orange")
plt.title(
    "Total monthly volume on positive (green), negative (red) and neutral (orange) announcement days"
)
plt.xlabel("Date")
plt.yticks([])  # Hide y-axis
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.xlim(start_date, end_date)  # Set x-axis limits
plt.grid(True)
plt.tight_layout()
plt.show()


# %% ## Make time series plot for specific period and variables

ticker = "AGG"
df = etf_sel_halfhourly[ticker]
start_date = "2014-01-01"
end_date = "2022-12-31"
title = ticker
y_1 = "abn_short_scaled"
y_2 = "PRICE"
vert_line = "2015-01-16 13:30:00"


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
    # vert_line,
)
test_fig.show()

# %%
for key in etf_sel_halfhourly.keys():
    mean = etf_sel_halfhourly[key]["abn_short_scaled"].mean()

    stdev = etf_sel_halfhourly[key]["abn_short_scaled"].std()
    min = etf_sel_halfhourly[key]["abn_short_scaled"].min()
    max = etf_sel_halfhourly[key]["abn_short_scaled"].max()
    print(
        f"Stats for {key}:\n"
        f"            mean: {mean}\n"
        f"          stdev: {stdev}\n"
        f"          min: {min}\n"
        f"          max: {max}"
    )
# %% ## Code to get many plots, either print and save or display them


## Specify parameters for the plotting function
ticker = "LQD"


results = get_extreme_values(
    etf_sel_halfhourly[ticker], "abn_short_scaled", 0.02, "highest"
)
event_date_list = [timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in results[0]]
# List of event dates to be evaluated
event_dt_list = event_date_list

col_list = [
    "abn_short_scaled"
]  # y1 variables to be included in the plots, if you input multiple variables, a plot will be created for each of them
y_2 = "PRICE"  # y2 variable to be included in the plots, this is just a string input, so it does not support multiple inputs as the y1
etf_list = [
    ticker
]  # List of ETFs to be plotted, a plot will be created for each of the ETFs
day_range = 5  # Range of days around the event date
display_dummy = False  # Dummy, if True the plots will be plotted, if False, the plots will be saved to a specified directory
parent_dir = rf"C:/Users/ROB7831/OneDrive - Robeco Nederland B.V/Documents/Thesis/Plots/Events/{ticker}"  # Specify parent directory (only necessary if display_dummy = False)


## Get plots of the selected etfs for pre-defined day range and for selected variables in col_list
get_eventday_plots(
    etf_dict,
    etf_sel_halfhourly,
    etf_list,
    event_dt_list,
    col_list,
    y_2,
    day_range,
    display_dummy,
    parent_dir,
)

# %% ## Get years and weekdays column in dataframes


for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key]["YEAR"] = etf_sel_halfhourly[key]["DT"].dt.year
    etf_sel_halfhourly[key]["Weekday"] = etf_sel_halfhourly[key]["DT"].dt.day_name()
# %% ## Get plots per year

grouped = (
    etf_sel_halfhourly["AGG"]
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

# %% ## Get average volume per weekday
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


# %% ## Add new column of average short_ratio


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



# %% #
for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key]["RETURN_100"] = etf_sel_halfhourly[key]["RETURN"] * 100

#%%
etf_sel_halfhourly_dict_stor = etf_sel_halfhourly.copy()
# %% Do regressions

# Specify regression parameters


# dep_vars = ["Short_Ratio_scaled_exp"]
dep_vars = [
    'future_ret_1halfhours',
    'future_ret_1days',
    'future_ret_1days_EOD',
            ]
# dep_vars = [
#     "future_ret_1halfhours",
#     "future_ret_2halfhours",
#     "future_ret_0days_EOD",
#     "future_ret_1halfhours_EOD",
#     "future_ret_1days",

#     "future_ret_1days_EOD",
#     "future_ret_3days_EOD",
#     "future_ret_5days_EOD",
#     "future_ret_10days_EOD",
#     "future_ret_20days_EOD",
# ]

# indep_vars = [
#     "Short_Ratio_scaled_exp",
# #     "Short_scaled_exp",
# #     "Volume_scaled_exp",
# #     "cum_ret_scaled_exp",
# #     "PRICE/NAV_pct_lag14",
# #     "prev_day_rv_Average_5day_scaled_exp",
# #     "vol_index",
# ]
indep_vars = [
    # "1percentile_pos_dummy_abn_short",
    # "1percentile_neg_dummy_abn_short",
    '0.5percentile_pos_dummy',
    '0.5percentile_neg_dummy',
    # '0.25percentile_pos_dummy_abn_short',
    # '0.25percentile_neg_dummy_abn_short',
    # '0.1percentile_pos_dummy_abn_short',
    # '0.1percentile_neg_dummy_abn_short',
    # '2.5percentile_pos_dummy_abn_short',
    # '2.5percentile_neg_dummy_abn_short',
    # '0.05percentile_pos_dummy_abn_short',
    # '0.05percentile_neg_dummy_abn_short',
    # "Volume_scaled_exp",
    # "cum_ret_scaled_exp",
    # "PRICE/NAV_pct_lag14",
#     "prev_day_rv_Average_5day_scaled_exp",
    # "vol_index",
]

# for interval in intervals:
#     indep_vars.append(f"d_{interval}")

# for interval in intervals:
#     for event in ['ISM', 'FOMC']:
#         indep_vars.append(f"d_{event}_{interval}")

# for interval in intervals:
#     indep_vars.append(f"i_lagReturn_{interval}")

# Specify error type
cov_type = "HC1"
ticker = None
start_date = '2014-01-01'
end_date = '2015-12-31'

for key in etf_sel_halfhourly.keys():
    df = etf_sel_halfhourly_dict_stor[key].copy()
    df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    etf_sel_halfhourly[key] = df



# Perform regressions
regression_result_dict = {}
if ticker is None:

    for key in etf_sel_halfhourly.keys():
        regression_result_dict[key] = do_regression(
            etf_sel_halfhourly[key], indep_vars, dep_vars, cov_type
        )
else:
    for key in ticker:
        regression_result_dict[key] = do_regression(
            etf_sel_halfhourly[key], indep_vars, dep_vars, cov_type
        )

# Show results for regression results. Ticker is None prints all results



show_reg_results(regression_result_dict, ticker)

#%%

latex_table = get_latex_table_stacked(regression_result_dict, dep_vars, indep_vars, True)
# latex_table = get_latex_table(regression_result_dict, dep_vars, indep_vars)
print(latex_table)
# %%
for key in etf_sel_halfhourly.keys():
    df = etf_sel_halfhourly[key].copy()
    resid = regression_result_dict[key]["Short_Ratio_scaled_exp"].resid.copy()
    resid.reindex(df.index, fill_value=np.nan)

    df["abn_short_ratio"] = resid

    etf_sel_halfhourly[key] = df

# %%
# Get latex table
latex_table, result_df = get_latex_table(regression_result_dict, dep_vars, indep_vars)

print(latex_table)

#%%
dummy_list = [
    "1percentile_pos_dummy",
    "1percentile_neg_dummy",
    '0.5percentile_pos_dummy',
    '0.5percentile_neg_dummy',
    '0.25percentile_pos_dummy',
    '0.25percentile_neg_dummy',
    '0.1percentile_pos_dummy',
    '0.1percentile_neg_dummy',
    '2.5percentile_pos_dummy',
    '2.5percentile_neg_dummy',
    '0.05percentile_pos_dummy',
    '0.05percentile_neg_dummy',
]

for key in etf_sel_halfhourly.keys():
    for dummy in dummy_list:
        etf_sel_halfhourly[key][f'{dummy}_abn_short'] = etf_sel_halfhourly[key][dummy] * etf_sel_halfhourly[key]['abn_short_scaled']

# %%
# threshold = 10
threshold_dict = {
    "AGG": 1.25,
    "HYG": 3,
    "IEF": 3,
    "LQD": 4,
    "SPY": 1.25,
    "SHY": 3,
    "TLT": 2.5,
}

abn_date_dict = {}
for key in etf_sel_halfhourly.keys():
    df = etf_sel_halfhourly[key].copy()
    filter_df = df[abs(df["abn_short_scaled"]) > threshold_dict[key]]

    count = filter_df.shape[0]
    indices = filter_df.index.tolist()
    dates = df.loc[indices, "DT"]
    abn_date_dict[key] = dates
    print(f"{key}: # times above threshold {threshold_dict[key]}: {count}")


# %% Load in MOVE and VIX data

move_df = pd.read_csv(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Other data\MOVE_data.csv"
)
vix_df = pd.read_excel(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Other data\VIX_data.xlsx"
)

# Drop irrelevant rows
vix_df = vix_df.drop(vix_df.index[:10]).reset_index(drop=True)

# Drop irrelevant columns
move_df = move_df.drop(columns=["Open", "High", "Low", "Close", "Volume"])

# Drop NaNs
vix_df = vix_df.dropna()
# Rename columns
vix_df = vix_df.rename(
    columns={"FRED Graph Observations": "DATE", "Unnamed: 1": "vol_index"}
)
move_df = move_df.rename(columns={"Date": "DATE", "Adj Close": "vol_index"})

# # Get dates in proper format
move_df["DATE"] = pd.to_datetime(move_df["DATE"], format="%d/%m/%Y")
move_df["DATE"] = move_df["DATE"].dt.strftime("%Y-%m-%d")
vix_df["DATE"] = pd.to_datetime(vix_df["DATE"], format="%d/%m/%Y")
vix_df["DATE"] = vix_df["DATE"].dt.strftime("%Y-%m-%d")

# %% Merge move and VIX data into general dataframe

for key in etf_sel_halfhourly.keys():
    df = etf_sel_halfhourly[key].copy()
    df_move = move_df.copy()
    df_vix = vix_df.copy()

    if key in ["SPY", "HYG"]:
        merged = pd.merge(df, df_vix, on="DATE", suffixes=("", "vol_index"))
    else:
        merged = pd.merge(df, df_move, on="DATE", suffixes=("", "vol_index"))

    merged["vol_index"] = merged["vol_index"].fillna(method="ffill")
    etf_sel_halfhourly[key] = merged


# %% Save to pickle files

with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_halfhourly_20240515.pkl",
    "wb",
) as f:
    pickle.dump(etf_sel_halfhourly, f)

with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_daily_20240515.pkl",
    "wb",
) as f:
    pickle.dump(etf_sel_daily, f)
# %% Add future cumulative returns
for key in etf_sel_halfhourly.keys():
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 0, 'days', True)
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 1, 'days', True)
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 3, 'days', True)
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 5, 'days', True)
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 1, 'halfhours', True)
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 0, 'days')
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 1, 'days')
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 3, 'days')
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 5, 'days')
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 1, 'halfhours')
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 2, 'halfhours')
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 1, 'halfhours', True, 1)
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 1, 'halfhours', False, 1)
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 2, 'halfhours', False, 1)
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 10, "days", True)
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 20, "days", True)
    # etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 0, "days", True, 1)
    etf_sel_halfhourly[key] = add_future_ret(etf_sel_halfhourly[key], 2, "halfhours", True, 1)

# %%


# %% Get dummies for large spikes in abnormal short volume

threshold_dict = {
    "AGG": 4,
    "HYG": 3,
    "IEF": 3,
    "LQD": 4,
    "SPY": 1.25,
    "SHY": 3,
    "TLT": 2.5,
}

for key in etf_sel_halfhourly.keys():
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 1, 'abn_short_scaled', True, 'abs')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 1, 'abn_short_scaled', True, 'pos')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 1, 'abn_short_scaled', True, 'neg')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 0.5, 'abn_short_scaled', True, 'abs')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 0.5, 'abn_short_scaled', True, 'pos')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 0.5, 'abn_short_scaled', True, 'neg')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 0.1, 'abn_short_scaled', True, 'abs')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 0.1, 'abn_short_scaled', True, 'pos')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 0.1, 'abn_short_scaled', True, 'neg')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 0.25, 'abn_short_scaled', True, 'abs')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 0.25, 'abn_short_scaled', True, 'pos')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 0.25, 'abn_short_scaled', True, 'neg')
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 4, "abn_short_scaled", False, "neg"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 4, "abn_short_scaled", False, "pos"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 4, "abn_short_scaled", False, "abs"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 3, "abn_short_scaled", False, "neg"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 3, "abn_short_scaled", False, "pos"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 3, "abn_short_scaled", False, "abs"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 2.5, "abn_short_scaled", False, "neg"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 2.5, "abn_short_scaled", False, "pos"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 2.5, "abn_short_scaled", False, "abs"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 1.25, "abn_short_scaled", False, "neg"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 1.25, "abn_short_scaled", False, "pos"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 1.25, "abn_short_scaled", False, "abs"
    )
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 0.05, 'abn_short_scaled', True, 'abs')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 0.05, 'abn_short_scaled', True, 'pos')
    # etf_sel_halfhourly[key] = create_peak_dummies(etf_sel_halfhourly[key], 0.05, 'abn_short_scaled', True, 'neg')
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 2.5, "abn_short_scaled", True, "abs"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 2.5, "abn_short_scaled", True, "pos"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 2.5, "abn_short_scaled", True, "neg"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 5, "abn_short_scaled", True, "abs"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 5, "abn_short_scaled", True, "pos"
    )
    etf_sel_halfhourly[key] = create_peak_dummies(
        etf_sel_halfhourly[key], 5, "abn_short_scaled", True, "neg"
    )


# %%
for key in etf_sel_halfhourly.keys():
    plt.figure()
    plt.hist(etf_sel_halfhourly[key]["abn_short_scaled"], bins=100)
    plt.title(f"Histogram of abnormal short volume - {key}")
    plt.show



#%%

percentile = 0.5
for key in etf_sel_halfhourly.keys():
    df = etf_sel_halfhourly[key].copy()        
    plt.figure(figsize=(10, 6))

    plt.plot(df['DT'], df[f'{percentile}percentile_pos_dummy_abn_short'], label=f'{percentile}_percentile', marker='o', color='green')
    plt.plot(df['DT'], df[f'{percentile}percentile_neg_dummy_abn_short'], label=f'{percentile}_percentile', marker='o', color='red')
    plt.xlabel('Date')
    plt.ylabel('Dummy x abnormal short')
    plt.title(f'{key} - Spikes Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


# %%

for key in etf_sel_halfhourly.keys():
    df = etf_sel_halfhourly[key].copy()
    for year in df['YEAR'].unique():
        num = df[df['YEAR'] == year]['Short_Ratio'].mean()
        print(f'{key}: {year}: {num}')
# %% Plot average short ratio over time

plt.figure(figsize=(10, 6))

for ticker, df in etf_sel_halfhourly.items():

    yearly_avg = df.groupby('YEAR')['Short_Ratio'].mean()
    plt.plot(yearly_avg.index, yearly_avg.values, marker='o', label=ticker)

plt.xlabel('Year')
plt.ylabel('Average Short Ratio')
plt.title('Average Short Ratio per year for the considered ETFs')
plt.legend()

plt.show()

#%% Calculate hit rate of very short term strategies

ret_col = 'future_ret_1halfhours'
for ticker, df in etf_sel_halfhourly.items():

    df_dummy = df[df['5percentile_pos_dummy'] == 1]
    total_num = df.shape[0]
    total_num_dummy = df_dummy.shape[0]
    pos_returns = df[df[ret_col] > 0].shape[0]
    pos_returns_dummy = df_dummy[df_dummy[ret_col] > 0].shape[0]

    hit_rate = pos_returns / total_num if total_num > 0 else 0
    hit_rate_dummy = pos_returns_dummy / total_num_dummy if total_num_dummy > 0 else 0
    average_ret = df[ret_col].mean()
    average_ret_dummy = df_dummy[ret_col].mean()

    print(f'Hit rate {ret_col} {ticker}: {hit_rate_dummy} (n={total_num_dummy}) total hit rate: {hit_rate}, diff: {hit_rate_dummy - hit_rate}')
    # print(f'{ticker} average return: {average_ret_dummy}, average tot return: {average_ret}, diff: {average_ret_dummy-average_ret}')
# %% Look at some aggregate daily stuff

var = "Volume"

intervals = [
                f"{var}_9first",
                f"{var}_FH",
                f"{var}_10first",
                f"{var}_10second",
                f"{var}_11first",
                f"{var}_11second",
                f"{var}_12first",
                f"{var}_12second",
                f"{var}_13first",
                f"{var}_13second",
                f"{var}_14first",
                f"{var}_14second",
                f"{var}_SLH",
                f"{var}_LH",
]

for key in etf_sel_daily.keys():
    etf_sel_daily[key][f"{var}_total"] = etf_sel_daily[key][intervals].sum(axis=1)
#%%
for key in etf_sel_daily.keys():
    etf_sel_daily[key]['Short_Ratio_total'] =  etf_sel_daily[key]['Short_total'] /  etf_sel_daily[key]['Volume_total']
# %%
for key in etf_sel_daily.keys():
    mean = etf_sel_daily[key]['Short_Ratio_total'].mean()
    std = etf_sel_daily[key]['Short_Ratio_total'].std()
    minimum =  etf_sel_daily[key]['Short_Ratio_total'].min()
    maximum =  etf_sel_daily[key]['Short_Ratio_total'].max()
    print(f'{key}:')
    print(f'Mean: {mean}')
    print(f'Stdev: {std}')
    print(f'Minimum: {min}')
    print(f'Maximum: {max}')
# %% Calculate cumulative daily returns
var = "Return"

intervals = [
                f"{var}_09first",
                f"{var}_FH",
                f"{var}_10first",
                f"{var}_10second",
                f"{var}_11first",
                f"{var}_11second",
                f"{var}_12first",
                f"{var}_12second",
                f"{var}_13first",
                f"{var}_13second",
                f"{var}_14first",
                f"{var}_14second",
                f"{var}_SLH",
                f"{var}_LH",
]

for key in etf_sel_daily.keys():
    etf_sel_daily[key][f"{var}_total"] = etf_sel_daily[key][intervals].apply(lambda row: np.prod(1 + row) - 1, axis=1)

#%% Add future returns

lead = 20
for key in etf_sel_daily.keys():
    etf_sel_daily[key][f'Return_total_lead{lead}'] = etf_sel_daily[key]['Return_total'].shift(-lead)


# %% Do regressions

# Specify regression parameters


# dep_vars = ["Short_Ratio_scaled_exp"]
dep_vars = [
    "Cum_Ret_20days"
            ]
# dep_vars = [
#     "future_ret_1halfhours",
#     "future_ret_2halfhours",
#     "future_ret_0days_EOD",
#     "future_ret_1halfhours_EOD",
#     "future_ret_1days",

#     "future_ret_1days_EOD",
#     "future_ret_3days_EOD",
#     "future_ret_5days_EOD",
#     "future_ret_10days_EOD",
#     "future_ret_20days_EOD",
# ]


indep_vars =  [
  "20day_diff_short_ratio"
]



# Specify error type
cov_type = "HC1"
ticker = None
start_date = '2014-01-01'
end_date = '2022-12-31'

# for key in etf_sel_daily.keys():
#     df = etf_sel_daily_dict_stor[key].copy()
#     df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
#     etf_sel_daily[key] = df



# Perform regressions
regression_result_dict = {}
if ticker is None:

    for key in etf_sel_daily.keys():
        regression_result_dict[key] = do_regression(
            etf_sel_daily[key], indep_vars, dep_vars, cov_type
        )
else:
    for key in ticker:
        regression_result_dict[key] = do_regression(
            etf_sel_daily[key], indep_vars, dep_vars, cov_type
        )

# Show results for regression results. Ticker is None prints all results



show_reg_results(regression_result_dict, ticker)

# %%

num_days = 3

for key in etf_sel_daily.keys():

    df = etf_sel_daily[key].copy()
    

    cumulative_returns = []
    for i in range(len(df)):
        cum_ret = 1
        z = i + num_days + 1
        y = len(df)
        end_index = min(y, z)
        for j in range(i+1, end_index):
            cum_ret *= df.loc[j, 'Return_total'] + 1
        cum_ret -= 1
        cumulative_returns.append(cum_ret)


    etf_sel_daily[key][f'Cum_Ret_{num_days}days'] = cumulative_returns
# %% Get rolling window averages for short_ratio over past 1 week and past 1 month

for key in etf_sel_daily.keys():
    df = etf_sel_daily[key].copy()

    df.set_index('DATE', inplace=True)

    df['5day_average_short_ratio'] = df['Short_Ratio_total'].rolling(window=5).mean()
    df['20day_average_short_ratio'] = df['Short_Ratio_total'].rolling(window=20).mean()

    df['5day_diff_short_ratio'] = df['Short_Ratio_total'] - df['5day_average_short_ratio']
    df['20day_diff_short_ratio'] = df['Short_Ratio_total'] - df['20day_average_short_ratio']

    df.reset_index(inplace=True)

    etf_sel_daily[key]['5day_average_short_ratio'] = df['5day_average_short_ratio']
    etf_sel_daily[key]['20day_average_short_ratio'] = df['20day_average_short_ratio']
    etf_sel_daily[key]['5day_diff_short_ratio'] = df['5day_diff_short_ratio']
    etf_sel_daily[key]['20day_diff_short_ratio'] = df['20day_diff_short_ratio']


    

# %%
