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

#%%
# Data per 27-05-2024, including buy-sell imbalance data
with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_halfhourly_20240529.pkl",
    "rb",
) as f:
    etf_sel_halfhourly = pickle.load(f)

with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_daily_20240529.pkl",
    "rb",
) as f:
    etf_sel_daily = pickle.load(f)


#%% Supress warnings
warnings.resetwarnings()
warnings.simplefilter("ignore")

# Define some dictionaries for mapping
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
etf_sel_halfhourly_dict_stor = etf_sel_halfhourly.copy()

# %% Do regressions

# Specify dependent variable, if multiple are added it will do a univariate regression for each of them
dep_vars = [
    # 'Short'
    # 'Short_scaled_rolling_window_interval_250days'
    #  'future_ret_0days_EOD',
#  'future_ret_1days_EOD',
#  'future_ret_3days_EOD',
#  'future_ret_5days_EOD',
#  'future_ret_1halfhours_EOD',
#  'future_ret_0days',
 'future_ret_1days',
#  'future_ret_3days',
#  'future_ret_5days',
#  'future_ret_1halfhours',
#  'future_ret_2halfhours',
#  'future_ret_1halfhours_EOD_lead1',
#  'future_ret_1halfhours_lead1',
#  'future_ret_2halfhours_lead1',
#  'future_ret_10days_EOD',
#  'future_ret_20days_EOD',


]

indep_vars = [
    # 'abn_short_absolute',
    'abn_short_absolute_scaled_rolling_window_interval_250days',
    # 'Volume',
    # 'Volume_scaled_rolling_window_interval_250days',
    'Buy_Sell_Imb_lag1',
    'PRICE/NAV_pct_lag14',
    'vol_index_lag14',
    # 'cum_ret',
    # 'prev_day_rv_Average_5day'
]

# for interval in intervals:
#     indep_vars.append(f'd_{interval}')

# Specify error type
cov_type = "HC1"
ticker = None
start_date = "2014-01-01"
end_date = "2022-12-31"

# # Get dataframe for correct timeframe
# for key in etf_sel_halfhourly.keys():
#     df = etf_sel_halfhourly_dict_stor[key].copy()
#     df = df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]
#     etf_sel_halfhourly[key] = df



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

# %%
# Create latex table out of stored regression results obtained by regression above
latex_table = get_latex_table_stacked(
    regression_result_dict, dep_vars, indep_vars, True
)

print(latex_table)

#%%
# Create a measure of abnormal short volume or other measure

# Define of which dependent variable the residual should be taken
var = ""

# Define the name of the abnormal volume measure column
abn_col_name = ""

# Add abnormal short measure to the dataframes
for key in etf_sel_halfhourly.keys():
    df = etf_sel_halfhourly[key].copy()
    resid = regression_result_dict[key][var].resid.copy()
    resid.reindex(df.index, fill_value=np.nan)

    df[abn_col_name] = resid

    etf_sel_halfhourly[key] = df


# %%
dep_vars = [
    # 'Short'
    # 'Short_scaled_rolling_window_interval_250days'
    #  'future_ret_0days_EOD',
#  'future_ret_1days_EOD',
#  'future_ret_3days_EOD',
#  'future_ret_5days_EOD',
 'future_ret_1halfhours_EOD',
#  'future_ret_0days',
#  'future_ret_1days',
#  'future_ret_3days',
#  'future_ret_5days',
#  'future_ret_1halfhours',
#  'future_ret_2halfhours',
#  'future_ret_1halfhours_EOD_lead1',
#  'future_ret_1halfhours_lead1',
#  'future_ret_2halfhours_lead1',
#  'future_ret_10days_EOD',
#  'future_ret_20days_EOD',
]

indep_vars = [
    # 'abn_short_absolute',
    'abn_short_absolute_scaled_rolling_window_interval_250days',
    # 'Volume',
    # 'Volume_scaled_rolling_window_interval_250days',
    'Buy_Sell_Imb_lag1',
    'PRICE/NAV_pct_lag14',
    'vol_index_lag14',
    # 'cum_ret',
    # 'prev_day_rv_Average_5day'
]

# for interval in intervals:
#     indep_vars.append(f'd_{interval}')





entity_effects = True
time_effects = False
cov_type = 'robust' # can be of the following types
# 'unadjusted' - assume homoskedasticity
# 'robust' - use White's estimator for heteroskedasticity robustness
# 'clustered' - one- or two-way clustering, takes cluster_entity and cluster_time (or clusters, but needs specification)
# 'kernel' - Driscoll-Kraay HAC estimator , takes kernel , default is Bartletts kernel



result = do_panel_regression(etf_sel_halfhourly, dep_vars=dep_vars, indep_vars=indep_vars, entity_effects=entity_effects, time_effects=time_effects, cov_type=cov_type)
result_df = result.resids

result.summary
#%%

new_col_name = "abn_short_absolute"
for key in etf_sel_halfhourly:
    filtered_df = result_df.loc[key].copy()

    filtered_df = filtered_df.reset_index(level=0)
    # filtered_df = filtered_df.reset_index()

    etf_sel_halfhourly[key] = pd.merge(etf_sel_halfhourly[key], filtered_df, on='DT', suffixes=("", "abn_short"))
    etf_sel_halfhourly[key] = etf_sel_halfhourly[key].rename(columns={"residual" : new_col_name})


# %%



vars_scale = ["abn_short_absolute"]

# Scale only over intervals, moving window of 1 year (250 trading days)
for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key] = scale_vars_exp_window(
        etf_sel_halfhourly[key],
        vars_scale,
        StandardScaler(),
        60,
        method="<=t",
        interval_col="TIME",
        inplace=False,
        rolling=True,
    )

for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key] = scale_vars_exp_window(
        etf_sel_halfhourly[key],
        vars_scale,
        StandardScaler(),
        20,
        method="<=t",
        interval_col="TIME",
        inplace=False,
        rolling=True,
    )

# %% Save to pickle files

with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_halfhourly_20240529.pkl",
    "wb",
) as f:
    pickle.dump(etf_sel_halfhourly, f)

with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_daily_20240529.pkl",
    "wb",
) as f:
    pickle.dump(etf_sel_daily, f)
# %%
