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
#%%
# Load most recent data
with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_halfhourly_20240605.pkl",
    "rb",
) as f:
    etf_sel_halfhourly = pickle.load(f)


# Open CFTC data
df_cftc = pd.read_excel(r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Other data\cftc_data.xlsx")


# #%%
# with open(r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_halfhourly_20240605.pkl", "wb") as f:
#     pickle.dump(etf_sel_halfhourly, f)

# with open(r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_daily_20240605.pkl", "wb") as f:
#     pickle.dump(etf_sel_daily, f)
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
code_dictionary = {
    "020601": "U.S. Treasury Bonds",
    "020604": "Long-term U.S. Treasury Bonds",
    "042601": "2-Year U.S. Treasury Notes",
    "043602": "10-Year U.S. Treasury Notes",
    "044601": "5-Year U.S. Treasury Notes",
    "13874+": "S&P 500 Consolidated",
    "138741": "S&P 500 Stock Index",
    "13874A": "S&P 500 Mini",
    "043607": "Ultra 10-year U.S. Treasury Note",
}
intervals = etf_sel_halfhourly["AGG"]["TIME"].unique()
etf_sel_halfhourly_dict_stor = etf_sel_halfhourly.copy()

# %% Do regressions

# Specify dependent variable, if multiple are added it will do a univariate regression for each of them
dep_vars = [
    # 'Short'
    # 'Short_scaled_rolling_window_interval_250days'
    'Short_Ratio',
    # 'abn_short_ratio_absolute',
    #  'future_ret_0days_EOD',
#  'future_ret_1days_EOD',
#  'future_ret_3days_EOD',
#  'future_ret_5days_EOD',
#  'future_ret_1halfhours_EOD',
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
    # 'abn_short_absolute_scaled_rolling_window_interval_250days',
    # 'Volume',
    # 'Short_lag1',
    # 'Volume_lag1',
    # 'Short_lag2',
    # 'Volume_lag2',

    'Short_Ratio_lag1',
    'Short_Ratio_lag2',

    # 'Short_Ratio',
    # 'abn_short_ratio',
    # 'abn_short',
    # 'abn_short_1',
    # 'abn_short_3',
    # 'Volume_scaled_rolling_window_interval_250days',
    'Buy_Sell_Imb_lag1',
    'PRICE/NAV_pct_lag14',
    'delta_NAV_PRICE_lag14',
    'vol_index_lag14',
    # 'cum_ret',
    'prev_day_rv_Average_5day',
    #  'abn_short_scaled_rolling_window_interval_250days',
    # 'abn_short_1_scaled_rolling_window_interval_250days',
    # 'abn_short_2_scaled_rolling_window_interval_250days',
    # 'abn_short_scaled_rolling_window_interval_20days',
    # 'abn_short_1_scaled_rolling_window_interval_20days',
    # 'abn_short_2_scaled_rolling_window_interval_20days',
    # 'vol_index_lag14_scaled_rolling_window_interval_250days',
    # 'prev_day_rv_Average_5day_scaled_rolling_window_interval_250days',
    # 'Buy_Sell_Imb_lag1_scaled_rolling_window_interval_250days',
    # 'PRICE/NAV_pct_lag14_scaled_rolling_window_interval_250days'
]

for interval in intervals:
    indep_vars.append(f'd_{interval}')

    


# Specify error type
cov_type = "HC1"
ticker = None
start_date = "2014-01-01"
end_date = "2022-12-31"
# time = '15:30:00'

# indep_vars.append(f'd_{time}')

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
# [etf_sel_halfhourly[key]['TIME'] == time]

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
var = "Short_Ratio"

# Define the name of the abnormal volume measure column
abn_col_name = "abn_short_ratio"

# Add abnormal short measure to the dataframes
for key in etf_sel_halfhourly.keys():
    df = etf_sel_halfhourly[key].copy()
    resid = regression_result_dict[key][var].resid.copy()
    resid.reindex(df.index, fill_value=np.nan)

    df[abn_col_name] = resid

    etf_sel_halfhourly[key] = df


# %%
# Do panel regression with different ETFs in different panels
dep_vars = [
    # 'Short'
    # 'Short_scaled_rolling_window_interval_250days',
    #  'future_ret_0days_EOD',
#  'future_ret_1days_EOD',
#  'future_ret_3days_EOD',
#  'future_ret_5days_EOD',
#  'future_ret_1halfhours_EOD',
#  'future_ret_0days',
#  'future_ret_1days',
#  'future_ret_3days',
#  'future_ret_5days',
'future_ret_1halfhours',
#  'future_ret_2halfhours',
#  'future_ret_1halfhours_EOD_lead1',
#  'future_ret_1halfhours_lead1',
#  'future_ret_2halfhours_lead1',
#  'future_ret_10days_EOD',
#  'future_ret_20days_EOD',
]

indep_vars = [
    # 'abn_short_absolute',
    # 'abn_short_absolute_scaled_rolling_window_interval_250days',
    # 'Volume',
    # 'Short_Ratio',
    # 'abn_short',
    # 'abn_short_1',
    'abn_short_3',
    # 'abn_short_scaled_rolling_window_interval_250days',
    # 'abn_short_1_scaled_rolling_window_interval_250days',
    # 'abn_short_2_scaled_rolling_window_interval_250days',
    # 'abn_short_scaled_rolling_window_interval_20days',
    # 'abn_short_1_scaled_rolling_window_interval_20days',
    # 'abn_short_2_scaled_rolling_window_interval_20days',
    # 'Volume_scaled_rolling_window_interval_250days',
    # 'Buy_Sell_Imb_lag1',
    # 'PRICE/NAV_pct_lag14',
    # 'vol_index_lag14',
    # 'cum_ret',
    # 'prev_day_rv_Average_5day',
    # 'vol_index_lag14_scaled_rolling_window_interval_250days',
    # 'prev_day_rv_Average_5day_scaled_rolling_window_interval_250days',
    # 'Buy_Sell_Imb_lag1_scaled_rolling_window_interval_250days',
    # 'PRICE/NAV_pct_lag14_scaled_rolling_window_interval_250days'

]

# for interval in intervals:
#     indep_vars.append(f'd_{interval}')





entity_effects = True
time_effects = True
cov_type = 'kernel' # can be of the following types
# 'unadjusted' - assume homoskedasticity
# 'robust' - use White's estimator for heteroskedasticity robustness
# 'clustered' - one- or two-way clustering, takes cluster_entity and cluster_time (or clusters, but needs specification)
# 'kernel' - Driscoll-Kraay HAC estimator , takes kernel , default is Bartletts kernel
time = None


result = do_panel_regression(etf_sel_halfhourly, dep_vars=dep_vars, indep_vars=indep_vars, entity_effects=entity_effects, time_effects=time_effects, cov_type=cov_type, time=time)
result_df = result.resids

result.summary
#%%

new_col_name = "resids_panel_entity_time_4"
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


# %%
# Do panel regression with different intervals in different panels
dep_vars = [
    # 'Short'
    # 'Short_scaled_rolling_window_interval_250days'
    #  'future_ret_0days_EOD',
#  'future_ret_1days_EOD',
#  'future_ret_3days_EOD',
#  'future_ret_5days_EOD',
#  'future_ret_1halfhours_EOD',
#  'future_ret_0days',
#  'future_ret_1days',
#  'future_ret_3days',
#  'future_ret_5days',
 'future_ret_1halfhours',
#  'future_ret_2halfhours',
#  'future_ret_1halfhours_EOD_lead1',
#  'future_ret_1halfhours_lead1',
#  'future_ret_2halfhours_lead1',
#  'future_ret_10days_EOD',
#  'future_ret_20days_EOD',
]

indep_vars = [
    # 'abn_short_absolute',
    # 'abn_short_absolute_scaled_rolling_window_interval_250days',
    # 'Volume',
    'Short_Ratio'
    # 'Volume_scaled_rolling_window_interval_250days',
    # 'Buy_Sell_Imb_lag1',
    # 'PRICE/NAV_pct_lag14',
    # 'vol_index_lag14',
    # 'cum_ret',
    # 'prev_day_rv_Average_5day'
]

# for interval in intervals:
#     indep_vars.append(f'd_{interval}')






entity_effects = True
time_effects = False
cov_type = 'kernel' # can be of the following types
# 'unadjusted' - assume homoskedasticity
# 'robust' - use White's estimator for heteroskedasticity robustness
# 'clustered' - one- or two-way clustering, takes cluster_entity and cluster_time (or clusters, but needs specification)
# 'kernel' - Driscoll-Kraay HAC estimator , takes kernel , default is Bartletts kernel
# time = '15:00:00'

panel_result_dict = {}

for key in etf_sel_halfhourly:
    panel_result_dict[key] = do_panel_regression_alt(etf_sel_halfhourly[key], dep_vars=dep_vars, indep_vars=indep_vars, entity_effects=entity_effects, time_effects=time_effects, cov_type=cov_type)

for key in panel_result_dict:
    summary = panel_result_dict[key].summary
    print(f"{key}: {summary}")
    

# %%

def do_panel_regression_alt(df, dep_vars, indep_vars, entity_col='ticker', time_col='DT', entity_effects=False, time_effects=False, cov_type='robust'):
    """
    Function to do a panel regression, given an input dictionary containing dfs for different financial items (here ETFs)
    Parameters:
    data_dict: dictionary containing dataframes with data
    dep_vars: list of one item containing the column name of the dependent variable
    indep_vars: list of column names of indepdendent variables
    entity_effects: boolean indicating if item fixed effects should be included in the panel regression
    time_effects: boolean indicating if time fixed effects should be included in the panel regression
    cov_type: string specifiying the covariance type that is used for the standard error calculation, can be of the following types
    - 'unadjusted' - assume homoskedasticity
    - 'robust' - use White's estimator for heteroskedasticity robustness
    - 'clustered' - one- or two-way clustering, takes cluster_entity and cluster_time (or clusters, but needs specification)
    - 'kernel' - Driscoll-Kraay HAC estimator , takes kernel , default is Bartletts kernel

    Returns: fitted model
    
    """
    import pandas as pd
    from linearmodels import PanelOLS
    # Combine dataframes from dictionary into one df, suitable for panel regression
    combined_df = df.copy()
    # Set multi-index, to allow for panel regression
    combined_df.set_index([entity_col, time_col], inplace=True)

    y = combined_df[dep_vars]
    x = combined_df[indep_vars]
    model = PanelOLS(y, x, entity_effects=entity_effects, time_effects=time_effects)

    results = model.fit(cov_type=cov_type)

    return results

#%%
combined_df = pd.concat([df.assign(ticker=ticker) for ticker, df in etf_sel_halfhourly.items()])
combined_df.set_index(['ticker', 'DT'], inplace=True)

# %% Individual OLS regressions


#%%
## Get some statistics on different residual measures

for key in etf_sel_halfhourly:
    panel_entity_mean = etf_sel_halfhourly[key]['resids_panel_entity_3'].mean()
    panel_entity_std = etf_sel_halfhourly[key]['resids_panel_entity_3'].std()
    panel_entity_time_mean = etf_sel_halfhourly[key]['resids_panel_entity_time_3'].mean()
    panel_entity_time_std = etf_sel_halfhourly[key]['resids_panel_entity_time_3'].std()
    pooled_mean = etf_sel_halfhourly[key]['resids_pooled_3'].mean()
    pooled_std = etf_sel_halfhourly[key]['resids_pooled_3'].std()
    individual_mean = etf_sel_halfhourly[key]['resid_individual_3'].mean()
    individual_std = etf_sel_halfhourly[key]['resid_individual_3'].std()
    print(f"RESULTS FOR {key}:")
    print(f"Mean panel entity fixed effect: {panel_entity_mean}")
    print(f"Std panel entity fixed effect: {panel_entity_std}")
    print(f"Mean panel entity & time fixed effect: {panel_entity_time_mean}")
    print(f"Std panel entity & time fixed effect: {panel_entity_time_std}")
    print(f"Mean pooled regression: {pooled_mean}")
    print(f"Std pooled regression: {pooled_std}")
    print(f"Mean individual regression: {individual_mean}")
    print(f"Std individual regression: {individual_std}")
    print("______________________________________________________")
    print("___________________________________________________________________")



# %% Add scaled variables

# vars_scale = ['PRICE/NAV_pct_lag14', 'Buy_Sell_Imb_lag1', 'prev_day_rv_Average_5day', 'vol_index_lag14', 'resid_individual_2',	'resids_panel_entity_time_2', 'resids_panel_entity_2', 'resids_pooled_2']
vars_scale = ['abn_short', 'abn_short_1', 'abn_short_2']
window_list = [250, 20]
# Scale only over intervals, moving window of 1 year (250 trading days)
for key in etf_sel_halfhourly.keys():
    for window_size in window_list:

        etf_sel_halfhourly[key] = scale_vars_exp_window(
            etf_sel_halfhourly[key],
            vars_scale,
            StandardScaler(),
            window_size=window_size,
            method="<=t",
            interval_col="TIME",
            inplace=False,
            rolling=True,
        )
# %%
# Add lags for different short values
# %% # Add columns containing aggregate short volume, aggregate short ratio, and average short ratio up to a certain

for key in etf_sel_halfhourly.keys():
    etf_sel_halfhourly[key]["cum_ret_factor"] = etf_sel_halfhourly[key]["RETURN"] + 1
    etf_sel_halfhourly[key]["agg_short"] = (
        etf_sel_halfhourly[key].groupby("DATE")["Short"].cumsum()
    )
    etf_sel_halfhourly[key].drop(columns=["cum_ret_factor"], inplace=True)
    # Add lag
    etf_sel_halfhourly[key] = add_lag(etf_sel_halfhourly[key], "cum_ret", 1)


#%%
df = etf_sel_halfhourly['AGG'].copy()
#%%


for key in etf_sel_halfhourly:

    etf_sel_halfhourly[key]['agg_short'] = (
        etf_sel_halfhourly[key].groupby("DATE")['Short'].cumsum()
    )
    etf_sel_halfhourly[key]['agg_short_scaled'] = (
    etf_sel_halfhourly[key].groupby("DATE")['Short_scaled_rolling_window_interval_250days'].cumsum()
    )
    etf_sel_halfhourly[key]['agg_short_ratio'] = (
    etf_sel_halfhourly[key].groupby("DATE")['Short_Ratio'].cumsum()
    )
    etf_sel_halfhourly[key]['ave_short_ratio'] = (
    etf_sel_halfhourly[key].groupby("DATE")['Short_Ratio'].cumsum() / (df.groupby("DATE").cumcount() + 1)
    )

#%%
agg_list = [
        'resid_individual_1',
 'resids_pooled_1',
 'resids_panel_entity_1',
 'resids_panel_entity_time_1',
 'resid_individual_2',
 'resids_panel_entity_time_2',
 'resids_panel_entity_2',
 'resids_pooled_2',
'resid_individual_2_scaled_rolling_window_interval_250days',
 'resids_panel_entity_time_2_scaled_rolling_window_interval_250days',
 'resids_panel_entity_2_scaled_rolling_window_interval_250days',
 'resids_pooled_2_scaled_rolling_window_interval_250days',
 'cum_ret_scaled_rolling_window_interval_250days',
 'resid_individual_4',
 'resids_pooled_4',
 'resids_panel_entity_4',
 'resids_panel_entity_time_4'
]

for key in etf_sel_halfhourly:
    for var in agg_list:
         etf_sel_halfhourly[key][f'agg_{var}'] = (
        etf_sel_halfhourly[key].groupby("DATE")[var].cumsum()
    )
# %% Add some lags
lag_list = [14]
for key in etf_sel_halfhourly:
    for lag in lag_list:

        etf_sel_halfhourly[key] = add_lag(etf_sel_halfhourly[key], "delta_NAV_PRICE", lag)



#%%
# Add a Delta NAV variable
for key in etf_sel_halfhourly:
        
    etf_sel_halfhourly[key]['delta_NAV_PRICE'] = etf_sel_halfhourly[key]['PRICE/NAV_pct'] - etf_sel_halfhourly[key]['PRICE/NAV_pct_lag14']

#%%
for key in etf_sel_halfhourly:
    etf_sel_halfhourly[key] = add_lag(etf_sel_halfhourly[key], "Short_Ratio", 2)


#%% Get correlation across ETFs
def get_cross_correlations(dict, col_name):
    extracted_cols = {key: df[col_name] for key, df in dict.items()}

    combined_df = pd.DataFrame(extracted_cols)
    corr_matrix = combined_df.corr()

    # print(corr_matrix)
    return corr_matrix

#%%
get_cross_correlations(etf_sel_halfhourly, "abn_short_2_scaled_rolling_window_interval_250days")

#%%
# get just the SPY data in dataframe
# df_spy = etf_sel_halfhourly['SPY']
# df_shy = etf_sel_halfhourly['SHY']
df_tlt = etf_sel_halfhourly['TLT']
# df_ief = etf_sel_halfhourly['IEF']

# Extract only the S&P 500 mini futures CFTC data (very similar to S&P 500 consolidated)
# df_cftf_spy = df_cftc[df_cftc['CFTC Contract Market Code'] == '13874A'].reset_index(drop=True)
# df_cftf_shy = df_cftc[df_cftc['CFTC Contract Market Code'] == '042601'].reset_index(drop=True)
df_cftf_tlt = df_cftc[df_cftc['CFTC Contract Market Code'] == '020601'].reset_index(drop=True)
# df_cftf_ief = df_cftc[df_cftc['CFTC Contract Market Code'] == '043602'].reset_index(drop=True)
# %%
# Add a column with the difference between the non-commercial and commercial short ratio
# df_cftf_spy['Short Ratio Difference'] = df_cftf_spy['Noncommercial Short Ratio'] - df_cftf_spy['Commercial Short Ratio']
# df_cftf_shy['Short Ratio Difference'] = df_cftf_shy['Noncommercial Short Ratio'] - df_cftf_shy['Commercial Short Ratio']
df_cftf_tlt['Short Ratio Difference'] = df_cftf_tlt['Noncommercial Short Ratio'] - df_cftf_tlt['Commercial Short Ratio']
# df_cftf_ief['Short Ratio Difference'] = df_cftf_ief['Noncommercial Short Ratio'] - df_cftf_ief['Commercial Short Ratio']

#%%
def plot_financial_products(
    df,
    columns_to_plot,
    product_list=None,
    title="Financial Product Trends Over Time",
    xlabel="Date",
):
    """
    Plots specific columns of a multi-index dataframe over time, with each financial product as a separate line.

    Parameters:
    df (pd.DataFrame): The multi-index dataframe to plot.
    columns_to_plot (list): List of columns to plot.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    """

    # Get the unique financial product types
    if product_list:
        product_types = product_list
    else:
        product_types = df.index.get_level_values(0).unique()

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Loop through each product type
    for product in product_types:
        # Extract the data for the current product type
        product_data = df.loc[product]

        # Plot each specified column for the current product type
        for column in columns_to_plot:
            plt.plot(
                product_data.index,
                product_data[column],
                label=f"{code_dictionary[product]} - {column}",
            )

    if len(columns_to_plot) == 1:
        ylabel = columns_to_plot[0]
    else:
        ylabel = "Data"

    # Add title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add legend
    plt.legend()

    # Show plot
    plt.show()


#%%
code_list = [
    # "13874A",
    # "042601",
    "020601",
    # "043602"
]

col_to_plot = [
    # 'Short_Ratio_last',
    # 'Short_Ratio_avg',
    'Short_Ratio_weekly_avg', 
    'Commercial Short Ratio', 
    'Noncommercial Short Ratio'
    ]

# df_plot = df_cftf_spy.copy()
# df_plot = df_plot.set_index(["CFTC Contract Market Code", "Date"])
# title = "Last half hour Tuesday SPY short ratio and CFTC data split"

# df_plot = df_cftf_shy.copy()
# df_plot = df_plot.set_index(["CFTC Contract Market Code", "Date"])
# title = "Last half hour Tuesday SHY short ratio and CFTC data split"
df_plot = df_cftf_ief.copy()
df_plot = df_plot.set_index(["CFTC Contract Market Code", "Date"])
title = "Last half hour Tuesday TLT short ratio and CFTC data split"
#%%

plot_financial_products(df_plot, col_to_plot, code_list, title="Difference between non-commercial and commercial short ratio over time - TLT")
# %%


def add_cols_to_weeklydf(df_weekly, df_to_merge, cols_to_add):
    df_cftc = df_weekly.copy()
    df_spy = df_to_merge.copy()

    # Set relevant columns to datetime
    df_spy['DATE'] = pd.to_datetime(df_spy['DATE'])
    df_cftc['Date'] = pd.to_datetime(df_cftc['Date'])


    # Only get specific columns
    cols_to_add.append('DATE') # ensure that the date column is kept
    df_spy = df_spy[cols_to_add]


    # Get last interval for each date
    last_interval_df = df_spy.groupby('DATE').tail(1)

    #Get daily averages
    daily_avg_df = df_spy.groupby('DATE').mean().reset_index()

    

    # Merge columns
    df_merged_first = df_cftc.merge(last_interval_df, left_on='Date', right_on='DATE', how='left')

    df_final = df_merged_first.merge(daily_avg_df, left_on='Date', right_on='DATE', suffixes=('_last', '_avg'), how='left')

    # Drop double date columns
    df_final.drop(columns=['DATE_last', 'DATE_avg'], inplace=True)

    daily_avg_df.set_index('DATE', inplace=True)
    rolling_avg_df = daily_avg_df.rolling(window=7, min_periods=1).mean().reset_index()

    # Rename columns to indicate rolling average
    rolling_avg_df.columns = [col + '_weekly_avg' if col != 'DATE' else col for col in rolling_avg_df.columns]

    # Merge rolling averages with the original weekly DataFrame
    df_final = df_final.merge(rolling_avg_df, left_on='Date', right_on='DATE', how='left')

    return df_final
    
#%%

cols_to_add = [
    'Short_Ratio',
    'Short',
    'abn_short_ratio'
]
df_cftf_tlt = add_cols_to_weeklydf(df_cftf_tlt, df_tlt, cols_to_add)






# %%
dep_var_list = [
    # 'Short_Ratio_weekly_avg',
    # 'Short_Ratio_avg',
    # 'Short_Ratio_last',
    'Commercial Short Ratio',
    # 'Noncommercial Short Ratio'
]

indep_var_list = [
    'Short_Ratio_last',
    'Short_Ratio_avg',
    'Short_Ratio_weekly_avg',
    # 'abn_short_ratio_last',
    # 'abn_short_ratio_avg',
    # 'abn_short_ratio_weekly_avg',

    # 'Commercial Short Ratio', 
    'Noncommercial Short Ratio'
]

covariance_type = 'HC1'


# result = do_regression(
#             df_cftf_spy, indep_var_list, dep_var_list, covariance_type
#         )
# result = do_regression(
#             df_cftf_shy, indep_var_list, dep_var_list, covariance_type
#         )
result = do_regression(
            df_cftf_tlt, indep_var_list, dep_var_list, covariance_type
        )
# result = do_regression(
#             df_cftf_ief, indep_var_list, dep_var_list, covariance_type
#         )

print(result[dep_var_list[0]].summary())

result_dict = {}
result_dict[dep_var_list[0]] = result

#%%
latex_table = get_latex_table(result_dict, dep_var_list, indep_var_list)

print(latex_table[0])
