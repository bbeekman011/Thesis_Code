# %%
import pyreadr
import pandas as pd
from collections import OrderedDict
from Functions import split_df_on_symbol, merge_df_on_vol_columns
from datetime import datetime

# %%
## Specify paths
sample_path_etf_prices_30min = r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\ETF data\ETF_prices_30min.rds"
sample_path_etf_shvol_30min = r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\ETF data\ETF_shvol_30min.rds"
sample_path_etf_vol_30min = r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\ETF data\ETF_vol_30min.rds"
sample_path_us_auctions = r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Other data\Bloomberg_releases_US_updated_2024.xlsx"
sample_path_us_releases = r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Other data\Bloomberg_releases_US_updated_2024.xlsx"
sample_path_ty = r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Other data\TY.csv"

### Read R files
# etf_prices_1min = pyreadr.read_r(r'C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\ETF data\ETF_prices_1min.rds')
etf_prices_30min = pyreadr.read_r(sample_path_etf_prices_30min)
etf_shvol_30min = pyreadr.read_r(sample_path_etf_shvol_30min)
etf_vol_30min = pyreadr.read_r(sample_path_etf_vol_30min)

### Read Excel and CSV files
df_us_auctions = pd.read_excel(sample_path_us_auctions)
df_us_releases = pd.read_excel(sample_path_us_releases)
df_ty = pd.read_csv(sample_path_ty)

# Convert ordered dictionaries to Pandas dataframe
# df_etf_prices_1min = etf_prices_1min[None]   ## This file doesn't seem to be correct
df_etf_prices_30min = etf_prices_30min[None]
df_etf_shvol_30min = etf_shvol_30min[None]
df_etf_vol_30min = etf_vol_30min[None]
df_etf_shvol_30min = df_etf_shvol_30min.rename(
    columns={"Date": "DATE"}
)  # Match date column name to other dataframes
# %%
etf_prices_30min_dict = split_df_on_symbol(df_etf_prices_30min, "SYMBOL")
etf_shvol_30min_dict = split_df_on_symbol(df_etf_shvol_30min, "SYMBOL")
etf_vol_30min_dict = split_df_on_symbol(df_etf_vol_30min, "SYMBOL")


# %%
## Split Datetime into separate date and time
for key, df in etf_prices_30min_dict.items():
    df["DATE"] = df["DT"].dt.date
    df["TIME"] = df["DT"].dt.time

    df = df.reindex(columns=["DT", "DATE", "TIME", "PRICE"])  # Reorder columns
    etf_prices_30min_dict[key] = df

# %%

### Calculate 30-minute returns
for key, df in etf_prices_30min_dict.items():
    df["RETURN"] = df["PRICE"] / df["PRICE"].shift(1) - 1

    etf_prices_30min_dict[key] = df

# %%
## Merge the volume and short volume dataframes
etf_shvol_vol_30min_dict = {}

for key in etf_shvol_30min_dict.keys():
    merged_df = pd.merge(etf_shvol_30min_dict[key], etf_vol_30min_dict[key], on="DATE")
    etf_shvol_vol_30min_dict[key] = merged_df


# %%

## Specify new columns, in line with the notation used in the columns of the volume data, use numbers only to support logic later on
new_columns = [
    "Return_09first",
    "Return_09second",
    "Return_10first",
    "Return_10second",
    "Return_11first",
    "Return_11second",
    "Return_12first",
    "Return_12second",
    "Return_13first",
    "Return_13second",
    "Return_14first",
    "Return_14second",
    "Return_15first",
    "Return_15second",
]

## Specify which columns to rename to match naming conventions used in volume data
columns_to_rename = {
    "Return_09second": "Return_FH",
    "Return_15first": "Return_SLH",
    "Return_15second": "Return_LH",
}

#%%

## Merge using the function
etf_merged_30min_daily_dict = merge_df_on_vol_columns(
    etf_prices_30min_dict,
    etf_shvol_vol_30min_dict,
    new_columns,
    columns_to_rename,
    "DATE",
    "TIME",
    "RETURN")

# %%
