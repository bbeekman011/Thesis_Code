# %%
import pyreadr
import pandas as pd
from collections import OrderedDict


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


# %%


## Functions
def split_df_on_symbol(df, symbol_tag):
    """Takes a dataframe containing data for multiple symbols or tickers and transfers these to a dictionary containing separate dataframes for each symbol, that symbol being the key
    Parameters:
    df (dataframe): Input dataframe containing some data on etfs with different symbols
    symbol_tag (string): String with the column name of the column on which the data needs to be split
    Returns:
    Dictionary: Contains a separate dataframe for each symbol (key)
    """

    output_dict = {}
    column_list = df.columns.tolist()
    column_list.remove(symbol_tag)

    for symbol in df[symbol_tag].unique():
        data = df[df[symbol_tag] == symbol][column_list].reset_index(drop=True)

        output_dict[symbol] = data

    return output_dict


# %%
etf_prices_30min_dict = split_df_on_symbol(df_etf_prices_30min, "SYMBOL")
etf_shvol_30min_dict = split_df_on_symbol(df_etf_shvol_30min, "SYMBOL")
etf_vol_30min_dict = split_df_on_symbol(df_etf_vol_30min, "SYMBOL")

# %%
for key, df in etf_prices_30min_dict.items():
    df["DATE"] = df["DT"].dt.date
    df["TIME"] = df["DT"].dt.time

    df = df.reindex(columns=["DT", "DATE", "TIME", "PRICE"])
    etf_prices_30min_dict[key] = df


# %%

### Get 30-minute returns
for key, df in etf_prices_30min_dict.items():
    df["RETURN"] = df["PRICE"] / df["PRICE"].shift(1) - 1

    etf_prices_30min_dict[key] = df
# %%
test=3
print(test)