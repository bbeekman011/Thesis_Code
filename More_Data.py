# %%
import pyreadr
import pandas as pd
from collections import OrderedDict
from Functions import *
from datetime import datetime, timedelta
import numpy as np
import time as tm
import pickle
import pytz
import cot_reports as cot
import matplotlib.pyplot as plt

# %%

# Overview of the report types and corresponding strings in that order respectively

# Legacy Report Futures-only:                          	            'legacy_fut'
# Legacy Report Futures-and-Options Combined:       	            'legacy_futopt'
# Supplemental Report Futures-and-Options combined (short format):	'supplemental_futopt'
# Disaggregated Report Futures-only:                            	'disaggregated_fut'
# Disaggregated Report Futures-and-Options Combined:            	'disaggregated_futopt'
# Traders in Financial Futures (TFF):                           	'traders_in_financial_futures_fut'
# Traders in Financial Futures (TFF) Futures-and-Options Combined: 	'traders_in_financial_futures_futopt'


# Overview of the possible commands
# cot_hist(): downloads the compressed bulk files for the specified report (cot_report_type) starting from 1986 (depending on the report type) up to 2016 and returns the data in a dataframe.

# cot_year(): downloads historical single year data for the specified report (cot_report_type) and returns the data in a dataframe (cot_report_type). If the current year is specified, the latest published data is fetched.

# cot_all(): downloads the complete available data, including the latest, of the specified report and returns the data in a dataframe (cot_report_type).

# cot_all_reports(): downloads all available historical information of the available COT reports and returns the data as seven dataframes.
# %%


# Example: cot_hist()
# df = cot.cot_hist(cot_report_type= 'traders_in_financial_futures_futopt')
# cot_hist() downloads the historical bulk file for the specified report type, in this example the Traders in Financial Futures Futures-and-Options Combined report. Returns the data as dataframe.

# # Example: cot_year()
# df = cot.cot_year(year = 2020, cot_report_type = 'traders_in_financial_futures_fut')
# cot_year() downloads the single year file of the specified report type and year. Returns the data as dataframe.

# # Example for collecting data of a few years, here from 2017 to 2020, of a specified report:

begin_year = 2014
end_year = 2022
data_frames = []

# Collect data for each year
for year in range(begin_year, end_year + 1):
    single_year = pd.DataFrame(cot.cot_year(year, cot_report_type="legacy_futopt"))
    data_frames.append(single_year)

# Concatenate all data into a single DataFrame
df_selection = pd.concat(data_frames, ignore_index=True)

# %%
df_cftc_raw = df_selection.copy()
# # Example: cot_all()
# df = cot.cot_all(cot_report_type='legacy_fut')
# # cot_all() downloads the historical bulk file and all remaining single year files of the specified report type.  Returns the data as dataframe.
# %%

# Specify list of market and exchange names we want to consider
market_list = [
    "2-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE ",
    "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE ",
    "5-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE ",
    "U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE ",
    "LONG-TERM U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE ",
    "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE ",
    "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE ",
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE ",
    "2-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
    "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
    "5-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
    "U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE",
    "LONG-TERM U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE",
    "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE",
    "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
    "UST 2Y NOTE - CHICAGO BOARD OF TRADE",
    "UST 10Y NOTE - CHICAGO BOARD OF TRADE",
    "ULTRA UST 10Y - CHICAGO BOARD OF TRADE",
    "UST 5Y NOTE - CHICAGO BOARD OF TRADE",
]

# List of contract market code corresponding to the instruments denoted in the list above (S&P 500 variations and different maturity US treasuries)
# Links are as follows:
#     "020601" : U.S. Treasury Bonds,
#     "020604": Long-term U.S. Treasury Bonds,
#     "042601": 2-Year U.S. Treasury Notes,
#     "043602": 10-Year U.S. Treasury Notes,
#     "044601" : 5-Year U.S. Treasury Notes,
#     "13874+" : S&P 500 Consolidated,
#     "138741" : S&P 500 Stock Index, Available until 14-09-2021
#     "13874A": S&P 500 Mini,
#     "043607": Ultra 10-year U.S. Treasury Note, Availabe from 08-03-2016
contract_code_list = [
    "020601",
    "020604",
    "042601",
    "043602",
    "044601",
    "13874+",
    "138741",
    "13874A",
    "043607",
]

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
#%%
# Filter dataframe to only contain relevant products (see specification above)
df_selection["CFTC Contract Market Code"] = df_selection[
    "CFTC Contract Market Code"
].str.strip()

# %%
df_cftc = df_selection[
    df_selection["CFTC Contract Market Code"].isin(contract_code_list)
]


# %%
df_cftc = df_cftc.rename(columns={"As of Date in Form YYYY-MM-DD": "Date"})

# %%
# Convert Date column to datetime
df_cftc["Date"] = pd.to_datetime(df_cftc["Date"])

# Sort values
df_cftc = df_cftc.sort_values(by="Date")

# %%
df_cftc = df_cftc.reset_index(drop=True)


# %%
# Set multi-index on contract code and time
df_cftc = df_cftc.set_index(["CFTC Contract Market Code", "Date"])
# %%
interesting_list = [
    "Market and Exchange Names",
    "Noncommercial Positions-Long (All)",
    "Noncommercial Positions-Short (All)",
    "Noncommercial Positions-Spreading (All)",
    "Commercial Positions-Long (All)",
    "Commercial Positions-Short (All)",
    "Change in Open Interest (All)",
    "Change in Noncommercial-Long (All)",
    "Change in Noncommercial-Short (All)",
    "Change in Noncommercial-Spreading (All)",
    "Change in Commercial-Long (All)",
    "Change in Commercial-Short (All)",
    "% of OI-Noncommercial-Long (All)",
    "% of OI-Noncommercial-Short (All)",
    "% of OI-Noncommercial-Spreading (All)",
    "% of OI-Commercial-Long (All)",
    "% of OI-Commercial-Short (All)",
    #     #  'Change in Noncommercial-Long (All)',
    # #  'Change in Noncommercial-Short (All)',
    # #  'Change in Noncommercial-Spreading (All)',
      'Open Interest (All)',
    # #    'Noncommercial Positions-Long (All)',
    # #  'Noncommercial Positions-Short (All)',
    #  'Noncommercial Positions-Spreading (All)',
    # #  'Commercial Positions-Long (All)',
    # #  'Commercial Positions-Short (All)',
    #  'Open Interest (Old)',
    #   'Open Interest (Other)',
    #   'Change in Open Interest (All)',
    #    ' Total Reportable Positions-Long (All)',
    #  'Total Reportable Positions-Short (All)',
    #  'Nonreportable Positions-Long (All)',
    #  'Nonreportable Positions-Short (All)',
    #  'Change in Total Reportable-Long (All)',
    #  'Change in Total Reportable-Short (All)',
    #  '% of Open Interest (OI) (All)',
    #  'Traders-Total (All)'
]
# %%

# Keep selection of relevant variables
df_cftc = df_cftc[interesting_list]

# %%
# Create a non-commercial short ratio variable

df_cftc["Noncommercial Short Ratio"] = df_cftc[
    "Noncommercial Positions-Short (All)"
] / (
    df_cftc["Noncommercial Positions-Short (All)"]
    + df_cftc["Noncommercial Positions-Long (All)"]
)
#%%
# Create a commercial short ratio variabkle
df_cftc["Commercial Short Ratio"] = df_cftc[
    "Commercial Positions-Short (All)"
] / (
    df_cftc["Commercial Positions-Short (All)"]
    + df_cftc["Commercial Positions-Long (All)"]
)
# %%
# Add a lagged version and a difference
df_cftc["Lagged Noncommercial Short Ratio"] = df_cftc.groupby(
    "CFTC Contract Market Code"
)["Noncommercial Short Ratio"].shift(1)
df_cftc["Change in Noncommercial-Short Ratio"] = (
    df_cftc["Noncommercial Short Ratio"] - df_cftc["Lagged Noncommercial Short Ratio"]
)

df_cftc["Lagged Commercial Short Ratio"] = df_cftc.groupby(
    "CFTC Contract Market Code"
)["Commercial Short Ratio"].shift(1)
df_cftc["Change in Commercial-Short Ratio"] = (
    df_cftc["Commercial Short Ratio"] - df_cftc["Lagged Commercial Short Ratio"]
)

#%%
#Save to excel
df_cftc.to_excel("cftc_data.xlsx")
#%%
# Load processed CFTC data

df_cftc = pd.read_excel(r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Other data\cftc_data.xlsx")
df_cftc = df_cftc.set_index(["CFTC Contract Market Code", "Date"])
# %% Function to plot
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


# %%
# Links are as follows:
#     "020601" : U.S. Treasury Bonds,
#     "020604": Long-term U.S. Treasury Bonds,
#     "042601": 2-Year U.S. Treasury Notes,
#     "043602": 10-Year U.S. Treasury Notes,
#     "044601" : 5-Year U.S. Treasury Notes,
#     "13874+" : S&P 500 Consolidated,
#     "138741" : S&P 500 Stock Index, Available until 14-09-2021
#     "13874A": S&P 500 Mini,
#     "043607": Ultra 10-year U.S. Treasury Note, Availabe from 08-03-2016
col_to_plot = ['Noncommercial Short Ratio', 'Commercial Short Ratio']
code_list = [
    "13874+",
    # "138741",
    "13874A",
    "020601",
    "020604",
    "042601",
    "043602",
    "044601",
    "043607",

]
plot_financial_products(df_cftc, col_to_plot, code_list)
# %%
df_cftc['Noncommercial Short Ratio (Pct)'] = df_cftc['% of OI-Noncommercial-Short (All)'] / (df_cftc['% of OI-Noncommercial-Short (All)'] + df_cftc['% of OI-Noncommercial-Long (All)'])
# %%
## Get multiple plots in one figure

def plot_financial_products_multiple(
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
    """

    # Get the unique financial product types
    if product_list:
        product_types = product_list
    else:
        product_types = df.index.get_level_values(0).unique()

    num_products = len(product_types)
    num_columns = 2  # Adjust the number of columns as needed
    num_rows = (num_products + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, product in enumerate(product_types):
        ax = axes[i]
        product_data = df.loc[product]

        for column in columns_to_plot:
            ax.plot(
                product_data.index,
                product_data[column],
                label=f"{code_dictionary[product]} - {column}",
            )

        ax.set_title(f"Product: {code_dictionary[product]}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Value")
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


#%%
plot_financial_products_multiple(df_cftc, col_to_plot, code_list)
# %%
with open(
    r"C:\Users\ROB7831\OneDrive - Robeco Nederland B.V\Documents\Thesis\Data\Processed\etf_sel_halfhourly_20240529.pkl",
    "rb",
) as f:
    etf_sel_halfhourly = pickle.load(f)
# %%
