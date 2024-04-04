## This file contains all the functions used in the thesis project of Bas Beekman
from datetime import datetime
import pandas as pd


def split_df_on_symbol(df, symbol_tag: str):
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


def merge_df_on_vol_columns(
    merge_dict: dict,
    merger_dict: dict,
    new_col_list: list,
    rename_col_dict: dict,
    date_string: str,
    time_string: str,
    return_string: str,
):
    """This function merges the dataframe containing 30-minute price and return data and merges with a dataframe containing (short) volume data by day in 30 minute intervals by column
    Parameters:
    merge_dict (dictionary): dictionary of dataframes with price data, to merge with other dictionary
    merger_dict (dictionary): dictionary of dataframes with (short) volume data, merge_dict will be merged into this dictionary
    new_col_list (list): list new column names to be added, should follow the naming convention description_2digitnumber_first/second, which corresponds to a value to be taken from the merge_dict corresponding to the first of second half hour of a particlar hour and day
    rename_col_dict (dictionary): dictionary stating which columns should be renamed from new_col_list, since new_col_list is restrictive in naming conventions due to necessary logic connecting to timestamps.
    date_string (string): Name of the column in the dataframes inside the two input dictionaries. Note that these should be the same for the merge_dict and merger_dict
    time_string (string): Name of the column in the merger_dict dataframes corresponding to time.
    return_string (string): Name of the column in the merger_dict which contains the values to be added to merger_dict.

    Returns:
    output_dict (dictionary): dictionary containing dataframes with volume value and price or other return data added
    """

    from datetime import datetime
    import pandas as pd

    output_dict = {}

    for key in merger_dict.keys():

        df1 = merge_dict[key]
        df2 = merger_dict[key]

        date_threshold = max(
            df1[date_string][0], df2[date_string][0]
        )  # Specify date threshold, based on earliest available data in either prices or volume
        df1 = df1[df1[date_string] >= date_threshold]
        df2 = df2[df2[date_string] >= date_threshold]

        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        merged_df = df2.copy()

        for column_name in new_col_list:
            hour = int(column_name.split("_")[1][:2])

            time = datetime.strptime(
                f"{hour + 1}:00:00" if "second" in column_name else f"{hour}:30:00",
                "%H:%M:%S",
            ).time()

            temp_df = pd.merge(
                df2, df1[df1[time_string] == time], on=date_string, how="left"
            )

            merged_df[column_name] = temp_df[return_string]

        merged_df.rename(columns=rename_col_dict, inplace=True)

        output_dict[key] = merged_df

    return output_dict


def get_column_name(time, category):
    """This function just hardcodes the shit out of a tough issue, apologies, it's terrible
    Parameters:
    time (datetime): input time to be linked to a column name
    category (string): name of the category the value needs to be taken from
    here specifically options are (Short, Short_dollar, Volume, Volume_dollar)
    Returns:
    string corresponding to correct column name in volume data
    """

    if time == pd.to_datetime("09:30:00").time():
        return f"{category}_FH"
    elif time == pd.to_datetime("10:00:00").time():
        return f"{category}_09second"
    elif time == pd.to_datetime("10:30:00").time():
        return f"{category}_10first"
    elif time == pd.to_datetime("11:00:00").time():
        return f"{category}_10second"
    elif time == pd.to_datetime("11:30:00").time():
        return f"{category}_11first"
    elif time == pd.to_datetime("12:00:00").time():
        return f"{category}_11second"
    elif time == pd.to_datetime("12:30:00").time():
        return f"{category}_12first"
    elif time == pd.to_datetime("13:00:00").time():
        return f"{category}_12second"
    elif time == pd.to_datetime("13:30:00").time():
        return f"{category}_13first"
    elif time == pd.to_datetime("14:00:00").time():
        return f"{category}_13second"
    elif time == pd.to_datetime("14:30:00").time():
        return f"{category}_14first"
    elif time == pd.to_datetime("15:00:00").time():
        return f"{category}_14second"
    elif time == pd.to_datetime("15:30:00").time():
        return f"{category}_SLH"
    elif time == pd.to_datetime("16:00:00").time():
        return f"{category}_LH"
    else:
        return None
