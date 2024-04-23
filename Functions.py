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
        return f"{category}_9first"
    elif time == pd.to_datetime("10:00:00").time():
        return f"{category}_FH"
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


def merge_df_on_price_rows(
    merge_dict: dict,
    merger_dict: dict,
    date_string: str,
    time_string: str,
    col_name_list: list,
):
    """This function merges the dataframe containing (short) volume data by day and merges with a dataframe with 30-minute price  intervals on the rows
    Parameters:
    merge_dict (dictionary): dictionary of dataframes with (short) volume data, to merge with other dictionary    dictionary of dataframes with price data
    merger_dict (dictionary):  dictionary of dataframes with price data, merge_dict will be merged into this dictionary
    date_string (string): Name of the column in the dataframes inside the two input dictionaries. Note that these should be the same for the merge_dict and merger_dict
    time_string (string): Name of the column in the merger_dict dataframes corresponding to time.
    col_name_list (list): List of column names, following the naming convention (without specification) of the (short) volume data, these columns will be added to the price dataframe

    Returns:
    output_dict (dictionary): dictionary containing dataframes price data with (short) volume data added for each 30-minute interval
    """

    import numpy as np

    output_dict = merger_dict.copy()

    for key in output_dict.keys():
        # Precompute date and time arrays
        dates = output_dict[key][date_string].values
        times = output_dict[key][time_string].values

        for col_name in col_name_list:
            # Precompute column names
            column_names = [get_column_name(time, col_name) for time in times]
            mask = np.array([col_name is not None for col_name in column_names])

            # Filter out None column names and corresponding dates
            valid_dates = dates[mask]
            valid_column_names = np.array(column_names)[mask]

            # Filter merge_dict[key] based on dates
            merge_dict_key_dates = merge_dict[key][
                merge_dict[key][date_string].isin(valid_dates)
            ]

            # Iterate through valid indices
            for i, date, column_name in zip(
                range(len(output_dict[key])), valid_dates, valid_column_names
            ):
                value = merge_dict_key_dates.loc[
                    merge_dict_key_dates[date_string] == date, column_name
                ].iloc[0]
                output_dict[key].at[i, col_name] = value

    return output_dict


def fill_missing_intervals(df):
    """This function loops through a dataframe with the format of dates and times in rows below each other.
    It checks if each date present in the data set has all 30 minute intervals between 09:30:00 and 16:00:00 present and if not, it adds these.
    Missing values are filled forward."""

    merged_dfs = []
    # Create a list of all possible half-hour intervals between 09:30:00 and 16:00:00
    all_intervals = pd.date_range(start="09:30:00", end="16:00:00", freq="30min").time

    for date in df["DATE"].unique():
        # Filter DataFrame for the current date
        date_df = df[df["DATE"] == date]

        # Create a DataFrame with all possible half-hour intervals for the current date
        all_intervals_df = pd.DataFrame({"TIME": all_intervals})
        all_intervals_df["DATE"] = date

        # Merge with the original DataFrame for the current date to fill in missing intervals
        merged = pd.merge(all_intervals_df, date_df, on="TIME", how="left")

        # Fill missing values (NaNs) forward
        merged["PRICE"] = merged["PRICE"].ffill()

        merged_dfs.append(merged)

    output_df = pd.concat(merged_dfs, ignore_index=True)

    ## Drop irrelevant columns, change names and reorder
    output_df["DT"] = pd.to_datetime(
        output_df["DATE_x"].astype(str) + " " + output_df["TIME"].astype(str)
    )
    output_df = output_df.drop(columns=["DATE_y"])
    output_df = output_df.rename(columns={"DATE_x": "DATE"})
    output_df = output_df.reindex(columns=["DT", "DATE", "TIME", "PRICE"])

    return output_df


def intraday_plot(
    df,
    dt_col: str,
    date_col: str,
    start_date: str,
    end_date: str,
    fig_title: str,
    y_ax1_col: str,
    y_ax1_label: str,
    y_ax1_title: str,
    y_ax2_col=None,
    y_ax2_label=None,
    y_ax2_title=None,
    vertical_line=None,
    mode="lines",
    x_ax_title="Date",
):

    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ## Get the correct range in the data
    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].reset_index(
        drop=True
    )

    ## Check if a second axis is necessary, add two traces if so
    if not y_ax2_col is None and not y_ax2_label is None and not y_ax2_title is None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df[dt_col], y=df[y_ax1_col], mode=mode, name=y_ax1_label),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df[dt_col], y=df[y_ax2_col], mode=mode, name=y_ax2_label),
            secondary_y=True,
        )

        # Add y-axes titles
        fig.update_yaxes(title_text=y_ax1_title, secondary_y=False)
        fig.update_yaxes(title_text=y_ax2_title, secondary_y=True)

    else:
        fig = go.Figure(data=go.Scatter(x=df[dt_col], y=df[y_ax1_col], mode=mode))

        # Add y-axis title
        fig.update_yaxes(title_text=y_ax1_title)

    # Don't show data points outside of trading hours, remove weekends
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=[16.5, 9], pattern="hour"),
            dict(bounds=["sat", "mon"]),
        ],
        title_text=x_ax_title,
    )

    # Update titles
    fig.update_layout(title=fig_title)

    # Add vertical line if specified
    if vertical_line:
        fig.add_vline(x=vertical_line, line_dash="dash", line_color="green")

    return fig


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


def get_eventday_plots(
    name_dict: dict,
    input_dict: dict,
    ticker_list: list,
    event_dt_list: list,
    y1_list: str,
    y_2: str,
    parent_dir: str,
    day_range: int,
):
    import os

    total_days = 2 * day_range + 1

    for event_dt in event_dt_list:

        # Create new directory for files to be stored in
        event_date = pd.to_datetime(event_dt).strftime("%Y-%m-%d")
        parent_dir = parent_dir
        new_dir = event_date
        path = os.path.join(parent_dir, new_dir)
        if not os.path.exists(path):
            os.mkdir(path)

        for ticker in ticker_list:

            df = input_dict[ticker]
            vert_line = event_dt

            start_date = pd.to_datetime(event_dt) - pd.Timedelta(days=day_range)
            start_date = start_date.strftime("%Y-%m-%d")
            end_date = pd.to_datetime(event_dt) + pd.Timedelta(days=day_range)
            end_date = end_date.strftime("%Y-%m-%d")
            title = f"{ticker} - {name_dict[ticker]}"
            y_2 = "PRICE"

            for y_1 in y1_list:
                fig = intraday_plot(
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

                fig.write_image(
                    rf"{path}/{event_date}_{ticker}_{y_1}_{total_days}day.png"
                )


def short_ratio(value1, value2):
    return value1 / value2


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


def rolling_avg_trading_days(series, window_size):
    trading_days = series.index.dayofweek < 5  # Filter only weekdays
    return series[trading_days].rolling(window=window_size, min_periods=1).mean()


def add_rolling_window_average_col(df_in, ave_col_name, window_size, dt_col):

    df = df_in.copy()
    
    df.set_index(dt_col, inplace=True)
    
    rolling_avg = df.groupby(df.index.time)[ave_col_name].apply(
        rolling_avg_trading_days, window_size=window_size
    )
    rolling_avg = rolling_avg.reset_index(level=0, drop=True)

    df[f"{window_size}day_Avg_{ave_col_name}"] = rolling_avg

    df.reset_index(inplace=True)

    return df


def intraday_barplot(dict, ticker, metric, start_date, end_date, event, non_event_def=True):
    import matplotlib.pyplot as plt
    
    df = dict[ticker]

    df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]


    event_df = df[df[event] == 1]
    
    if non_event_def is True:
        non_event_df = df[df["EVENT"] == 0]
    else:
        non_event_df = df[df[event] == 0]

    event_grouped = event_df.groupby(event_df['TIME'])[metric].mean().reset_index()
    non_event_grouped = non_event_df.groupby(non_event_df['TIME'])[metric].mean().reset_index()
    


    # Plotting
    plt.figure(figsize=(12, 6))
    bar_width = 0.4


    x_event = range(len(event_grouped))
    x_non_event = [x + bar_width  for x in x_event]


    plt.bar(x_event, event_grouped[metric], width=bar_width, label='Event Days', color='blue')
    plt.bar(x_non_event, non_event_grouped[metric], width=bar_width, label='Non-Event Days', color='orange')

    plt.xlabel('Time')
    plt.ylabel(metric)
    plt.title(f'Average Value of {metric} for {ticker} on {event} and non-{event} days ')
    # Use original x-coordinates for x-axis ticks
    plt.xticks(x_event, event_grouped['TIME'], rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()
