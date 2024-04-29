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
    """Function to make an intraday interactive plot for a specific date and specific sets of variables
    Parameters:
    df: dataframe containing relevant data
    dt_col: column name of Datetime column
    date_col: column name of Date column
    start_date: start date of plot
    end_date: end date of plot
    fig_title: title of the figure
    y_ax1_col: 
    """

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
    name_dict,
    input_dict,
    ticker_list,
    event_dt_list,
    y1_list,
    y_2: str,
    day_range: int,
    display_dummy = True,
    parent_dir = None,
):
    """ This function is used to get a large number of intraday plots, and either save these to a directory or plot them directly.
    Parameters:
    name_dict: dictionairy linking ETF tickers to descriptions, used for clarity in plot titles
    input_dict: dictionairy containing dataframes of data on ETFs, ETF tickers are used as keys
    ticker_list: list of tickers to be plotted
    event_dt_list: list of event dates to be plotted
    y1_list: list of parameters to be plotted on the first y-axis
    y2: parameter to be plotted on the second y-axis
    day_range: number of days to plotted around each event date (2-sided, so day_range 2 will plot 5 days)
    display_dummy: dummy variable indicating if plots should be displayed. If true, plots are displayed, if False, plots are saved
    parent_dir: only necessary if display_dummy is False. Specifies the parent directory of where the plots should be saved
    """

    total_days = 2 * int(day_range) + 1

    for event_dt in event_dt_list:

        
        event_date = pd.to_datetime(event_dt).strftime("%Y-%m-%d")

        if display_dummy is False:
            # Create new directory for files to be stored in
            import os    
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
        else:
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

                    fig.show()



def short_ratio(value1, value2):
    return value1 / value2


def add_daily_cols(df, suffix_list, func, input_col1, input_col2, new_col):
    """ This function adds columns consisting of an operation on existing columns to the 'daily' dataframe
    Parameters:
    df: dataframe to which column is added
    suffix_list: list of relevant suffixes related to half-hour intervals
    func: function specifying the operation to be done
    input_col1: first input column (order is relevant depending on operation to be done)
    input_col2: second input column (order is relevant depending on operation to be done)
    new_col: name of the new column (excluding suffixes)

    Returns:
    output_df: dataframe with new column added for each suffix
    """
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

    return series.rolling(window=window_size, min_periods=window_size).mean()


def add_rolling_window_average_col(df_in, ave_col_name, window_size, dt_col):
    """This function adds columns containing the rolling average for an x amount of days for a specific time interval, e.g. the 10-day rolling average of the 11:00 interval
    Parameters:
    df_in: input dataframe to which columns are added
    ave_col_name: Name of the column for which the rolling average needs to be added
    window_size: length of rolling window
    dt_col: name of the column in df_in containing the datetime variable

    Returns:
    df: input df with new columns added to it
    """
    def rolling_avg_trading_days(series, window_size):
        return series.rolling(window=window_size, min_periods=window_size).mean()

    df = df_in.copy()
    df.set_index(dt_col, inplace=True)

    rolling_grouped = df.groupby(df.index.time)[ave_col_name]
    rolling_avg = (
        rolling_grouped.apply(rolling_avg_trading_days, window_size=window_size)
        .reset_index(level=0, drop=True)
        .sort_index()
    )

    rolling_avg = rolling_avg.to_frame()
    df.reset_index(inplace=True)
    rolling_avg.reset_index(inplace=True)

    merged = pd.merge(
        df,
        rolling_avg,
        on=dt_col,
        suffixes=("", f"_Average_{window_size}day"),
    )

    df = merged

    return df


def intraday_barplot(
    dict, ticker, metric, start_date, end_date, event, non_event_def=True, lag_bar=False, surprise_split=False, surprise_col = None
):
    """This function is used to make a barplot of the average value of a variable (e.g. Volume, Short volume, Return etc.)
    for each half hour interval during a trading day (09:30:00 - 16:00:00). Specifically, it is used to compare such values on event days
    to non-event days, where an event corresponds to a major macroeconomic announcement.
    Parameters:
    dict: dictionairy containing dataframes of data with tickers as keys
    ticker: specific ticker to be plotted, must be a key of the input dataframe
    start_date: start date of the to be plotted sample
    end_date: end date of the to be plotted sample
    event: event to be investigated, choose from "ISM", "FOMC", "NFP", "CPI", "GDP", "IP", "PI", "HST", "PPI", "EVENT" or any of the lags, e.g. ISM_lag
    non_event_def: dummy variable determinging definition of 'non-event days'. If True, non-event days are days on which there is no event at all, if False, non-event days are all days except the days on which the specific input event occurs.
    lag_bar: dummy indicating if a third bar is added showing also the lagged event day
    surprise_split: dummy indicating if barplot shows different bars for positive and negative surprise
    surprise_col: suffix of the colname used for the surprise column, used to differentiate between different types of surprise definition
    Output:
    Shows barplot
    """
    import matplotlib.pyplot as plt

    df = dict[ticker]

    df = df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]

    event_df = df[df[event] == 1]
    
    

    if non_event_def is True:
        non_event_df = df[df["EVENT"] == 0]
    else:
        non_event_df = df[df[event] == 0]

    event_grouped = event_df.groupby(event_df["TIME"])[metric].mean().reset_index()
    non_event_grouped = (
        non_event_df.groupby(non_event_df["TIME"])[metric].mean().reset_index()
    )

    if surprise_split and (event == 'FOMC' or event == "ISM"):
        event_pos_df = df[(df[event] == 1) & (df[f'{event}_{surprise_col}'] == 1)]
        event_neg_df = df[(df[event] == 1) & (df[f'{event}_{surprise_col}'] == -1)]
        event_neu_df = df[(df[event] == 1) & (df[f'{event}_{surprise_col}'] == 0)]

        n_pos = len(event_pos_df['DATE'].unique())
        n_neg = len(event_neg_df['DATE'].unique())
        n_neu = len(event_neu_df['DATE'].unique())
        

        event_pos_grouped = event_pos_df.groupby(event_pos_df["TIME"])[metric].mean().reset_index()
        event_neg_grouped = event_neg_df.groupby(event_neg_df["TIME"])[metric].mean().reset_index()
        event_neu_grouped = event_neu_df.groupby(event_neu_df["TIME"])[metric].mean().reset_index()

        # Plotting
        plt.figure(figsize=(12, 6))
        bar_width = 0.15
        num_bars = len(event_grouped)
        counter = -2


        # Calculate the x-axis positions for each category
        

        
        x_event = [x + counter * bar_width for x in range(num_bars)]
        counter += 1

        plt.bar(
            x_event,
            event_grouped[metric],
            width=bar_width,
            label=f"{event} Days",
            color="blue",
        )
        
        if n_neg > 0:
            x_neg_event = [x + counter * bar_width for x in range(num_bars)]
            plt.bar(
            x_neg_event,
            event_neg_grouped[metric],
            width=bar_width,
            label=f"Negative {event} Days (n={n_neg})",
            color="red",
            )
            counter += 1
        

        if n_pos > 0:
            x_pos_event = [x + counter * bar_width for x in range(num_bars)]
            plt.bar(
                x_pos_event,
                event_pos_grouped[metric],
                width=bar_width,
                label=f"Positive {event} Days (n={n_pos})",
                color="green",
            )
            counter += 1

        

        if n_neu > 0:
            x_neu_event = [x + counter * bar_width for x in range(num_bars)]    
            plt.bar(
                x_neu_event,
                event_neu_grouped[metric],
                width=bar_width,
                label=f"Neutral {event} Days (n={n_neu})",
                color="orange",
            )
            counter += 1
        
        x_non_event = [x + counter * bar_width for x in range(num_bars)] 
        plt.bar(
            x_non_event,
            non_event_grouped[metric],
            width=bar_width,
            label=f"Non-{event} Days",
            color="cyan",
        )


    else:

        if lag_bar:

            event_lag_df = df[df[f'{event}_lag'] == 1]
            event_lag_grouped = event_lag_df.groupby(event_lag_df["TIME"])[metric].mean().reset_index()

            # Plotting
            plt.figure(figsize=(12, 6))
            bar_width = 0.3

            x_event = range(len(event_grouped))
            x_event_lag = [x - bar_width for x in x_event] 
            x_non_event = [x + bar_width for x in x_event]

            plt.bar(
                x_event,
                event_grouped[metric],
                width=bar_width,
                label=f"{event} Days",
                color="blue",
            )
            
            
            plt.bar(
                x_event_lag,
                event_lag_grouped[metric],
                width=bar_width,
                label=f"Lagged {event} Days",
                color="red",
            )
            plt.bar(
                x_non_event,
                non_event_grouped[metric],
                width=bar_width,
                label=f"Non-{event} Days",
                color="orange",
            )
        else:
            # Plotting
            plt.figure(figsize=(12, 6))
            bar_width = 0.4

            x_event = range(len(event_grouped))
            x_non_event = [x + bar_width for x in x_event]

            plt.bar(
                x_event,
                event_grouped[metric],
                width=bar_width,
                label=f"{event} Days",
                color="blue",
            )

            plt.bar(
                x_non_event,
                non_event_grouped[metric],
                width=bar_width,
                label=f"Non-{event} Days",
                color="orange",
            )

    plt.xlabel("Time")
    plt.ylabel(metric)
    plt.title(
        f"Average Value of {metric} for {ticker} on {event} and non-{event} days in sample period {start_date} - {end_date}"
    )
    # Use original x-coordinates for x-axis ticks
    plt.xticks(x_event, event_grouped["TIME"], rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()
