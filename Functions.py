## This file contains all the functions used in the thesis project of Bas Beekman
from datetime import datetime, date
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

        df1 = merge_dict[key].copy()
        df2 = merger_dict[key].copy()


        if isinstance(df1[date_string][0], str):
            df1[date_string] = pd.to_datetime(df1[date_string]).dt.date

        if isinstance(df2[date_string][0], str):
            df2[date_string] = pd.to_datetime(df2[date_string]).dt.date
        
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
    display_dummy=True,
    parent_dir=None,
):
    """This function is used to get a large number of intraday plots, and either save these to a directory or plot them directly.
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
            # new_dir = event_date
            # path = os.path.join(parent_dir, new_dir)
            path = parent_dir
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
    """This function adds columns consisting of an operation on existing columns to the 'daily' dataframe
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
    dict,
    ticker,
    metric,
    start_date,
    end_date,
    event,
    non_event_def=True,
    lag_bar=False,
    lag_num = None,
    surprise_split=False,
    surprise_col=None,
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

    if surprise_split:
        event_pos_df = df[(df[event] == 1) & (df[f"{event}_{surprise_col}"] == 1)]
        event_neg_df = df[(df[event] == 1) & (df[f"{event}_{surprise_col}"] == -1)]
        event_neu_df = df[(df[event] == 1) & (df[f"{event}_{surprise_col}"] == 0)]

        n_pos = len(event_pos_df["DATE"].unique())
        n_neg = len(event_neg_df["DATE"].unique())
        n_neu = len(event_neu_df["DATE"].unique())

        event_pos_grouped = (
            event_pos_df.groupby(event_pos_df["TIME"])[metric].mean().reset_index()
        )
        event_neg_grouped = (
            event_neg_df.groupby(event_neg_df["TIME"])[metric].mean().reset_index()
        )
        event_neu_grouped = (
            event_neu_df.groupby(event_neu_df["TIME"])[metric].mean().reset_index()
        )

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
                label=f"Negative {surprise_col} {event} Days (n={n_neg})",
                color="red",
            )
            counter += 1

        if n_pos > 0:
            x_pos_event = [x + counter * bar_width for x in range(num_bars)]
            plt.bar(
                x_pos_event,
                event_pos_grouped[metric],
                width=bar_width,
                label=f"Positive {surprise_col} {event} Days (n={n_pos})",
                color="green",
            )
            counter += 1

        if n_neu > 0:
            x_neu_event = [x + counter * bar_width for x in range(num_bars)]
            plt.bar(
                x_neu_event,
                event_neu_grouped[metric],
                width=bar_width,
                label=f"Neutral {surprise_col} {event} Days (n={n_neu})",
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

            event_lag_df = df[df[f"{event}_lag{lag_num}"] == 1]
            event_lag_grouped = (
                event_lag_df.groupby(event_lag_df["TIME"])[metric].mean().reset_index()
            )

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
    ax = plt.gca()  # Get the current axis object
    return ax
    


def event_date_transformation(df, start_date: str, end_date: str):
    """Function to get dataframe in right sample period
    Parameters:
    df: dataframe to be transformed, should have a "DATE" column containing dates
    start_date (string): start of sample date in (YYYY-MM-DD) format
    end_date (string): end of sample date in (YYYY-MM-DD) format
    Returns:
    df: original df in correct sample period
    """

    df = df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]

    df.reset_index(drop=True, inplace=True)

    return df


def add_event_dummies(dict_in, df_event, event_dict, lag: int):
    """This function adds (lagged) dummy variables for specific events to each dataframe in a dictionary of dataframes.
    Paremters:
    dict_in: input dictionary containing dataframes with data to which event dummies should be added
    df_event: dataframe containing data on different events (here obtained from Bloomberg)
    event_dict: dictionary linking abbrevations for different events to their description as given in the "Event" column in the Bloomberg data
    lag (int): parameter specifying the number of lags of the dummy, input 0 if you want a dummy for the event date itself
    Returns:
    dict_in: dictionary of dataframes, where to each dataframe the event dummy variables are added
    """
    from datetime import datetime, date

    ## Lag the DATE column to match the pre-specified number of lags, makes it easier later to write general code for adding the dummies
    event_df = df_event.copy()
    event_df["DATE"] = pd.to_datetime(event_df["DATE"])

    def previous_business_day(date, lag):
        return date - pd.offsets.BusinessDay(lag)

    event_df["DATE"] = event_df["DATE"].apply(previous_business_day, lag=lag)
    event_df["DATE"] = event_df["DATE"].dt.strftime("%Y-%m-%d")
    event_df.reset_index(drop=True, inplace=True)

    
    # Loop over the dataframes in the dictionary to add event dummy variables
    for key in dict_in.keys():
        df1 = dict_in[key].copy()

        event_map = {}

        # Populate the event map with date-event pairs
        for index, row in event_df.iterrows():
            dates = row["DATE"]
            event = row["Event"]
            if dates in event_map:
                event_map[dates].append(event)
            else:
                event_map[dates] = [event]

        
        # Function to check if an event exists on a given date
        def check_event(date, event_list):
            if date in event_map and any(event in event_map[date] for event in event_list):
                return 1
            return 0
        
        # Variable to remember if date column transformation was applied
        date_trans = False

        # Check if DATE column is in correct format
        if isinstance(df1["DATE"][0], date):
            df1["DATE"] = df1["DATE"].apply(lambda x: x.strftime("%Y-%m-%d"))
            date_trans = True

        if lag == 0:
            # Loop through the dictionary with events and descriptions as in Bloomberg data
            for key2 in event_dict.keys():
                # Apply the check_event function to create dummy variables for each event
                df1[key2] = df1["DATE"].apply(
                    lambda x: check_event(x, [event_dict[key2]])
                )

            # Create a combined 'EVENT' column
            df1["EVENT"] = df1[list(event_dict.keys())].max(axis=1)

            

        else:
            # Loop through the dictionary with events and descriptions as in Bloomberg data
            for key2 in event_dict.keys():
                # Apply the check_event function to create dummy variables for each event
                df1[f"{key2}_lag{lag}"] = df1["DATE"].apply(
                    lambda x: check_event(x, [event_dict[key2]])
                )

            # Create a list of col names
            lag_col_list = []
            for key2 in event_dict.keys():
                lag_col_list.append(f"{key2}_lag{lag}")

            # Create combined 'EVENT' column
            df1[f"EVENT_lag{lag}"] = df1[lag_col_list].max(axis=1)

        # Change back the DATE column to its original format if necessary
        if date_trans:
            df1["DATE"] = pd.to_datetime(df1["DATE"], format="%Y-%m-%d").dt.date

        dict_in[key] = df1

    return dict_in

def add_surprise_dummies(dict_in, event_dict, event_df, event_list, surprise_def: str):
    """This function adds a dummy variable indicating if a certain event is a positive, negative or neutral surprise
    based on either the surprise compared to analyst forecasts or the market reaction.
    Parameters:
    dict_in: input dictionary containing dataframes with data to which surprise dummies should be added
    event_dict: dictionary linking abbrevations for different events to their description as given in the "Event" column in the Bloomberg data
    df_event: dataframe containing data on different events (here obtained from Bloomberg) 
    event_list: list of events for which surprise dummies should be added
    surprise_def (str): string indicating which definition of a surprise should be considered, should be from the following options:
        absolute: surprise is seen as the sign of the surprise variable
        (int)_stdev: positive (negative) surprise is defined as surprise larger (smaller) than (-)(int), depending on the type of macroeconomic variable
        marketfh: positive (negative) surprise is defined as a positive (negative) market reaction to a macroeconomic announcement in the half hour after the announcement
        (float)_marketfh: positive (negative) surprise is defined as a positive (negative) market reaction to a macroeconomic announcement of at least abs((float)) in the half hour after the announcement
        marketrod: positive (negative) surprise is defined as a positive (negative) market reaction to a macroeconomic announcement in the cumulative rest-of-day return after the announcement
        (float)_marketrod: positive (negative) surprise is defined as a positive (negative) market reaction to a macroeconomic announcement of at least abs((float)) in the cumulative rest-of-day return after the announcement
    """

    import re
    from datetime import datetime, date

    df_events = event_df.copy()
    ## Lists of events for which a positive surprise is seen as positive or negative respectively
    positive_event_list = ['ISM', 'NFP', 'GDP', 'IP', 'PI', 'HST']
    negative_event_list = ['FOMC', 'CPI', 'PPI']
    ## Create FOMC and ISM surprise columns based on forecaster surprise, 1: positive surprise, -1: negative surprise, 0: no surprise (median forecaster correct, or no dispersion in forecasts)
    for key in dict_in.keys():
        df = dict_in[key].copy()

        # Variable to remember if date column transformation was applied
        date_trans = False

        # Check if DATE column is in correct format
        if isinstance(df["DATE"][0], date):
            df["DATE"] = df["DATE"].apply(lambda x: x.strftime("%Y-%m-%d"))
            date_trans = True

        event_date_dict = {}
        for event in event_list:
            event_date_dict[event] = df[df[event] == 1]['DATE'].unique()

        # fomc_event_dates = df[df['FOMC'] == 1]['DATE'].unique()
        # ism_event_dates = df[df['ISM'] == 1]['DATE'].unique()

        # Function to get surprise for given date and event
        
        def get_surprise(event_df, date, event):
            try:
                return event_df[(event_df['DATE'] == date) & (event_df['Event'] == event_dict[event])]['Surprise'].iloc[0]
            except IndexError:
                return None
        
        # Function to extract number from input string (thx ChatGPT)
        def extract_number_and_string(input_string):
            # Regular expression pattern to match either an integer or a float
            pattern = r'([-+]?\d*\.\d+|\d+)_([^_]*)'

            # Search for the pattern in the input string
            match = re.search(pattern, input_string)

            if match:
                # Extract the matched number and string from the regex match
                number = match.group(1)
                string_after_number = match.group(2)
                
                # Convert the extracted number to either integer or float
                if '.' in number:
                    number = float(number)
                else:
                    number = int(number)

                return number, string_after_number
            else:
                # Return None if no match is found
                return None, None
        
        
        def get_return(df, event, date, length):
            from datetime import datetime

            interval_mapping_dict = {
                "09:30:00":"09first",
                "10:00:00":"FH",
                "10:30:00":"10first",
                "11:00:00":"10second",
                "11:30:00":"11first",
                "12:00:00":"11second",
                "12:30:00":"12first",
                "13:00:00":"12second",
                "13:30:00":"13first",
                "14:00:00":"13second",
                "14:30:00":"14first",
                "15:00:00":"14second",
                "15:30:00":"SLH",
                "16:00:00":"LH",
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


            time = event_after_release_dict[event]

            if length == 'fh':

                if 'DT' in df.columns:
                    dt = date + ' ' + time
                    timestamp = pd.to_datetime(dt)
                    return df.loc[df['DT'] == timestamp, 'RETURN'].values[0]
                else:
                    col_name = f'Return_{interval_mapping_dict[time]}'
                    return df.loc[df['DATE'] == date, col_name].values[0]
                
            elif length == 'rod':
                if 'DT' in df.columns:
                    start_index = list(interval_mapping_dict.keys()).index(time)
                    ret = 1
                    for interval in list(interval_mapping_dict.keys())[start_index:]:
                        dt = date + ' ' + interval
                        timestamp = pd.to_datetime(dt)
                        ret = ret * (1 + df.loc[df['DT'] == timestamp, 'RETURN'].values[0])
                        
                    return (ret - 1)
                
                else:
                    start_index = list(interval_mapping_dict.keys()).index(time)
                    ret = 1
                    for interval in list(interval_mapping_dict.keys())[start_index:]:

                        col_name = f'Return_{interval_mapping_dict[interval]}'
                        ret = ret * (1 + df.loc[df['DATE'] == date, col_name].values[0])

                    return (ret - 1)

            else:
                return None




        # Extract the number and string from the surprise definition, returns None if no number is in there
        surprise_def_num, surprise_def_string = extract_number_and_string(surprise_def)
        
        # Loop over events in event list for which surprise dummies must be added
        for event in event_list:
            # Loop over event dates for that particular event
            for dates in event_date_dict[event]:
                # Get the surprise value for a specific event for a specific date
                surprise = get_surprise(df_events, dates, event)
                fh_return = get_return(df, event, dates, 'fh')
                rod_return = get_return(df, event, dates, 'rod')

                # Check if there is a number in the surprise_def string, if not we just look at the string itself, 'general definitions' are implemented
                if surprise_def_num is None:

                    if surprise_def == 'absolute':

                        if surprise is not None:
                            if surprise > 0:
                                if event in positive_event_list:
                                    df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 1
                                    
                                elif event in negative_event_list:
                                    df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = -1
                            
                            elif surprise < 0:
                                if event in positive_event_list:
                                    df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = -1
                                elif event in negative_event_list:
                                    df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 1
                            else:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 0
                    elif surprise_def == 'marketfh':
                        if fh_return is not None:
                            if fh_return > 0:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 1
                            elif fh_return < 0:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = -1
                            else: 
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 0
                        
                    elif surprise_def == 'marketrod':
                        if rod_return is not None:
                            if rod_return> 0:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 1
                            elif rod_return< 0:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = -1
                            else: 
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 0
                
                # Implement number-specific logic (stdev and market return thresholds)
                else:
                    if surprise_def_string == 'stdev':
                        if surprise is not None:
                            if abs(surprise) < surprise_def_num:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 0
                            elif surprise > 0:
                                if event in positive_event_list:
                                    df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 1
                                elif event in negative_event_list:
                                    df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = -1
                            elif surprise < 0:
                                if event in positive_event_list:
                                    df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = -1
                                elif event in negative_event_list:
                                    df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 1
                            else:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 0
                    elif surprise_def_string == 'marketfh':
                        if fh_return is not None:
                            if abs(fh_return) < surprise_def_num:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 0
                            elif fh_return > 0:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 1
                            elif fh_return < 0:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = -1
                        
                    elif surprise_def_string == 'marketrod':
                        if rod_return is not None:
                            if abs(rod_return) < surprise_def_num:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 0
                            elif rod_return > 0:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = 1
                            elif rod_return < 0:
                                df.loc[df['DATE'] == dates, f'{event}_surprise_{surprise_def}'] = -1
        
        # Change back the DATE column to its original format if necessary
        if date_trans:
            df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d").dt.date

        dict_in[key] = df

    return dict_in


def create_grid_barplots(plot_title, dict_in, tickers, metrics, events, start_date, end_date, non_event_def, lag_bar, lag_num, surprise_split, surprise_col):
    from PIL import Image
    import math
    import matplotlib.pyplot as plt
    import os
    num_plots = len(tickers) * len(metrics) * len(events)
    num_rows = len(tickers)
    num_cols = math.ceil(num_plots / num_rows)  

    # Create a figure with appropriate number of subplots
    # fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

    plot_index = 0
    for i, ticker in enumerate(tickers):
        for j, metric in enumerate(metrics):
            for k, event in enumerate(events):

                # Generate the plot
                intraday_barplot(
                    dict_in,
                    ticker,
                    metric,
                    start_date,
                    end_date,
                    event,
                    non_event_def=non_event_def,
                    lag_bar=lag_bar,
                    lag_num=lag_num,
                    surprise_split=surprise_split,
                    surprise_col=surprise_col
                )
                    

                # Save the plot as an image
                plt.savefig(f"plot_{plot_index}.png")
                plt.close()
                plot_index += 1

    

    # Combine the images into a single figure
    images = [Image.open(f"plot_{i}.png") for i in range(plot_index)]
    widths, heights = zip(*(i.size for i in images))

    max_height = max(heights)
    max_width = max(widths)
    cell_width = max_width 
    cell_height = max_height 

    total_height = cell_height * num_rows

    new_im = Image.new('RGB', (max_width * num_cols, total_height))

    for i, im in enumerate(images):
    # Calculate grid position
        col_index = i % num_cols
        row_index = i // num_cols

        # Calculate paste coordinates
        paste_x = col_index * cell_width
        paste_y = row_index * cell_height

        # Paste image onto new_im
        new_im.paste(im, (paste_x, paste_y))

    # Save the image
    save_path = r'C:/Users/ROB7831/OneDrive - Robeco Nederland B.V/Documents/Thesis/Plots/Combined Plots/'
    new_im.save(f'{save_path}{plot_title}.png')
    # Display the combined image
    # new_im.show()
    for i in range(plot_index):
        os.remove(f"plot_{i}.png")

    return new_im



# def scale_vars(df_in, vars_to_scale, inplace=True):
#     """Function used to scale a specified set of columns in a dataframe using sklearn StandardScaler
#     Parameters:
#     df_in: dataframe containing columns which need to be scaled
#     vars_to_scale: list of column names specifying which columns need to be scaled
#     Returns:
#     df: original dataframe with specified columns scaled
#     """
#     import pandas as pd
#     from sklearn.preprocessing import StandardScaler 

#     df = df_in.copy()
#     scaler = StandardScaler()

#     scaled_df = pd.DataFrame(scaler.fit_transform(df[vars_to_scale]), columns=vars_to_scale)
#     if inplace:
#         df[vars_to_scale] = scaled_df[vars_to_scale]
#     else:
#         for var in vars_to_scale:
#             df[f'{var}_scaled'] = scaled_df[var]

            
#     return df




def do_regression(df_in, indep_vars, dep_vars, cov_type, max_lag = 1):
    import statsmodels.api as sm
    results_dict = {}
    df = df_in.copy()

    for dep_var in dep_vars:

        df.dropna(subset=indep_vars+[dep_var], inplace=True)
        x = df[indep_vars]
        y = df[dep_var]

        x = sm.add_constant(x)

        model =sm.OLS(y, x)
        if cov_type == 'HAC':
            results_dict[dep_var] = model.fit(cov_type=cov_type, cov_kwds={'maxlags': max_lag})
        else:
            results_dict[dep_var] = model.fit(cov_type=cov_type)

    return results_dict






def get_latex_table(result_dict, dep_vars, indep_vars):
    
    import pandas as pd

    result_df = pd.DataFrame(index = result_dict.keys(), columns= indep_vars + ['R-squared'])
    for name in result_dict.keys():
        results = result_dict[name][dep_vars[0]]
        for var in indep_vars:
            coef = results.params[var]
            t_stats = results.tvalues[var]

            p_value = results.pvalues[var]
            if p_value < 0.01:
                significance = "***"
            elif 0.01 <= p_value < 0.05:
                significance = "**"
            elif 0.05 <= p_value < 0.1:
                significance = "*"
            else:
                significance = ""
            
            result_df.loc[name, var] = f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{coef:.4f}{significance} \\\ ({t_stats:.4f})\\end{{tabular}}"
        
        result_df.loc[name, 'R-squared'] = f"{results.rsquared:.4f}"
        

    
    result_df.columns = result_df.columns.str.replace('_', '\\_')
    result_df = result_df.T
    

    latex_table = result_df.to_latex(caption=f'Table with regression results of {dep_vars}. T-statistics are between brackets. *, ** and *** indicate significance at 10, 5 and 1\% respectively.')
    
    return latex_table, result_df
    


def show_reg_results(result_dict, ticker=None):
    if ticker is None:
        for ticker in result_dict.keys():
            for var in result_dict[ticker].keys():

                summary = result_dict[ticker][var].summary()
                print(f'Regression results for dependent variable {var} and {ticker}: {summary}')
    else:
        for var in result_dict[ticker].keys():

            summary = result_dict[ticker][var].summary()
            print(f'Regression results for dependent variable {var} and {ticker}: {summary}')

def add_lag(df_in, var, lag_num):
    df = df_in.copy()
    df[f'{var}_lag{lag_num}'] = df[var].shift(lag_num)

    return df

import pandas as pd
from sklearn.base import BaseEstimator

class WalkForwardTransformer:
    def __init__(self, transformer: BaseEstimator, window_size: int, method: str = '<=t', interval_col: str = None, rolling: bool = False):
        self.transformer = transformer
        self.method = method
        self.window_size = window_size
        self.interval_col = interval_col
        self.rolling = rolling

    def generate_walkforward_chunks(self, X: pd.DataFrame):
        if self.interval_col:
            grouped = X.groupby(self.interval_col)

            for name, group in grouped:
                group = group.drop(columns=self.interval_col)
                if self.rolling:
                    start_index = max(0, self.window_size)
                    for i in range(start_index, len(group)):
                        yield group.iloc[i - self.window_size:i + 1], name
                else:
                    start_index = max(0, self.window_size)
                    for i in range(start_index, len(group)):
                        yield group.iloc[:i + 1], name
        else:
            if self.rolling:
                start_index = max(0, self.window_size)
                for i in range(start_index, len(X)):
                    yield X.iloc[i - self.window_size:i + 1], None
            else:
                start_index = max(0, self.window_size)
                for i in range(start_index, len(X)):
                    yield X.iloc[:i + 1], None


    def transform(self, X: pd.DataFrame, verbose: int = 0):
        if self.interval_col:
            ix = X.groupby(self.interval_col).apply(lambda df: df.index[self.window_size:]).explode()
            # Ensure ix is a Series before attempting to drop levels
            if isinstance(ix, pd.Series) and ix.index.nlevels > 1:
                ix = ix.droplevel(0)
        else:
            ix = X.index[self.window_size:]

        Xgen = self.generate_walkforward_chunks(X)
        Z = []

        for i, (Xi, interval) in enumerate(Xgen):
            if self.method == '<t':
                self.transformer.fit(Xi.iloc[:-1])
            elif self.method == '<=t':
                self.transformer.fit(Xi)
            else:
                raise NotImplementedError(self.method)
            
            Xil = Xi.iloc[[-1]]
            # Reshape Xil to 2D array if it is a single row
            Xil_reshaped = Xil.values.reshape(1, -1)
            Zil = self.transformer.transform(Xil_reshaped)
            Z.append(Zil.tolist()[0])

            if verbose == 1:
                print('Progress: %0.2f%%' % ((i + 1) * 100. / (len(X) - self.window_size)), end='\r')

        Z = pd.DataFrame(Z, index=ix, columns=[col for col in X.columns if col != self.interval_col])
        return Z


def scale_vars_exp_window(df_in: pd.DataFrame, scale_vars: list, scaler: BaseEstimator, window_size: int, method: str = "<=t", interval_col: str = None, inplace: bool = False, rolling: bool = False) -> pd.DataFrame:
    df = df_in.copy()
    wft = WalkForwardTransformer(scaler, window_size=window_size, method=method, interval_col=interval_col, rolling=rolling)
    
    if interval_col:
        scale_vars += [interval_col]
    
    
    df_scaled = wft.transform(df[scale_vars], verbose=1)

    if interval_col:
        scale_vars.remove(interval_col)

    if rolling:
        col_string = 'rolling_window'
    else:
        col_string = 'expanding_window'


    if inplace:
        df[scale_vars] = df_scaled
    else:
        for var in scale_vars:
            if interval_col:
                df[f'{var}_scaled_{col_string}_interval_{window_size}days'] = df_scaled[var]
            else:
                df[f'{var}_scaled_exp_{col_string}_{window_size}'] = df_scaled[var]

    return df


def get_extreme_values(df_in, column_name, percentile, direction):


    # number of observations related to percentile
    n = int(len(df_in) * (percentile / 100))
    if direction == "pos":
        indices = df_in[column_name].nlargest(n).index
        dates = df_in.loc[indices, 'DT']
        values = df_in[column_name].nlargest(n).values
        return dates.tolist(), values.tolist()
    
    elif direction == "neg":
        indices = df_in[column_name].nsmallest(n).index
        dates = df_in.loc[indices, 'DT']
        values = df_in[column_name].nsmallest(n).values
        return dates.tolist(), values.tolist()
    
    elif direction == "abs":
        indices = df_in[column_name].abs().nlargest(n).index
        dates = df_in.loc[indices, 'DT']
        values = df_in[column_name].abs().nlargest(n).values
        return dates.tolist(), values.tolist()
    else:
        print("Incorrect direction argument passed, please pass:  pos, neg or abs.")
        return None


def add_future_ret(df_in, count, interval_type, EOD_dummy=False, start_lead=0):
    """
    Function to add column with future cumulative returns for pre-specified number of days and specification for EOD or not. Here if EOD is true, the cumulative returns 
    until EOD and then a full day will be calculated. The start_lead variable specifies if the cumulative return is calculated from the time of the respective row, or if a lag is
    introduced, so e.g. starting from 1 half hour later. 
    
    """
    
    df = df_in.copy()
    # Determine number of intervals in one day
    if interval_type == 'days':    
        interval_size = len(df['TIME'].unique())
    elif interval_type == 'halfhours':
        interval_size = 1
    else:
        print("Unknown interval length, please input 'days' or 'halfhours'.")
        return None

    # Number each interval
    interval_to_num = {interval: i for i, interval in enumerate(df['TIME'].unique())}

    # Map numbers to intervals, and calculate interval distance to end of day
    df['TIME_num'] = df['TIME'].map(interval_to_num)
    df['EOD_diff'] = interval_size - 1 - df['TIME_num']

    # Get the total difference in rows between the data until which cumulative return is calculated and current date

    if EOD_dummy:
        df['TIME_diff'] = df['EOD_diff'] + interval_size*count
        for index, row in df.iterrows():
            df.at[index, f'future_price_{count}{interval_type}_EOD'] = df['PRICE'].shift(-row['TIME_diff']).loc[index]

        df[f'future_ret_{count}{interval_type}_EOD'] = df[f'future_price_{count}{interval_type}_EOD'] / df['PRICE'] -1

        if start_lead != 0:
            df[f'future_ret_{count}{interval_type}_EOD_lead{start_lead}'] = df[f'future_ret_{count}{interval_type}_EOD'].shift(-start_lead)

        df = df.drop(columns=[f'future_price_{count}{interval_type}_EOD'])
        
    else:
        df['TIME_diff'] = interval_size * count
        for index, row in df.iterrows():
            df.at[index, f'future_price_{count}{interval_type}'] = df['PRICE'].shift(-row['TIME_diff']).loc[index]

        df[f'future_ret_{count}{interval_type}'] = df[f'future_price_{count}{interval_type}'] / df['PRICE'] -1

        if start_lead != 0:
            df[f'future_ret_{count}{interval_type}_lead{start_lead}'] = df[f'future_ret_{count}{interval_type}'].shift(-start_lead)
        
        df = df.drop(columns=[f'future_price_{count}{interval_type}'])
    
    df = df.drop(columns=['TIME_num', 'EOD_diff', 'TIME_diff'])

    return df


def create_peak_dummies(df_in, level, col_name, percentile=True, threshold_direction='abs'):
    """
    level: level of the threshold, refers to percentage point percentil if percentile is True, otherwise absolute level of the threshold
    col_name: name of the column for which the dummy should be created
    percentile: dummy variable determining type of threshold: percentile or absolute. Percentile returns dummy variables for the x% percentile, absolute returns
    dummy variables for an absolute threshold level
    abs: can take (abs, pos, neg), determines if spikes in absolute value, only positive or only negative spikes should be taken into account
    """
    def get_extreme_values(df_in, column_name, percentile, direction):


        # number of observations related to percentile
        n = int(len(df_in) * (percentile / 100))
        if direction == "pos":
            indices = df_in[column_name].nlargest(n).index
            dates = df_in.loc[indices, 'DT']
            values = df_in[column_name].nlargest(n).values
            return dates.tolist(), values.tolist()
        
        elif direction == "neg":
            indices = df_in[column_name].nsmallest(n).index
            dates = df_in.loc[indices, 'DT']
            values = df_in[column_name].nsmallest(n).values
            return dates.tolist(), values.tolist()
        
        elif direction == "abs":
            indices = df_in[column_name].abs().nlargest(n).index
            dates = df_in.loc[indices, 'DT']
            values = df_in[column_name].abs().nlargest(n).values
            return dates.tolist(), values.tolist()
        else:
            print("Incorrect direction argument passed, please pass: 'highest' or 'lowest'.")
            return None
    
    df = df_in.copy()

    if percentile:
        date_list, value_list = get_extreme_values(df, col_name, level, threshold_direction)
        df[f'{level}percentile_{threshold_direction}_dummy'] = df['DT'].isin(date_list).astype(int)
    
    else:
        if threshold_direction == 'abs':

            df[f'{level}threshold_{threshold_direction}_dummy'] = (df[col_name].abs() > level).astype(int)
        elif threshold_direction == 'pos':
            df[f'{level}threshold_{threshold_direction}_dummy'] = (df[col_name] > level).astype(int)
        
        elif threshold_direction == 'neg':
            df[f'{level}threshold_{threshold_direction}_dummy'] = (df[col_name] < -level).astype(int)
    
    return df







def convert_variable_label(variable_name):
    import re
    import math

    label_dict = {
     "PRICE/NAV_pct_lag14" : r"$\beta^{PRICE/NAV \%}_{t-1}$",
     "cum_ret_scaled_exp" : r"$\beta^{cum\_return}_{t, 1\text{ \textit{to} }k-1}$",
     "vol_index" : r"$\beta^{VIX/MOVE}_{t-1}$",
     "prev_day_rv_Average_5day_scaled_exp" : r"$\beta^{\Bar{RV}_5day}_{t-1}$",
     "Volume_scaled_exp" : r"$\beta^{Volume}_{t, k}$",
    }
    label = ""
    if variable_name in label_dict.keys():
        label = label_dict[variable_name]

    

    # Check if it's a future_ret variable
    future_ret_match = re.match(r"future_ret_(\d+)([a-zA-Z_]*?)(?:_(\w+))?$", variable_name)
    if future_ret_match:
        num = int(future_ret_match.group(1))
        time_unit = future_ret_match.group(2)
        eod = future_ret_match.group(3)

        if time_unit == "days":
            if eod == "EOD":
                if num == 0:
                    label = f"$R_{{t, 14}}$"
                else: 
                    label = f"$R_{{t+{num}, 14}}$" 
            else:
                if num == 0:
                    label = f"R_{{t, k}}$"
                else:
                    label = f"$R_{{t+{num}, k}}$"
            
        elif time_unit == "halfhours":
            if eod == "EOD":
                if num == 0:
                    label = f"$R_{{t, 14}}$"
                else:
                    label = f"$R_{{t + {math.ceil(num / 14)}, {num % 14}}}$"
            else:
                if num == 0:
                    label = f"$R_{{t, k}}$"
                elif num < 14:
                    label = f"$R_{{t, k + {num % 14}}}$"
                else:
                    if (num % 14) == 0:
                        label = f"$R_{{t + {math.ceil(num / 14) - 1}, k}}$"
                    else:
                        label = f"$R_{{t + {math.ceil(num / 14) - 1}, k + {num % 14}}}$"

                
    else:
        # Check if it's a percentile_pos or percentile_neg variable
        percentile_match = re.match(r"(\d*\.?\d+)percentile_(pos|neg)_dummy", variable_name)
        if percentile_match:
            percentile = percentile_match.group(1)
            sign = "+" if percentile_match.group(2) == "pos" else "-"
            label = f"$\\beta_{{\\mathds{{1}}^{{{sign}}}_{{P_{{{percentile}}}}}}}$"
        
    
    return label




def get_latex_table_stacked(result_dict, dep_vars, indep_vars, include_r_squared):
    import pandas as pd

    all_rows = []

    for dep_var in dep_vars:
        dep_var_label = convert_variable_label(dep_var)

        # Add dependent variable name to the top left
        all_rows.append(f'\\multicolumn{{{len(result_dict) + 1}}}{{l}}{{{dep_var_label}}} \\\\ \\hline')

        headers = [''] + list(result_dict.keys())
        all_rows.append(' & '.join(headers) + ' \\\\ \\hline')

        for var in indep_vars:
            coef_values = []
            t_stat_values = []

            for ticker, ticker_results in result_dict.items():
                results = ticker_results.get(dep_var, None)
                if results is not None:
                    coef = results.params[var]
                    t_stats = results.tvalues[var]
                    p_value = results.pvalues[var]
                    if p_value < 0.01:
                        significance = "***"
                    elif 0.01 <= p_value < 0.05:
                        significance = "**"
                    elif 0.05 <= p_value < 0.1:
                        significance = "*"
                    else:
                        significance = ""
                    coef_values.append(f"{coef:.4f}{significance}")
                    t_stat_values.append(f"({t_stats:.4f})")
                else:
                    coef_values.append("")
                    t_stat_values.append("")

            # Add coefficients row
            row_values = [convert_variable_label(var)] + coef_values
            all_rows.append(' & '.join(row_values) + ' \\\\')

            # Add t-statistics row
            row_values = [""] + t_stat_values
            all_rows.append(' & '.join(row_values) + ' \\\\')
        
        # Add R-squared values if necessary
        if include_r_squared:
            rsquared_values = [f'{result_dict[ticker].get(dep_var, "").rsquared:.4f}' for ticker in result_dict]
            all_rows.append(f'$R^2$ & {" & ".join(rsquared_values)} \\\\')
            
        # Add an empty line between panels
        all_rows.append("\\hline \n \\multicolumn{" + str(len(result_dict) + 1) + "}{l}{} \\\\")

        

    latex_table = "\n".join(all_rows)
    caption = f"This table shows regression results of future cumulative returns from $R_{{t, k}}$ for different window lengths on dummy variables indicating whether abnormal short volume at time t, interval k breaches the interval $P_{{i}}$. *, ** and *** indicate significance at the 10\%, 5\% and 1\% level respectively, t-statistics are indicated below the coefficients."
    latex_table = f"\\begin{{table}}[H]\n\\centering\n\\caption{{{caption}}}\n\\begin{{tabular}}{{l{'c' * len(result_dict.keys())}}}\n\\hline\n{latex_table}\n\\end{{tabular}}\n\\end{{table}}"
    
    return latex_table

def do_panel_regression(data_dict, dep_vars, indep_vars, entity_effects=False, time_effects=False, cov_type='robust', time=None):
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
    combined_df = pd.concat([df.assign(ticker=ticker) for ticker, df in data_dict.items()])

    if time:
        combined_df = combined_df[combined_df['TIME'] == time]
    # Set multi-index, to allow for panel regression
    combined_df.set_index(['ticker', 'DT'], inplace=True)

    y = combined_df[dep_vars]
    x = combined_df[indep_vars]
    model = PanelOLS(y, x, entity_effects=entity_effects, time_effects=time_effects)

    results = model.fit(cov_type=cov_type)

    return results


def do_panel_regression_alt(df, dep_vars, indep_vars, entity_effects=False, time_effects=False, cov_type='robust'):
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
    combined_df['TIME'] = pd.to_datetime(combined_df['TIME'])
    # Set multi-index, to allow for panel regression
    combined_df.set_index(['TIME', 'DT'], inplace=True)

    y = combined_df[dep_vars]
    x = combined_df[indep_vars]
    model = PanelOLS(y, x, entity_effects=entity_effects, time_effects=time_effects)

    results = model.fit(cov_type=cov_type)

    return results


def define_mappings():
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

    return etf_dict, event_dict, suffix_list, interval_mapping_dict, event_after_release_dict, code_dictionary


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
    import matplotlib.pyplot as plt
    code_dictionary = define_mappings()[5]
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


# Do plots
def plot_time_series(df: pd.DataFrame, date_col: str, value_col: str):
    import pandas as pd
    import matplotlib.pyplot as plt
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort the DataFrame by the date column
    df = df.sort_values(by=date_col)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df[date_col], df[value_col], marker='o', linestyle='-')
    
    # Set the plot title and labels
    plt.title(f'Time Series Plot of {value_col} Over Time')
    plt.xlabel('Date')
    plt.ylabel(value_col)
    
    # Rotate the date labels for better readability
    plt.xticks(rotation=45)
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def add_daily_sums(interval_df: pd.DataFrame, interval_date_col: str, interval_value_col: str, 
                daily_df: pd.DataFrame, daily_date_col: str, output_col_name: str) -> pd.DataFrame:
    
    import pandas as pd
    
    # Ensure the date columns are in datetime format
    interval_df[interval_date_col] = pd.to_datetime(interval_df[interval_date_col])
    daily_df[daily_date_col] = pd.to_datetime(daily_df[daily_date_col])
    
    # Group the interval DataFrame by the date column and sum the values for each day
    daily_sums = interval_df.groupby(interval_date_col)[interval_value_col].sum().reset_index()
    daily_sums.columns = [daily_date_col, output_col_name]
    
    # Merge the daily DataFrame with the aggregated sums
    result_df = pd.merge(daily_df, daily_sums, left_on=daily_date_col, right_on=daily_date_col, how='left')
    
    return result_df

    



def calculate_daily_return(interval_df: pd.DataFrame, interval_date_col: str, interval_time_col: str, interval_value_col: str, 
                           daily_df: pd.DataFrame, daily_date_col: str) -> pd.DataFrame:
    
    import pandas as pd
    # Ensure the date and time columns are in datetime format
    interval_df[interval_date_col] = pd.to_datetime(interval_df[interval_date_col])
    # interval_df[interval_time_col] = pd.to_datetime(interval_df[interval_time_col], format='%H:%M').dt.time
    daily_df[daily_date_col] = pd.to_datetime(daily_df[daily_date_col])
    
    # Filter the interval DataFrame to only include rows where the time is 16:00
    interval_df_16 = interval_df[interval_df[interval_time_col] == '16:00:00']
    interval_df_0930 = interval_df[interval_df[interval_time_col] == '09:30:00']
    
    # Merge the filtered interval DataFrame with the daily DataFrame on the date column
    merged_df = pd.merge(daily_df, interval_df_16[[interval_date_col, interval_value_col]], 
                         left_on=daily_date_col, right_on=interval_date_col, how='left', suffixes=('', '_16'))
    merged_df = pd.merge(merged_df, interval_df_0930[[interval_date_col, interval_value_col]], 
                         left_on=daily_date_col, right_on=interval_date_col, how='left', suffixes=('', '_0930'))
    
    # Sort the DataFrame by the date column to ensure correct calculation of returns
    merged_df = merged_df.sort_values(by=daily_date_col)
    
    # Calculate the return values
    merged_df['Return_close_close'] = (merged_df[interval_value_col] / merged_df[interval_value_col].shift(1)) - 1
    merged_df['Return_open_close'] = (merged_df[interval_value_col] / merged_df[interval_value_col + '_0930']) - 1
    
    return merged_df