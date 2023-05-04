"""
DATA PROCESSING
---------------
The `data_processing.py` script provides several airflow tasks to process the stock-market-dataset.
The goal initially was to run the data processing task in parallel, however there was insufficient RAM.
Each task was divided into sub-tasks in order to visualize the data pipeline with Airflow. Each task 
receives or passes a task intance which are dependent on one another.  The symbols, etfs and stocks data
are received from the `ingest.py` script (task) and the output is used to merge the desired data structure.
As shown below:

                Symbol: string
                Security Name: string
                Date: string (YYYY-MM-DD)
                Open: float
                High: float
                Low: float
                Close: float
                Adj Close: float
                Volume: int

Then utilizing the data, I extract additional features such as the vol_moving_avg (volume moving average) by
leveraging pandas groupby() to group the data based on the symbols and utilized a 30 day window exponential 
moving average which returns a more precise average than the simple moving average and we do not loose data 
(first 30 days).  I had a similar approach for the adj_close_rolling_med (adjacent rolling median), however
I leveraged the pandas rolling() with a window of 30 days to extract the median for each value.  For the 
rolling median I used the min_periods parameter and set it to one in order to onle have one non-NA observation 
to work with. This is important because if min_periods is set to a value greater than 1 and there are missing 
values within the rolling window, the rolling statistic for that window will also be missing. This can result 
in incomplete or inaccurate results.
"""
import os
import pandas as pd
from airflow.decorators import task

@task
def process_symbols_data(ti):
    """
    Processes the symbols metadata.

    Parameters
    ----------
    ti : Task instance
        Path to raw data extracted from ingest task.

    Returns
    -------
    ti : Task Instance.
        Json serialized pandas dataframe. 
    """
    symbols_meta = ti.xcom_pull(task_ids='ingest_data', key='symbols_data')
    df_symbols = pd.read_csv(symbols_meta)[['Symbol', 'Security Name']]
    ti.xcom_push(key='symbols_df', value=df_symbols.to_json())

@task
def process_etfs_data(ti):
    """
    Processes the etfs data.

    Parameters
    ----------
    ti : Task instance
        Path to raw data extracted from ingest task.

    Returns
    -------
    ti : Task Instance.
        Json serialized pandas dataframe. 
    """
    etfs_dir = ti.xcom_pull(task_ids='ingest_data', key='etfs_data')
    df_market = pd.read_json(ti.xcom_pull(task_ids='process_stocks', key='market_df'))
    for csv_file in os.listdir(etfs_dir):
        if csv_file.endswith('.csv'):
            symbol = csv_file.strip('.csv')
            df_temp = pd.read_csv(os.path.join(etfs_dir,csv_file))
            df_temp['Symbol'] = symbol
            df_market = pd.concat([df_market, df_temp], axis=0, ignore_index=True)
    ti.xcom_push(key='market_df', value=df_market.to_json())

@task
def process_stocks_data(ti):
    """
    Processes the stocks data.

    Parameters
    ----------
    ti : Task instance
        Path to raw data extracted from ingest task.

    Returns
    -------
    ti : Task Instance.
        Json serialized pandas dataframe. 
    """
    stocks_dir = ti.xcom_pull(task_ids='ingest_data', key='stocks_data')
    df_market = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Symbol'])
    for csv_file in os.listdir(stocks_dir):
        if csv_file.endswith('.csv'):
            symbol = csv_file.strip('.csv')
            df_temp = pd.read_csv(os.path.join(stocks_dir,csv_file))
            df_temp['Symbol'] = symbol
            df_market = pd.concat([df_market, df_temp], axis=0, ignore_index=True)
    ti.xcom_push(key='market_df', value=df_market.to_json())

@task
def merge_data(ti):
    """
    Merge data.

    Parameters
    ----------
    ti : Task instance(s)
        A json serialized dataframe for etfs plus stocks dataframe, and the symbols dataframe.

    Returns
    -------
    ti : Task Instance.
        Json serialized pandas dataframe for merged data. 
    """
    df_market = pd.read_json(ti.xcom_pull(task_ids='process_etfs', key='market_df'))
    symbols = pd.read_json(ti.xcom_pull(task_ids='process_symbols', key='symbols_df'))
    df_merged = pd.merge(symbols, df_market)
    ti.xcom_push(key='df_merged', value=df_merged.to_json())

@task
def extract_new_features(ti):
    """
    Extract features from original data.

    Parameters
    ----------
    ti : Task instance(s)
        A json serialized dataframe for original dataframe.

    Returns
    -------
    ti : Task Instance.
        Json serialized pandas dataframe for new feature dataset. 
    """
    df = pd.read_json(ti.xcom_pull(task_ids='merge_data', key='df_merged'))
    df = df.assign(vol_moving_avg=df.groupby('Symbol')['Volume'].transform(lambda x: x.ewm(span=30, adjust=False).mean()),
                   adj_close_rolling_med=df.groupby('Symbol')['Adj Close'].transform(lambda x: x.rolling(window=30, min_periods=1).median()))
    ti.xcom_push(key='features_df', value=df.to_json())

@task
def save_to_parquet(path, filename, task_id, key, ti):
    """
    Saves dataframe to parquet format and saves as compressed 'gzip' file.

    Parameters
    ----------
    path : string
        Path to save.

    filename : string
        Name of file.

    task_id : string
        Task id which to retrieve data from.

    key : string
        Specified key from the task_id.

    """
    df = pd.read_json(ti.xcom_pull(task_ids=task_id, key=key))
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_parquet(os.path.join(path,filename), compression='gzip')