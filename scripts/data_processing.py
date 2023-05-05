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
import logging
import time

from utils import save_to_parquet

def process_data(raw_data_path):
    """
    Processes the raw data.

    Parameters
    ----------
    raw_data_path : Path to raw data.

    Returns
    -------
    df_merged : A merged dataframe from different datasources. 
    """
    files = os.listdir(raw_data_path)
    df_market = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Symbol'])
    for file in files:
        if file.endswith('.csv'):
            df_symbols = pd.read_csv(os.path.join(raw_data_path, file))[['Symbol', 'Security Name']]
        else:
            subdir = os.path.join(raw_data_path, file)
            for csv_file in os.listdir(subdir):
                if csv_file.endswith('.csv'):
                    symbol = csv_file.strip('.csv')
                    df_temp = pd.read_csv(os.path.join(subdir,csv_file))
                    df_temp['Symbol'] = symbol
                    df_market = pd.concat([df_market, df_temp], axis=0, ignore_index=True)
    df_merged = pd.merge(df_symbols, df_market)
    return df_merged
if __name__ == '__main__':
    
    log_filename = '/opt/airflow/logs/pipeline_execution.log'
    raw_path = '/opt/airflow/data/raw/'
    processed_data_path = '/opt/airflow/data/processed/'
    filename = 'stock-market-prices.parquet.gzip'

    logging.basicConfig(filename=log_filename, level=logging.INFO)
    logging.info(f'Starting Data Processing Task')
    logging.info(f'Starting Data Processing Task...')
    t1 = time.time()
    df_final = process_data(raw_path)
    t2 = time.time()
    t_delta = (t2-t1)/60
    logging.info(f'Data Processing completed in {t_delta} minutes')
    logging.info('Saving data')
    save_to_parquet(df_final, processed_data_path, filename)
    logging.info(f'Original data saved to {os.path.join(processed_data_path, filename)}')
    logging.shutdown()