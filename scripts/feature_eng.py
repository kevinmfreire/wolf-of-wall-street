import os
import time
import logging

from utils import save_to_parquet, load_parquet_data

def extract_new_features(df):
    """
    Extracts features from original dataset.

    Parameters
    ----------
    df : Pandas Data Frame

    Returns
    -------
    df : A new Data Frame with added features.
            - vol_moving_avg
            - adj_close_rolling_med
    """
    df = df.assign(vol_moving_avg=df.groupby('Symbol')['Volume'].transform(lambda x: x.ewm(span=30, adjust=False).mean()),
                   adj_close_rolling_med=df.groupby('Symbol')['Adj Close'].transform(lambda x: x.rolling(window=30, min_periods=1).median()))
    
    return df

if __name__ == '__main__':
    processed_data = '/opt/airflow/data/processed/stock-market-prices.parquet.gzip'
    feature_extraction_data_path = '/opt/airflow/data/feature_data/'
    feature_data_file = '30-day-window-stock-market-prices.parquet.gzip'
    log_filename = '/opt/airflow/logs/pipeline_execution.log'

    logging.basicConfig(filename=log_filename, level=logging.INFO)
    logging.info('Starting Feature Engineering Task')

    df_processed = load_parquet_data(processed_data)
    t1 = time.time()
    df_features = extract_new_features(df_processed)
    t_delta = (time.time() - t1)/60
    logging.info(f'Feature Engineering Task completed in {t_delta} minutes.')
    logging.info(f'Saving data to {os.path.join(feature_extraction_data_path, feature_data_file)}')
    save_to_parquet(df_features, feature_extraction_data_path, feature_data_file)
    logging.info('Data saved.')