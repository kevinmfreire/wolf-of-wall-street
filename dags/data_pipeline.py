from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from scripts.ingest import ingest_data
from scripts.data_processing import *
from scripts.train import train

default_args = {
    'owner': 'Kevin Freire',
    'start_date': days_ago(1)
}

# Define the DAG
with DAG(
    dag_id='stock_market_pipeline_v21',
    default_args=default_args,
    schedule_interval=None,
    ) as dag:

    data_ingest = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
        op_kwargs={'path_to_raw_data':'data/raw/'}
    )

    symbols_data_processing = PythonOperator(
        task_id='process_symbols',
        python_callable=process_symbols_data
    )

    etfs_data_processing = PythonOperator(
        task_id='process_etfs',
        python_callable=process_etfs_data
    )

    stocks_data_processing = PythonOperator(
        task_id='process_stocks',
        python_callable=process_stocks_data
    )

    merge_processed_data = PythonOperator(
        task_id='merge_data',
        python_callable=merge_data
    )

    save_processed_data = PythonOperator(
        task_id='save_processed_data',
        python_callable=save_to_parquet,
        op_kwargs={'path': 'data/processed/',
                   'filename': 'stock-market-prices.parquet.gzip',
                   'task_id': 'merge_data',
                   'key': 'df_merged'
                   }
    )

    extract_features = PythonOperator(
        task_id='extract_features',
        python_callable=extract_new_features
    )

    save_feature_data = PythonOperator(
        task_id='save_feature_data',
        python_callable=save_to_parquet,
        op_kwargs={'path': 'data/feature_data/',
                   'filename': '30-day-window-stock-market-prices.parquet.gzip',
                   'task_id': 'extract_features',
                   'key': 'features_df'
                   }
    )

    train_model = PythonOperator(
        task_id='train_save_model',
        python_callable=train,
        op_kwargs={'log_filename': 'logs/predictions.log',
                   'model_filename': 'data/volume_rf_reg.pkl',
                   'scaler_filename': 'data/scaler_model.bin'
                   }
    )

    data_ingest >> [symbols_data_processing, stocks_data_processing]
    
    stocks_data_processing >> etfs_data_processing

    [symbols_data_processing, etfs_data_processing] >> merge_processed_data

    merge_processed_data >> [save_processed_data, extract_features]

    extract_features >> [save_feature_data, train_model]
