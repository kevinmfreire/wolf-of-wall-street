from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash_operator import BashOperator    
from airflow.utils.dates import days_ago

# from scripts.ingest import ingest_data
# from scripts.data_processing import *
# from scripts.train import train

default_args = {
    'owner': 'Kevin Freire',
    'start_date': days_ago(1)
}

# Define the DAG
with DAG(
    dag_id='stock_market_pipeline_v1.1',
    default_args=default_args,
    schedule_interval=None,
    ) as dag:

    data_processing = BashOperator(
        task_id='data_processing',
        bash_command='python3 /opt/airflow/scripts/data_processing.py'
    )

    extract_features = BashOperator(
        task_id='extract_features',
        bash_command='python3 /opt/airflow/scripts/feature_eng.py'
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command='python3 /opt/airflow/scripts/train.py'
    )

    data_processing >> extract_features >> train_model
