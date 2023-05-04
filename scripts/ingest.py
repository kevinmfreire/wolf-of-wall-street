"""
DATA INGESTION
--------------

This airflow task `ingest_data` takes the path directories from the `data/raw` path to
allow the data processing tasks efficiently take the appropriate path rather than having
to loop through the directory one at a time.  

Future work
-----------
I want to use this task to ingest data from a Cloud Service provider such as AWS or GCP
in order for anyone to access the data from a remote directory and process the data without
having to waste memory in their local system.  
"""
import os
from airflow.decorators import task

@task
def ingest_data(path_to_raw_data: str, ti):
    """
    parameters
    ----------
    path_to_raw_data : string
        The path to the raw data.
    
    returns
    -------
    ti : Task Instance
        - stocks_data : The path to the stocks .csv files
        - etfs_data : The path to the etfs .csv files
        - symbols_data : The path to the symbols metadata which includes the security name and it's symbol.
    """
    for file in os.listdir(path_to_raw_data):
        if file=='stocks':
            ti.xcom_push(key='stocks_data', value=os.path.join(path_to_raw_data,file))
        elif file=='etfs':
            ti.xcom_push(key='etfs_data', value=os.path.join(path_to_raw_data, file))
        elif file.endswith('.csv'):
            ti.xcom_push(key='symbols_data', value=os.path.join(path_to_raw_data, file))
