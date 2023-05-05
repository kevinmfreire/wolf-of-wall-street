import os
import pandas as pd
import joblib

def save_to_parquet(df, path, filename):
    """
    Saves dataframe to parquet
    """
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_parquet(os.path.join(path,filename), compression='gzip')

def load_parquet_data(path_to_file):
    """Loads parquet file into a pandas dataframe"""
    return pd.read_parquet(path_to_file)

def save_model(model, scaler, save_path, model_file_path, scaler_file_path):
    """
    Saves both the trained regressor model and the scaler model.
    """
    joblib.dump(model, os.path.join(save_path, model_file_path), compress=True)
    joblib.dump(scaler, os.path.join(save_path, scaler_file_path), compress=True)