"""
MODEL TRAINING
--------------

The Airflow task takes the values from the `extract_new_features` task
that returns a pandas dataframe that has been json serialized.  It then
gets deserialized into a pandas dataframe to extract the feature values
and the target value.  In order for the RandomForrestRegressor to train
the features must first be normalized to improve model performance by 
ensuring that features are on the same scale (0-1).

The MinMaxScaler from scikit-learn was used to preserve the shape of the
original distribution data.  The scaler is designed to work well with both
training and unseen data, which is important for the accuracy and reliability
of machine learning models.

I used the RandomForrestRegressor because is well suited for modelling 
non-linear data because it is able to capture complex relationships between
features and target values (ie. stock prices, trading values, etc.)
"""
import pandas as pd
import logging
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from airflow.decorators import task

@task
def train(log_filename: str , model_filename: str, scaler_filename: str, ti):
    """
    parameters
    ----------
    log_file_name : string
        The path to predictions.log

    model_filename : string
        The path and filename to saving trained model.

    scaler_filename : string
        The path and file name to saving scaler model.

    ti : Task Instance
        The task intance that allows the train task perform once a previous task
        completes.
    
    returns
    -------
    model : Saves the trained model to a pcikle file.
    scaler : Saves the trainined scaler for normalizing unseen data.
    """
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    data = pd.read_json(ti.xcom_pull(task_ids='extract_features', key='features_df'))
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data.dropna(inplace=True)

    features = ['vol_moving_avg', 'adj_close_rolling_med']
    target = 'Volume'

    scaler = MinMaxScaler()

    X = data[features]
    X = scaler.fit_transform(X)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    logging.info(f'Model Performance Metrics: Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    logging.info(f'Model Prediction Metrics: Mean Absolute Error: {mae}, Mean Squared Error: {mse}')
    logging.info(f'Saving Models.')

    joblib.dump(model, model_filename, compress=True)
    joblib.dump(scaler, scaler_filename, compress=True)

    logging.info(f'RandomForrestRegressor Model saved to {model_filename}')
    logging.info(f'MinMaxScaler Model saved to {scaler_filename}')
 
    logging.shutdown()