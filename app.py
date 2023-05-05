import joblib
import streamlit as st
import boto3

class RandomForrestReg():
    """
    Main Class for serving the Machine Learning Model

    Attributes
    ----------
    model_file: string
        The path to saved model.
    scaler_file : string
        The Path to saved scaler model for normalizing input data.
    model : Object
        The Random Forrest Regressor model.
    scaler : Object
        The Scaler model.

    Methods
    -------
    Predict()
        It normalizes input data (vol_moving_avg, adj_close_rolling_med) and then predicts volume based on inputs.
    """
    def __init__(self):
        self.s3 = boto3.resource(
            service_name='s3',
            region_name='us-east-2',
            aws_access_key_id='AWS_ACCESS_KEY_ID',
            aws_secret_access_key='AWS_SECRET_ACCESS_KEY'
        )
        self.model_filename = 'volume_rf_reg.pkl'
        self.scaler_filename = 'scaler_model.bin'
        self.bucket = 'rtaistockmarket'

        model_obj = self.s3.Bucket(self.bucket).Object(self.model_filename).download_file(self.model_filename)
        scaler_obj = self.s3.Bucket(self.bucket).Object(self.scaler_filename).download_file(self.scaler_filename)

        self.model_file = 'volume_rf_reg.pkl'
        self.scaler_file = 'scaler_model.bin'
        self.model = joblib.load(self.model_file)
        self.scaler = joblib.load(self.scaler_file)
    
    def predict(self, vol_moving_avg, adj_close_rolling_med):
        """
        Predict volume based on input data.

        Parameters
        ----------
        vol_moving_avg : int
            The Volume moving average (30 Days).

        adj_close_rolling_med : int
            The Adjacent Close rolling median

        Returns
        -------
        Volume : Prediction of Random Forrest Regressor. 
        """
        input = self.scaler.fit_transform([[vol_moving_avg, adj_close_rolling_med]])
        return {'Volume': self.model.predict(input)[0]}

if __name__ == '__main__':

    rf = RandomForrestReg()

    st.set_page_config(layout="wide")
    st.write("""
            # Stock Market Volume Prediction
            """)
    vol_avg = st.text_input('Enter Volume Moving Average: ')
    close_med = st.text_input('Enter Close Median')

    if vol_avg and close_med:
        pred = rf.predict(vol_avg, close_med)
        st.write('The Volume prediction is:', pred)
    else:
        st.write('Please fill in both values')