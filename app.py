import joblib
import streamlit as st

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
        self.model_file = 'models/volume_rf_reg.bin'
        self.scaler_file = 'models/scaler_model.bin'
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
        return {'Volume': int(self.model.predict(input)[0])}

if __name__ == '__main__':

    rf = RandomForrestReg()

    st.set_page_config(layout="wide")
    st.write("""
            # Stock Market Volume Prediction
            """)
    vol_avg = st.text_input('Enter Volume 30 Day Moving Average: ')
    close_med = st.text_input('Enter Adj Close Median')

    if vol_avg and close_med:
        pred = rf.predict(vol_avg, close_med)
        st.write('The Volume prediction is:', pred)
    else:
        st.write('Please fill in both values')