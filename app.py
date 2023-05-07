import joblib
import streamlit as st

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    '''Using Streamlit cache_resource decorator to preserve RAM memory and for efficiency.'''
    return joblib.load('models/volume_rf_reg.bin')

@st.cache_resource
def load_scaler():
    '''Using Streamlit cache_resource decorator to preserve RAM memory and for efficiency.'''
    return joblib.load('models/scaler_model.bin')

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
        self.model = load_model()
        self.scaler = load_scaler()
    
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
        return {'Volume':  float('{:.2f}'.format(self.model.predict(input)[0]))}

if __name__ == '__main__':

    rf = RandomForrestReg()
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