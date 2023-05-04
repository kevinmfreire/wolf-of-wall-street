# Wolf of Wall Street

## Requirements

* Install docker engine through the officical [documentation](https://docs.docker.com/engine/install/ubuntu/).
* Linux OS

## Table of Content
1. [Overview]()
2. [Goal]()
3. [Methodology]()
4. [Practical Applications]()
5. [Usage]()
6. [Conclusion]()

## Overview

The rise of a wealthy stock-broker living the high life to his fall involving crime, corruption and the federal government...

This project won't give you an adrenaline rush, nor make you a millionare. :money_with_wings:  

It simply demonstrates a way to solve a data pipeline problem. Using a combination of Apache Airflow, Docker, Python, and libraries such as pandas and scikit-learn.  I built a data pipeline from ingesting data, to data processing, feature extraction, model training and then saving the model to access it via an API.  

## Goals
The following are the problems to solve.

 1. **Raw Data Processing**: Extract data from the [stock-market-dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset) provided by kaggle. Process the data into a structure format and save it.

 2. **Feature Engineering**: Extract additional features on top of the processed data.  Features include the `vol_moving_avg` and the `adj_close_rolling_median`. Save it once more.

 3. **Integrate ML Training**: Integrate a ML model training step within the pipeline to predict the trading volume in the stock market.

 4. **Model Serving**: Build an API to serve the trained predictive model.

## Methodology
### Data Processing
I leveraged `pandas` to structure my data in the format as shown below:
```
    Symbol: string
    Security Name: string
    Date: string (YYYY-MM-DD)
    Open: float
    High: float
    Low: float
    Close: float
    Adj Close: float
    Volume: int
```
### Feature Engineering
Utilizing the processed data, I extract additional features such as the `vol_moving_avg` (volume moving average) by leveraging `pandas groupby()` to group the data based on the symbols and utilized a 30 day window exponential moving average which returns a more precise average than the simple moving average and we do not loose data (first 30 days).  

I had a similar approach for the `adj_close_rolling_med` (adjacent rolling median), however I leveraged the pandas `rolling()` with a window of 30 days to extract the median for each value.  For the rolling median I used the `min_periods` parameter and set it to one in order to onle have one non-NA observation to work with. This is important because if `min_periods` is set to a value greater than 1 and there are missing 
values within the rolling window, the rolling statistic for that window will also be missing. This can result in incomplete or inaccurate results.
### ML Training
In order to use the features I first normalized the data to improve model performance by ensuring that features are on the same scale (0-1).  To do so I used the `MinMaxScaler` from `scikit-learn` to preserve the shape of the original distribution data.  The scaler is designed to work well with both training and unseen data, which is important for the accuracy and reliability of machine learning models.

I used `scikit-learn RandomForrestRegressor` to train using the features from the [Feature Engineering]() section. I used the model because it is well suited for modelling non-linear data because it is able to capture complex relationships between features and target values (ie. stock prices, trading values, etc.).

### Data Pipeline
Leveraged Docker and Apache Ariflow to automate the entire process.

### Model Serving
Saving both the `MinMaxScaler` model and the `RandomForrestRegressor` model, I built a class that caa be called via an `/predict` API to performTrading Volume predicitons based on the `vol_moving_avg`, and `adj_close_rolling_med` inputs.

## Practical Applications
There are several use cases for training a machine learning model to predict the stock market trading value:

1. Investment Decisions.
2. Portfolio Optimization.
3. Risk Managment.
4. Trading Strategies.
5. Market Research.
5. Algorithmic Trading.

## Usage
To replicate the work, please follow the steps below:
1. Clone repo
```
git clone github-repo
```
2. Create a data directory in the root directory as follows:
```bash
mkdir data
mkdir data/raw
cd data/raw
```
3. Download the [stock-market-dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset) and place it in the `data/raw` directory.  Ensure your data directory follows the same structure below.
```
data/
|   raw/
|   |   etfs/
|   |   stocks/
|   |   symbols_meta_data.csv
```
4. Build the docker image as follows:
```bash
docker build . --tag extending_airflow:latest
```
5. Once it is done, initialize airflow.
```bash
docker compose up airflow-init
```
6. Finally run the airflow docker.
```bash
docker compose up
```
7. Once it finishes you can visit [localhost:8080](localhost:8080) on your web browser and use `airflow` as both the user and password.

## Conclusion
In summary, this project involves collecting data from the stock market, processing the data using Apache Airflow, Docker, and Python, extracting features from the data, training a machine learning model, and serving the model through an API. This project can be useful for investors who want to predict stock market trends and make informed investment decisions.