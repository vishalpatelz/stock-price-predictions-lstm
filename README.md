
# Stock Price Prediction using LSTM

This project involves using Long Short-Term Memory (LSTM) neural networks to predict stock prices. We have focused on predicting the closing prices of stocks from Yahoo Finance.

## Table of Contents

- [Overview](#overview)
- [Data Collection](#data-collection)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)
- [Run](#run)

## Overview

This project aims to predict the closing prices of stocks using historical data. We use LSTM neural networks, a type of recurrent neural network (RNN) capable of learning order dependence in sequence prediction problems.

## Data Collection

We use the `yfinance` library to download historical stock data from Yahoo Finance. The data includes the daily closing prices of the stocks.

```python
import yfinance as yf

# Download Google stock data from Yahoo Finance
ticker = "GOOGL"
data = yf.download(ticker, start="2018-01-01", end="2023-01-01")
```

## Data Preparation

The data preparation involves the following steps:
1. Selecting the 'Close' price column.
2. Scaling the data using MinMaxScaler.
3. Creating a dataset with a specified time step for the LSTM model.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Prepare the data
data = data[['Close']]
data = data.dropna()

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create the dataset
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape the data for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)
```

## Model Training

The LSTM model is trained using the prepared dataset. The model is saved after training for later use.

## Model Evaluation

The model's performance is evaluated using Mean Squared Error (MSE) and visualizations of the predicted vs. actual stock prices.

```python
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

# Load the model
model = load_model('LSTM_Best_Model_1.keras')

# Make predictions
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

# Create a DataFrame to compare the predictions with the actual prices
df = pd.DataFrame(data[time_step + 1:])
df['Predictions'] = predictions

# Calculate MSE
mse = mean_squared_error(df['Close'], df['Predictions'])
print("Mean Squared Error:", mse)

print(df.head())
```

## Usage

To use the code, ensure you have all the dependencies installed. Load the dataset, prepare it, load the pre-trained model, and make predictions.

## Results

The results of the model are visualized using Matplotlib, showing the actual vs. predicted stock prices.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))
plt.plot(df.index, df['Close'], label='Actual Price')
plt.plot(df.index, df['Predictions'], label='Predicted Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
```

## Dependencies

- numpy
- pandas
- yfinance
- sklearn
- tensorflow
- matplotlib

Install the dependencies using pip:

```bash
pip install numpy pandas yfinance sklearn tensorflow matplotlib
```

## Acknowledgements

We would like to acknowledge the use of Yahoo Finance for providing the stock data and the various Python libraries that made this project possible.

## Run

The file structure as below,

#1. LSTMStockipynb.ipynb
#2. LSTM_Test.ipynb
#3. LSTM_Best_Model_1.keras

Run main file #1 which will test 3 different models using LSTM. It uses the yfinance library and downloads data of "AAPL"
The best-performing model will be saved for traying on #2.
Now run #2 file, which will use #3 as pre-trained model #3. It will try to predict the prices of "GOOGL" & "META"

If you have any questions, contact us!!