import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Data Collection
ticker = input("Enter stock ticker symbol: ")
start_date = "2010-01-01"
end_date = "2023-01-01"
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Feature Engineering and Handling Non-Stationary Data
stock_data['50MA'] = stock_data['Close'].rolling(window=50).mean()
stock_data['200MA'] = stock_data['Close'].rolling(window=200).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

stock_data['RSI'] = calculate_rsi(stock_data)

stock_data['Close_diff'] = stock_data['Close'].diff()
stock_data = stock_data.dropna()

# Check stationarity
result = adfuller(stock_data['Close_diff'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Model Selection and Training (ARIMA)
train_size = int(0.8 * len(stock_data))
train_data = stock_data[:train_size]['Close_diff']
test_data = stock_data[train_size:]['Close_diff']

order = (2, 1, 2)
model = ARIMA(train_data, order=order)
fitted_model = model.fit()

forecast_steps = len(test_data)
forecast = fitted_model.forecast(steps=forecast_steps)
forecasted_prices = stock_data['Close'].iloc[train_size - 1] + np.cumsum(forecast)

rmse = np.sqrt(mean_squared_error(stock_data['Close'].iloc[train_size:], forecasted_prices))
print("Root Mean Squared Error:", rmse)

# Visualization and Prediction
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index[train_size:], stock_data['Close'].iloc[train_size:], label="Actual Prices")
plt.plot(stock_data.index[train_size:], forecasted_prices, color='red', label="Predicted Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title(f"{ticker} Stock Price Prediction")
plt.legend()
plt.show()
print("Predicted Price for the next day:", forecasted_prices.iloc[-1])
