# Stock Price Prediction using Linear Regression

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Download Apple stock data
import yfinance as yf
import pandas as pd

try:
    df = yf.download("AAPL", start="2015-01-01", end="2024-12-31", progress=False)
    if df.empty:
        raise ValueError("Downloaded dataframe is empty. Try again later or check network.")
except Exception as e:
    print("Error downloading data:", e)
    exit()

df = df[['Close']]
df.dropna(inplace=True)


# Create lag feature for previous day's close
df['Close_t-1'] = df['Close'].shift(1)
df.dropna(inplace=True)

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

X_train = train[['Close_t-1']]
y_train = train['Close']
X_test = test[['Close_t-1']]
y_test = test['Close']

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
preds = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Prices")
plt.plot(preds, label="Predicted Prices")
plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

