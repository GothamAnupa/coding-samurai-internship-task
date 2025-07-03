import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import yfinance as yf # Import the yfinance library

# --- Configuration ---
# Define the name of your stock data CSV file.
# Make sure this file is in the same directory as your script,
# or provide the full path to the file.
STOCK_DATA_FILE = 'stock_data.csv'
# The column in your CSV that contains the closing price.
CLOSE_PRICE_COLUMN = 'Close'
# The column in your CSV that contains the date.
DATE_COLUMN = 'Date' # This will be the index name if using yfinance data
# Number of previous days' closing prices to use as features.
# This creates 'lagged' features for time series analysis.
N_LAG_FEATURES = 5

# Set to True to download real data using yfinance, False to use existing CSV or dummy data
DOWNLOAD_REAL_DATA = True
# If DOWNLOAD_REAL_DATA is True, specify the stock ticker and date range
STOCK_TICKER = 'AAPL' # Example: Apple Inc.
START_DATE = '2020-01-01'
END_DATE = '2023-01-01'


# --- 1. Data Loading ---
def load_data(file_path, download_real=False, ticker=None, start=None, end=None):
    """
    Loads stock data from a CSV file or downloads real data using yfinance.
    """
    if download_real and ticker and start and end:
        print(f"Attempting to download real data for {ticker} from {start} to {end}...")
        try:
            data = yf.download(ticker, start=start, end=end)
            if data.empty:
                print(f"No data downloaded for {ticker}. Check ticker symbol and date range.")
                return None
            # yfinance returns 'Date' as index, and 'Close' column
            # Reset index to make 'Date' a column for consistency with previous logic
            df = data.reset_index()
            # Rename 'Date' column to match DATE_COLUMN if it's different (yfinance uses 'Date')
            df.rename(columns={'Date': DATE_COLUMN}, inplace=True)
            print(f"Successfully downloaded data for {ticker}.")
            # Optionally save to CSV for future use without re-downloading
            df.to_csv(file_path, index=False)
            print(f"Downloaded data saved to '{file_path}'.")
            return df
        except Exception as e:
            print(f"Error downloading data with yfinance: {e}")
            print("Falling back to existing CSV or dummy data.")
            download_real = False # Fallback to file system logic

    if os.path.exists(file_path) and not download_real:
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded data from existing '{file_path}'.")
            return df
        except Exception as e:
            print(f"Error loading data from '{file_path}': {e}")
            print("Generating dummy data for demonstration purposes...")
            # Fallback to dummy data generation if CSV exists but fails to load
            dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=50, freq='D'))
            close_prices = np.linspace(100, 150, 50) + np.random.randn(50) * 5
            data = pd.DataFrame({DATE_COLUMN: dates, CLOSE_PRICE_COLUMN: close_prices})
            data.to_csv(file_path, index=False)
            print(f"Dummy data saved to '{file_path}'.")
            return data
    else:
        print(f"'{file_path}' not found or DOWNLOAD_REAL_DATA is False and file doesn't exist.")
        print("Generating dummy data for demonstration purposes...")
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=50, freq='D'))
        close_prices = np.linspace(100, 150, 50) + np.random.randn(50) * 5
        data = pd.DataFrame({DATE_COLUMN: dates, CLOSE_PRICE_COLUMN: close_prices})
        data.to_csv(file_path, index=False)
        print(f"Dummy data saved to '{file_path}'.")
        return data


# --- 2. Data Preprocessing and Feature Engineering ---
def preprocess_and_engineer_features(df, close_col, date_col, n_lags):
    """
    Preprocesses the data and creates lagged features.
    - Converts 'Date' column to datetime objects and sets as index.
    - Sorts data by date.
    - Creates 'n_lags' new features, each representing the closing price
      from a previous day.
    - Drops rows with NaN values resulting from lagging.
    """
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    df.set_index(date_col, inplace=True)

    # Create lagged features
    for i in range(1, n_lags + 1):
        df[f'Lag_{i}'] = df[close_col].shift(i)

    # Drop rows with NaN values (these are the first 'n_lags' rows)
    df.dropna(inplace=True)

    # Define features (X) and target (y)
    # X will be the lagged features
    X = df[[f'Lag_{i}' for i in range(1, n_lags + 1)]]
    # y will be the current closing price (the one we want to predict)
    y = df[close_col]

    print(f"Created {n_lags} lagged features.")
    print(f"Shape of features (X): {X.shape}")
    print(f"Shape of target (y): {y.shape}")
    return X, y, df.index # Return dates for plotting (after dropping NaNs)

# --- 3. Model Training and Evaluation ---
def train_and_evaluate_model(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets, trains a Linear Regression model,
    and evaluates its performance.
    """
    # Split data into training and testing sets
    # shuffle=False is crucial for time series data to maintain chronological order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    print(f"\nTraining data size: {len(X_train)} samples")
    print(f"Testing data size: {len(X_test)} samples")

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\nLinear Regression model trained successfully.")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    return model, X_test, y_test, y_pred

# --- 4. Visualization ---
def plot_predictions(dates, y_true, y_pred, n_lags):
    """
    Plots the actual vs. predicted stock prices.
    """
    # The dates passed to this function are already aligned with X and y after preprocessing.
    # We need to slice them to match the test set.
    test_dates = dates[len(dates) - len(y_true):]

    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_true, label='Actual Prices', color='blue')
    plt.plot(test_dates, y_pred, label='Predicted Prices', color='red', linestyle='--')
    plt.title(f'Stock Price Prediction (Linear Regression with {n_lags} Lagged Features)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    df = load_data(STOCK_DATA_FILE, DOWNLOAD_REAL_DATA, STOCK_TICKER, START_DATE, END_DATE)
    if df is None:
        print("Data loading failed. Exiting.")
        exit()

    # 2. Preprocess and Engineer Features
    # Ensure the dataframe passed is a copy to avoid SettingWithCopyWarning
    X, y, valid_dates = preprocess_and_engineer_features(df.copy(), CLOSE_PRICE_COLUMN, DATE_COLUMN, N_LAG_FEATURES)

    # 3. Train and Evaluate Model
    model, X_test, y_test, y_pred = train_and_evaluate_model(X, y)

    # 4. Visualize Predictions
    plot_predictions(valid_dates, y_test, y_pred, N_LAG_FEATURES)

    print("\n--- Next Steps ---")
    print("1. If DOWNLOAD_REAL_DATA is True, the script will attempt to download data.")
    print("   Otherwise, it will try to load from 'stock_data.csv' or generate dummy data.")
    print("2. Experiment with N_LAG_FEATURES to see how it affects performance.")
    print("3. Try other models like RandomForestRegressor, SVR, or simple Neural Networks.")
    print("4. Add more features (e.g., moving averages, volume, technical indicators).")
    print("5. Implement a proper time series cross-validation strategy.")
    print("6. Push this code to your GitHub repository: 'coding-samurai-internship-task'.")
