# 📈 Stock Price Prediction using Machine Learning

This project predicts Apple's stock closing price using historical data with Linear Regression.

## 🔧 Tech Stack

- Python
- Jupyter Notebook
- scikit-learn
- yfinance
- matplotlib
- pandas

## 📁 Folder Structure

StockPricePrediction/
├── data/ # Optional CSV or saved data
├── models/ # Future: Save trained models
├── plots/ # Prediction plots saved here
├── stock_predictor.ipynb
├── requirements.txt
└── README.md

## 🔍 Features

- Download historical stock data using Yahoo Finance
- Create lag features for next-day prediction
- Train/test split for time series
- Evaluate with MAE, MSE, RMSE
- Visualize predictions vs actual stock prices

## 🚀 Run the Project

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the notebook:
    ```bash
    jupyter notebook stock_predictor.ipynb
    ```
3. Check `plots/prediction.png` for visual results

---Feel free to enhance this with:
- More lag features (e.g., t-2, t-3)
- Moving averages
- Neural Networks (MLP or LSTM)
