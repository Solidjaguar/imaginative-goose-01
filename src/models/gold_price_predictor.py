import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

class GoldPricePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.lstm_model = None
        self.arima_model = None

    def fetch_data(self, start_date, end_date):
        gold = yf.download("GC=F", start=start_date, end=end_date)
        usd_index = yf.download("DX-Y.NYB", start=start_date, end=end_date)["Close"]
        oil = yf.download("CL=F", start=start_date, end=end_date)["Close"]
        sp500 = yf.download("^GSPC", start=start_date, end=end_date)["Close"]
        
        df = pd.DataFrame({
            'Gold_Price': gold['Close'],
            'USD_Index': usd_index,
            'Oil_Price': oil,
            'SP500': sp500
        })
        return df.dropna()

    def create_features(self, df):
        df['MA7'] = df['Gold_Price'].rolling(window=7).mean()
        df['MA30'] = df['Gold_Price'].rolling(window=30).mean()
        df['Volatility'] = df['Gold_Price'].rolling(window=30).std()
        df['USD_Index_Change'] = df['USD_Index'].pct_change()
        df['Oil_Price_Change'] = df['Oil_Price'].pct_change()
        df['SP500_Change'] = df['SP500'].pct_change()
        return df.dropna()

    def prepare_data(self, df, target_col='Gold_Price', feature_cols=None):
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_models(self, X, y):
        # Train Random Forest
        self.rf_model.fit(X, y)
        
        # Train LSTM
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        self.lstm_model = self.create_lstm_model((1, X.shape[1]))
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        self.lstm_model.fit(X_reshaped, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        
        # Train ARIMA
        self.arima_model = ARIMA(y, order=(5,1,0))
        self.arima_model = self.arima_model.fit()

    def predict(self, X):
        rf_pred = self.rf_model.predict(X)
        
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        lstm_pred = self.lstm_model.predict(X_reshaped).flatten()
        
        arima_pred = self.arima_model.forecast(steps=len(X))
        
        # Ensemble prediction (simple average)
        ensemble_pred = (rf_pred + lstm_pred + arima_pred) / 3
        return ensemble_pred

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'MSE': mse, 'MAE': mae, 'R2': r2}

    def plot_predictions(self, y_true, y_pred):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true.values, label='Actual')
        plt.plot(y_true.index, y_pred, label='Predicted')
        plt.title('Gold Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    predictor = GoldPricePredictor()
    
    # Fetch and prepare data
    data = predictor.fetch_data('2020-01-01', '2023-08-29')
    data = predictor.create_features(data)
    
    # Prepare features and target
    X, y = predictor.prepare_data(data)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train models
    predictor.train_models(X_train, y_train)
    
    # Make predictions
    predictions = predictor.predict(X_test)
    
    # Evaluate
    eval_results = predictor.evaluate(y_test, predictions)
    print("Evaluation results:", eval_results)
    
    # Plot
    predictor.plot_predictions(y_test, predictions)