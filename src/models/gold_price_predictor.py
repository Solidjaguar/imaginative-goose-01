import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from loguru import logger

class GoldPricePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'linear': None,
            'random_forest': None,
            'lstm': None
        }

    def prepare_data(self, data, sequence_length=10):
        X = []
        y = []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        X = np.array(X)
        y = np.array(y)
        X = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        return X, y

    def create_lstm_model(self, input_shape, units=50, dropout=0.2, learning_rate=0.001):
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units),
            Dropout(dropout),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model

    def optimize_random_forest(self, X, y):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            model = RandomForestRegressor(**params, random_state=42)
            return -np.mean(cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        return study.best_params

    def train_models(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info("Training Linear Regression model...")
        self.models['linear'] = LinearRegression()
        self.models['linear'].fit(X_train.reshape(X_train.shape[0], -1), y_train)

        logger.info("Optimizing and training Random Forest model...")
        rf_params = self.optimize_random_forest(X_train.reshape(X_train.shape[0], -1), y_train)
        self.models['random_forest'] = RandomForestRegressor(**rf_params, random_state=42)
        self.models['random_forest'].fit(X_train.reshape(X_train.shape[0], -1), y_train)

        logger.info("Training LSTM model...")
        self.models['lstm'] = self.create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        self.models['lstm'].fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )

    def predict(self, X):
        predictions = {}
        for name, model in self.models.items():
            if name == 'lstm':
                predictions[name] = model.predict(X)
            else:
                predictions[name] = model.predict(X.reshape(X.shape[0], -1))
        return predictions

    def evaluate(self, y_true, predictions):
        results = {}
        for name, y_pred in predictions.items():
            results[name] = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        return results

    def plot_predictions(self, y_true, predictions):
        # Implement this method in the visualizer module
        pass