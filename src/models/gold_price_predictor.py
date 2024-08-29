import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import json

class GoldPricePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        with open('config.json', 'r') as f:
            self.config = json.load(f)

    def prepare_data(self, data):
        X = data.drop('Gold_Price', axis=1)
        y = data['Gold_Price']
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y.values

    def create_sequences(self, X, y, time_steps=10):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    def train_models(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-self.config['model']['train_size'], random_state=self.config['model']['random_state'])

        # Linear Regression
        self.models['linear'] = LinearRegression()
        self.models['linear'].fit(X_train, y_train)

        # Random Forest
        self.models['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=self.config['model']['random_state'])
        self.models['random_forest'].fit(X_train, y_train)

        # LSTM
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)

        self.models['lstm'] = Sequential([
            LSTM(units=self.config['model']['lstm_units'], return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            Dropout(self.config['model']['lstm_dropout']),
            LSTM(units=self.config['model']['lstm_units']),
            Dropout(self.config['model']['lstm_dropout']),
            Dense(1)
        ])
        self.models['lstm'].compile(optimizer='adam', loss='mse')
        self.models['lstm'].fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

        # XGBoost
        self.models['xgboost'] = XGBRegressor(random_state=self.config['model']['random_state'])
        self.models['xgboost'].fit(X_train, y_train)

        # LightGBM
        self.models['lightgbm'] = LGBMRegressor(random_state=self.config['model']['random_state'])
        self.models['lightgbm'].fit(X_train, y_train)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        predictions = {}
        for name, model in self.models.items():
            if name == 'lstm':
                X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))
                predictions[name] = model.predict(X_seq).flatten()
            else:
                predictions[name] = model.predict(X_scaled)
        return predictions

def train_and_evaluate(data):
    predictor = GoldPricePredictor()
    X, y = predictor.prepare_data(data)
    predictor.train_models(X, y)
    return predictor