from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class GoldPricePredictor:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.scaler = StandardScaler()

    def preprocess_data(self, data):
        features = data.drop(['Close', 'Date'], axis=1)
        target = data['Close']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_and_evaluate(self, data):
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
        
        self.best_model = 'Random Forest'  # For simplicity, we're using Random Forest as the best model
        return self, self.models

    def predict(self, features):
        scaled_features = self.scaler.transform(features)
        return self.models[self.best_model].predict(scaled_features)

def train_and_evaluate(data):
    predictor = GoldPricePredictor()
    return predictor.train_and_evaluate(data)