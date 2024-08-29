from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from src.utils.data_fetcher import fetch_all_data
from src.models.gold_price_predictor import train_and_evaluate
from loguru import logger

app = Flask(__name__)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Fetch data and train models
logger.info("Fetching data and training models...")
data = fetch_all_data(config['data']['start_date'], config['data']['end_date'])
predictor = train_and_evaluate(data)
logger.info("Data fetched and models trained successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json
    df = pd.DataFrame([features])
    predictions = predictor.predict(df)
    return jsonify(predictions)

@app.route('/backtest')
def backtest():
    # Placeholder for backtesting functionality
    results = {
        "linear": {"total_return": 0.15, "sharpe_ratio": 0.8},
        "random_forest": {"total_return": 0.18, "sharpe_ratio": 0.9},
        "lstm": {"total_return": 0.20, "sharpe_ratio": 1.0},
        "xgboost": {"total_return": 0.22, "sharpe_ratio": 1.1},
        "lightgbm": {"total_return": 0.21, "sharpe_ratio": 1.05}
    }
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)