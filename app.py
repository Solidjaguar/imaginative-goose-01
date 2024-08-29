from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from src.utils.data_fetcher import fetch_all_data
from src.models.gold_price_predictor import train_and_evaluate
from src.backtesting.backtester import run_backtest

app = Flask(__name__)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Fetch data and train models
data = fetch_all_data(config['data']['start_date'], config['data']['end_date'])
predictor = train_and_evaluate(data)

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
    results = run_backtest(data, config)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)