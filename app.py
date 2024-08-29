from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from src.utils.data_fetcher import fetch_all_data
from src.models.gold_price_predictor import train_and_evaluate
from src.backtesting.backtester import run_backtests
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Fetch data and train models
logger.info("Fetching data and training models...")
data = fetch_all_data(config['data']['start_date'], config['data']['end_date'])
predictor, models = train_and_evaluate(data)
logger.info("Data fetched and models trained successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.json
        df = pd.DataFrame([features])
        predictions = predictor.predict(df)
        return jsonify(predictions.tolist())
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/backtest')
def backtest():
    try:
        results = run_backtests(models, data)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/visualize')
def visualize():
    try:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=data.index, y=data['Close'])
        plt.title('Gold Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        
        return render_template('visualize.html', plot_url=plot_url)
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)