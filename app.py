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
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

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
        # Create multiple plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

        # Plot 1: Gold Price Over Time
        sns.lineplot(x=data.index, y=data['Close'], ax=ax1)
        ax1.set_title('Gold Price Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')

        # Plot 2: Correlation Heatmap
        correlation = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_title('Correlation Heatmap')

        # Plot 3: Model Performance Comparison
        model_names = []
        mse_scores = []
        r2_scores = []

        X = data.drop(['Close', 'Date'], axis=1)
        y = data['Close']

        for name, model in models.items():
            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            model_names.append(name)
            mse_scores.append(mse)
            r2_scores.append(r2)

        x = np.arange(len(model_names))
        width = 0.35

        ax3.bar(x - width/2, mse_scores, width, label='MSE')
        ax3.bar(x + width/2, r2_scores, width, label='R2')
        ax3.set_ylabel('Scores')
        ax3.set_title('Model Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.legend()

        plt.tight_layout()
        
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