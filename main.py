import sys
import json
from loguru import logger
from src.utils.data_fetcher import fetch_all_data
from src.utils.data_processor import prepare_data
from src.models.gold_price_predictor import GoldPricePredictor
from src.utils.visualizer import plot_predictions, plot_performance
from src.strategies.trading_strategy import MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy
from src.strategies.paper_trader import PaperTrader

def setup_logging():
    logger.remove()
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    logger.add("logs/gold_predictor.log", rotation="500 MB", level="DEBUG")

def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Config file not found. Using default values.")
        return {
            "start_date": "2010-01-01",
            "end_date": None,
            "train_size": 0.8,
            "initial_balance": 10000,
            "risk_per_trade": 0.02
        }

def main():
    setup_logging()
    config = load_config()

    logger.info("Fetching and preparing data...")
    try:
        data = fetch_all_data(config['start_date'], config['end_date'])
        prepared_data = prepare_data(data)
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        return

    logger.info("Creating and training the model...")
    predictor = GoldPricePredictor()
    try:
        X, y = predictor.prepare_data(prepared_data)
        train_size = int(len(X) * config['train_size'])
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        predictor.train_models(X_train, y_train)
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return

    logger.info("Making predictions...")
    try:
        predictions = predictor.predict(X_test)
        eval_results = predictor.evaluate(y_test, predictions)
        logger.info(f"Evaluation results: {eval_results}")
        plot_predictions(y_test, predictions)
    except Exception as e:
        logger.error(f"Error in making predictions: {str(e)}")
        return

    logger.info("Simulating trading strategies...")
    try:
        strategies = [
            MovingAverageCrossover(),
            RSIStrategy(),
            BollingerBandsStrategy()
        ]
        paper_trader = PaperTrader(initial_balance=config['initial_balance'])

        for i in range(len(predictions)):
            price = predictions[i]
            signals = [strategy.generate_signal(price) for strategy in strategies]
            combined_signal = sum(signals) / len(signals)
            paper_trader.execute_trade(combined_signal, price, config['risk_per_trade'])

        performance_summary = paper_trader.get_performance_summary()
        logger.info(f"Paper trading results: {performance_summary}")
        plot_performance(paper_trader.balance_history)
    except Exception as e:
        logger.error(f"Error in trading simulation: {str(e)}")

if __name__ == "__main__":
    main()