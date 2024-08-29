import json
from src.utils.data_fetcher import fetch_all_data
from src.utils.data_processor import prepare_data
from src.models.gold_price_predictor import GoldPricePredictor
from src.utils.visualizer import plot_predictions
from src.strategies.trading_strategy import MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy
from src.strategies.paper_trader import PaperTrader

def main():
    # Fetch and prepare data
    data = fetch_all_data()
    prepared_data = prepare_data(data)

    # Create and train the model
    predictor = GoldPricePredictor()
    X, y = predictor.prepare_data(prepared_data)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    predictor.train_models(X_train, y_train)

    # Make predictions
    predictions = predictor.predict(X_test)

    # Evaluate and plot results
    eval_results = predictor.evaluate(y_test, predictions)
    print("Evaluation results:", eval_results)
    predictor.plot_predictions(y_test, predictions)

    # Initialize trading strategies
    ma_strategy = MovingAverageCrossover()
    rsi_strategy = RSIStrategy()
    bb_strategy = BollingerBandsStrategy()

    # Initialize paper trader
    paper_trader = PaperTrader(initial_balance=10000)

    # Simulate trading
    for i in range(len(predictions)):
        price = predictions[i]
        ma_signal = ma_strategy.generate_signal(price)
        rsi_signal = rsi_strategy.generate_signal(price)
        bb_signal = bb_strategy.generate_signal(price)

        # Combine signals (you can implement your own logic here)
        combined_signal = (ma_signal + rsi_signal + bb_signal) / 3

        paper_trader.execute_trade(combined_signal, price)

    # Print paper trading results
    print(paper_trader.get_performance_summary())

if __name__ == "__main__":
    main()