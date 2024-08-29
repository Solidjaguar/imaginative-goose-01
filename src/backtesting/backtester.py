import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from loguru import logger

def calculate_returns(prices):
    return np.log(prices / prices.shift(1))

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate / 252  # Assuming 252 trading days in a year
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def backtest(model, data, model_name):
    logger.info(f"Starting backtesting for {model_name}")
    
    # Prepare features and target
    features = data.drop(['Close', 'Date'], axis=1)
    target = data['Close']
    
    # Use 80% of the data for training, 20% for testing
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Train the model
    model.fit(train_data.drop(['Close', 'Date'], axis=1), train_data['Close'])
    
    # Make predictions
    predictions = model.predict(test_data.drop(['Close', 'Date'], axis=1))
    
    # Calculate returns
    actual_returns = calculate_returns(test_data['Close'])
    predicted_returns = calculate_returns(pd.Series(predictions))
    
    # Calculate metrics
    mse = mean_squared_error(test_data['Close'], predictions)
    sharpe_ratio = calculate_sharpe_ratio(predicted_returns)
    total_return = (predictions[-1] / predictions[0]) - 1
    
    logger.info(f"Backtesting results for {model_name}:")
    logger.info(f"MSE: {mse}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio}")
    logger.info(f"Total Return: {total_return}")
    
    return {
        "mse": mse,
        "sharpe_ratio": sharpe_ratio,
        "total_return": total_return
    }

def run_backtests(models, data):
    results = {}
    for model_name, model in models.items():
        results[model_name] = backtest(model, data, model_name)
    return results