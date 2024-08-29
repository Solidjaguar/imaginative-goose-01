import pytest
import pandas as pd
import numpy as np
from src.backtesting.backtester import calculate_returns, calculate_sharpe_ratio, backtest
from sklearn.linear_model import LinearRegression

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Date': pd.date_range(start='2021-01-01', periods=100),
        'Open': np.random.randint(1000, 2000, 100),
        'High': np.random.randint(1000, 2000, 100),
        'Low': np.random.randint(1000, 2000, 100),
        'Close': np.random.randint(1000, 2000, 100),
        'Volume': np.random.randint(1000000, 5000000, 100)
    })

def test_calculate_returns():
    prices = pd.Series([100, 110, 105, 115])
    returns = calculate_returns(prices)
    assert len(returns) == len(prices)
    assert np.isclose(returns.iloc[1], np.log(110/100))

def test_calculate_sharpe_ratio():
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    sharpe = calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    assert sharpe > 0

def test_backtest(sample_data):
    model = LinearRegression()
    results = backtest(model, sample_data, 'Linear Regression')
    assert 'mse' in results
    assert 'sharpe_ratio' in results
    assert 'total_return' in results