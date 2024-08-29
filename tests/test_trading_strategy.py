import unittest
import pandas as pd
import numpy as np
from src.strategies.trading_strategy import MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy, TrendFollowingStrategy

class TestTradingStrategies(unittest.TestCase):
    def setUp(self):
        self.data = pd.Series(np.random.rand(100))

    def test_moving_average_crossover(self):
        strategy = MovingAverageCrossover()
        signal = strategy.generate_signal(self.data)
        self.assertIn(signal, [-1, 0, 1])

    def test_rsi_strategy(self):
        strategy = RSIStrategy()
        signal = strategy.generate_signal(self.data)
        self.assertIn(signal, [-1, 0, 1])

    def test_bollinger_bands_strategy(self):
        strategy = BollingerBandsStrategy()
        signal = strategy.generate_signal(self.data)
        self.assertIn(signal, [-1, 0, 1])

    def test_trend_following_strategy(self):
        strategy = TrendFollowingStrategy()
        signal = strategy.generate_signal(self.data)
        self.assertIn(signal, [-1, 0, 1])

if __name__ == '__main__':
    unittest.main()