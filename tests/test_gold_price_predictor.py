import unittest
import numpy as np
import pandas as pd
from src.models.gold_price_predictor import GoldPricePredictor

class TestGoldPricePredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = GoldPricePredictor()
        self.sample_data = pd.DataFrame(np.random.rand(100, 10))

    def test_prepare_data(self):
        X, y = self.predictor.prepare_data(self.sample_data)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape[0], y.shape[0])

    def test_train_models(self):
        X, y = self.predictor.prepare_data(self.sample_data)
        self.predictor.train_models(X, y)
        self.assertIsNotNone(self.predictor.models['linear'])
        self.assertIsNotNone(self.predictor.models['random_forest'])
        self.assertIsNotNone(self.predictor.models['lstm'])

    def test_predict(self):
        X, y = self.predictor.prepare_data(self.sample_data)
        self.predictor.train_models(X, y)
        predictions = self.predictor.predict(X)
        self.assertIsInstance(predictions, dict)
        self.assertEqual(len(predictions), 3)  # linear, random_forest, lstm

if __name__ == '__main__':
    unittest.main()