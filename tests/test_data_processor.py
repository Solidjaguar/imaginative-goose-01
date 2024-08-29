import unittest
import pandas as pd
import numpy as np
from src.utils.data_processor import prepare_data

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'Gold_Price': np.random.rand(100),
            'USD_Index': np.random.rand(100),
            'Oil_Price': np.random.rand(100),
            'SP500': np.random.rand(100),
            'Treasury_Yield': np.random.rand(100),
            'VIX': np.random.rand(100)
        })

    def test_prepare_data(self):
        result = prepare_data(self.sample_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(result.shape[1], self.sample_data.shape[1])
        self.assertLess(result.shape[0], self.sample_data.shape[0])  # Due to NaN dropping

if __name__ == '__main__':
    unittest.main()