import unittest
from unittest.mock import patch
import pandas as pd
from src.utils.data_fetcher import DataFetcher, fetch_all_data

class TestDataFetcher(unittest.TestCase):
    @patch('yfinance.download')
    def test_fetch_data(self, mock_download):
        mock_download.return_value = pd.DataFrame({'Close': [1, 2, 3]})
        result = DataFetcher.fetch_data('AAPL', '2021-01-01', '2021-01-03')
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)

    @patch('src.utils.data_fetcher.DataFetcher.fetch_data')
    def test_fetch_gold_data(self, mock_fetch_data):
        mock_fetch_data.return_value = pd.Series([1, 2, 3])
        result = DataFetcher.fetch_gold_data('2021-01-01', '2021-01-03')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[1], 6)

if __name__ == '__main__':
    unittest.main()