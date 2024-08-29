import unittest
from src.strategies.paper_trader import PaperTrader

class TestPaperTrader(unittest.TestCase):
    def setUp(self):
        self.trader = PaperTrader(initial_balance=10000)

    def test_execute_trade(self):
        self.trader.execute_trade(1, 100)
        self.assertGreater(self.trader.position, 0)
        self.assertEqual(self.trader.entry_price, 100)

        self.trader.execute_trade(-1, 110)
        self.assertLess(self.trader.position, 0)
        self.assertEqual(self.trader.entry_price, 110)

        self.trader.execute_trade(0, 105)
        self.assertEqual(self.trader.position, 0)

    def test_get_performance_summary(self):
        self.trader.execute_trade(1, 100)
        self.trader.execute_trade(0, 110)
        summary = self.trader.get_performance_summary()
        self.assertIn('total_return', summary)
        self.assertIn('sharpe_ratio', summary)
        self.assertIn('max_drawdown', summary)
        self.assertIn('num_trades', summary)
        self.assertIn('final_balance', summary)

if __name__ == '__main__':
    unittest.main()