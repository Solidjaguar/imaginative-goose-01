import pandas as pd
import numpy as np
from loguru import logger

class PaperTrader:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.balance_history = [initial_balance]

    def execute_trade(self, signal, price, risk_per_trade=0.02):
        if signal == 1 and self.position <= 0:
            # Open long position
            position_size = self.calculate_position_size(risk_per_trade)
            self.position = position_size / price
            self.entry_price = price
            self.trades.append(('buy', price, self.position))
            logger.info(f"Opening long position: {self.position} units at {price}")
        elif signal == -1 and self.position >= 0:
            # Open short position
            position_size = self.calculate_position_size(risk_per_trade)
            self.position = -position_size / price
            self.entry_price = price
            self.trades.append(('sell', price, self.position))
            logger.info(f"Opening short position: {self.position} units at {price}")
        elif signal == 0 and self.position != 0:
            # Close position
            pnl = self.calculate_pnl(price)
            self.balance += pnl
            self.trades.append(('close', price, self.position))
            logger.info(f"Closing position: PnL = {pnl}")
            self.position = 0
            self.entry_price = 0

        self.balance_history.append(self.balance)

    def calculate_position_size(self, risk_per_trade):
        return self.balance * risk_per_trade

    def calculate_pnl(self, current_price):
        if self.position > 0:
            return self.position * (current_price - self.entry_price)
        elif self.position < 0:
            return -self.position * (self.entry_price - current_price)
        else:
            return 0

    def get_performance_summary(self):
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown = self.calculate_max_drawdown()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'final_balance': self.balance
        }

    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        returns = pd.Series(self.balance_history).pct_change().dropna()
        excess_returns = returns - (risk_free_rate / 252)  # Assuming 252 trading days in a year
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    def calculate_max_drawdown(self):
        balance_series = pd.Series(self.balance_history)
        cumulative_max = balance_series.cummax()
        drawdown = (cumulative_max - balance_series) / cumulative_max
        max_drawdown = drawdown.max()
        return max_drawdown