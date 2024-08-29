import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class TradingStrategy(ABC):
    @abstractmethod
    def generate_signal(self, data):
        pass

class MovingAverageCrossover(TradingStrategy):
    def __init__(self, short_window=10, long_window=30):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data
        signals['short_mavg'] = data.rolling(window=self.short_window, min_periods=1).mean()
        signals['long_mavg'] = data.rolling(window=self.long_window, min_periods=1).mean()
        signals['signal'] = np.where(signals['short_mavg'] > signals['long_mavg'], 1, -1)
        signals['position'] = signals['signal'].diff()
        return signals['position'].iloc[-1]

class RSIStrategy(TradingStrategy):
    def __init__(self, window=14, oversold=30, overbought=70):
        self.window = window
        self.oversold = oversold
        self.overbought = overbought

    def generate_signal(self, data):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        if rsi.iloc[-1] < self.oversold:
            return 1
        elif rsi.iloc[-1] > self.overbought:
            return -1
        else:
            return 0

class BollingerBandsStrategy(TradingStrategy):
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std

    def generate_signal(self, data):
        rolling_mean = data.rolling(window=self.window).mean()
        rolling_std = data.rolling(window=self.window).std()
        upper_band = rolling_mean + (rolling_std * self.num_std)
        lower_band = rolling_mean - (rolling_std * self.num_std)
        
        if data.iloc[-1] < lower_band.iloc[-1]:
            return 1
        elif data.iloc[-1] > upper_band.iloc[-1]:
            return -1
        else:
            return 0

class TrendFollowingStrategy(TradingStrategy):
    def __init__(self, window=50):
        self.window = window

    def generate_signal(self, data):
        trend = data.rolling(window=self.window).mean()
        if data.iloc[-1] > trend.iloc[-1]:
            return 1
        elif data.iloc[-1] < trend.iloc[-1]:
            return -1
        else:
            return 0

def apply_risk_management(position, entry_price, current_price, stop_loss=0.02, take_profit=0.04):
    if position == 1:  # Long position
        if current_price <= entry_price * (1 - stop_loss):
            return 0  # Close position (stop loss)
        elif current_price >= entry_price * (1 + take_profit):
            return 0  # Close position (take profit)
    elif position == -1:  # Short position
        if current_price >= entry_price * (1 + stop_loss):
            return 0  # Close position (stop loss)
        elif current_price <= entry_price * (1 - take_profit):
            return 0  # Close position (take profit)
    
    return position  # Maintain current position

def calculate_position_size(account_balance, risk_per_trade=0.02):
    return account_balance * risk_per_trade