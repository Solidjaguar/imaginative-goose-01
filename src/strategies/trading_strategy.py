import pandas as pd
import numpy as np

class TradingStrategy:
    def generate_signal(self, price):
        raise NotImplementedError("Subclass must implement abstract method")

class MovingAverageCrossover(TradingStrategy):
    def __init__(self, short_window=10, long_window=30):
        self.short_window = short_window
        self.long_window = long_window
        self.short_ma = []
        self.long_ma = []

    def generate_signal(self, price):
        self.short_ma.append(price)
        self.long_ma.append(price)
        
        if len(self.short_ma) > self.short_window:
            self.short_ma.pop(0)
        if len(self.long_ma) > self.long_window:
            self.long_ma.pop(0)
        
        if len(self.short_ma) < self.short_window or len(self.long_ma) < self.long_window:
            return 0
        
        short_ma = sum(self.short_ma) / len(self.short_ma)
        long_ma = sum(self.long_ma) / len(self.long_ma)
        
        if short_ma > long_ma:
            return 1  # Buy signal
        elif short_ma < long_ma:
            return -1  # Sell signal
        else:
            return 0  # Hold

class RSIStrategy(TradingStrategy):
    def __init__(self, window=14, oversold=30, overbought=70):
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
        self.prices = []
        self.gains = []
        self.losses = []

    def generate_signal(self, price):
        self.prices.append(price)
        if len(self.prices) > self.window:
            self.prices.pop(0)
        
        if len(self.prices) < 2:
            return 0
        
        change = self.prices[-1] - self.prices[-2]
        gain = max(change, 0)
        loss = max(-change, 0)
        
        self.gains.append(gain)
        self.losses.append(loss)
        
        if len(self.gains) > self.window:
            self.gains.pop(0)
            self.losses.pop(0)
        
        if len(self.gains) < self.window:
            return 0
        
        avg_gain = sum(self.gains) / len(self.gains)
        avg_loss = sum(self.losses) / len(self.losses)
        
        if avg_loss == 0:
            return 0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        if rsi < self.oversold:
            return 1  # Buy signal
        elif rsi > self.overbought:
            return -1  # Sell signal
        else:
            return 0  # Hold

class BollingerBandsStrategy(TradingStrategy):
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
        self.prices = []

    def generate_signal(self, price):
        self.prices.append(price)
        if len(self.prices) > self.window:
            self.prices.pop(0)
        
        if len(self.prices) < self.window:
            return 0
        
        mean = sum(self.prices) / len(self.prices)
        std = np.std(self.prices)
        
        upper_band = mean + (self.num_std * std)
        lower_band = mean - (self.num_std * std)
        
        if price < lower_band:
            return 1  # Buy signal
        elif price > upper_band:
            return -1  # Sell signal
        else:
            return 0  # Hold

def apply_risk_management(suggested_position, entry_price, current_price, stop_loss=0.02, take_profit=0.04):
    if suggested_position == 1:  # Long position
        if current_price <= entry_price * (1 - stop_loss):
            return 0  # Close position (stop loss)
        elif current_price >= entry_price * (1 + take_profit):
            return 0  # Close position (take profit)
    elif suggested_position == -1:  # Short position
        if current_price >= entry_price * (1 + stop_loss):
            return 0  # Close position (stop loss)
        elif current_price <= entry_price * (1 - take_profit):
            return 0  # Close position (take profit)
    
    return suggested_position  # Maintain current position

def calculate_position_size(account_balance, risk_per_trade=0.02):
    return account_balance * risk_per_trade