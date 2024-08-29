import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from loguru import logger

def plot_predictions(y_true, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    for name, pred in predictions.items():
        plt.plot(pred, label=f'{name} Prediction')
    plt.title('Gold Price Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('predictions.png')
    plt.close()
    logger.info("Predictions plot saved as 'predictions.png'")

def plot_performance(balance_history):
    plt.figure(figsize=(12, 6))
    plt.plot(balance_history)
    plt.title('Paper Trading Performance')
    plt.xlabel('Trade')
    plt.ylabel('Account Balance')
    plt.savefig('performance.png')
    plt.close()
    logger.info("Performance plot saved as 'performance.png'")

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        logger.info("Feature importance plot saved as 'feature_importance.png'")
    else:
        logger.warning("Model does not have feature_importances_ attribute")

def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    logger.info("Correlation matrix plot saved as 'correlation_matrix.png'")

def plot_trading_signals(data, signals):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Gold_Price'], label='Gold Price')
    plt.scatter(data.index[signals == 1], data['Gold_Price'][signals == 1], marker='^', color='g', label='Buy Signal')
    plt.scatter(data.index[signals == -1], data['Gold_Price'][signals == -1], marker='v', color='r', label='Sell Signal')
    plt.title('Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('trading_signals.png')
    plt.close()
    logger.info("Trading signals plot saved as 'trading_signals.png'")