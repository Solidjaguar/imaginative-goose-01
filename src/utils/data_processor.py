import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.preprocessing import StandardScaler
from loguru import logger

def prepare_data(data):
    try:
        # Add all technical analysis features
        data = add_all_ta_features(
            data, open="Gold_Price", high="Gold_Price", low="Gold_Price", close="Gold_Price", volume="Gold_Price"
        )
        
        # Add custom features
        data['Gold_Returns'] = data['Gold_Price'].pct_change()
        data['USD_Returns'] = data['USD_Index'].pct_change()
        data['Oil_Returns'] = data['Oil_Price'].pct_change()
        data['SP500_Returns'] = data['SP500'].pct_change()
        
        # Add lagged features
        for col in ['Gold_Price', 'USD_Index', 'Oil_Price', 'SP500', 'Treasury_Yield', 'VIX']:
            for lag in [1, 5, 10]:
                data[f'{col}_Lag_{lag}'] = data[col].shift(lag)
        
        # Add rolling statistics
        for col in ['Gold_Price', 'USD_Index', 'Oil_Price', 'SP500', 'Treasury_Yield', 'VIX']:
            for window in [5, 10, 20]:
                data[f'{col}_Rolling_Mean_{window}'] = data[col].rolling(window=window).mean()
                data[f'{col}_Rolling_Std_{window}'] = data[col].rolling(window=window).std()
        
        # Add exponential moving averages
        for col in ['Gold_Price', 'USD_Index', 'Oil_Price', 'SP500', 'Treasury_Yield', 'VIX']:
            for span in [5, 10, 20]:
                data[f'{col}_EMA_{span}'] = data[col].ewm(span=span, adjust=False).mean()
        
        # Drop rows with NaN values
        data = dropna(data)
        
        # Normalize the data
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
        
        logger.info(f"Prepared data shape: {data_scaled.shape}")
        logger.info(f"Columns: {data_scaled.columns}")
        
        return data_scaled
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise