from sklearn.metrics import mean_squared_error, r2_score

def run_backtests(models, data):
    results = {}
    X = data.drop(['Close', 'Date'], axis=1)
    y = data['Close']
    
    for name, model in models.items():
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        results[name] = {'mse': mse, 'r2': r2}
    
    return results