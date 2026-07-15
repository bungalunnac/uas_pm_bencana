import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def aggregate_by_granularity(data, granularity):
    """Group data into temporal bins according to the selected granularity."""
    temp = data.copy()
    temp['Bin'] = (temp['Year'] // granularity) * granularity
    return temp.groupby('Bin')['Disaster_Count'].sum().reset_index()

def evaluate_granularity(data, granularity_list=[1, 2, 3, 5]):
    """Evaluate multiple granularities and return the one with the smallest MAE."""
    results = []
    for g in granularity_list:
        binned = aggregate_by_granularity(data, g)
        if len(binned) < 4:
            continue
        X = binned['Bin'].values.reshape(-1, 1)
        y = binned['Disaster_Count'].values
        n_test = max(1, int(len(binned) * 0.3))
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]
        if len(X_train) < 2:
            continue
        model = LinearRegression()
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        results.append({
            'granularity': g, 
            'mae': mae,
            'model': model, 
            'binned_data': binned
        })
    return min(results, key=lambda x: x['mae']) if results else None
