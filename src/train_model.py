import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import aggregate_by_granularity, evaluate_granularity

PREDICTION_YEAR = 2027
MIN_DATA_POINTS = 5

def train_and_evaluate(processed_dir, models_dir):
    """
    Load processed aggregation data, train Linear Regression and KNN models
    for every valid Country × Disaster Type combination, select the best model,
    save results to CSV, and persist trained models to disk via pickle.
    """
    os.makedirs(models_dir, exist_ok=True)

    agg_path = os.path.join(processed_dir, 'processed_agg.csv')
    if not os.path.exists(agg_path):
        raise FileNotFoundError(
            f"Aggregated data not found at {agg_path}. "
            "Please run data_preprocessing.py first."
        )

    agg = pd.read_csv(agg_path)

    # ── Filter: only combinations with enough data points ─────────────────────
    count_per_combo = (
        agg.groupby(['Country', 'Disaster Type'])['Year']
        .count()
        .reset_index(name='n_years')
    )
    valid_combinations = count_per_combo[count_per_combo['n_years'] >= MIN_DATA_POINTS]
    print(f"Processing {len(valid_combinations)} combinations with 2 models...\n")

    comparison_results = []
    trained_models = {}          # {(country, disaster): model_object}

    for _, row in valid_combinations.iterrows():
        country = row['Country']
        disaster  = row['Disaster Type']
        subset = agg[(agg['Country'] == country) & (agg['Disaster Type'] == disaster)]

        best_gran = evaluate_granularity(subset)
        if best_gran is None:
            continue

        g      = best_gran['granularity']
        binned = aggregate_by_granularity(subset, g)
        if len(binned) < 4:
            continue

        X = binned['Bin'].values.reshape(-1, 1)
        y = binned['Disaster_Count'].values
        n_test = max(1, int(len(binned) * 0.3))
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]
        if len(X_train) < 2:
            continue

        # ── Linear Regression ─────────────────────────────────────────────────
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        mae_lr    = mean_absolute_error(y_test, y_pred_lr)
        rmse_lr   = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        r2_lr     = r2_score(y_test, y_pred_lr) if len(y_test) > 1 else 0
        pred_lr   = max(0, round(lr.predict([[(PREDICTION_YEAR // g) * g]])[0], 2))

        # ── KNN Regression ────────────────────────────────────────────────────
        k   = min(3, len(X_train))
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        mae_knn    = mean_absolute_error(y_test, y_pred_knn)
        rmse_knn   = np.sqrt(mean_squared_error(y_test, y_pred_knn))
        r2_knn     = r2_score(y_test, y_pred_knn) if len(y_test) > 1 else 0
        pred_knn   = max(0, round(knn.predict([[(PREDICTION_YEAR // g) * g]])[0], 2))

        # ── Select best model by MAE ──────────────────────────────────────────
        if mae_lr <= mae_knn:
            best_model_name = 'Linear Regression'
            best_prediction  = pred_lr
            best_model    = lr
        else:
            best_model_name = 'KNN Regression'
            best_prediction  = pred_knn
            best_model    = knn

        comparison_results.append({
            'Country': country,
            'Disaster Type': disaster,
            'Granularity (years)': g,
            'MAE_LR':   round(mae_lr, 3),
            'RMSE_LR':  round(rmse_lr, 3),
            'R2_LR':    round(r2_lr, 3),
            'MAE_KNN':  round(mae_knn, 3),
            'RMSE_KNN': round(rmse_knn, 3),
            'R2_KNN':   round(r2_knn, 3),
            'Best_Model': best_model_name,
            f'Prediction_{PREDICTION_YEAR}': best_prediction,
        })

        trained_models[(country, disaster)] = {
            'model':       best_model,
            'model_name':  best_model_name,
            'granularity': g,
            'lr_model':    lr,
            'knn_model':   knn,
        }

    # ── Save comparison CSV ───────────────────────────────────────────────────
    comparison_df = pd.DataFrame(comparison_results)
    csv_path = os.path.join(processed_dir, 'complete_prediction_results.csv')
    comparison_df.to_csv(csv_path, index=False)

    # ── Persist trained models ────────────────────────────────────────────────
    pkl_path = os.path.join(models_dir, 'best_models.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(trained_models, f)

    print(f"[SUCCESS] Done! {len(comparison_df)} combinations processed.")
    print(f"   - Saved results : {csv_path}")
    print(f"   - Saved models  : {pkl_path}")

    return comparison_df, trained_models

if __name__ == '__main__':
    project_root  = os.environ.get('PROJECT_ROOT', os.path.abspath('.'))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir    = os.path.join(project_root, 'models')
    train_and_evaluate(processed_dir, models_dir)
