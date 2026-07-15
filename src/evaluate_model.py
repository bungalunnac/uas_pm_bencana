import os
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for saving files
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_model_comparison(comparison_df, reports_dir):
    """Generate boxplot comparing LR vs KNN across MAE, RMSE, R² metrics."""
    os.makedirs(reports_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Performance Comparison: Linear Regression vs KNN Regression',
        fontsize=13, fontweight='bold'
    )
    for ax, (col_lr, col_knn), label in zip(
        axes,
        [('MAE_LR', 'MAE_KNN'), ('RMSE_LR', 'RMSE_KNN'), ('R2_LR', 'R2_KNN')],
        ['MAE', 'RMSE', 'R2']
    ):
        ax.boxplot(
            [comparison_df[col_lr], comparison_df[col_knn]]
        )
        ax.set_xticklabels(['Linear Regression', 'KNN Regression'])
        ax.set_title(f'{label} Distribution')
        ax.set_ylabel(label)
    plt.tight_layout()
    path = os.path.join(reports_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")

def plot_feature_importance(comparison_df, processed_dir, reports_dir):
    """Generate the exact same double-plot Feature Importance as Google Colab using regression coefficients."""
    os.makedirs(reports_dir, exist_ok=True)
    agg_path = os.path.join(processed_dir, 'processed_agg.csv')
    if not os.path.exists(agg_path):
        print("⚠️ processed_agg.csv not found, skipping coefficient-based feature importance.")
        return

    agg = pd.read_csv(agg_path)
    from sklearn.linear_model import LinearRegression

    koef_list = []
    # Loop over all valid combinations in comparison_df
    for idx, row in comparison_df.iterrows():
        negara = row['Country']
        jenis = row['Disaster Type']
        g = row.get('Granularity (years)', row.get('Granularity_Best', 1))
        
        subset = agg[(agg['Country'] == negara) & (agg['Disaster Type'] == jenis)]
        # Binned data aggregation logic
        temp = subset.copy()
        temp['Bin'] = (temp['Year'] // g) * g
        binned = temp.groupby('Bin')['Disaster_Count'].sum().reset_index()
        
        if len(binned) < 4:
            continue
        X_fi = binned['Bin'].values.reshape(-1, 1)
        y_fi = binned['Disaster_Count'].values
        m = LinearRegression()
        m.fit(X_fi, y_fi)
        koef_list.append({'Country': negara, 'Disaster Type': jenis, 'Coefficient': m.coef_[0]})

    if not koef_list:
        print("⚠️ No valid coefficients computed.")
        return

    koef_df = pd.DataFrame(koef_list)

    # Visualization - Match Colab
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Feature Importance — Tren Waktu per Disaster Type', fontsize=13, fontweight='bold')

    avg_koef = koef_df.groupby('Disaster Type')['Coefficient'].mean().sort_values()
    colors   = ['#E74C3C' if x > 0 else '#2E86C1' for x in avg_koef.values]
    axes[0].barh(avg_koef.index, avg_koef.values, color=colors, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=0.8)
    axes[0].set_title('Average Coefficient by Disaster Type\n(Red=Increasing, Blue=Decreasing)')
    axes[0].set_xlabel('Average Coefficient')

    axes[1].hist(koef_df['Coefficient'], bins=30, color='#9B59B6', edgecolor='black', alpha=0.8)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Coefficient=0')
    axes[1].axvline(x=koef_df['Coefficient'].mean(), color='orange', linestyle='--',
                    linewidth=1.5, label=f"Mean={koef_df['Coefficient'].mean():.4f}")
    axes[1].set_title('Coefficient Distribution Across All Combinations')
    axes[1].set_xlabel('Coefficient')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(reports_dir, 'feature_importance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")

def evaluate_and_save_reports(processed_dir, reports_dir):
    """Load comparison CSV and generate all evaluation report charts."""
    csv_path = os.path.join(processed_dir, 'complete_prediction_results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Prediction CSV not found at {csv_path}. "
            "Please run train_model.py first."
        )

    df = pd.read_csv(csv_path)

    # Rename column for convenience (handles variants)
    if 'Prediction_2027' not in df.columns:
        pred_col = [c for c in df.columns if c.startswith('Prediction_') or c.startswith('Prediction_')]
        if pred_col:
            df = df.rename(columns={pred_col[0]: 'Prediction_2027'})

    plot_model_comparison(df, reports_dir)
    plot_feature_importance(df, processed_dir, reports_dir)

    # Print summary stats
    lr_wins  = (df['Best_Model'] == 'Linear Regression').sum()
    knn_wins = (df['Best_Model'] == 'KNN Regression').sum()
    total    = len(df)
    winner   = 'Linear Regression' if lr_wins >= knn_wins else 'KNN Regression'

    print("\n=== PERFORMANCE COMPARISON ===\n")
    print(f"{'Metric':<10} {'Linear Regression':>20} {'KNN Regression':>15}")
    print("-" * 48)
    print(f"{'MAE':<10} {df['MAE_LR'].mean():>20.3f} {df['MAE_KNN'].mean():>15.3f}")
    print(f"{'RMSE':<10} {df['RMSE_LR'].mean():>20.3f} {df['RMSE_KNN'].mean():>15.3f}")
    print(f"{'R2':<10} {df['R2_LR'].mean():>20.3f} {df['R2_KNN'].mean():>15.3f}")
    print(f"\nBest model  : {winner}")
    print(f"   LR  wins in {lr_wins}/{total} combinations ({lr_wins/total*100:.1f}%)")
    print(f"   KNN wins in {knn_wins}/{total} combinations ({knn_wins/total*100:.1f}%)")

if __name__ == '__main__':
    processed_dir = os.path.join('data', 'processed')
    reports_dir   = os.path.join('reports')
    evaluate_and_save_reports(processed_dir, reports_dir)
