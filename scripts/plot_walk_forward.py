#!/usr/bin/env python3
"""
Visualize Walk-Forward Optimization Results
Shows complete out-of-sample predictions across all folds
"""
import os
os.environ['KERAS_BACKEND'] = 'jax'

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

def plot_walk_forward_time_series():
    """Plot complete OOS predictions vs actuals across all folds"""
    print("\n=== Loading Walk-Forward Results ===")

    results_file = 'outputs/walk_forward_results.pkl'
    if not Path(results_file).exists():
        print(f"ERROR: {results_file} not found. Run main.py with walk_forward enabled first.")
        return

    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    fold_results = results['fold_results']
    oos_predictions_scaled = results['oos_predictions']
    oos_actuals_scaled = results['oos_actuals']
    scalers = results['scalers']

    print(f"Loaded {len(fold_results)} folds with {len(oos_actuals_scaled)} total OOS predictions")

    # Inverse transform to original scale
    scaler = scalers['standard']

    # Assuming total_gmv_log is first in scaled features
    # Find index properly
    try:
        # Recreate the scaled features list to find index
        from src.data.data_loader import DataLoader
        from src.data.data_preprocessing import DataPreprocessor
        from src.data.feature_engineering import FeatureEngineer

        loader = DataLoader('data/raw')
        data = loader.load_secondfile()
        preprocessor = DataPreprocessor({})
        data = preprocessor.handle_missing_values(data)
        engineer = FeatureEngineer({})
        data = engineer.engineer_all_features(data)
        data_scaled, _ = preprocessor.scale_features(data)

        scaled_features = [col for col in data_scaled.columns if col.endswith('_log')]
        other_features = ['avg_price', 'nps_score', 'time_index', 'NPS', 'month_sin', 'month_cos']
        all_scaled = scaled_features + [f for f in other_features if f in data_scaled.columns]

        gmv_idx = all_scaled.index('total_gmv_log')
        mean_gmv = scaler.mean_[gmv_idx]
        std_gmv = scaler.scale_[gmv_idx]

        # Inverse transform
        actuals_log = oos_actuals_scaled * std_gmv + mean_gmv
        predictions_log = oos_predictions_scaled * std_gmv + mean_gmv

        actuals_original = np.expm1(actuals_log)
        predictions_original = np.expm1(predictions_log)

        print(f"Inverse transformed to original GMV scale")
        print(f"  Actuals range: {actuals_original.min():,.0f} to {actuals_original.max():,.0f}")
        print(f"  Predictions range: {predictions_original.min():,.0f} to {predictions_original.max():,.0f}")

    except Exception as e:
        print(f"Warning: Could not inverse transform: {e}")
        actuals_original = oos_actuals_scaled
        predictions_original = oos_predictions_scaled

    # Create comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Complete OOS Time Series
    ax = axes[0, 0]
    indices = np.arange(len(actuals_original))
    ax.plot(indices, actuals_original, 'o-', label='Actual GMV', linewidth=2, markersize=8, color='#2E86AB')
    ax.plot(indices, predictions_original, 's--', label='Predicted GMV', linewidth=2, markersize=8, color='#A23B72', alpha=0.8)

    # Add fold boundaries
    samples_per_fold = []
    for fold in fold_results:
        samples_per_fold.append(1)  # 1 month per fold

    cum_samples = np.cumsum(samples_per_fold)
    for i, boundary in enumerate(cum_samples[:-1]):
        ax.axvline(boundary - 0.5, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Month Index (Out-of-Sample)', fontsize=12, fontweight='bold')
    ax.set_ylabel('GMV (Original Scale)', fontsize=12, fontweight='bold')
    ax.set_title(f'Walk-Forward: Out-of-Sample Predictions ({len(fold_results)} Folds)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: Scatter Plot
    ax = axes[0, 1]
    ax.scatter(actuals_original, predictions_original, alpha=0.7, s=150, edgecolors='k', linewidth=1.5)

    # Perfect prediction line
    min_val = min(actuals_original.min(), predictions_original.min())
    max_val = max(actuals_original.max(), predictions_original.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(actuals_original, predictions_original)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Actual GMV', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted GMV', fontsize=12, fontweight='bold')
    ax.set_title('Actual vs Predicted (OOS)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Fold-by-Fold Performance
    ax = axes[1, 0]
    fold_indices = [f['fold'] for f in fold_results]
    fold_r2s = [f['r2'] for f in fold_results]
    fold_mapes = [f['mape'] for f in fold_results]

    ax.bar(fold_indices, fold_r2s, alpha=0.7, color='#06A77D', edgecolor='k')
    ax.axhline(np.mean(fold_r2s), color='red', linestyle='--', linewidth=2, label=f'Mean R² = {np.mean(fold_r2s):.3f}')
    ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('R² by Fold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Metrics Summary
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate overall metrics
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

    mape = mean_absolute_percentage_error(actuals_original, predictions_original) * 100
    rmse = np.sqrt(mean_squared_error(actuals_original, predictions_original))
    mae = np.mean(np.abs(actuals_original - predictions_original))

    summary_text = "WALK-FORWARD VALIDATION SUMMARY\n\n"
    summary_text += f"Total Folds: {len(fold_results)}\n"
    summary_text += f"Total OOS Samples: {len(actuals_original)}\n\n"

    summary_text += "OVERALL METRICS:\n"
    summary_text += f"  R²:   {r2:.4f}\n"
    summary_text += f"  MAPE: {mape:.2f}%\n"
    summary_text += f"  RMSE: {rmse:,.0f}\n"
    summary_text += f"  MAE:  {mae:,.0f}\n\n"

    summary_text += "FOLD STABILITY:\n"
    summary_text += f"  R² Mean:  {np.mean(fold_r2s):.4f}\n"
    summary_text += f"  R² Std:   {np.std(fold_r2s):.4f}\n"
    summary_text += f"  R² Range: [{np.min(fold_r2s):.3f}, {np.max(fold_r2s):.3f}]\n\n"

    summary_text += "MAPE BY FOLD:\n"
    summary_text += f"  Mean:  {np.mean(fold_mapes):.2f}%\n"
    summary_text += f"  Std:   {np.std(fold_mapes):.2f}%\n"
    summary_text += f"  Range: [{np.min(fold_mapes):.1f}%, {np.max(fold_mapes):.1f}%]"

    ax.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.8, edgecolor='#2E86AB', linewidth=2))

    plt.tight_layout()
    output_path = 'outputs/figures/walk_forward_complete.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Also create a detailed time series plot
    create_detailed_time_series_plot(actuals_original, predictions_original, fold_results)


def create_detailed_time_series_plot(actuals, predictions, fold_results):
    """Create detailed time series plot with dates"""
    print("\n=== Creating Detailed Time Series Plot ===")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Plot 1: Actual vs Predicted over time
    ax = axes[0]
    indices = np.arange(len(actuals))

    ax.plot(indices, actuals, 'o-', label='Actual', linewidth=3, markersize=10, color='#2E86AB')
    ax.plot(indices, predictions, 's--', label='Predicted', linewidth=3, markersize=10, color='#A23B72', alpha=0.8)

    # Shade each fold
    colors = plt.cm.Set3(np.linspace(0, 1, len(fold_results)))
    start_idx = 0
    for i, fold in enumerate(fold_results):
        end_idx = start_idx + 1  # 1 sample per fold
        ax.axvspan(start_idx - 0.5, end_idx - 0.5, alpha=0.1, color=colors[i], label=f'Fold {i}' if i < 3 else '')
        start_idx = end_idx

    ax.set_xlabel('Out-of-Sample Month Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('GMV', fontsize=13, fontweight='bold')
    ax.set_title('Complete Out-of-Sample Predictions (All Walk-Forward Folds)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.4)

    # Plot 2: Prediction Errors
    ax = axes[1]
    errors = actuals - predictions
    error_pct = (errors / actuals) * 100

    ax.bar(indices, error_pct, alpha=0.7, color=['green' if e < 0 else 'red' for e in errors], edgecolor='k')
    ax.axhline(0, color='k', linestyle='-', linewidth=2)
    ax.set_xlabel('Out-of-Sample Month Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Error (%)', fontsize=13, fontweight='bold')
    ax.set_title('Prediction Error by Month', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.4, axis='y')

    # Add statistics
    mean_error_pct = np.mean(np.abs(error_pct))
    ax.text(0.02, 0.98, f'Mean Absolute Error: {mean_error_pct:.1f}%',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    output_path = 'outputs/figures/walk_forward_detailed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("WALK-FORWARD OPTIMIZATION VISUALIZATION")
    print("="*70)

    plot_walk_forward_time_series()

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. outputs/figures/walk_forward_complete.png")
    print("  2. outputs/figures/walk_forward_detailed.png")
    print("\nThese charts show the COMPLETE out-of-sample time series")
    print("across all walk-forward folds for proper trend analysis!")
