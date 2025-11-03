#!/usr/bin/env python3
"""
Comprehensive visualization of NAM training results
Creates all diagnostic plots for model evaluation
"""
import os
os.environ['KERAS_BACKEND'] = 'jax'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_training_history(history_file='outputs/training_log.csv'):
    """Plot training and validation loss curves"""
    print("\n=== Plotting Training History ===")

    if not Path(history_file).exists():
        print(f"Warning: {history_file} not found. Skipping training history plot.")
        return

    # Load training history
    history = pd.read_csv(history_file)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Loss curves
    axes[0].plot(history['epoch'], history['loss'], label='Training Loss', linewidth=2, marker='o', markersize=3)
    if 'val_loss' in history.columns:
        axes[0].plot(history['epoch'], history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=3)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: MAE curves
    axes[1].plot(history['epoch'], history['mae'], label='Training MAE', linewidth=2, marker='o', markersize=3)
    if 'val_mae' in history.columns:
        axes[1].plot(history['epoch'], history['val_mae'], label='Validation MAE', linewidth=2, marker='s', markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'outputs/figures/training_history.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_predictions_vs_actual(model_path='outputs/models/final_nam_model.keras'):
    """Plot predictions vs actual for train, val, and test sets"""
    print("\n=== Plotting Predictions vs Actual ===")

    import keras
    from src.data.data_loader import DataLoader
    from src.data.data_preprocessing import DataPreprocessor
    from src.data.feature_engineering import FeatureEngineer
    from src.training.trainer import NAMTrainer

    # Load data and process
    loader = DataLoader('data/raw')
    data = loader.load_secondfile()

    preprocessor = DataPreprocessor({})
    data = preprocessor.handle_missing_values(data)
    data = preprocessor.treat_outliers(data)

    engineer = FeatureEngineer({})
    data = engineer.engineer_all_features(data)

    data_scaled, scalers = preprocessor.scale_features(data)

    # Split data
    train_data, val_data, test_data = preprocessor.time_series_split(data_scaled)

    # Load model
    model = keras.models.load_model(model_path)

    # Get predictions for all sets
    X_train, y_train = NAMTrainer.prepare_data_for_keras(train_data)
    X_val, y_val = NAMTrainer.prepare_data_for_keras(val_data)
    X_test, y_test = NAMTrainer.prepare_data_for_keras(test_data)

    train_pred = model.predict(X_train).flatten()
    val_pred = model.predict(X_val).flatten()
    test_pred = model.predict(X_test).flatten()

    # Inverse transform all to original scale
    scaler = scalers['standard']
    scaled_features = [col for col in data_scaled.columns if col.endswith('_log')]
    other_features = ['avg_price', 'nps_score', 'time_index', 'NPS', 'month_sin', 'month_cos']
    all_scaled = scaled_features + [f for f in other_features if f in data_scaled.columns]

    try:
        gmv_idx = all_scaled.index('total_gmv_log')
        mean_gmv = scaler.mean_[gmv_idx]
        std_gmv = scaler.scale_[gmv_idx]

        # Inverse transform
        def inverse_transform(scaled_vals):
            log_vals = scaled_vals * std_gmv + mean_gmv
            return np.expm1(log_vals)

        y_train_orig = inverse_transform(y_train)
        y_val_orig = inverse_transform(y_val)
        y_test_orig = inverse_transform(y_test)
        train_pred_orig = inverse_transform(train_pred)
        val_pred_orig = inverse_transform(val_pred)
        test_pred_orig = inverse_transform(test_pred)

    except:
        # Fallback
        y_train_orig, y_val_orig, y_test_orig = y_train, y_val, y_test
        train_pred_orig, val_pred_orig, test_pred_orig = train_pred, val_pred, test_pred

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: All data time series
    ax = axes[0, 0]
    all_actual = np.concatenate([y_train_orig, y_val_orig, y_test_orig])
    all_pred = np.concatenate([train_pred_orig, val_pred_orig, test_pred_orig])
    indices = np.arange(len(all_actual))

    ax.plot(indices[:len(y_train_orig)], y_train_orig, 'o-', label='Train Actual', linewidth=2)
    ax.plot(indices[:len(y_train_orig)], train_pred_orig, 's--', label='Train Predicted', linewidth=2, alpha=0.7)
    ax.plot(indices[len(y_train_orig):len(y_train_orig)+len(y_val_orig)], y_val_orig, 'o-', label='Val Actual', linewidth=2)
    ax.plot(indices[len(y_train_orig):len(y_train_orig)+len(y_val_orig)], val_pred_orig, 's--', label='Val Predicted', linewidth=2, alpha=0.7)
    ax.plot(indices[-len(y_test_orig):], y_test_orig, 'o-', label='Test Actual', linewidth=2)
    ax.plot(indices[-len(y_test_orig):], test_pred_orig, 's--', label='Test Predicted', linewidth=2, alpha=0.7)

    ax.axvline(len(y_train_orig)-0.5, color='red', linestyle=':', alpha=0.5, label='Train/Val Split')
    ax.axvline(len(y_train_orig)+len(y_val_orig)-0.5, color='orange', linestyle=':', alpha=0.5, label='Val/Test Split')

    ax.set_xlabel('Time Index', fontsize=12)
    ax.set_ylabel('GMV (Original Scale)', fontsize=12)
    ax.set_title('Predicted vs Actual - Full Time Series', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: Scatter plot (all data)
    ax = axes[0, 1]
    ax.scatter(y_train_orig, train_pred_orig, alpha=0.6, s=100, label='Train', edgecolors='k')
    ax.scatter(y_val_orig, val_pred_orig, alpha=0.6, s=100, label='Val', edgecolors='k')
    ax.scatter(y_test_orig, test_pred_orig, alpha=0.6, s=150, label='Test', edgecolors='k', marker='s')

    # Perfect prediction line
    all_vals = np.concatenate([y_train_orig, y_val_orig, y_test_orig])
    min_val, max_val = all_vals.min(), all_vals.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual GMV', fontsize=12)
    ax.set_ylabel('Predicted GMV', fontsize=12)
    ax.set_title('Predicted vs Actual Scatter', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Residuals
    ax = axes[1, 0]
    train_resid = y_train_orig - train_pred_orig
    val_resid = y_val_orig - val_pred_orig
    test_resid = y_test_orig - test_pred_orig

    ax.scatter(range(len(train_resid)), train_resid, alpha=0.6, s=100, label='Train', edgecolors='k')
    ax.scatter(range(len(train_resid), len(train_resid)+len(val_resid)), val_resid, alpha=0.6, s=100, label='Val', edgecolors='k')
    ax.scatter(range(len(train_resid)+len(val_resid), len(train_resid)+len(val_resid)+len(test_resid)), test_resid, alpha=0.6, s=150, label='Test', edgecolors='k', marker='s')

    ax.axhline(0, color='k', linestyle='--', linewidth=2)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Metrics summary
    ax = axes[1, 1]
    ax.axis('off')

    # Compute metrics
    from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

    metrics_text = "MODEL PERFORMANCE SUMMARY\n\n"

    # Train metrics
    train_r2 = r2_score(y_train_orig, train_pred_orig)
    train_mape = mean_absolute_percentage_error(y_train_orig, train_pred_orig) * 100
    train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred_orig))

    metrics_text += f"TRAIN SET (n={len(y_train_orig)}):\n"
    metrics_text += f"  R²:   {train_r2:.4f}\n"
    metrics_text += f"  MAPE: {train_mape:.2f}%\n"
    metrics_text += f"  RMSE: {train_rmse:,.0f}\n\n"

    # Val metrics
    if len(y_val_orig) > 1:
        val_r2 = r2_score(y_val_orig, val_pred_orig)
    else:
        val_r2 = np.nan
    val_mape = mean_absolute_percentage_error(y_val_orig, val_pred_orig) * 100
    val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig))

    metrics_text += f"VALIDATION SET (n={len(y_val_orig)}):\n"
    metrics_text += f"  R²:   {val_r2:.4f}\n"
    metrics_text += f"  MAPE: {val_mape:.2f}%\n"
    metrics_text += f"  RMSE: {val_rmse:,.0f}\n\n"

    # Test metrics
    if len(y_test_orig) > 1:
        test_r2 = r2_score(y_test_orig, test_pred_orig)
    else:
        test_r2 = np.nan
    test_mape = mean_absolute_percentage_error(y_test_orig, test_pred_orig) * 100
    test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))

    metrics_text += f"TEST SET (n={len(y_test_orig)}):\n"
    metrics_text += f"  R²:   {test_r2:.4f}\n"
    metrics_text += f"  MAPE: {test_mape:.2f}%\n"
    metrics_text += f"  RMSE: {test_rmse:,.0f}\n"

    ax.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = 'outputs/figures/comprehensive_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_loss_convergence(history_file='outputs/training_log.csv'):
    """Plot detailed loss convergence with log scale"""
    print("\n=== Plotting Loss Convergence ===")

    if not Path(history_file).exists():
        print(f"Warning: {history_file} not found.")
        return

    history = pd.read_csv(history_file)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Linear scale
    ax = axes[0]
    ax.plot(history['epoch'], history['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history.columns:
        ax.plot(history['epoch'], history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Convergence (Linear Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale
    ax = axes[1]
    ax.semilogy(history['epoch'], history['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history.columns:
        ax.semilogy(history['epoch'], history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Loss Convergence (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = 'outputs/figures/loss_convergence.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all plots"""
    print("="*70)
    print("GENERATING COMPREHENSIVE TRAINING VISUALIZATIONS")
    print("="*70)

    Path('outputs/figures').mkdir(parents=True, exist_ok=True)

    # Plot 1: Training history
    plot_training_history()

    # Plot 2: Loss convergence
    plot_loss_convergence()

    # Plot 3: Predictions vs Actual
    plot_predictions_vs_actual()

    print("\n" + "="*70)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("="*70)
    print("\nGenerated files:")
    print("  1. outputs/figures/training_history.png")
    print("  2. outputs/figures/loss_convergence.png")
    print("  3. outputs/figures/comprehensive_results.png")
    print("  4. outputs/figures/actual_vs_predicted.png (from main.py)")
    print("\nView with: start outputs/figures/*.png")


if __name__ == "__main__":
    main()
