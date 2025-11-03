#!/usr/bin/env python3
"""
Extract and visualize price elasticity curves from trained NAM model
Shows elasticity for aggregate and individual products
"""
import os
os.environ['KERAS_BACKEND'] = 'jax'

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras

# Import custom classes before unpickling
from src.models.simple_nam import SimpleNAM
from src.training.loss_functions import NAMLoss

sns.set_style("whitegrid")


def extract_feature_contribution(model, X_baseline, feature_idx, feature_range):
    """
    Extract contribution of a single feature by varying it

    Args:
        model: Trained NAM model
        X_baseline: Baseline feature values (scaled)
        feature_idx: Index of feature to vary
        feature_range: Range of values to test

    Returns:
        predictions: Model predictions for each feature value
    """
    predictions = []

    for value in feature_range:
        # Create input with this feature varied
        X_test = X_baseline.copy()
        X_test[:, feature_idx] = value

        # Get prediction
        pred = model.predict(X_test, verbose=0).flatten()[0]
        predictions.append(pred)

    return np.array(predictions)


def plot_price_elasticity(model_path='outputs/elasticity_data.pkl'):
    """Plot price elasticity curves"""
    print("\n=== Extracting Price Elasticity Curves ===")

    # Load saved data
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    data_scaled = data['data_scaled']
    scalers = data['scalers']

    print(f"Model loaded: {model.count_params():,} parameters")

    # Prepare baseline (use median values)
    from src.training.trainer import NAMTrainer
    X_all, y_all = NAMTrainer.prepare_data_for_keras(data_scaled)

    X_baseline = np.median(X_all, axis=0, keepdims=True)

    # Find price-related features
    numeric_cols = data_scaled.select_dtypes(include=[np.number]).columns.tolist()

    # Drop target columns to get feature list
    target_cols = ['total_gmv_log', 'total_gmv', 'GMV_log', 'GMV']
    feature_cols = [col for col in numeric_cols if col not in target_cols]

    # Identify price features
    price_features = []
    for i, col in enumerate(feature_cols):
        if 'Price' in col or 'price' in col or 'MRP' in col:
            price_features.append((i, col))

    if not price_features:
        print("No price features found. Using first 3 features for demonstration.")
        price_features = [(i, feature_cols[i]) for i in range(min(3, len(feature_cols)))]

    print(f"Found {len(price_features)} price-related features")

    # Create figure
    n_plots = min(len(price_features), 10)  # Top 10
    n_rows = (n_plots + 2) // 3  # 3 columns
    n_cols = min(3, n_plots)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_plots > 1 else axes

    for plot_idx, (feat_idx, feat_name) in enumerate(price_features[:n_plots]):
        ax = axes[plot_idx] if n_plots > 1 else axes[0]

        # Vary this feature from -3 to +3 (scaled range)
        feature_range = np.linspace(-3, 3, 50)

        # Get predictions
        contributions = extract_feature_contribution(model, X_baseline, feat_idx, feature_range)

        # Plot
        ax.plot(feature_range, contributions, linewidth=3, color='#2E86AB')
        ax.axvline(X_baseline[0, feat_idx], color='red', linestyle='--', linewidth=2, label='Current Value')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

        # Mark optimal point
        optimal_idx = np.argmax(contributions)
        ax.scatter([feature_range[optimal_idx]], [contributions[optimal_idx]],
                   s=200, color='gold', edgecolor='k', linewidth=2, zorder=5, label='Optimal')

        ax.set_xlabel(f'{feat_name} (Scaled)', fontsize=10)
        ax.set_ylabel('GMV Contribution', fontsize=10)
        ax.set_title(f'{feat_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Calculate elasticity (slope at current point)
        current_idx = np.argmin(np.abs(feature_range - X_baseline[0, feat_idx]))
        if current_idx > 0 and current_idx < len(feature_range) - 1:
            elasticity = (contributions[current_idx+1] - contributions[current_idx-1]) / (feature_range[current_idx+1] - feature_range[current_idx-1])
            ax.text(0.05, 0.95, f'Elasticity: {elasticity:.3f}',
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('NAM Feature Contribution Curves (Price Elasticity)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = 'outputs/figures/elasticity_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_marketing_response_curves(model_path='outputs/elasticity_data.pkl'):
    """Plot marketing investment response curves"""
    print("\n=== Extracting Marketing Response Curves ===")

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    data_scaled = data['data_scaled']

    # Prepare baseline
    from src.training.trainer import NAMTrainer
    X_all, y_all = NAMTrainer.prepare_data_for_keras(data_scaled)
    X_baseline = np.median(X_all, axis=0, keepdims=True)

    # Get feature names
    numeric_cols = data_scaled.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = ['total_gmv_log', 'total_gmv', 'GMV_log', 'GMV']
    feature_cols = [col for col in numeric_cols if col not in target_cols]

    # Find marketing features
    marketing_features = []
    for i, col in enumerate(feature_cols):
        if any(ch in col for ch in ['TV', 'Digital', 'SEM', 'Radio', 'Sponsor', 'marketing', 'Investment']):
            marketing_features.append((i, col))

    if not marketing_features:
        print("No marketing features found.")
        return

    print(f"Found {len(marketing_features)} marketing features")

    # Create figure
    n_plots = min(len(marketing_features), 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for plot_idx, (feat_idx, feat_name) in enumerate(marketing_features[:n_plots]):
        ax = axes[plot_idx]

        feature_range = np.linspace(-2, 3, 50)
        contributions = extract_feature_contribution(model, X_baseline, feat_idx, feature_range)

        ax.plot(feature_range, contributions, linewidth=3, color='#06A77D')
        ax.axvline(X_baseline[0, feat_idx], color='red', linestyle='--', linewidth=2, label='Current Investment')
        ax.fill_between(feature_range, contributions, alpha=0.2, color='#06A77D')

        ax.set_xlabel(f'{feat_name} (Scaled)', fontsize=10)
        ax.set_ylabel('GMV Contribution', fontsize=10)
        ax.set_title(f'{feat_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Marketing Investment Response Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = 'outputs/figures/marketing_response_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("ELASTICITY & RESPONSE CURVE EXTRACTION")
    print("="*70)

    # Generate elasticity curves
    plot_price_elasticity()

    # Generate marketing response curves
    plot_marketing_response_curves()

    print("\n" + "="*70)
    print("ELASTICITY ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. outputs/figures/elasticity_curves.png")
    print("  2. outputs/figures/marketing_response_curves.png")
    print("\nThese show the learned feature shapes from your NAM model!")
    print("Use these curves for investment optimization and pricing decisions.")
