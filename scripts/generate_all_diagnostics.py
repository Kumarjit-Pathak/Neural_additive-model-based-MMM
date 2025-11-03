#!/usr/bin/env python
"""
Generate All Diagnostic Plots and Analyses for NAM Model
This script generates comprehensive visualizations and metrics
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle

warnings.filterwarnings('ignore')
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Import our modules
from src.models.simple_nam import SimpleNAM
from src.data.unified_pipeline import UnifiedDataPipeline
from src.evaluation.metrics import ModelEvaluator
from loguru import logger

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


class DiagnosticGenerator:
    """Generate comprehensive diagnostics for NAM model"""

    def __init__(self, model_path=None):
        """Initialize diagnostic generator"""
        self.model_path = model_path or self._find_latest_model()
        self.pipeline = UnifiedDataPipeline()
        self.evaluator = ModelEvaluator()
        self.model = None
        self.data_dict = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directories
        os.makedirs('plots/diagnostics', exist_ok=True)
        os.makedirs('outputs/analysis', exist_ok=True)

    def _find_latest_model(self):
        """Find the most recent model file"""
        model_files = [f for f in os.listdir('models') if f.endswith('.keras')]
        if not model_files:
            raise ValueError("No model files found in models/ directory")

        # Sort by modification time
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join('models', x)))
        latest = os.path.join('models', model_files[-1])
        logger.info(f"Using latest model: {latest}")
        return latest

    def load_model_and_data(self):
        """Load model and prepare data"""
        logger.info("Loading model and data...")

        # Load model
        self.model = keras.models.load_model(self.model_path, compile=False)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        logger.info(f"Model loaded: {self.model_path}")

        # Load and prepare data
        self.data_dict = self.pipeline.get_full_pipeline(verbose=False)
        logger.info(f"Data loaded: {self.data_dict['X_train'].shape[0]} train samples")

    def generate_training_history_plots(self):
        """Generate training history visualizations"""
        logger.info("Generating training history plots...")

        # Try to load training history from CSV
        try:
            # Find latest training log
            log_files = [f for f in os.listdir('outputs/logs') if f.startswith('training_')]
            if log_files:
                log_files.sort()
                latest_log = os.path.join('outputs/logs', log_files[-1])
                history_df = pd.read_csv(latest_log)

                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                # Loss plot
                axes[0,0].plot(history_df.index + 1, history_df['loss'], label='Train Loss', alpha=0.7)
                if 'val_loss' in history_df.columns:
                    axes[0,0].plot(history_df.index + 1, history_df['val_loss'], label='Val Loss', alpha=0.7)
                axes[0,0].set_xlabel('Epoch')
                axes[0,0].set_ylabel('Loss')
                axes[0,0].set_title('Training Loss History')
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)

                # MAE plot
                if 'mae' in history_df.columns:
                    axes[0,1].plot(history_df.index + 1, history_df['mae'], label='Train MAE', alpha=0.7)
                    if 'val_mae' in history_df.columns:
                        axes[0,1].plot(history_df.index + 1, history_df['val_mae'], label='Val MAE', alpha=0.7)
                    axes[0,1].set_xlabel('Epoch')
                    axes[0,1].set_ylabel('MAE')
                    axes[0,1].set_title('Mean Absolute Error')
                    axes[0,1].legend()
                    axes[0,1].grid(True, alpha=0.3)

                # MAPE plot
                if 'mape' in history_df.columns:
                    axes[1,0].plot(history_df.index + 1, history_df['mape'], label='Train MAPE', alpha=0.7)
                    if 'val_mape' in history_df.columns:
                        axes[1,0].plot(history_df.index + 1, history_df['val_mape'], label='Val MAPE', alpha=0.7)
                    axes[1,0].set_xlabel('Epoch')
                    axes[1,0].set_ylabel('MAPE (%)')
                    axes[1,0].set_title('Mean Absolute Percentage Error')
                    axes[1,0].legend()
                    axes[1,0].grid(True, alpha=0.3)

                # Learning rate plot
                if 'learning_rate' in history_df.columns:
                    axes[1,1].plot(history_df.index + 1, history_df['learning_rate'], 'g-', alpha=0.7)
                    axes[1,1].set_xlabel('Epoch')
                    axes[1,1].set_ylabel('Learning Rate')
                    axes[1,1].set_title('Learning Rate Schedule')
                    axes[1,1].grid(True, alpha=0.3)

                plt.suptitle('Training History Analysis', fontsize=14, y=1.02)
                plt.tight_layout()
                plt.savefig('plots/diagnostics/training_history_complete.png', dpi=100, bbox_inches='tight')
                plt.close()
                logger.info("  Saved: plots/diagnostics/training_history_complete.png")

        except Exception as e:
            logger.warning(f"Could not generate training history plots: {e}")

    def generate_prediction_plots(self):
        """Generate prediction vs actual plots"""
        logger.info("Generating prediction plots...")

        # Get predictions
        y_train_pred = self.model.predict(self.data_dict['X_train'], verbose=0)
        y_val_pred = self.model.predict(self.data_dict['X_val'], verbose=0)
        y_test_pred = self.model.predict(self.data_dict['X_test'], verbose=0)

        # Flatten predictions
        y_train_pred = y_train_pred.flatten()
        y_val_pred = y_val_pred.flatten()
        y_test_pred = y_test_pred.flatten()

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Scatter plots
        for ax, y_true, y_pred, title in zip(
            axes[0],
            [self.data_dict['y_train'], self.data_dict['y_val'], self.data_dict['y_test']],
            [y_train_pred, y_val_pred, y_test_pred],
            ['Train Set', 'Validation Set', 'Test Set']
        ):
            # Calculate R²
            r2 = r2_score(y_true, y_pred)

            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.5, s=10)
            ax.plot([y_true.min(), y_true.max()],
                   [y_true.min(), y_true.max()],
                   'r--', lw=2, label='Perfect Fit')
            ax.set_xlabel('Actual (log GMV)')
            ax.set_ylabel('Predicted (log GMV)')
            ax.set_title(f'{title} (R² = {r2:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Residual plots
        for ax, y_true, y_pred, title in zip(
            axes[1],
            [self.data_dict['y_train'], self.data_dict['y_val'], self.data_dict['y_test']],
            [y_train_pred, y_val_pred, y_test_pred],
            ['Train Residuals', 'Validation Residuals', 'Test Residuals']
        ):
            residuals = y_true - y_pred
            ax.scatter(y_pred, residuals, alpha=0.5, s=10)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Predicted (log GMV)')
            ax.set_ylabel('Residual')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Add ±2 std lines
            std_resid = np.std(residuals)
            ax.axhline(y=2*std_resid, color='orange', linestyle=':', alpha=0.5)
            ax.axhline(y=-2*std_resid, color='orange', linestyle=':', alpha=0.5)

        plt.suptitle('Prediction Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('plots/diagnostics/predictions_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: plots/diagnostics/predictions_analysis.png")

    def generate_feature_importance_plot(self):
        """Generate feature importance visualization"""
        logger.info("Generating feature importance plot...")

        try:
            # Calculate feature importance using permutation
            y_test_pred_base = self.model.predict(self.data_dict['X_test'], verbose=0).flatten()
            base_score = r2_score(self.data_dict['y_test'], y_test_pred_base)

            importances = []
            feature_names = self.data_dict['feature_cols']

            for i in range(len(feature_names)):
                X_test_permuted = self.data_dict['X_test'].copy()
                np.random.shuffle(X_test_permuted[:, i])

                y_pred_permuted = self.model.predict(X_test_permuted, verbose=0).flatten()
                permuted_score = r2_score(self.data_dict['y_test'], y_pred_permuted)

                importance = base_score - permuted_score
                importances.append(importance)

            # Create DataFrame and sort
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)

            # Plot top 20 features
            top_features = importance_df.tail(20)

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Permutation Importance (R² decrease)')
            plt.title('Top 20 Feature Importances')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig('plots/diagnostics/feature_importance.png', dpi=100, bbox_inches='tight')
            plt.close()
            logger.info("  Saved: plots/diagnostics/feature_importance.png")

            # Save importance data
            importance_df.to_csv('outputs/analysis/feature_importance.csv', index=False)

        except Exception as e:
            logger.warning(f"Could not generate feature importance: {e}")

    def generate_saturation_curves(self):
        """Generate marketing saturation curves for Beta-Gamma features"""
        logger.info("Generating saturation curves...")

        # Identify Beta-Gamma features
        beta_gamma_indices = [i for i, ft in enumerate(self.data_dict['feature_types'])
                             if ft == 'parametric_beta_gamma']

        if beta_gamma_indices:
            n_features = len(beta_gamma_indices)
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

            for idx, (ax, feat_idx) in enumerate(zip(axes, beta_gamma_indices)):
                if idx < n_features:
                    feature_name = self.data_dict['feature_cols'][feat_idx]

                    # Create test range for this feature
                    x_min = self.data_dict['X_train'][:, feat_idx].min()
                    x_max = self.data_dict['X_train'][:, feat_idx].max()
                    x_range = np.linspace(x_min, x_max, 100)

                    # Create input with all features at mean except current
                    X_test = np.tile(self.data_dict['X_train'].mean(axis=0), (100, 1))
                    X_test[:, feat_idx] = x_range

                    # Get predictions
                    y_pred = self.model.predict(X_test, verbose=0).flatten()

                    # Plot
                    ax.plot(x_range, y_pred, 'b-', linewidth=2)
                    ax.set_xlabel(feature_name[:20])
                    ax.set_ylabel('Contribution')
                    ax.set_title(f'Saturation: {feature_name[:30]}', fontsize=10)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off')

            plt.suptitle('Marketing Saturation Curves (Beta-Gamma Features)', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig('plots/diagnostics/saturation_curves.png', dpi=100, bbox_inches='tight')
            plt.close()
            logger.info("  Saved: plots/diagnostics/saturation_curves.png")

    def generate_price_elasticity_curves(self):
        """Generate price elasticity curves for monotonic features"""
        logger.info("Generating price elasticity curves...")

        # Identify monotonic features
        monotonic_indices = [i for i, ft in enumerate(self.data_dict['feature_types'])
                            if ft == 'monotonic_negative']

        if monotonic_indices:
            fig, axes = plt.subplots(1, len(monotonic_indices), figsize=(6 * len(monotonic_indices), 5))
            if len(monotonic_indices) == 1:
                axes = [axes]

            for ax, feat_idx in zip(axes, monotonic_indices):
                feature_name = self.data_dict['feature_cols'][feat_idx]

                # Create test range
                x_min = self.data_dict['X_train'][:, feat_idx].min()
                x_max = self.data_dict['X_train'][:, feat_idx].max()
                x_range = np.linspace(x_min, x_max, 100)

                # Create input
                X_test = np.tile(self.data_dict['X_train'].mean(axis=0), (100, 1))
                X_test[:, feat_idx] = x_range

                # Get predictions
                y_pred = self.model.predict(X_test, verbose=0).flatten()

                # Calculate elasticity
                price_pct_change = (x_range - x_range[0]) / x_range[0] * 100
                demand_pct_change = (y_pred - y_pred[0]) / y_pred[0] * 100

                # Plot
                ax.plot(price_pct_change, demand_pct_change, 'r-', linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('Price Change (%)')
                ax.set_ylabel('Demand Change (%)')
                ax.set_title(f'Price Elasticity: {feature_name}')
                ax.grid(True, alpha=0.3)

            plt.suptitle('Price Elasticity Curves', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig('plots/diagnostics/price_elasticity.png', dpi=100, bbox_inches='tight')
            plt.close()
            logger.info("  Saved: plots/diagnostics/price_elasticity.png")

    def generate_error_distribution_plots(self):
        """Generate error distribution analysis"""
        logger.info("Generating error distribution plots...")

        # Get predictions and errors
        y_test_pred = self.model.predict(self.data_dict['X_test'], verbose=0).flatten()
        errors = self.data_dict['y_test'] - y_test_pred

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Error histogram
        axes[0,0].hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0,0].set_xlabel('Prediction Error')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title(f'Error Distribution (μ={errors.mean():.3f}, σ={errors.std():.3f})')
        axes[0,0].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot (Normal Distribution)')
        axes[0,1].grid(True, alpha=0.3)

        # Error vs predicted
        axes[1,0].scatter(y_test_pred, errors, alpha=0.5, s=10)
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1,0].set_xlabel('Predicted Value')
        axes[1,0].set_ylabel('Error')
        axes[1,0].set_title('Heteroscedasticity Check')
        axes[1,0].grid(True, alpha=0.3)

        # Absolute error by percentile
        percentiles = np.percentile(self.data_dict['y_test'], np.arange(0, 101, 10))
        abs_errors_by_percentile = []

        for i in range(len(percentiles)-1):
            mask = (self.data_dict['y_test'] >= percentiles[i]) & (self.data_dict['y_test'] < percentiles[i+1])
            if mask.sum() > 0:
                abs_errors_by_percentile.append(np.abs(errors[mask]).mean())
            else:
                abs_errors_by_percentile.append(0)

        axes[1,1].bar(range(len(abs_errors_by_percentile)), abs_errors_by_percentile, alpha=0.7)
        axes[1,1].set_xlabel('Target Decile')
        axes[1,1].set_ylabel('Mean Absolute Error')
        axes[1,1].set_title('Error by Target Value Decile')
        axes[1,1].grid(True, alpha=0.3)

        plt.suptitle('Error Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('plots/diagnostics/error_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: plots/diagnostics/error_analysis.png")

    def generate_metrics_summary(self):
        """Generate comprehensive metrics summary"""
        logger.info("Generating metrics summary...")

        # Calculate metrics for all datasets
        datasets = ['train', 'val', 'test']
        X_sets = [self.data_dict['X_train'], self.data_dict['X_val'], self.data_dict['X_test']]
        y_sets = [self.data_dict['y_train'], self.data_dict['y_val'], self.data_dict['y_test']]

        metrics_summary = {}

        for name, X, y in zip(datasets, X_sets, y_sets):
            y_pred = self.model.predict(X, verbose=0).flatten()

            # Calculate metrics
            metrics = self.evaluator.compute_metrics(y, y_pred)

            # Add additional metrics
            metrics['samples'] = len(y)
            metrics['y_mean'] = float(y.mean())
            metrics['y_std'] = float(y.std())
            metrics['pred_mean'] = float(y_pred.mean())
            metrics['pred_std'] = float(y_pred.std())

            metrics_summary[name] = metrics

        # Save metrics
        with open('outputs/analysis/metrics_summary.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info("  Saved: outputs/analysis/metrics_summary.json")

        # Create metrics table plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data
        metrics_names = ['R²', 'MAE', 'RMSE', 'MAPE (%)', 'Mean Error', 'Std Error']
        table_data = []

        for metric, display in zip(['r2', 'mae', 'rmse', 'mape', 'mean_error', 'std_error'], metrics_names):
            row = [display]
            for dataset in datasets:
                value = metrics_summary[dataset].get(metric, 0)
                if metric == 'mape':
                    row.append(f'{value:.2f}')
                else:
                    row.append(f'{value:.4f}')
            table_data.append(row)

        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Train', 'Validation', 'Test'],
                        loc='center',
                        cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)

        plt.title('Model Performance Metrics', fontsize=14, pad=20)
        plt.savefig('plots/diagnostics/metrics_table.png', dpi=100, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: plots/diagnostics/metrics_table.png")

    def run_all_diagnostics(self):
        """Run all diagnostic analyses"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING COMPREHENSIVE DIAGNOSTICS")
        logger.info("="*80)

        # Load model and data
        self.load_model_and_data()

        # Generate all plots and analyses
        self.generate_training_history_plots()
        self.generate_prediction_plots()
        self.generate_feature_importance_plot()
        self.generate_saturation_curves()
        self.generate_price_elasticity_curves()
        self.generate_error_distribution_plots()
        self.generate_metrics_summary()

        logger.info("\n" + "="*80)
        logger.info("DIAGNOSTIC GENERATION COMPLETE")
        logger.info("="*80)
        logger.info("\nOutputs generated:")
        logger.info("  Plots: plots/diagnostics/")
        logger.info("  Data: outputs/analysis/")

        return True


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate NAM model diagnostics')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (uses latest if not specified)')

    args = parser.parse_args()

    # Create diagnostic generator
    generator = DiagnosticGenerator(model_path=args.model)

    # Run all diagnostics
    success = generator.run_all_diagnostics()

    if success:
        print("\n[SUCCESS] All diagnostics generated successfully!")
    else:
        print("\n[WARNING] Some diagnostics may have failed. Check logs.")


if __name__ == "__main__":
    main()