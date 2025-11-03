"""
Visualization for NAM
Agent 4: Evaluation Engineer
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from loguru import logger


class NAMVisualizer:
    """Visualization suite for NAM"""

    def __init__(self, output_dir='outputs/figures/'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300

    def plot_actual_vs_predicted(self, y_true, y_pred, dates=None, title='Actual vs Predicted'):
        """Plot actual vs predicted values"""
        logger.info("Creating actual vs predicted plot")

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Top: Actual vs Predicted
        x = dates if dates is not None else range(len(y_true))
        axes[0].plot(x, y_true, label='Actual', marker='o', linewidth=2)
        axes[0].plot(x, y_pred, label='Predicted', marker='s', linewidth=2, alpha=0.7)
        axes[0].set_ylabel('GMV')
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Bottom: Residuals
        residuals = y_true - y_pred
        axes[1].bar(x, residuals, color=['red' if r < 0 else 'green' for r in residuals], alpha=0.6)
        axes[1].axhline(y=0, color='black', linestyle='--')
        axes[1].set_ylabel('Residual')
        axes[1].set_xlabel('Date' if dates is not None else 'Sample')
        axes[1].set_title('Prediction Residuals')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / 'actual_vs_predicted.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot to {filepath}")

    def plot_walk_forward_results(self, fold_results):
        """Visualize walk-forward optimization results"""
        logger.info("Creating walk-forward results plot")

        fold_nums = [f['fold'] for f in fold_results]
        fold_r2s = [f['r2'] for f in fold_results]
        fold_mapes = [f['mape'] for f in fold_results]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # R² by fold
        axes[0].bar(fold_nums, fold_r2s, alpha=0.7, color='steelblue')
        axes[0].axhline(y=np.mean(fold_r2s), color='r', linestyle='--', label=f'Mean R² = {np.mean(fold_r2s):.3f}')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('R²')
        axes[0].set_title('Walk-Forward R² by Fold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAPE by fold
        axes[1].bar(fold_nums, fold_mapes, alpha=0.7, color='coral')
        axes[1].axhline(y=np.mean(fold_mapes), color='r', linestyle='--', label=f'Mean MAPE = {np.mean(fold_mapes):.2f}%')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('MAPE (%)')
        axes[1].set_title('Walk-Forward MAPE by Fold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / 'walk_forward_results.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot to {filepath}")

    def plot_model_comparison(self, comparison_results):
        """Plot model comparison"""
        logger.info("Creating model comparison plot")

        models = list(comparison_results.keys())
        r2_scores = [comparison_results[m]['r2'] for m in models]
        mapes = [comparison_results[m]['mape'] for m in models]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # R² comparison
        axes[0].barh(models, r2_scores, alpha=0.7, color='steelblue')
        axes[0].set_xlabel('R² Score')
        axes[0].set_title('Model Comparison: R²')
        axes[0].grid(True, alpha=0.3)

        # MAPE comparison
        axes[1].barh(models, mapes, alpha=0.7, color='coral')
        axes[1].set_xlabel('MAPE (%)')
        axes[1].set_title('Model Comparison: MAPE')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / 'model_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot to {filepath}")
