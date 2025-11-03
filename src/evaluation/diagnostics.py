"""
Model diagnostics
Agent 4: Evaluation Engineer
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from loguru import logger


class ModelDiagnostics:
    """Diagnostic analysis for model validation"""

    def __init__(self, output_dir='outputs/figures/'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_residuals(self, y_true, y_pred):
        """
        Comprehensive residual analysis

        Returns:
            Dictionary of diagnostic statistics
        """
        logger.info("Performing residual analysis")

        residuals = y_true - y_pred

        # Normality test
        _, p_normality = stats.shapiro(residuals)

        # Autocorrelation (Durbin-Watson)
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(residuals)

        diagnostics = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'normality_p_value': p_normality,
            'durbin_watson': dw_stat,
            'is_normal': p_normality > 0.05,
            'has_autocorrelation': (dw_stat < 1.5 or dw_stat > 2.5)
        }

        logger.info(f"Residual diagnostics: mean={diagnostics['mean_residual']:.4f}, std={diagnostics['std_residual']:.4f}")

        return diagnostics

    def plot_residual_diagnostics(self, y_true, y_pred):
        """Plot comprehensive residual diagnostics"""
        logger.info("Creating residual diagnostic plots")

        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Residuals vs predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)

        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # Histogram
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals over time (or index)
        axes[1, 1].plot(residuals, marker='o', linestyle='-', alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Over Time')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / 'residual_diagnostics.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved diagnostic plots to {filepath}")
