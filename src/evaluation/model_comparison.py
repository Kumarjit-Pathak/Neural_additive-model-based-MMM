"""
Model comparison utilities
Agent 4: Evaluation Engineer
"""
import pandas as pd
from loguru import logger


class ModelComparator:
    """Compare multiple models"""

    def __init__(self):
        self.comparison_table = None

    def create_comparison_table(self, comparison_results) -> pd.DataFrame:
        """
        Create comparison table from results

        Args:
            comparison_results: Dict of {model_name: metrics_dict}

        Returns:
            DataFrame with comparison
        """
        logger.info("Creating model comparison table")

        comparison_df = pd.DataFrame(comparison_results).T

        # Sort by R²
        comparison_df = comparison_df.sort_values('r2', ascending=False)

        self.comparison_table = comparison_df

        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison_df.to_string()}")

        return comparison_df

    def save_comparison(self, filepath='outputs/model_comparison.csv'):
        """Save comparison table to CSV"""
        if self.comparison_table is not None:
            self.comparison_table.to_csv(filepath)
            logger.info(f"Saved comparison table to {filepath}")
