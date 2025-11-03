"""
ROI simulation for marketing channels
Agent 5: Business Tools
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger


class ROISimulator:
    """Simulate ROI curves for marketing channels"""

    def __init__(self, model, output_dir='outputs/figures/roi/'):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def simulate_channel_roi(self,
                             channel: str,
                             spend_range: np.ndarray,
                             current_features: dict,
                             brand_id: str,
                             sku_id: str) -> pd.DataFrame:
        """
        Simulate ROI curve for a specific channel

        Args:
            channel: Marketing channel name
            spend_range: Array of spend values to simulate
            current_features: Dict of other feature values
            brand_id: Brand identifier
            sku_id: SKU identifier

        Returns:
            DataFrame with spend, GMV, incremental GMV, ROI
        """
        logger.info(f"Simulating ROI curve for {channel}")

        roi_curve = []

        for spend in spend_range:
            # Set channel spend
            features = current_features.copy()
            features[f'{channel}_adstock'] = spend

            # Predict GMV with this spend
            model_input = {
                'features': {k: np.array([[v]]) for k, v in features.items()},
                'brand_id': brand_id,
                'sku_id': sku_id
            }

            gmv_with_spend = float(self.model(model_input, training=False).numpy()[0, 0])

            # Predict GMV with zero spend (baseline)
            features_zero = current_features.copy()
            features_zero[f'{channel}_adstock'] = 0

            model_input_zero = {
                'features': {k: np.array([[v]]) for k, v in features_zero.items()},
                'brand_id': brand_id,
                'sku_id': sku_id
            }

            gmv_zero = float(self.model(model_input_zero, training=False).numpy()[0, 0])

            # Calculate incremental and ROI
            incremental_gmv = gmv_with_spend - gmv_zero
            roi = incremental_gmv / spend if spend > 0 else 0

            roi_curve.append({
                'spend': spend,
                'gmv': gmv_with_spend,
                'incremental_gmv': incremental_gmv,
                'roi': roi
            })

        roi_df = pd.DataFrame(roi_curve)

        # Calculate marginal ROI
        roi_df['marginal_roi'] = roi_df['incremental_gmv'].diff() / roi_df['spend'].diff()

        logger.info(f"ROI curve simulated for {channel}")

        return roi_df

    def plot_roi_curve(self, roi_df: pd.DataFrame, channel: str):
        """Plot ROI curve"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROI curve
        axes[0].plot(roi_df['spend'], roi_df['roi'], linewidth=2, label='Average ROI')
        axes[0].plot(roi_df['spend'], roi_df['marginal_roi'], linewidth=2, linestyle='--', label='Marginal ROI')
        axes[0].set_xlabel('Marketing Spend')
        axes[0].set_ylabel('ROI')
        axes[0].set_title(f'{channel} ROI Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Incremental GMV
        axes[1].plot(roi_df['spend'], roi_df['incremental_gmv'], linewidth=2, color='green')
        axes[1].set_xlabel('Marketing Spend')
        axes[1].set_ylabel('Incremental GMV')
        axes[1].set_title(f'{channel} Incremental GMV')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / f'{channel}_roi_curve.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved ROI curve to {filepath}")
