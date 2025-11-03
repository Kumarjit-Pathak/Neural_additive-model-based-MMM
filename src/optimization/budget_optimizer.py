"""
Budget optimization for marketing allocation
Agent 5: Business Tools
"""
import numpy as np
from scipy.optimize import minimize
from loguru import logger
from typing import Dict


class BudgetOptimizer:
    """Optimize marketing budget allocation to maximize GMV"""

    def __init__(self, model, channels=None):
        """
        Args:
            model: Trained NAM model
            channels: List of marketing channels
        """
        self.model = model
        self.channels = channels or ['TV', 'Digital', 'SEM', 'Radio', 'Sponsorship']

    def optimize_allocation(self, total_budget: float, current_features: Dict, brand_id: str, sku_id: str) -> Dict:
        """
        Find optimal budget allocation

        Args:
            total_budget: Total marketing budget
            current_features: Dict of other feature values (price, NPS, etc.)
            brand_id: Brand identifier
            sku_id: SKU identifier

        Returns:
            Optimal allocation dict
        """
        logger.info(f"Optimizing budget allocation for total budget: ${total_budget:,.0f}")

        n_channels = len(self.channels)

        # Initial guess (equal split)
        x0 = np.ones(n_channels) * (total_budget / n_channels)

        # Objective: Maximize GMV (minimize negative GMV)
        def objective(x):
            # Prepare features with current marketing spend
            features = current_features.copy()

            for i, channel in enumerate(self.channels):
                features[f'{channel}_adstock'] = x[i]

            # Predict GMV
            model_input = {
                'features': {k: np.array([[v]]) for k, v in features.items()},
                'brand_id': brand_id,
                'sku_id': sku_id
            }

            gmv = float(self.model(model_input, training=False).numpy()[0, 0])

            return -gmv  # Negative for maximization

        # Constraints
        constraints = [
            # Budget constraint
            {'type': 'eq', 'fun': lambda x: x.sum() - total_budget},
            # Minimum 10% per channel
            {'type': 'ineq', 'fun': lambda x: x - 0.10 * total_budget / n_channels},
            # Maximum 50% per channel
            {'type': 'ineq', 'fun': lambda x: 0.50 * total_budget - x}
        ]

        bounds = [(0, total_budget) for _ in range(n_channels)]

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        # Format results
        optimal_allocation = dict(zip(self.channels, result.x))
        predicted_gmv = -result.fun

        logger.info("Optimization complete:")
        for channel, allocation in optimal_allocation.items():
            logger.info(f"  {channel}: ${allocation:,.0f} ({allocation/total_budget*100:.1f}%)")
        logger.info(f"  Predicted GMV: ${predicted_gmv:,.0f}")

        return {
            'allocation': optimal_allocation,
            'predicted_gmv': predicted_gmv,
            'optimization_success': result.success
        }
