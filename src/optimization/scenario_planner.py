"""
Scenario planning for marketing strategies
Agent 5: Business Tools
"""
import pandas as pd
from loguru import logger
from typing import Dict, List


class ScenarioPlanner:
    """Plan and simulate different business scenarios"""

    def __init__(self, model):
        self.model = model
        self.scenarios = self._define_scenarios()

    def _define_scenarios(self) -> Dict:
        """Define predefined scenarios"""
        scenarios = {
            'baseline': {},

            'aggressive_digital': {
                'Digital_adstock': 2.0,
                'SEM_adstock': 1.5
            },

            'price_reduction': {
                'avg_price': -0.15,
                'discount_pct': 0.20
            },

            'premium_positioning': {
                'avg_price': 0.20,
                'TV_adstock': 1.3,
                'Sponsorship_adstock': 1.5
            },

            'festival_push': {
                'total_investment': 1.5,
                'discount_pct': 0.25,
                'is_festival': 1
            }
        }

        return scenarios

    def run_scenario(self, scenario_name: str, base_data: pd.DataFrame, brand_id: str, sku_id: str) -> Dict:
        """
        Run a specific scenario

        Args:
            scenario_name: Name of scenario
            base_data: Baseline data
            brand_id: Brand identifier
            sku_id: SKU identifier

        Returns:
            Scenario results
        """
        logger.info(f"Running scenario: {scenario_name}")

        adjustments = self.scenarios.get(scenario_name, {})

        # Apply adjustments
        scenario_data = base_data.copy()
        for feature, multiplier in adjustments.items():
            if feature in scenario_data.columns:
                if isinstance(multiplier, (int, float)) and multiplier < 1:
                    # Percentage change
                    scenario_data[feature] = scenario_data[feature] * (1 + multiplier)
                else:
                    scenario_data[feature] = scenario_data[feature] * multiplier

        # Predict outcomes
        predictions = self.model.predict(scenario_data)

        # Calculate metrics
        result = {
            'scenario': scenario_name,
            'total_gmv': predictions.sum(),
            'avg_gmv': predictions.mean(),
            'gmv_lift_pct': ((predictions.sum() / base_data['GMV'].sum()) - 1) * 100 if 'GMV' in base_data else 0
        }

        logger.info(f"  Total GMV: ${result['total_gmv']:,.0f}")
        logger.info(f"  GMV Lift: {result['gmv_lift_pct']:.2f}%")

        return result

    def compare_scenarios(self, scenarios: List[str], base_data: pd.DataFrame, brand_id: str, sku_id: str) -> pd.DataFrame:
        """
        Compare multiple scenarios

        Returns:
            DataFrame with scenario comparison
        """
        logger.info(f"Comparing {len(scenarios)} scenarios")

        results = []

        for scenario in scenarios:
            result = self.run_scenario(scenario, base_data, brand_id, sku_id)
            results.append(result)

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('total_gmv', ascending=False)

        logger.info("\nScenario Comparison:")
        logger.info(f"\n{comparison_df.to_string()}")

        return comparison_df
