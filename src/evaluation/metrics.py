"""
Evaluation metrics for NAM
Agent 4: Evaluation Engineer
"""
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from loguru import logger
from typing import Dict


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self):
        self.results = {}

    def compute_metrics(self, y_true, y_pred) -> Dict:
        """
        Compute comprehensive regression metrics

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        metrics = {
            # Standard metrics
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,

            # Normalized RMSE
            'nrmse': np.sqrt(mean_squared_error(y_true, y_pred)) / np.std(y_true),

            # Error statistics
            'mean_error': np.mean(y_pred - y_true),
            'median_error': np.median(y_pred - y_true),
            'std_error': np.std(y_pred - y_true),

            # Correlation
            'pearson_corr': np.corrcoef(y_true, y_pred)[0, 1]
        }

        return metrics

    def evaluate_model(self, model, test_data, target_col='GMV') -> Dict:
        """
        Evaluate model on test data

        Args:
            model: Trained model
            test_data: Test dataset
            target_col: Target column name

        Returns:
            Metrics dictionary
        """
        logger.info("Evaluating model on test data")

        # Get predictions
        predictions = model.predict(test_data)

        # Get actuals
        if target_col in test_data.columns:
            actuals = test_data[target_col].values
        else:
            actuals = test_data.values

        # Flatten if needed
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()

        # Compute metrics
        metrics = self.compute_metrics(actuals, predictions)

        # Log results
        logger.info("Evaluation Results:")
        logger.info(f"  R²:   {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  RMSE: {metrics['rmse']:.2f}")
        logger.info(f"  MAE:  {metrics['mae']:.2f}")

        self.results = metrics
        return metrics

    def compare_models(self, models_dict: Dict, test_data) -> Dict:
        """
        Compare multiple models

        Args:
            models_dict: Dict of {model_name: model}
            test_data: Test dataset

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(models_dict)} models")

        comparison = {}

        for model_name, model in models_dict.items():
            logger.info(f"Evaluating: {model_name}")
            metrics = self.evaluate_model(model, test_data)
            comparison[model_name] = metrics

        return comparison
