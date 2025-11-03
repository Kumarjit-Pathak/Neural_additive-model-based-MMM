"""
Baseline models for comparison
Agent 2: Model Architect
"""
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
import keras
from keras import layers
from loguru import logger


class BaselineModels:
    """Collection of baseline models for comparison"""

    @staticmethod
    def create_linear_regression():
        """Simple linear regression"""
        logger.info("Creating Linear Regression baseline")
        return LinearRegression()

    @staticmethod
    def create_ridge_regression(alpha=1.0):
        """Ridge regression with L2 regularization"""
        logger.info(f"Creating Ridge Regression baseline (alpha={alpha})")
        return Ridge(alpha=alpha)

    @staticmethod
    def create_gradient_boosting(n_estimators=100, max_depth=6):
        """Gradient Boosting baseline"""
        logger.info(f"Creating Gradient Boosting baseline (n_estimators={n_estimators})")
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42
        )

    @staticmethod
    def create_simple_nn(input_dim, hidden_dims=[128, 64, 32]):
        """Simple neural network baseline"""
        logger.info(f"Creating Simple NN baseline (hidden_dims={hidden_dims})")

        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden_dims[0], activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(hidden_dims[1], activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(hidden_dims[2], activation='relu'),
            layers.Dense(1)
        ], name='simple_nn_baseline')

        return model
