"""
Custom Keras callbacks
Agent 3: Training Specialist
"""
import os
# Use TensorFlow backend for consistency
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from loguru import logger


class ConstraintMonitorCallback(keras.callbacks.Callback):
    """Monitor constraint violations during training"""

    def __init__(self, log_freq=10):
        super().__init__()
        self.log_freq = log_freq

    def on_epoch_end(self, epoch, logs=None):
        """Log constraint violations"""
        if epoch % self.log_freq == 0:
            # Placeholder for constraint checking
            # This would check monotonicity violations
            logger.info(f"Epoch {epoch}: Checking constraints...")


# Make MLflow callback optional
try:
    import mlflow

    class MLflowCallback(keras.callbacks.Callback):
        """Log metrics to MLflow"""

        def __init__(self, run=None):
            super().__init__()
            self.run = run

        def on_epoch_end(self, epoch, logs=None):
            """Log metrics to MLflow"""
            if self.run and logs:
                mlflow.log_metrics(logs, step=epoch)

except ImportError:
    # Dummy callback if mlflow not available
    class MLflowCallback(keras.callbacks.Callback):
        def __init__(self, run=None):
            super().__init__()
            logger.warning("MLflow not available, MLflowCallback disabled")
