"""
Loss functions for NAM training
Agent 3: Training Specialist
"""
import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="NAM")
class NAMLoss(keras.losses.Loss):
    """
    Custom loss function for NAM with multiple components
    """

    def __init__(self,
                 lambda_constraint=0.5,
                 lambda_hierarchical=0.3,
                 lambda_smooth=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.lambda_constraint = lambda_constraint
        self.lambda_hierarchical = lambda_hierarchical
        self.lambda_smooth = lambda_smooth

    def call(self, y_true, y_pred):
        """
        Compute loss

        Args:
            y_true: True GMV values
            y_pred: Predicted GMV values

        Returns:
            Total loss (scalar)
        """
        # Main prediction loss (MSE)
        loss_fit = ops.mean(ops.square(y_true - y_pred))

        # Note: Constraint and hierarchical losses are added via add_loss()
        # in the training loop

        return loss_fit

    def get_config(self):
        config = super().get_config()
        config.update({
            'lambda_constraint': self.lambda_constraint,
            'lambda_hierarchical': self.lambda_hierarchical,
            'lambda_smooth': self.lambda_smooth
        })
        return config
