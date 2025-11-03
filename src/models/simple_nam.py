"""
Simplified Neural Additive Model for initial training
Agent 2: Model Architect
"""
import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
from keras import layers
from .constrained_layers import MonotonicPositiveLayer, MonotonicNegativeLayer, BetaGammaLayer
from loguru import logger


@keras.saving.register_keras_serializable(package="NAM")
class SimpleNAM(keras.Model):
    """
    Simplified Neural Additive Model that works with standard Keras fit()

    Uses numeric input arrays instead of complex dict structure
    """

    def __init__(self, n_features, feature_types=None, hidden_dims=[64, 32], **kwargs):
        super().__init__(**kwargs)

        self.n_features = n_features
        self.feature_types = feature_types or ['unconstrained'] * n_features
        self.hidden_dims = hidden_dims

        logger.info(f"Building SimpleNAM with {n_features} features")

        # Create feature networks - use proper architecture based on type
        self.feature_networks = []
        for i, feat_type in enumerate(self.feature_types):

            if feat_type == 'parametric_beta_gamma':
                # Marketing investment - Beta-Gamma for saturation
                network = keras.Sequential([
                    BetaGammaLayer(init_a=1.0, init_alpha=0.5, init_beta=0.1, name=f'feat_{i}_betagamma')
                ], name=f'feature_{i}_marketing')
                logger.info(f"Feature {i}: Using Beta-Gamma (saturation modeling)")

            elif feat_type == 'monotonic_negative':
                # Price - Monotonic decreasing
                network = keras.Sequential([
                    MonotonicNegativeLayer(hidden_dims[0], name=f'feat_{i}_mono_neg_1'),
                    layers.ReLU(),
                    MonotonicNegativeLayer(1, name=f'feat_{i}_mono_neg_out')
                ], name=f'feature_{i}_price')
                logger.info(f"Feature {i}: Using Monotonic Negative (price elasticity)")

            elif feat_type == 'monotonic_positive':
                # Discount - Monotonic increasing
                network = keras.Sequential([
                    MonotonicPositiveLayer(hidden_dims[0], name=f'feat_{i}_mono_pos_1'),
                    layers.ReLU(),
                    MonotonicPositiveLayer(1, name=f'feat_{i}_mono_pos_out')
                ], name=f'feature_{i}_discount')
                logger.info(f"Feature {i}: Using Monotonic Positive (discount effect)")

            else:  # unconstrained
                # Generic features - Unconstrained network
                network_layers = []
                for j, dim in enumerate(hidden_dims):
                    network_layers.append(layers.Dense(dim, activation='relu', name=f'feat_{i}_layer{j+1}'))
                network_layers.append(layers.Dense(1, name=f'feat_{i}_out'))
                network = keras.Sequential(network_layers, name=f'feature_{i}')

            self.feature_networks.append(network)

    def call(self, inputs, training=None):
        """
        Forward pass

        Args:
            inputs: Tensor of shape (batch_size, n_features)

        Returns:
            predictions: Tensor of shape (batch_size, 1)
        """
        # Split input into individual features
        feature_outputs = []

        # Safety check: only iterate over available features
        n_available_features = inputs.shape[1] if len(inputs.shape) > 1 else 0
        n_networks = len(self.feature_networks)

        if n_available_features != n_networks:
            logger.warning(f"Feature mismatch: model expects {n_networks}, got {n_available_features}")

        # Iterate only over available features
        for i in range(min(n_networks, n_available_features)):
            # Extract single feature
            feature_input = inputs[:, i:i+1]

            # Get contribution from this feature network
            contribution = self.feature_networks[i](feature_input, training=training)
            feature_outputs.append(contribution)

        # Sum all contributions (NAM additive structure - NO CHANGE TO LOGIC)
        output = keras.ops.sum(keras.ops.stack(feature_outputs, axis=0), axis=0)

        return output

    def get_feature_contributions(self, inputs):
        """Get individual feature contributions"""
        contributions = []

        for i, network in enumerate(self.feature_networks):
            feature_input = inputs[:, i:i+1]
            contribution = network(feature_input, training=False)
            contributions.append(contribution.numpy())

        return contributions

    def get_config(self):
        """Get model configuration for serialization"""
        config = super().get_config()
        config.update({
            'n_features': self.n_features,
            'feature_types': self.feature_types,
            'hidden_dims': self.hidden_dims
        })
        return config
