"""
Hierarchical Neural Additive Model using Keras 3
Agent 2: Model Architect
"""
import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
from keras import layers
from .constrained_layers import MonotonicPositiveLayer, MonotonicNegativeLayer, BetaGammaLayer
from loguru import logger


class HierarchicalNAM(keras.Model):
    """
    Hierarchical Neural Additive Model

    Architecture:
    - Brand-level subnetworks (shared patterns across brand)
    - SKU-level subnetworks (product-specific variations)
    - Hierarchical combination with learnable weight
    """

    def __init__(self, feature_configs, brand_ids, sku_to_brand_mapping, hier_weight=0.7, **kwargs):
        super().__init__(**kwargs)

        self.feature_configs = feature_configs
        self.brand_ids = brand_ids
        self.sku_to_brand = sku_to_brand_mapping
        self.hier_weight_init = hier_weight

        logger.info(f"Building HierarchicalNAM with {len(brand_ids)} brands, {len(sku_to_brand_mapping)} SKUs")

        # Build brand-level networks
        self.brand_networks = {}
        for brand in brand_ids:
            self.brand_networks[brand] = self._create_feature_networks(
                feature_configs,
                name_prefix=f'brand_{brand}'
            )

        # Build SKU-level networks
        self.sku_networks = {}
        for sku in sku_to_brand_mapping.keys():
            self.sku_networks[sku] = self._create_feature_networks(
                feature_configs,
                name_prefix=f'sku_{sku}'
            )

    def _create_feature_networks(self, feature_configs, name_prefix):
        """Factory for creating feature-specific networks"""
        networks = {}

        for feat_name, config in feature_configs.items():
            feat_type = config.get('type', 'unconstrained')
            hidden_dims = config.get('hidden_dims', [64, 32])
            dropout = config.get('dropout', 0.1)

            if feat_type == 'monotonic_positive':
                # Monotonic increasing network
                network = keras.Sequential([
                    layers.Input(shape=(1,)),
                    MonotonicPositiveLayer(hidden_dims[0], name=f'{name_prefix}_{feat_name}_1'),
                    layers.ReLU(),
                    layers.Dropout(dropout),
                    MonotonicPositiveLayer(hidden_dims[1], name=f'{name_prefix}_{feat_name}_2'),
                    layers.ReLU(),
                    MonotonicPositiveLayer(1, name=f'{name_prefix}_{feat_name}_out')
                ], name=f'{name_prefix}_{feat_name}')

            elif feat_type == 'monotonic_negative':
                # Monotonic decreasing network (for price)
                network = keras.Sequential([
                    layers.Input(shape=(1,)),
                    MonotonicNegativeLayer(hidden_dims[0], name=f'{name_prefix}_{feat_name}_1'),
                    layers.ReLU(),
                    layers.Dropout(dropout),
                    MonotonicNegativeLayer(hidden_dims[1], name=f'{name_prefix}_{feat_name}_2'),
                    layers.ReLU(),
                    MonotonicNegativeLayer(1, name=f'{name_prefix}_{feat_name}_out')
                ], name=f'{name_prefix}_{feat_name}')

            elif feat_type == 'parametric_beta_gamma':
                # Parametric Beta-Gamma function
                network = keras.Sequential([
                    layers.Input(shape=(1,)),
                    BetaGammaLayer(name=f'{name_prefix}_{feat_name}_betagamma')
                ], name=f'{name_prefix}_{feat_name}')

            else:
                # Unconstrained network
                network = keras.Sequential([
                    layers.Input(shape=(1,)),
                    layers.Dense(hidden_dims[0], activation='relu', name=f'{name_prefix}_{feat_name}_1'),
                    layers.Dropout(dropout),
                    layers.Dense(hidden_dims[1], activation='relu', name=f'{name_prefix}_{feat_name}_2'),
                    layers.Dense(1, name=f'{name_prefix}_{feat_name}_out')
                ], name=f'{name_prefix}_{feat_name}')

            networks[feat_name] = network

        return networks

    def call(self, inputs, training=None):
        """
        Forward pass

        Args:
            inputs: Dict with 'features' (dict of tensors), 'brand_id', 'sku_id'

        Returns:
            prediction: Final GMV prediction
        """
        features = inputs['features']
        brand_id = inputs['brand_id']
        sku_id = inputs['sku_id']

        # Brand-level prediction
        brand_output = 0.0
        for feat_name, feat_value in features.items():
            if feat_name in self.brand_networks[brand_id]:
                brand_contrib = self.brand_networks[brand_id][feat_name](feat_value, training=training)
                brand_output = brand_output + brand_contrib

        # SKU-level prediction
        sku_output = 0.0
        for feat_name, feat_value in features.items():
            if feat_name in self.sku_networks[sku_id]:
                sku_contrib = self.sku_networks[sku_id][feat_name](feat_value, training=training)
                sku_output = sku_output + sku_contrib

        # Hierarchical combination (70% brand, 30% SKU by default)
        final_output = self.hier_weight_init * brand_output + (1.0 - self.hier_weight_init) * sku_output

        return final_output

    def get_feature_contributions(self, inputs, brand_id, sku_id, training=False):
        """Get individual feature contributions for interpretability"""
        features = inputs['features']
        contributions = {}

        for feat_name, feat_value in features.items():
            if feat_name in self.brand_networks[brand_id]:
                brand_contrib = self.brand_networks[brand_id][feat_name](feat_value, training=training)
                sku_contrib = self.sku_networks[sku_id][feat_name](feat_value, training=training)

                total_contrib = self.hier_weight_init * brand_contrib + (1.0 - self.hier_weight_init) * sku_contrib
                contributions[feat_name] = total_contrib

        return contributions
