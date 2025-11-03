"""
Unit tests for Keras models
Agent 6: Test Automation
"""
import pytest
import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
import numpy as np
from src.models.constrained_layers import MonotonicPositiveLayer, MonotonicNegativeLayer, BetaGammaLayer
from src.models.hierarchical_nam import HierarchicalNAM


def test_monotonic_positive_layer():
    """Test monotonic increasing constraint"""
    layer = MonotonicPositiveLayer(units=16)
    layer.build((None, 1))

    # Sorted inputs
    x = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
    outputs = layer(x).numpy()

    # Check monotonicity
    diffs = outputs[1:] - outputs[:-1]
    violations = (diffs < 0).sum()

    assert violations == 0, f"Monotonicity violated {violations} times"


def test_monotonic_negative_layer():
    """Test monotonic decreasing constraint"""
    layer = MonotonicNegativeLayer(units=16)
    layer.build((None, 1))

    x = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
    outputs = layer(x).numpy()

    diffs = outputs[1:] - outputs[:-1]
    violations = (diffs > 0).sum()

    assert violations == 0, f"Decreasing monotonicity violated {violations} times"


def test_beta_gamma_layer():
    """Test Beta-Gamma investment function"""
    layer = BetaGammaLayer()
    layer.build((None, 1))

    x = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
    outputs = layer(x).numpy()

    # Check outputs are non-negative
    assert (outputs >= 0).all(), "Beta-Gamma outputs should be non-negative"

    # Check concavity (second derivative should be mostly negative)
    # Simplified check: output should increase then plateau/decrease
    max_idx = outputs.argmax()
    assert 10 < max_idx < 90, "Beta-Gamma should have diminishing returns"


def test_hierarchical_nam_creation(sample_feature_configs):
    """Test HierarchicalNAM model creation"""
    model = HierarchicalNAM(
        feature_configs=sample_feature_configs,
        brand_ids=['Brand1'],
        sku_to_brand_mapping={'SKU1': 'Brand1'}
    )

    assert model is not None
    assert 'Brand1' in model.brand_networks
    assert 'SKU1' in model.sku_networks


def test_model_save_load(sample_feature_configs, tmp_path):
    """Test Keras model serialization"""
    model = HierarchicalNAM(
        feature_configs=sample_feature_configs,
        brand_ids=['Brand1'],
        sku_to_brand_mapping={'SKU1': 'Brand1'}
    )

    # Save
    model_path = tmp_path / "test_model.keras"
    model.save(model_path)

    assert model_path.exists(), "Model file should be created"

    # Load
    loaded_model = keras.models.load_model(
        model_path,
        custom_objects={
            'HierarchicalNAM': HierarchicalNAM,
            'MonotonicPositiveLayer': MonotonicPositiveLayer,
            'MonotonicNegativeLayer': MonotonicNegativeLayer,
            'BetaGammaLayer': BetaGammaLayer
        }
    )

    assert loaded_model is not None, "Model should load successfully"
