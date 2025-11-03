"""
Model utility functions for NAM
Includes feature type mapping for proper MMM architecture
"""
import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
import joblib
from pathlib import Path
from typing import List, Dict
from loguru import logger


def map_feature_types(feature_names: List[str], model_config: Dict = None) -> List[str]:
    """
    Map feature names to appropriate NAM architecture types for MMM

    For Marketing Mix Models:
    - Marketing channels → parametric_beta_gamma (saturation)
    - Price features → monotonic_negative (price elasticity)
    - Discount features → monotonic_positive
    - Other features → unconstrained

    Args:
        feature_names: List of feature column names
        model_config: Optional model configuration dict

    Returns:
        List of feature types matching feature_names order
    """
    feature_types = []

    logger.info(f"Mapping {len(feature_names)} features to NAM types")

    for feat_name in feature_names:
        feat_lower = feat_name.lower()

        # Marketing investment features - Beta-Gamma for saturation
        if any(keyword in feat_lower for keyword in ['adstock', '_log']):
            # Check if it's a marketing channel
            if any(ch in feat_lower for ch in ['tv', 'digital', 'sem', 'radio', 'sponsor',
                                                'content', 'online', 'affiliate', 'investment']):
                feature_types.append('parametric_beta_gamma')
                logger.debug(f"  {feat_name} → parametric_beta_gamma (saturation modeling)")
                continue

        # Price features - Monotonic negative (price ↑ → GMV ↓)
        if any(keyword in feat_lower for keyword in ['price', 'mrp']) and 'discount' not in feat_lower:
            feature_types.append('monotonic_negative')
            logger.debug(f"  {feat_name} → monotonic_negative (price elasticity)")
            continue

        # Discount features - Monotonic positive (discount ↑ → GMV ↑)
        if 'discount' in feat_lower:
            feature_types.append('monotonic_positive')
            logger.debug(f"  {feat_name} → monotonic_positive (discount effect)")
            continue

        # All other features - Unconstrained
        feature_types.append('unconstrained')
        logger.debug(f"  {feat_name} → unconstrained")

    # Summary
    beta_gamma_count = sum(1 for ft in feature_types if ft == 'parametric_beta_gamma')
    monotonic_count = sum(1 for ft in feature_types if 'monotonic' in ft)
    unconstrained_count = sum(1 for ft in feature_types if ft == 'unconstrained')

    logger.info(f"Feature type mapping complete:")
    logger.info(f"  Beta-Gamma (saturation): {beta_gamma_count}")
    logger.info(f"  Monotonic (constraints): {monotonic_count}")
    logger.info(f"  Unconstrained: {unconstrained_count}")

    return feature_types


def get_feature_names_from_data(data_scaled):
    """
    Extract feature names from scaled data (excluding target)

    Args:
        data_scaled: Scaled DataFrame

    Returns:
        List of feature column names
    """
    import numpy as np

    numeric_cols = data_scaled.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = ['total_gmv_log', 'total_gmv', 'GMV_log', 'GMV']
    feature_names = [col for col in numeric_cols if col not in target_cols]

    return feature_names


def save_model(model, filepath: str, save_format='keras'):
    """
    Save Keras model

    Args:
        model: Keras model
        filepath: Path to save model
        save_format: 'keras' or 'h5'
    """
    logger.info(f"Saving model to {filepath}")
    model.save(filepath, save_format=save_format)
    logger.info("Model saved successfully")


def load_model(filepath: str):
    """
    Load Keras model

    Args:
        filepath: Path to saved model

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {filepath}")
    model = keras.models.load_model(filepath)
    logger.info("Model loaded successfully")
    return model
