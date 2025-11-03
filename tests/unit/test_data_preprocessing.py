"""
Unit tests for data preprocessing
Agent 6: Test Automation
"""
import pytest
import numpy as np
import pandas as pd
from src.data.adstock import apply_adstock
from src.data.data_preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer


def test_adstock_transformation():
    """Test adstock decay function"""
    # Create simple input: spike at index 2
    x = np.array([0, 0, 100, 0, 0], dtype=float)

    adstocked = apply_adstock(x, decay_rate=0.5, max_lag=3)

    # Check properties
    assert adstocked[2] > adstocked[3], "Adstock should decay"
    assert adstocked[3] > adstocked[4], "Adstock should continue decaying"
    assert adstocked[0] == 0, "No carryover before spike"
    assert adstocked.sum() > x.sum(), "Total adstock should be >= original"


def test_missing_value_handling():
    """Test missing value imputation"""
    data = pd.DataFrame({
        'GMV': [100, np.nan, 200],
        'MRP': [120, 130, 240],
        'Radio': [1000, np.nan, 3000]
    })

    preprocessor = DataPreprocessor({})
    processed = preprocessor.handle_missing_values(data)

    assert processed['GMV'].isna().sum() == 0, "GMV missing values should be imputed"
    assert processed['Radio'].isna().sum() == 0, "Radio missing values should be filled"


def test_outlier_treatment():
    """Test outlier detection and treatment"""
    # Create data with outliers
    data = pd.DataFrame({
        'GMV': [100, 110, 105, 1000, 108, 102]  # 1000 is outlier
    })

    preprocessor = DataPreprocessor({})
    treated = preprocessor.treat_outliers(data, method='iqr', threshold=1.5)

    # Check that outlier is capped
    assert treated['GMV'].max() < 1000, "Outlier should be capped"
    assert treated['GMV'].min() >= 0, "Values should be non-negative"


def test_feature_engineering_price():
    """Test price feature creation"""
    data = pd.DataFrame({
        'GMV': [100, 200, 300],
        'Units': [10, 20, 30],
        'MRP': [120, 240, 360]
    })

    engineer = FeatureEngineer({})
    features = engineer.create_price_features(data)

    assert 'avg_price' in features.columns
    assert 'discount_pct' in features.columns
    assert (features['avg_price'] == features['GMV'] / features['Units']).all()
