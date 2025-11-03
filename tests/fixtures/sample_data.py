"""
Test fixtures and sample data
Agent 6: Test Automation
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_monthly_data():
    """Create sample monthly aggregated data"""
    np.random.seed(42)

    dates = pd.date_range('2015-07-01', '2016-06-01', freq='MS')
    n = len(dates)

    data = pd.DataFrame({
        'Date': dates,
        'GMV': np.random.uniform(10000, 50000, n),
        'Units': np.random.randint(100, 1000, n),
        'avg_price': np.random.uniform(40, 60, n),
        'discount_pct': np.random.uniform(0, 0.3, n),
        'TV': np.random.uniform(1000000, 10000000, n),
        'Digital': np.random.uniform(500000, 5000000, n),
        'SEM': np.random.uniform(300000, 3000000, n),
        'nps_score': np.random.uniform(40, 60, n),
        'month_sin': np.sin(2 * np.pi * np.arange(n) / 12),
        'month_cos': np.cos(2 * np.pi * np.arange(n) / 12)
    })

    return data


@pytest.fixture
def sample_feature_configs():
    """Sample feature configurations for testing"""
    return {
        'avg_price': {
            'type': 'monotonic_negative',
            'hidden_dims': [32, 16],
            'dropout': 0.1
        },
        'TV_adstock': {
            'type': 'parametric_beta_gamma'
        },
        'nps_score': {
            'type': 'unconstrained',
            'hidden_dims': [32, 16],
            'dropout': 0.1
        }
    }


@pytest.fixture
def sample_training_config():
    """Sample training configuration"""
    return {
        'learning_rate': 0.001,
        'max_epochs': 10,
        'batch_size': 32,
        'lambda_constraint': 0.5,
        'lambda_hierarchical': 0.3,
        'lambda_smooth': 0.1,
        'early_stopping': {
            'enabled': True,
            'patience': 5
        },
        'checkpoint': {
            'enabled': False
        }
    }
