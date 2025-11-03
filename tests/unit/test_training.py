"""
Unit tests for training
Agent 6: Test Automation
"""
import pytest
import os
os.environ['KERAS_BACKEND'] = 'jax'

from src.training.walk_forward import WalkForwardSplitter
import pandas as pd
import numpy as np


def test_walk_forward_splitter(sample_monthly_data):
    """Test walk-forward splitter"""
    splitter = WalkForwardSplitter(
        initial_train_size=6,
        test_size=1,
        step_size=1,
        window_type='expanding'
    )

    folds = list(splitter.split(sample_monthly_data))

    # Check we get expected number of folds
    # 12 months total, 6 initial train, 1 test = 12-6 = 6 possible folds
    assert len(folds) == 6, f"Expected 6 folds, got {len(folds)}"

    # Check first fold
    train_data, test_data, fold_info = folds[0]
    assert len(train_data) == 6, "First fold should have 6 months training"
    assert len(test_data) == 1, "Test should have 1 month"
    assert fold_info['fold'] == 0

    # Check last fold (expanding window)
    train_data, test_data, fold_info = folds[-1]
    assert len(train_data) == 11, "Last fold should have 11 months training (expanding)"
    assert len(test_data) == 1


def test_walk_forward_rolling_window(sample_monthly_data):
    """Test walk-forward with rolling window"""
    splitter = WalkForwardSplitter(
        initial_train_size=6,
        test_size=1,
        window_type='rolling'
    )

    folds = list(splitter.split(sample_monthly_data))

    # Check rolling window maintains size
    for train_data, test_data, fold_info in folds:
        assert len(train_data) <= 6, "Rolling window should not exceed initial size"
        assert len(test_data) == 1
