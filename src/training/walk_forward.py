"""
Walk-Forward Optimization for time-series validation
Agent 3: Training Specialist
"""
import pandas as pd
import numpy as np
from typing import Tuple, Generator, Dict
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from loguru import logger


class WalkForwardSplitter:
    """Time-series walk-forward cross-validation splitter"""

    def __init__(self,
                 initial_train_size: int = 6,
                 test_size: int = 1,
                 step_size: int = 1,
                 window_type: str = 'expanding'):
        """
        Args:
            initial_train_size: Initial training window size (months)
            test_size: Test period size (months)
            step_size: How much to move forward each iteration
            window_type: 'expanding' or 'rolling'
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.window_type = window_type

    def split(self, data: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict], None, None]:
        """
        Generate train/test splits

        Args:
            data: DataFrame with datetime index or 'Date' column

        Yields:
            (train_data, test_data, fold_info)
        """
        # Ensure data is sorted by date
        if 'Date' in data.columns:
            data = data.sort_values('Date').reset_index(drop=True)
            date_col = 'Date'
        else:
            data = data.sort_index()
            date_col = data.index.name or 'index'

        # Get unique months
        if date_col == 'Date':
            unique_months = sorted(data['Date'].dt.to_period('M').unique())
        else:
            unique_months = sorted(data.index.to_period('M').unique())

        n_months = len(unique_months)
        fold = 0
        train_end_idx = self.initial_train_size

        logger.info(f"Walk-Forward Split: {n_months} months, initial_train={self.initial_train_size}, test={self.test_size}")

        while train_end_idx + self.test_size <= n_months:
            # Determine training window
            if self.window_type == 'expanding':
                train_start_idx = 0
            else:  # rolling
                train_start_idx = train_end_idx - self.initial_train_size

            # Get date ranges
            train_start = unique_months[train_start_idx].to_timestamp()
            train_end = unique_months[train_end_idx - 1].to_timestamp('M')
            test_start = unique_months[train_end_idx].to_timestamp()
            test_end = unique_months[train_end_idx + self.test_size - 1].to_timestamp('M')

            # Split data
            if date_col == 'Date':
                train_mask = (data['Date'] >= train_start) & (data['Date'] <= train_end)
                test_mask = (data['Date'] >= test_start) & (data['Date'] <= test_end)
            else:
                train_mask = (data.index >= train_start) & (data.index <= train_end)
                test_mask = (data.index >= test_start) & (data.index <= test_end)

            train_data = data[train_mask].copy()
            test_data = data[test_mask].copy()

            fold_info = {
                'fold': fold,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': len(train_data),
                'test_size': len(test_data)
            }

            yield train_data, test_data, fold_info

            # Move forward
            train_end_idx += self.step_size
            fold += 1


class WalkForwardNAMTrainer:
    """Walk-forward optimization trainer for NAM"""

    def __init__(self, model_fn, model_config, training_config):
        """
        Args:
            model_fn: Function that creates a fresh model instance
            model_config: Model configuration dict
            training_config: Training configuration dict
        """
        self.model_fn = model_fn
        self.model_config = model_config
        self.training_config = training_config
        self.fold_results = []

    def run_walk_forward(self, data: pd.DataFrame) -> Tuple[list, float]:
        """
        Execute walk-forward optimization

        Args:
            data: Complete dataset

        Returns:
            (fold_results, overall_r2)
        """
        from .trainer import NAMTrainer

        splitter = WalkForwardSplitter(
            initial_train_size=self.training_config.get('walk_forward', {}).get('initial_train_size', 6),
            test_size=self.training_config.get('walk_forward', {}).get('test_size', 1),
            window_type=self.training_config.get('walk_forward', {}).get('window_type', 'expanding')
        )

        all_oos_predictions = []
        all_oos_actuals = []

        logger.info("="*70)
        logger.info("Starting Walk-Forward Optimization")
        logger.info("="*70)

        for train_data, test_data, fold_info in splitter.split(data):
            logger.info(f"\nFold {fold_info['fold']}: {fold_info['test_start'].date()} to {fold_info['test_end'].date()}")
            logger.info(f"Train: {fold_info['train_size']} samples, Test: {fold_info['test_size']} samples")

            # Create fresh model
            model = self.model_fn(self.model_config)

            # Train
            trainer = NAMTrainer(model, self.training_config)
            trainer.train(train_data, None, epochs=self.training_config.get('max_epochs', 50))

            # Evaluate on OOS test - prepare data properly
            X_test, y_test = NAMTrainer.prepare_data_for_keras(test_data)
            test_pred = model.predict(X_test)
            test_actual = y_test

            # Flatten if needed
            if len(test_pred.shape) > 1:
                test_pred = test_pred.flatten()

            # Compute metrics
            fold_r2 = r2_score(test_actual, test_pred)
            fold_mape = mean_absolute_percentage_error(test_actual, test_pred) * 100

            fold_result = {
                'fold': fold_info['fold'],
                'test_period': f"{fold_info['test_start'].date()} to {fold_info['test_end'].date()}",
                'r2': fold_r2,
                'mape': fold_mape
            }

            self.fold_results.append(fold_result)

            all_oos_predictions.extend(test_pred)
            all_oos_actuals.extend(test_actual)

            logger.info(f"  R²: {fold_r2:.3f}, MAPE: {fold_mape:.2f}%")

        # Aggregate results
        overall_r2 = r2_score(all_oos_actuals, all_oos_predictions)
        overall_mape = mean_absolute_percentage_error(all_oos_actuals, all_oos_predictions) * 100

        fold_r2s = [f['r2'] for f in self.fold_results]

        logger.info("\n" + "="*70)
        logger.info("Walk-Forward Optimization Results")
        logger.info("="*70)
        logger.info(f"Overall OOS R²:    {overall_r2:.3f}")
        logger.info(f"Overall OOS MAPE:  {overall_mape:.2f}%")
        logger.info(f"R² Mean ± Std:     {np.mean(fold_r2s):.3f} ± {np.std(fold_r2s):.3f}")
        logger.info(f"R² Range:          [{np.min(fold_r2s):.3f}, {np.max(fold_r2s):.3f}]")
        logger.info("="*70)

        # Save all OOS predictions and actuals for visualization
        self.all_oos_predictions = np.array(all_oos_predictions)
        self.all_oos_actuals = np.array(all_oos_actuals)

        return self.fold_results, overall_r2
