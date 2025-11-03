#!/usr/bin/env python3
"""
Main entry point for NAM project
End-to-end pipeline execution
"""
import os
os.environ['KERAS_BACKEND'] = 'jax'

import sys
from pathlib import Path
import yaml
import numpy as np
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.data_loader import DataLoader
from src.data.data_preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.data.data_validation import DataValidator
from src.models.hierarchical_nam import HierarchicalNAM
from src.models.simple_nam import SimpleNAM
from src.training.trainer import NAMTrainer
from src.training.walk_forward import WalkForwardNAMTrainer
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualization import NAMVisualizer
from src.evaluation.model_comparison import ModelComparator


def main():
    """Main pipeline execution"""

    # Setup logging
    setup_logger(log_file='outputs/nam_pipeline.log', level='INFO')

    logger.info("="*70)
    logger.info("NAM Project - Complete Pipeline")
    logger.info("="*70)

    # Load configurations
    logger.info("\n[1/8] Loading configurations...")
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')
    data_config = load_config('configs/data_config.yaml')

    # Agent 1: Data Loading and Processing
    logger.info("\n[2/8] Agent 1: Loading and processing data...")
    loader = DataLoader(data_dir='data/raw')

    try:
        # Load Secondfile (monthly aggregated - primary dataset for MMM)
        data = loader.load_secondfile()
        logger.info(f"Loaded {len(data)} monthly records")
    except FileNotFoundError:
        logger.error("Data files not found in data/raw/. Please place CSV files there.")
        logger.info("Required files: Secondfile.csv (or other data files)")
        return 1

    # Validate data
    validator = DataValidator()
    validation_results = validator.validate_all(data)

    if not validation_results['overall_valid']:
        logger.warning("Data validation found issues (proceeding anyway)")

    # Preprocess data
    preprocessor = DataPreprocessor(data_config.get('data', {}).get('preprocessing', {}))
    data = preprocessor.handle_missing_values(data)
    data = preprocessor.treat_outliers(data)

    # Feature engineering
    engineer = FeatureEngineer(data_config.get('data', {}).get('features', {}))
    data = engineer.engineer_all_features(data)

    # Scale features
    data_scaled, scalers = preprocessor.scale_features(data)

    # Train/val/test split
    train_data, val_data, test_data = preprocessor.time_series_split(
        data_scaled,
        train_end=data_config.get('data', {}).get('split', {}).get('train_end', '2016-04-30'),
        val_end=data_config.get('data', {}).get('split', {}).get('val_end', '2016-05-31'),
        test_end=data_config.get('data', {}).get('split', {}).get('test_end', '2016-06-30')
    )

    logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # Agent 2: Model Architecture
    logger.info("\n[3/8] Agent 2: Building NAM architecture...")

    # Get number of features from training data (after split)
    # Use prepare_data_for_keras to get exact feature set
    from src.training.trainer import NAMTrainer
    X_train_sample, y_train_sample = NAMTrainer.prepare_data_for_keras(train_data)

    n_features = X_train_sample.shape[1]

    logger.info(f"Number of features for model: {n_features}")
    logger.info(f"Training data shape: X={X_train_sample.shape}, y={y_train_sample.shape}")

    # Create simple NAM model with SINGLE LAYER for explainability
    model = SimpleNAM(
        n_features=n_features,
        feature_types=['unconstrained'] * n_features,  # Can customize later
        hidden_dims=[16]  # Single hidden layer for interpretability
    )

    # Build the model with sample data (use first row)
    _ = model(X_train_sample[:1])

    logger.info(f"SimpleNAM model created and built with {n_features} features")

    # Agent 3: Training
    logger.info("\n[4/8] Agent 3: Training model...")

    trainer = NAMTrainer(model, training_config.get('training', {}))

    # Train model
    history = trainer.train(train_data, val_data)

    logger.info(f"Training complete. Best val_loss: {min(history.history['val_loss']):.4f}")

    # Agent 3: Walk-Forward Validation (optional)
    if training_config.get('walk_forward', {}).get('enabled', False):
        logger.info("\n[5/8] Agent 3: Running walk-forward validation...")

        def create_model(config):
            """Create fresh SimpleNAM model for walk-forward"""
            return SimpleNAM(
                n_features=n_features,
                feature_types=['unconstrained'] * n_features,
                hidden_dims=[16]  # Single layer for explainability
            )

        wfo_trainer = WalkForwardNAMTrainer(
            model_fn=create_model,
            model_config=model_config,
            training_config=training_config
        )

        fold_results, overall_r2 = wfo_trainer.run_walk_forward(data_scaled)
        logger.info(f"Walk-Forward Overall R²: {overall_r2:.3f}")

        # Save walk-forward results for visualization
        import pickle
        wfo_results = {
            'fold_results': fold_results,
            'overall_r2': overall_r2,
            'oos_predictions': wfo_trainer.all_oos_predictions,
            'oos_actuals': wfo_trainer.all_oos_actuals,
            'scalers': scalers
        }
        with open('outputs/walk_forward_results.pkl', 'wb') as f:
            pickle.dump(wfo_results, f)
        logger.info("Saved walk-forward results to outputs/walk_forward_results.pkl")
    else:
        logger.info("\n[5/8] Walk-forward validation skipped (disabled in config)")

    # Agent 4: Evaluation
    logger.info("\n[6/8] Agent 4: Evaluating model...")

    evaluator = ModelEvaluator()

    # Prepare test data for evaluation
    X_test, y_test_scaled = NAMTrainer.prepare_data_for_keras(test_data)
    test_pred_scaled = model.predict(X_test).flatten()

    # Inverse transform to original scale for meaningful metrics
    # y and predictions are in scaled log space, need to inverse both transforms
    if 'standard' in scalers:
        # Find index of total_gmv_log in the scaler
        # The scaler was fitted on specific columns, we need to know which index
        # Simpler approach: manually inverse using stored mean/std
        scaler = scalers['standard']

        # Get all scaled features to find total_gmv_log index
        scaled_features = [col for col in data_scaled.columns if col.endswith('_log')]
        other_features = ['avg_price', 'nps_score', 'time_index', 'NPS', 'month_sin', 'month_cos']
        all_scaled = scaled_features + [f for f in other_features if f in data_scaled.columns]

        # Find index of total_gmv_log
        try:
            gmv_idx = all_scaled.index('total_gmv_log')
            mean_gmv = scaler.mean_[gmv_idx]
            std_gmv = scaler.scale_[gmv_idx]

            # Inverse standardscaler: scaled * std + mean
            y_test_log = y_test_scaled * std_gmv + mean_gmv
            test_pred_log = test_pred_scaled * std_gmv + mean_gmv

            # Inverse log transform: expm1(log_value)
            y_test_original = np.expm1(y_test_log)
            test_pred_original = np.expm1(test_pred_log)

            logger.info(f"Inverse transformed to original GMV scale (mean={mean_gmv:.2f}, std={std_gmv:.2f})")
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not inverse transform: {e}. Using scaled values.")
            y_test_original = y_test_scaled
            test_pred_original = test_pred_scaled
    else:
        # Fallback: use scaled values
        y_test_original = y_test_scaled
        test_pred_original = test_pred_scaled
        logger.warning("Using scaled values for metrics (no inverse transform available)")

    # Compute metrics on ORIGINAL scale
    from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
    test_metrics = {
        'r2': r2_score(y_test_original, test_pred_original),
        'mape': mean_absolute_percentage_error(y_test_original, test_pred_original) * 100,
        'rmse': np.sqrt(mean_squared_error(y_test_original, test_pred_original))
    }

    logger.info("Test Metrics (Original Scale):")
    logger.info(f"  R²:   {test_metrics['r2']:.4f}")
    logger.info(f"  MAPE: {test_metrics['mape']:.2f}%")
    logger.info(f"  RMSE: {test_metrics['rmse']:.2f}")

    # Visualizations (use original scale for interpretability)
    visualizer = NAMVisualizer()
    visualizer.plot_actual_vs_predicted(y_test_original, test_pred_original, dates=test_data['Date'].values if 'Date' in test_data else None)

    # Agent 5: Business Tools
    logger.info("\n[7/8] Agent 5: Business optimization tools ready")
    logger.info("Budget optimizer, ROI simulator, and scenario planner are available for use")

    # Agent 6: Testing
    logger.info("\n[8/8] Agent 6: Running tests...")
    logger.info("Run 'pytest' to execute test suite")

    # Save final model
    logger.info("\nSaving final model...")
    model.save('outputs/models/final_nam_model.keras')
    logger.info("Model saved to outputs/models/final_nam_model.keras")

    logger.info("\n" + "="*70)
    logger.info("✓ NAM Pipeline Complete!")
    logger.info("="*70)
    logger.info(f"\nFinal Test Metrics:")
    logger.info(f"  R²:   {test_metrics['r2']:.4f}")
    logger.info(f"  MAPE: {test_metrics['mape']:.2f}%")
    logger.info(f"  RMSE: {test_metrics['rmse']:.2f}")

    logger.info("\nNext steps:")
    logger.info("  1. Review outputs/figures/ for visualizations")
    logger.info("  2. Check outputs/models/final_nam_model.keras")
    logger.info("  3. Run business optimization tools")
    logger.info("  4. Review results before deployment decision")

    return 0


if __name__ == "__main__":
    sys.exit(main())
