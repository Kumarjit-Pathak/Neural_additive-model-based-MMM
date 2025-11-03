#!/usr/bin/env python3
"""
Main entry point for NAM project - DAILY DATA VERSION
End-to-end pipeline with daily granularity for better statistical power
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
from src.models.simple_nam import SimpleNAM
from src.training.trainer import NAMTrainer
from src.training.walk_forward import WalkForwardNAMTrainer
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualization import NAMVisualizer


def main():
    """Main pipeline execution with DAILY data"""

    # Setup logging
    setup_logger(log_file='outputs/nam_pipeline_daily.log', level='INFO')

    logger.info("="*70)
    logger.info("NAM Project - DAILY DATA Pipeline")
    logger.info("="*70)

    # Load configurations
    logger.info("\n[1/8] Loading configurations...")
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')
    data_config = load_config('configs/data_config.yaml')

    # Agent 1: Data Loading and Processing
    logger.info("\n[2/8] Agent 1: Loading DAILY sales data...")
    loader = DataLoader(data_dir='data/raw')

    try:
        # Load DAILY data (250 records vs 12 monthly)
        data = loader.load_daily_sales()
        logger.info(f"✓ Loaded {len(data)} DAILY records (massive improvement!)")
    except Exception as e:
        logger.error(f"Failed to load daily data: {e}")
        logger.info("Falling back to monthly data...")
        data = loader.load_secondfile()

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

    # Train/val/test split for DAILY data
    # Use 70/15/15 split (more standard for larger datasets)
    train_size = int(len(data_scaled) * 0.70)
    val_size = int(len(data_scaled) * 0.15)

    train_data = data_scaled.iloc[:train_size]
    val_data = data_scaled.iloc[train_size:train_size+val_size]
    test_data = data_scaled.iloc[train_size+val_size:]

    logger.info(f"Daily split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    logger.info(f"Sample/feature ratio: Train={len(train_data)/45:.1f}, Total={len(data_scaled)/45:.1f}")

    # Agent 2: Model Architecture
    logger.info("\n[3/8] Agent 2: Building NAM architecture...")

    # Get number of features
    from src.training.trainer import NAMTrainer
    X_train_sample, y_train_sample = NAMTrainer.prepare_data_for_keras(train_data)

    n_features = X_train_sample.shape[1]

    logger.info(f"Number of features for model: {n_features}")
    logger.info(f"Training data shape: X={X_train_sample.shape}, y={y_train_sample.shape}")

    # Create NAM with PROPER MMM ARCHITECTURE (Beta-Gamma for marketing!)
    from src.models.model_utils import map_feature_types, get_feature_names_from_data

    # Get feature names to map types
    feature_names = get_feature_names_from_data(data_scaled)

    # Map to proper types (Beta-Gamma for marketing, Monotonic for price)
    feature_types = map_feature_types(feature_names, model_config)

    logger.info(f"✓ Using MMM-specific architectures:")
    beta_gamma_count = sum(1 for ft in feature_types if ft == 'parametric_beta_gamma')
    monotonic_count = sum(1 for ft in feature_types if 'monotonic' in ft)
    logger.info(f"  Beta-Gamma (marketing saturation): {beta_gamma_count}")
    logger.info(f"  Monotonic (price/discount constraints): {monotonic_count}")

    model = SimpleNAM(
        n_features=n_features,
        feature_types=feature_types,  # ✓ Proper MMM architecture!
        hidden_dims=[16]  # For unconstrained features
    )

    # Build the model
    _ = model(X_train_sample[:1])

    logger.info(f"Single-layer NAM created: {n_features} features, {model.count_params():,} parameters")

    # Agent 3: Training
    logger.info("\n[4/8] Agent 3: Training model on DAILY data...")

    trainer = NAMTrainer(model, training_config.get('training', {}))

    # Train model
    history = trainer.train(train_data, val_data)

    logger.info(f"Training complete. Best val_loss: {min(history.history['val_loss']):.4f}")

    # Agent 3: Walk-Forward Validation
    if training_config.get('walk_forward', {}).get('enabled', False):
        logger.info("\n[5/8] Agent 3: Running walk-forward validation on DAILY data...")

        def create_model(config):
            """Create fresh SimpleNAM model for walk-forward with MMM architecture"""
            return SimpleNAM(
                n_features=n_features,
                feature_types=feature_types,  # Use same MMM architecture
                hidden_dims=[16]
            )

        wfo_trainer = WalkForwardNAMTrainer(
            model_fn=create_model,
            model_config=model_config,
            training_config=training_config
        )

        fold_results, overall_r2 = wfo_trainer.run_walk_forward(data_scaled)
        logger.info(f"Walk-Forward Overall R²: {overall_r2:.3f}")

        # Save walk-forward results
        import pickle
        wfo_results = {
            'fold_results': fold_results,
            'overall_r2': overall_r2,
            'oos_predictions': wfo_trainer.all_oos_predictions,
            'oos_actuals': wfo_trainer.all_oos_actuals,
            'scalers': scalers,
            'data_type': 'daily'
        }
        with open('outputs/walk_forward_results_daily.pkl', 'wb') as f:
            pickle.dump(wfo_results, f)
        logger.info("Saved daily walk-forward results")
    else:
        logger.info("\n[5/8] Walk-forward validation skipped (disabled in config)")

    # Agent 4: Evaluation
    logger.info("\n[6/8] Agent 4: Evaluating model...")

    evaluator = ModelEvaluator()

    # Prepare test data
    X_test, y_test_scaled = NAMTrainer.prepare_data_for_keras(test_data)
    test_pred_scaled = model.predict(X_test).flatten()

    # Inverse transform
    if 'standard' in scalers:
        scaler = scalers['standard']
        scaled_features = [col for col in data_scaled.columns if col.endswith('_log')]
        other_features = ['avg_price', 'nps_score', 'time_index', 'NPS', 'month_sin', 'month_cos']
        all_scaled = scaled_features + [f for f in other_features if f in data_scaled.columns]

        try:
            gmv_idx = all_scaled.index('total_gmv_log')
            mean_gmv = scaler.mean_[gmv_idx]
            std_gmv = scaler.scale_[gmv_idx]

            y_test_log = y_test_scaled * std_gmv + mean_gmv
            test_pred_log = test_pred_scaled * std_gmv + mean_gmv

            y_test_original = np.expm1(y_test_log)
            test_pred_original = np.expm1(test_pred_log)

            logger.info(f"Inverse transformed to original GMV scale")
        except Exception as e:
            logger.warning(f"Could not inverse transform: {e}")
            y_test_original = y_test_scaled
            test_pred_original = test_pred_scaled
    else:
        y_test_original = y_test_scaled
        test_pred_original = test_pred_scaled

    # Compute comprehensive metrics
    from src.evaluation.advanced_metrics import compute_all_metrics, print_metrics_report

    # Get training data for MASE calculation
    X_train_all, y_train_all = NAMTrainer.prepare_data_for_keras(train_data)

    # Inverse transform training data for MASE
    if 'standard' in scalers:
        try:
            y_train_log = y_train_all * std_gmv + mean_gmv
            y_train_original = np.expm1(y_train_log)
        except:
            y_train_original = y_train_all
    else:
        y_train_original = y_train_all

    # Compute all metrics
    test_metrics = compute_all_metrics(y_test_original, test_pred_original, y_train_original)

    # Print comprehensive report
    print_metrics_report(test_metrics, f"Daily NAM Performance ({len(test_data)} test days)")

    # Visualizations
    visualizer = NAMVisualizer()
    visualizer.plot_actual_vs_predicted(y_test_original, test_pred_original, dates=test_data['Date'].values if 'Date' in test_data else None)

    # Agent 5: Business Tools + Elasticity Analysis
    logger.info("\n[7/8] Agent 5: Generating elasticity curves...")

    # Save model and data for elasticity analysis
    import pickle
    elasticity_data = {
        'model': model,
        'data_scaled': data_scaled,
        'scalers': scalers,
        'test_data': test_data,
        'feature_names': [col for col in data_scaled.select_dtypes(include=[np.number]).columns]
    }
    with open('outputs/elasticity_data.pkl', 'wb') as f:
        pickle.dump(elasticity_data, f)

    logger.info("Elasticity data saved. Run scripts/plot_elasticity.py to generate curves")

    # Agent 6: Testing
    logger.info("\n[8/8] Agent 6: Testing available")
    logger.info("Run 'pytest' to execute test suite")

    # Save final model
    logger.info("\nSaving final model...")
    model.save('outputs/models/final_nam_model_daily.keras')
    logger.info("Model saved to outputs/models/final_nam_model_daily.keras")

    logger.info("\n" + "="*70)
    logger.info("✓ NAM DAILY Pipeline Complete!")
    logger.info("="*70)
    logger.info(f"\nFinal Test Metrics ({len(test_data)} daily test samples):")
    logger.info(f"  R²:   {test_metrics['r2']:.4f}")
    logger.info(f"  MAPE: {test_metrics['mape']:.2f}%")
    logger.info(f"  RMSE: {test_metrics['rmse']:.2f}")

    logger.info("\nNext steps:")
    logger.info("  1. python scripts/plot_training_results.py  (training charts)")
    logger.info("  2. python scripts/plot_walk_forward.py      (time series trends)")
    logger.info("  3. python scripts/plot_elasticity.py        (elasticity curves)")
    logger.info("  4. Review business optimization tools")

    return 0


if __name__ == "__main__":
    sys.exit(main())
