#!/usr/bin/env python
"""
Optimized NAM Training Script
Replaces the 590-line monolithic train_300_epochs_full_test.py
Uses modular components from src/ to eliminate code duplication
"""
import os
import sys
import warnings
import argparse
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Set TensorFlow backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.orchestrator import TrainingOrchestrator
from loguru import logger


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from file or use defaults

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        # Default configuration matching original script behavior
        config = {
            'training': {
                'epochs': 300,
                'batch_size': 32,
                'learning_rate': 0.001,
                'early_stopping': {
                    'patience': 50,
                    'restore_best': True
                },
                'reduce_lr': {
                    'patience': 20,
                    'factor': 0.5,
                    'min_lr': 1e-6
                }
            },
            'model': {
                'type': 'simple',
                'hidden_dims': [32, 16]
            },
            'data': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'beta_gamma_keywords': [
                    'TV', 'Digital', 'SEM', 'Sponsorship',
                    'Content', 'Online', 'Radio', 'Affiliates',
                    'adstock', 'log'
                ],
                'monotonic_keywords': ['price', 'mrp'],
                'target_columns': ['GMV', 'GMV_log'],
                'exclude_columns': [
                    'Date', 'product_category', 'product_subcategory',
                    'GMV', 'GMV_log'
                ]
            }
        }
        logger.info("Using default configuration")

    return config


def main():
    """Main training function"""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train NAM model')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--test-system', action='store_true',
                       help='Run system tests after training')

    args = parser.parse_args()

    print("=" * 80)
    print(f"OPTIMIZED NAM TRAINING - {args.epochs} EPOCHS")
    print("=" * 80)

    # Load configuration
    config = load_config(args.config)

    # Override epochs if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs

    # Create orchestrator
    logger.info("Initializing training orchestrator...")
    orchestrator = TrainingOrchestrator(config)

    # Run complete pipeline
    logger.info(f"Starting {args.epochs}-epoch training pipeline...")
    model, metrics = orchestrator.run_complete_pipeline(epochs=args.epochs)

    # Print final summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Epochs Trained: {args.epochs}")
    print(f"\nFinal Performance:")
    print(f"  Train R²: {metrics['train']['r2']:.4f}")
    print(f"  Val R²:   {metrics['validation']['r2']:.4f}")
    print(f"  Test R²:  {metrics['test']['r2']:.4f}")
    print(f"  Test MAPE: {metrics['test']['mape']:.2f}%")
    print(f"\nModel Stats:")
    print(f"  Parameters: {model.count_params():,}")
    print(f"  Features: {orchestrator.data_dict['n_features']}")

    # Run system tests if requested
    if args.test_system:
        test_system(orchestrator)

    print("\n[SUCCESS] Training completed successfully!")
    print("=" * 80)


def test_system(orchestrator: TrainingOrchestrator):
    """
    Test system components (replaces test_complete_system function)

    Args:
        orchestrator: Trained orchestrator instance
    """
    print("\n" + "=" * 80)
    print("TESTING SYSTEM COMPONENTS")
    print("=" * 80)

    tests_passed = []
    tests_failed = []

    # Test 1: Check directories
    print("\n[1] Testing directory structure...")
    required_dirs = ['src', 'configs', 'scripts', 'models', 'plots', 'outputs']
    for d in required_dirs:
        if os.path.exists(d):
            print(f"    [OK] {d}")
            tests_passed.append(f"Directory: {d}")
        else:
            print(f"    [X] {d} missing")
            tests_failed.append(f"Directory: {d}")

    # Test 2: Check model components
    print("\n[2] Testing model components...")
    if orchestrator.model is not None:
        print("    [OK] Model built")
        tests_passed.append("Model building")
    else:
        print("    [X] Model not built")
        tests_failed.append("Model building")

    if orchestrator.history is not None:
        print("    [OK] Training history")
        tests_passed.append("Training history")
    else:
        print("    [X] No training history")
        tests_failed.append("Training history")

    # Test 3: Check outputs
    print("\n[3] Testing outputs...")
    if os.path.exists('plots') and len(os.listdir('plots')) > 0:
        print(f"    [OK] Plots generated ({len(os.listdir('plots'))} files)")
        tests_passed.append("Plot generation")
    else:
        print("    [X] No plots generated")
        tests_failed.append("Plot generation")

    if os.path.exists('models') and any(f.endswith('.keras') for f in os.listdir('models')):
        print("    [OK] Model saved")
        tests_passed.append("Model saving")
    else:
        print("    [X] Model not saved")
        tests_failed.append("Model saving")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {len(tests_passed)}")
    print(f"Tests Failed: {len(tests_failed)}")

    if len(tests_failed) == 0:
        print("\n[SUCCESS] All system tests passed!")
    else:
        print("\n[WARNING] Some tests failed:")
        for t in tests_failed:
            print(f"  - {t}")


if __name__ == "__main__":
    main()