"""
Training Orchestrator for NAM
Orchestrates the entire training pipeline using existing modular components
"""
import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from datetime import datetime
from typing import Dict, Optional, Tuple
from loguru import logger
from pathlib import Path

# Set TensorFlow backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.simple_nam import SimpleNAM
from src.models.hierarchical_nam import HierarchicalNAM
from src.data.unified_pipeline import UnifiedDataPipeline
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualization import NAMVisualizer


class TrainingOrchestrator:
    """Orchestrate NAM training using existing components"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize training orchestrator

        Args:
            config: Training configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.pipeline = UnifiedDataPipeline(self.config.get('data', {}))
        self.evaluator = ModelEvaluator()
        self.visualizer = NAMVisualizer('plots/')
        self.model = None
        self.history = None
        self.data_dict = None

    def _get_default_config(self) -> Dict:
        """Get default training configuration"""
        return {
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
                'type': 'simple',  # or 'hierarchical'
                'hidden_dims': [32, 16]
            },
            'data': {
                'train_ratio': 0.7,
                'val_ratio': 0.15
            }
        }

    def prepare_data(self, verbose: bool = True) -> Dict:
        """
        Prepare data using unified pipeline

        Returns:
            Dictionary with prepared data components
        """
        logger.info("Preparing data using unified pipeline...")
        start_time = time.time()

        # Use unified pipeline to get all data
        self.data_dict = self.pipeline.get_full_pipeline(verbose=verbose)

        elapsed = time.time() - start_time
        logger.info(f"Data preparation completed in {elapsed:.1f} seconds")

        return self.data_dict

    def build_model(self) -> keras.Model:
        """
        Build NAM model based on configuration

        Returns:
            Built Keras model
        """
        logger.info("Building NAM model...")

        model_type = self.config['model']['type']
        n_features = self.data_dict['n_features']
        feature_types = self.data_dict['feature_types']
        hidden_dims = self.config['model']['hidden_dims']

        if model_type == 'hierarchical':
            self.model = HierarchicalNAM(
                n_features=n_features,
                feature_types=feature_types,
                hidden_dims=hidden_dims
            )
        else:
            self.model = SimpleNAM(
                n_features=n_features,
                feature_types=feature_types,
                hidden_dims=hidden_dims
            )

        # Build model to initialize weights
        self.model.build(input_shape=(None, n_features))

        # Run dummy forward pass
        _ = self.model(tf.zeros((1, n_features)))

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config['training']['learning_rate']
            ),
            loss='mse',
            metrics=['mae', 'mape']
        )

        logger.info(f"Model built with {self.model.count_params():,} parameters")

        return self.model

    def create_callbacks(self) -> list:
        """
        Create training callbacks

        Returns:
            List of Keras callbacks
        """
        callbacks = []

        training_config = self.config['training']

        # Early stopping
        if training_config.get('early_stopping'):
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=training_config['early_stopping']['patience'],
                restore_best_weights=training_config['early_stopping']['restore_best'],
                verbose=1
            ))

        # Reduce learning rate
        if training_config.get('reduce_lr'):
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=training_config['reduce_lr']['factor'],
                patience=training_config['reduce_lr']['patience'],
                min_lr=training_config['reduce_lr']['min_lr'],
                verbose=1
            ))

        # Model checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        callbacks.append(ModelCheckpoint(
            f'models/nam_best_{timestamp}.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ))

        # CSV logger
        callbacks.append(CSVLogger(
            f'outputs/logs/training_{timestamp}.csv',
            separator=',',
            append=False
        ))

        return callbacks

    def train(self, epochs: Optional[int] = None) -> keras.callbacks.History:
        """
        Train the model

        Args:
            epochs: Number of epochs (overrides config if provided)

        Returns:
            Training history
        """
        if self.data_dict is None:
            self.prepare_data()

        if self.model is None:
            self.build_model()

        epochs = epochs or self.config['training']['epochs']
        batch_size = self.config['training']['batch_size']

        logger.info(f"Starting training for {epochs} epochs...")
        logger.info("=" * 80)
        logger.info(f"TRAINING NAM MODEL FOR {epochs} EPOCHS")
        logger.info("=" * 80)

        # Train model
        self.history = self.model.fit(
            self.data_dict['X_train'],
            self.data_dict['y_train'],
            validation_data=(self.data_dict['X_val'], self.data_dict['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.create_callbacks(),
            verbose=2
        )

        # Log results
        train_loss = self.history.history['loss'][-1]
        val_loss = self.history.history['val_loss'][-1]
        best_val_loss = min(self.history.history['val_loss'])
        best_epoch = self.history.history['val_loss'].index(best_val_loss) + 1

        logger.info(f"Training complete!")
        logger.info(f"  Final train loss: {train_loss:.4f}")
        logger.info(f"  Final val loss: {val_loss:.4f}")
        logger.info(f"  Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")

        return self.history

    def evaluate(self) -> Dict:
        """
        Evaluate model on test set

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set...")

        # Get test predictions
        test_loss, test_mae, test_mape = self.model.evaluate(
            self.data_dict['X_test'],
            self.data_dict['y_test'],
            verbose=0
        )

        # Get predictions for all splits
        y_train_pred = self.model.predict(self.data_dict['X_train'], verbose=0)
        y_val_pred = self.model.predict(self.data_dict['X_val'], verbose=0)
        y_test_pred = self.model.predict(self.data_dict['X_test'], verbose=0)

        # Calculate metrics using ModelEvaluator
        train_metrics = self.evaluator.compute_metrics(
            self.data_dict['y_train'],
            y_train_pred.flatten()
        )
        val_metrics = self.evaluator.compute_metrics(
            self.data_dict['y_val'],
            y_val_pred.flatten()
        )
        test_metrics = self.evaluator.compute_metrics(
            self.data_dict['y_test'],
            y_test_pred.flatten()
        )

        # Compile results
        results = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics,
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'test_mape': float(test_mape)
        }

        # Log summary
        logger.info("Evaluation Results:")
        logger.info(f"  Train R²: {train_metrics['r2']:.4f}")
        logger.info(f"  Val R²: {val_metrics['r2']:.4f}")
        logger.info(f"  Test R²: {test_metrics['r2']:.4f}")
        logger.info(f"  Test MAPE: {test_metrics['mape']:.2f}%")

        return results

    def generate_plots(self) -> None:
        """Generate diagnostic plots using NAMVisualizer"""
        logger.info("Generating diagnostic plots...")

        # Training history plot
        if self.history:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Loss plot
            axes[0].plot(self.history.history['loss'], label='Train Loss', alpha=0.7)
            axes[0].plot(self.history.history['val_loss'], label='Val Loss', alpha=0.7)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training History')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # MAE plot
            if 'mae' in self.history.history:
                axes[1].plot(self.history.history['mae'], label='Train MAE', alpha=0.7)
                axes[1].plot(self.history.history['val_mae'], label='Val MAE', alpha=0.7)
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('MAE')
                axes[1].set_title('Mean Absolute Error')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('plots/training_history.png', dpi=100)
            plt.close()
            logger.info("  Saved: plots/training_history.png")

        # Predictions vs actual
        y_test_pred = self.model.predict(self.data_dict['X_test'], verbose=0)
        self.visualizer.plot_actual_vs_predicted(
            self.data_dict['y_test'],
            y_test_pred.flatten(),
            title='Test Set: Actual vs Predicted'
        )

    def save_results(self, save_path: Optional[str] = None) -> None:
        """
        Save model and results

        Args:
            save_path: Path to save model (optional)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        if save_path:
            model_path = save_path
        else:
            model_path = f'models/nam_final_{timestamp}.keras'

        self.model.save(model_path)
        logger.info(f"Model saved: {model_path}")

        # Save metrics
        if hasattr(self, 'results'):
            # Convert numpy types to Python types for JSON serialization
            def convert_to_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            metrics_path = f'outputs/metrics_{timestamp}.json'
            with open(metrics_path, 'w') as f:
                json.dump(convert_to_json_serializable(self.results), f, indent=2)
            logger.info(f"Metrics saved: {metrics_path}")

    def run_complete_pipeline(self, epochs: Optional[int] = None) -> Tuple[keras.Model, Dict]:
        """
        Run complete training pipeline

        Args:
            epochs: Number of training epochs

        Returns:
            Tuple of (model, metrics)
        """
        logger.info("Starting complete training pipeline...")
        start_time = time.time()

        # Execute pipeline steps
        self.prepare_data()
        self.build_model()
        self.train(epochs)
        self.results = self.evaluate()
        self.generate_plots()
        self.save_results()

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed/60:.1f} minutes")

        return self.model, self.results