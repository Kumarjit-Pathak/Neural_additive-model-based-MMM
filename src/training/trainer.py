"""
NAM Trainer using Keras 3
Agent 3: Training Specialist
"""
import os
# Use TensorFlow backend for consistency
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from .loss_functions import NAMLoss
from .callbacks import ConstraintMonitorCallback
from loguru import logger
from typing import Dict, Optional

# Make MLflow optional
try:
    import mlflow
    from .callbacks import MLflowCallback
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available, tracking disabled")


class NAMTrainer:
    """Trainer for Neural Additive Model"""

    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.history = None

        # Setup optimizer
        self.optimizer = keras.optimizers.Adam(
            learning_rate=config.get('learning_rate', 0.001)
        )

        # Setup loss
        self.loss_fn = NAMLoss(
            lambda_constraint=config.get('lambda_constraint', 0.5),
            lambda_hierarchical=config.get('lambda_hierarchical', 0.3),
            lambda_smooth=config.get('lambda_smooth', 0.1)
        )

        # Compile model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=['mse', 'mae']
        )

        logger.info("NAMTrainer initialized")

    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = []

        # Early stopping
        if self.config.get('early_stopping', {}).get('enabled', True):
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get('early_stopping', {}).get('patience', 15),
                    restore_best_weights=True,
                    verbose=1
                )
            )

        # Model checkpoint
        if self.config.get('checkpoint', {}).get('enabled', True):
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=self.config.get('checkpoint', {}).get('filepath', 'outputs/models/best_model.keras'),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )

        # Learning rate scheduler
        if self.config.get('scheduler', {}).get('type') == 'reduce_on_plateau':
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.get('scheduler', {}).get('factor', 0.5),
                    patience=self.config.get('scheduler', {}).get('patience', 5),
                    min_lr=self.config.get('scheduler', {}).get('min_lr', 1e-6),
                    verbose=1
                )
            )

        # CSV logger
        callbacks.append(keras.callbacks.CSVLogger('outputs/training_log.csv'))

        # Constraint monitor
        callbacks.append(ConstraintMonitorCallback(log_freq=10))

        return callbacks

    @staticmethod
    def prepare_data_for_keras(data):
        """
        Prepare DataFrame for Keras training

        Args:
            data: DataFrame with features and target

        Returns:
            (X, y) tuple with numpy arrays
        """
        import numpy as np

        # Drop non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        # Separate features and target
        # Prefer total_gmv_log (log-transformed and scaled) for proper training
        target_cols = ['total_gmv_log', 'total_gmv', 'GMV_log', 'GMV']  # Check in priority order
        target_col = None

        for col in target_cols:
            if col in numeric_data.columns:
                target_col = col
                logger.info(f"Using target column: {target_col}")
                break

        if target_col:
            y = numeric_data[target_col].values

            # Drop target columns AND their raw (unscaled) versions
            cols_to_drop = [col for col in target_cols if col in numeric_data.columns]

            # Also drop raw versions of log-transformed columns to avoid scale mismatch
            # If we have Revenue_Camera_log, drop Revenue_Camera
            log_cols = [col for col in numeric_data.columns if col.endswith('_log')]
            for log_col in log_cols:
                raw_col = log_col.replace('_log', '')
                if raw_col in numeric_data.columns:
                    cols_to_drop.append(raw_col)

            # Also drop index column if it exists
            if 'Unnamed: 0' in numeric_data.columns:
                cols_to_drop.append('Unnamed: 0')

            # Drop Year and Month (already have temporal features)
            for col in ['Year', 'Month']:
                if col in numeric_data.columns:
                    cols_to_drop.append(col)

            X = numeric_data.drop(columns=list(set(cols_to_drop))).values
            logger.info(f"Dropped {len(set(cols_to_drop))} columns (raw + target)")
        else:
            # Fallback: use first column as target
            logger.warning("No valid target column found! Using first column as fallback")
            y = numeric_data.iloc[:, 0].values
            X = numeric_data.iloc[:, 1:].values

        return X.astype(np.float32), y.astype(np.float32)

    def train(self, train_data, val_data=None, epochs=None):
        """
        Train the model

        Args:
            train_data: Training dataset (DataFrame)
            val_data: Validation dataset (DataFrame, optional)
            epochs: Number of epochs (overrides config)

        Returns:
            Training history
        """
        logger.info("Starting training")

        # Prepare data for Keras
        X_train, y_train = self.prepare_data_for_keras(train_data)
        logger.info(f"Training data: X={X_train.shape}, y={y_train.shape}")

        if val_data is not None:
            X_val, y_val = self.prepare_data_for_keras(val_data)
            validation_data = (X_val, y_val)
            logger.info(f"Validation data: X={X_val.shape}, y={y_val.shape}")
        else:
            validation_data = None

        epochs = epochs or self.config.get('max_epochs', 200)
        batch_size = self.config.get('batch_size', 32)

        # MLflow tracking
        if MLFLOW_AVAILABLE and self.config.get('mlflow', {}).get('enabled', False):
            mlflow.set_experiment(self.config.get('mlflow', {}).get('experiment_name', 'nam_mmm'))

            with mlflow.start_run():
                # Log parameters
                mlflow.log_params({
                    'learning_rate': self.config.get('learning_rate'),
                    'batch_size': batch_size,
                    'max_epochs': epochs
                })

                # Train
                self.history = self.model.fit(
                    X_train,
                    y_train,
                    validation_data=validation_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=self.create_callbacks(),
                    verbose=1
                )

                # Log final metrics
                if validation_data is not None:
                    final_val_loss = self.history.history['val_loss'][-1]
                    mlflow.log_metric('final_val_loss', final_val_loss)
        else:
            # Train without MLflow
            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.create_callbacks(),
                verbose=1
            )

        logger.info("Training complete")
        return self.history
