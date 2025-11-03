"""
Hyperparameter tuning for NAM
Agent 3: Training Specialist
"""
import optuna
from loguru import logger
from typing import Dict, Callable


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna"""

    def __init__(self, model_fn: Callable, train_data, val_data, config: Dict):
        """
        Args:
            model_fn: Function that creates model given config
            train_data: Training data
            val_data: Validation data
            config: Tuning configuration
        """
        self.model_fn = model_fn
        self.train_data = train_data
        self.val_data = val_data
        self.config = config

    def objective(self, trial):
        """Optuna objective function"""
        from .trainer import NAMTrainer

        # Suggest hyperparameters
        suggested_config = {
            'learning_rate': trial.suggest_float('lr', 0.0001, 0.01, log=True),
            'lambda_constraint': trial.suggest_float('lambda_constraint', 0.1, 1.0),
            'lambda_hierarchical': trial.suggest_float('lambda_hierarchical', 0.1, 0.5),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'max_epochs': 50  # Reduced for tuning
        }

        # Create model
        model_config = self.config.get('model_config', {})
        model_config['hidden_dims'] = [
            trial.suggest_int('hidden_dim_1', 32, 128),
            trial.suggest_int('hidden_dim_2', 16, 64)
        ]

        model = self.model_fn(model_config)

        # Train
        trainer = NAMTrainer(model, suggested_config)
        history = trainer.train(self.train_data, self.val_data, epochs=50)

        # Return best validation loss
        best_val_loss = min(history.history['val_loss'])

        return best_val_loss

    def tune(self, n_trials: int = 50):
        """
        Run hyperparameter tuning

        Args:
            n_trials: Number of Optuna trials

        Returns:
            Best parameters dict
        """
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials")

        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params
