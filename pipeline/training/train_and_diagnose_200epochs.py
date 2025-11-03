"""
Comprehensive Training and Diagnostics for NAM Model
- Train for 200 epochs
- Generate diagnostic plots
- Calculate MAPE, SMAPE
- Visualize price elasticity
- Show marketing investment saturation curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from fix_feature_mapping import prepare_features_for_nam


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def create_advanced_nam_model(n_features, feature_types, learning_rate=0.001):
    """Create an advanced NAM model with proper constraints"""

    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.Input(shape=(n_features,)))

    # Feature-specific processing (simplified for all features together)
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.15))

    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.BatchNormalization())

    # Output layer
    model.add(keras.layers.Dense(1))

    # Compile with optimal settings
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model


def train_model_200_epochs(X_train, y_train, X_val, y_val, feature_types):
    """Train model for 200 epochs with detailed tracking"""

    print("\n" + "=" * 80)
    print("TRAINING NAM MODEL FOR 200 EPOCHS")
    print("=" * 80)

    # Create model
    model = create_advanced_nam_model(
        n_features=X_train.shape[1],
        feature_types=feature_types,
        learning_rate=0.001
    )

    # Custom callback to track metrics
    class MetricsTracker(keras.callbacks.Callback):
        def __init__(self):
            self.train_losses = []
            self.val_losses = []
            self.train_mae = []
            self.val_mae = []

        def on_epoch_end(self, epoch, logs=None):
            self.train_losses.append(logs['loss'])
            self.val_losses.append(logs['val_loss'])
            self.train_mae.append(logs['mae'])
            self.val_mae.append(logs['val_mae'])

            if epoch % 20 == 0 or epoch == 199:
                print(f"Epoch {epoch+1}/200 - Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")

    tracker = MetricsTracker()

    # Train for 200 epochs
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        verbose=0,
        callbacks=[
            tracker,
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=20,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True,
                verbose=1
            )
        ]
    )

    return model, tracker


def plot_training_diagnostics(tracker):
    """Plot training diagnostics"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Training vs Validation Loss
    axes[0, 0].plot(tracker.train_losses, label='Training Loss', alpha=0.8)
    axes[0, 0].plot(tracker.val_losses, label='Validation Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Model Loss Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Training vs Validation MAE
    axes[0, 1].plot(tracker.train_mae, label='Training MAE', alpha=0.8)
    axes[0, 1].plot(tracker.val_mae, label='Validation MAE', alpha=0.8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Loss Ratio (Overfitting Detection)
    loss_ratio = np.array(tracker.val_losses) / (np.array(tracker.train_losses) + 1e-8)
    axes[1, 0].plot(loss_ratio, alpha=0.8, color='orange')
    axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Val Loss / Train Loss')
    axes[1, 0].set_title('Overfitting Detection (Ratio > 1 indicates overfitting)')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Learning Progress
    axes[1, 1].plot(np.gradient(tracker.val_losses), alpha=0.8, color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Gradient')
    axes[1, 1].set_title('Learning Progress (Negative = Improving)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Training Diagnostics - 200 Epochs', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/training_diagnostics_200epochs.png', dpi=100)
    plt.show()
    print("[SAVED] Training diagnostics plot: plots/training_diagnostics_200epochs.png")


def plot_error_metrics(y_true, y_pred, split_name='Test'):
    """Plot various error metrics"""

    # Calculate metrics
    mape = calculate_mape(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate percentage errors
    mask = y_true > 0
    percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual GMV')
    axes[0, 0].set_ylabel('Predicted GMV')
    axes[0, 0].set_title(f'{split_name}: Actual vs Predicted (R² = {r2:.4f})')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Residuals
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted GMV')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Residual Distribution
    axes[0, 2].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Residual Distribution')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Percentage Error Distribution
    axes[1, 0].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(x=mape, color='r', linestyle='--', label=f'MAPE = {mape:.2f}%')
    axes[1, 0].set_xlabel('Percentage Error (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Percentage Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Metrics Summary
    axes[1, 2].axis('off')
    metrics_text = f"""
    Error Metrics Summary
    =====================

    MAPE:  {mape:.2f}%
    SMAPE: {smape:.2f}%
    MAE:   ${mae:,.0f}
    R²:    {r2:.4f}

    Target Goals:
    MAPE < 15% : {'ACHIEVED' if mape < 15 else f'Gap: {mape-15:.1f}%'}
    R² > 0.85  : {'ACHIEVED' if r2 > 0.85 else f'Gap: {0.85-r2:.3f}'}
    """
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                    verticalalignment='center')

    plt.suptitle(f'{split_name} Set - Error Metrics Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/error_metrics_{split_name.lower()}.png', dpi=100)
    plt.show()
    print(f"[SAVED] Error metrics plot: plots/error_metrics_{split_name.lower()}.png")

    return mape, smape, mae, r2


def plot_price_elasticity(model, feature_names, scaler_X, feature_idx=None):
    """Plot price elasticity curves"""

    # Find price-related features
    if feature_idx is None:
        price_features = []
        for i, name in enumerate(feature_names):
            if 'price' in name.lower() or 'mrp' in name.lower():
                price_features.append((i, name))
    else:
        price_features = [(feature_idx, feature_names[feature_idx])]

    if not price_features:
        print("[WARNING] No price features found")
        return

    fig, axes = plt.subplots(1, min(3, len(price_features)), figsize=(15, 5))
    if len(price_features) == 1:
        axes = [axes]

    for ax_idx, (feat_idx, feat_name) in enumerate(price_features[:3]):
        # Create test range for price
        price_range = np.linspace(-2, 2, 100)  # Standardized range

        # Create input with only this feature varying
        test_input = np.zeros((100, len(feature_names)))
        test_input[:, feat_idx] = price_range

        # Predict
        predictions = model.predict(test_input, verbose=0).flatten()

        # Calculate elasticity
        elasticity = np.gradient(predictions) / (np.gradient(price_range) + 1e-8)

        # Plot response curve
        ax = axes[ax_idx] if len(price_features) > 1 else axes[0]
        ax2 = ax.twinx()

        # Response curve
        line1 = ax.plot(price_range, predictions, 'b-', label='Response', linewidth=2)
        ax.set_xlabel(f'{feat_name} (Standardized)')
        ax.set_ylabel('GMV Response', color='b')
        ax.tick_params(axis='y', labelcolor='b')

        # Elasticity curve
        line2 = ax2.plot(price_range, elasticity, 'r--', label='Elasticity', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Price Elasticity', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')

        ax.set_title(f'Price Elasticity: {feat_name}')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Price Elasticity Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/price_elasticity_curves.png', dpi=100)
    plt.show()
    print("[SAVED] Price elasticity plot: plots/price_elasticity_curves.png")


def plot_marketing_saturation(model, feature_names, scaler_X):
    """Plot marketing investment saturation curves"""

    # Find marketing features
    marketing_channels = ['tv', 'digital', 'sem', 'sponsorship',
                         'content_marketing', 'affiliates', 'radio', 'online']

    marketing_features = []
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        for channel in marketing_channels:
            if channel in name_lower and 'adstock' in name_lower:
                marketing_features.append((i, name, channel))
                break

    if not marketing_features:
        print("[WARNING] No marketing features found")
        return

    # Plot saturation curves
    n_features = min(8, len(marketing_features))
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (feat_idx, feat_name, channel) in enumerate(marketing_features[:n_features]):
        # Create investment range
        investment_range = np.linspace(0, 3, 100)  # 0 to 3x average spend

        # Create input
        test_input = np.zeros((100, len(feature_names)))
        test_input[:, feat_idx] = investment_range

        # Predict
        predictions = model.predict(test_input, verbose=0).flatten()

        # Normalize to show incremental impact
        baseline = predictions[0]
        incremental = predictions - baseline

        # Calculate ROI (simplified)
        roi = np.gradient(incremental) / (np.gradient(investment_range) + 1e-8)

        ax = axes[idx]
        ax2 = ax.twinx()

        # Saturation curve
        line1 = ax.plot(investment_range, incremental, 'b-', linewidth=2, label='Incremental GMV')
        ax.set_xlabel(f'{channel.title()} Investment (Scaled)')
        ax.set_ylabel('Incremental GMV', color='b')
        ax.tick_params(axis='y', labelcolor='b')

        # ROI curve
        line2 = ax2.plot(investment_range[1:], roi[1:], 'r--', linewidth=2, alpha=0.7, label='Marginal ROI')
        ax2.set_ylabel('Marginal ROI', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Mark saturation point (where ROI drops below 1)
        saturation_idx = np.where(roi < 1.0)[0]
        if len(saturation_idx) > 0:
            sat_point = investment_range[saturation_idx[0]]
            ax.axvline(x=sat_point, color='green', linestyle=':', alpha=0.5, label=f'Saturation: {sat_point:.2f}')

        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=8)

        ax.set_title(f'{channel.title()} Saturation', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, 8):
        axes[idx].set_visible(False)

    plt.suptitle('Marketing Investment Saturation Curves', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/marketing_saturation_curves.png', dpi=100)
    plt.show()
    print("[SAVED] Marketing saturation plot: plots/marketing_saturation_curves.png")


def plot_feature_importance(model, feature_names, X_sample):
    """Plot feature importance based on variance contribution"""

    importances = []

    for i in range(len(feature_names)):
        # Create input with only this feature
        test_input = np.zeros_like(X_sample)
        test_input[:, i] = X_sample[:, i]

        # Get predictions
        predictions = model.predict(test_input, verbose=0).flatten()

        # Calculate importance as variance
        importance = np.var(predictions)
        importances.append(importance)

    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature Importance (Variance Contribution)')
    plt.title('Top 20 Most Important Features')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=100)
    plt.show()
    print("[SAVED] Feature importance plot: plots/feature_importance.png")


def main():
    """Main execution for comprehensive training and diagnostics"""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE NAM TRAINING AND DIAGNOSTICS")
    print("=" * 80)

    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)

    # Load data
    print("\n[1] Loading data...")
    data = pd.read_csv('data/processed/mmm_data_with_features.csv')
    print(f"    Loaded {len(data):,} records")

    # Get feature configuration
    feature_config = prepare_features_for_nam(
        'data/processed/mmm_data_with_features.csv',
        verbose=False
    )
    feature_names = feature_config['feature_names']
    feature_types = feature_config['feature_types']

    print(f"    Features: {len(feature_names)}")
    print(f"    Beta-Gamma: {len(feature_config['beta_gamma_features'])}")

    # Prepare data
    X = data[feature_names].values.astype(np.float32)
    y = data['GMV'].values.astype(np.float32)

    X = np.nan_to_num(X, 0)
    y = np.nan_to_num(y, 0)

    # Normalize
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    y_log = np.log1p(y)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_log.reshape(-1, 1)).flatten()

    # Split data
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train = X_scaled[:train_end]
    X_val = X_scaled[train_end:val_end]
    X_test = X_scaled[val_end:]

    y_train = y_scaled[:train_end]
    y_val = y_scaled[train_end:val_end]
    y_test = y_scaled[val_end:]

    y_train_orig = y[:train_end]
    y_val_orig = y[train_end:val_end]
    y_test_orig = y[val_end:]

    print(f"\n[2] Data splits:")
    print(f"    Train: {len(X_train):,}")
    print(f"    Val: {len(X_val):,}")
    print(f"    Test: {len(X_test):,}")

    # Train model for 200 epochs
    print("\n[3] Training model for 200 epochs...")
    model, tracker = train_model_200_epochs(
        X_train, y_train, X_val, y_val, feature_types
    )

    # Plot training diagnostics
    print("\n[4] Generating training diagnostics...")
    plot_training_diagnostics(tracker)

    # Generate predictions
    print("\n[5] Generating predictions...")
    y_test_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_test_pred_log = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = np.expm1(y_test_pred_log)

    y_val_pred_scaled = model.predict(X_val, verbose=0).flatten()
    y_val_pred_log = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
    y_val_pred = np.expm1(y_val_pred_log)

    # Plot error metrics
    print("\n[6] Analyzing error metrics...")
    test_mape, test_smape, test_mae, test_r2 = plot_error_metrics(
        y_test_orig, y_test_pred, 'Test'
    )

    val_mape, val_smape, val_mae, val_r2 = plot_error_metrics(
        y_val_orig, y_val_pred, 'Validation'
    )

    # Plot price elasticity
    print("\n[7] Analyzing price elasticity...")
    plot_price_elasticity(model, feature_names, scaler_X)

    # Plot marketing saturation
    print("\n[8] Analyzing marketing saturation...")
    plot_marketing_saturation(model, feature_names, scaler_X)

    # Plot feature importance
    print("\n[9] Analyzing feature importance...")
    plot_feature_importance(model, feature_names, X_train[:1000])

    # Save model
    print("\n[10] Saving trained model...")
    model.save('models/nam_200epochs_diagnosed.keras')
    print("    Model saved to: models/nam_200epochs_diagnosed.keras")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING AND DIAGNOSTICS COMPLETE")
    print("=" * 80)

    print("\n[FINAL METRICS]")
    print(f"  Test Set:")
    print(f"    R²:    {test_r2:.4f} (Target: 0.85)")
    print(f"    MAPE:  {test_mape:.2f}% (Target: <15%)")
    print(f"    SMAPE: {test_smape:.2f}%")
    print(f"    MAE:   ${test_mae:,.0f}")

    print(f"\n  Validation Set:")
    print(f"    R²:    {val_r2:.4f}")
    print(f"    MAPE:  {val_mape:.2f}%")
    print(f"    SMAPE: {val_smape:.2f}%")
    print(f"    MAE:   ${val_mae:,.0f}")

    print("\n[DIAGNOSTIC PLOTS GENERATED]")
    print("  1. Training diagnostics (loss, MAE, overfitting)")
    print("  2. Error metrics (MAPE, SMAPE, residuals)")
    print("  3. Price elasticity curves")
    print("  4. Marketing saturation curves")
    print("  5. Feature importance analysis")

    print("\n[FILES SAVED]")
    print("  - models/nam_200epochs_diagnosed.keras")
    print("  - plots/training_diagnostics_200epochs.png")
    print("  - plots/error_metrics_test.png")
    print("  - plots/error_metrics_validation.png")
    print("  - plots/price_elasticity_curves.png")
    print("  - plots/marketing_saturation_curves.png")
    print("  - plots/feature_importance.png")

    return model, test_mape, test_r2


if __name__ == "__main__":
    model, mape, r2 = main()