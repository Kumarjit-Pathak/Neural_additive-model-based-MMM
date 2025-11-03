"""
Advanced evaluation metrics for time series forecasting
Includes weighted MAPE, sMAPE, MASE, and other robust metrics
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def weighted_mape(y_true, y_pred):
    """
    Weighted MAPE - gives more weight to larger actual values
    More appropriate for revenue/GMV forecasting
    """
    weights = np.abs(y_true)
    weighted_abs_error = np.abs(y_true - y_pred) * weights
    return (weighted_abs_error.sum() / weights.sum()) * 100 / np.mean(y_true)


def symmetric_mape(y_true, y_pred):
    """
    Symmetric MAPE (sMAPE) - bounds errors symmetrically
    Better for comparing over/under predictions
    Range: 0-200%
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-10)) * 100


def mase(y_true, y_pred, y_train):
    """
    Mean Absolute Scaled Error (MASE)
    Scales errors by naive forecast performance
    < 1: Better than naive
    > 1: Worse than naive
    """
    # Naive forecast error (using training data)
    naive_errors = np.abs(np.diff(y_train))
    mae_naive = np.mean(naive_errors)

    # Model errors
    mae_model = mean_absolute_error(y_true, y_pred)

    return mae_model / (mae_naive + 1e-10)


def rmsse(y_true, y_pred, y_train):
    """
    Root Mean Squared Scaled Error
    Similar to MASE but using squared errors
    """
    naive_errors = np.diff(y_train) ** 2
    mse_naive = np.mean(naive_errors)

    mse_model = mean_squared_error(y_true, y_pred)

    return np.sqrt(mse_model / (mse_naive + 1e-10))


def bias_metric(y_true, y_pred):
    """
    Forecast bias - positive means over-forecasting, negative means under-forecasting
    """
    return np.mean(y_pred - y_true)


def bias_percentage(y_true, y_pred):
    """Bias as percentage of mean actual"""
    return (np.mean(y_pred - y_true) / np.mean(y_true)) * 100


def compute_all_metrics(y_true, y_pred, y_train=None):
    """
    Compute comprehensive metrics for model evaluation

    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_train: Training values (for MASE/RMSSE)

    Returns:
        dict: All computed metrics
    """
    metrics = {}

    # Standard metrics
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    # Percentage errors
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    metrics['wmape'] = weighted_mape(y_true, y_pred)
    metrics['smape'] = symmetric_mape(y_true, y_pred)

    # Scaled errors (if training data available)
    if y_train is not None and len(y_train) > 1:
        metrics['mase'] = mase(y_true, y_pred, y_train)
        metrics['rmsse'] = rmsse(y_true, y_pred, y_train)
    else:
        metrics['mase'] = None
        metrics['rmsse'] = None

    # Bias metrics
    metrics['bias'] = bias_metric(y_true, y_pred)
    metrics['bias_pct'] = bias_percentage(y_true, y_pred)

    # Relative metrics
    metrics['relative_rmse'] = metrics['rmse'] / np.mean(y_true) * 100
    metrics['relative_mae'] = metrics['mae'] / np.mean(y_true) * 100

    # Direction accuracy
    if len(y_true) > 1:
        actual_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        metrics['direction_accuracy'] = np.mean(actual_direction == pred_direction) * 100
    else:
        metrics['direction_accuracy'] = None

    return metrics


def print_metrics_report(metrics, title="Model Performance Metrics"):
    """Pretty print metrics report"""
    print("\n" + "="*70)
    print(title)
    print("="*70)

    print("\n📊 ACCURACY METRICS:")
    print(f"  R² Score:              {metrics['r2']:>10.4f}")
    print(f"  MAE:                   {metrics['mae']:>10,.2f}")
    print(f"  RMSE:                  {metrics['rmse']:>10,.2f}")

    print("\n📈 PERCENTAGE ERRORS:")
    print(f"  MAPE:                  {metrics['mape']:>10.2f}%")
    print(f"  Weighted MAPE:         {metrics['wmape']:>10.2f}%")
    print(f"  Symmetric MAPE:        {metrics['smape']:>10.2f}%")
    print(f"  Relative RMSE:         {metrics['relative_rmse']:>10.2f}%")
    print(f"  Relative MAE:          {metrics['relative_mae']:>10.2f}%")

    print("\n⚖️ SCALED ERRORS:")
    if metrics['mase'] is not None:
        print(f"  MASE (vs naive):       {metrics['mase']:>10.4f}  {'✓ Better' if metrics['mase'] < 1 else '✗ Worse'}")
        print(f"  RMSSE:                 {metrics['rmsse']:>10.4f}")
    else:
        print("  MASE:                  N/A (need training data)")

    print("\n🎯 BIAS ANALYSIS:")
    print(f"  Bias:                  {metrics['bias']:>10,.2f}")
    print(f"  Bias %:                {metrics['bias_pct']:>10.2f}%  {'(Over-forecast)' if metrics['bias_pct'] > 0 else '(Under-forecast)'}")

    print("\n📍 DIRECTIONAL ACCURACY:")
    if metrics['direction_accuracy'] is not None:
        print(f"  Direction Accuracy:    {metrics['direction_accuracy']:>10.1f}%")
    else:
        print("  Direction Accuracy:    N/A (need 2+ samples)")

    print("="*70)
