"""
Phase 2: Create Marketing Adstock Features
This script creates adstock transformations and other marketing features
to enable Beta-Gamma activation in the NAM model
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

def apply_adstock(x: np.ndarray, decay_rate: float = 0.7, max_lag: int = 3) -> np.ndarray:
    """
    Apply adstock transformation to model carryover effects

    Args:
        x: Time series of marketing spend
        decay_rate: How much effect decays each period (0-1)
        max_lag: Number of periods to consider carryover

    Returns:
        Adstocked series incorporating carryover effects
    """
    adstocked = np.zeros_like(x, dtype=np.float64)

    for lag in range(max_lag + 1):
        decay = decay_rate ** lag
        if lag == 0:
            adstocked += decay * x
        else:
            # Shift and apply decay
            shifted = np.zeros_like(x)
            shifted[lag:] = x[:-lag]  # Shift forward by lag periods
            adstocked += decay * shifted

    return adstocked


def create_marketing_adstock_features(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Create marketing adstock features with channel-specific decay rates
    """

    if verbose:
        print("=" * 80)
        print("PHASE 2: CREATING MARKETING ADSTOCK FEATURES")
        print("=" * 80)

    # Make a copy to avoid modifying original
    df = data.copy()

    # Define marketing channels and their decay rates
    channel_configs = {
        # Brand building (slower decay)
        'TV': {'decay': 0.7, 'max_lag': 4, 'type': 'brand'},
        'Sponsorship': {'decay': 0.8, 'max_lag': 5, 'type': 'brand'},
        'Content_Marketing': {'decay': 0.6, 'max_lag': 3, 'type': 'brand'},

        # Performance marketing (faster decay)
        'Digital': {'decay': 0.5, 'max_lag': 3, 'type': 'performance'},
        'SEM': {'decay': 0.3, 'max_lag': 2, 'type': 'performance'},
        'Affiliates': {'decay': 0.4, 'max_lag': 2, 'type': 'performance'},
        'Online_marketing': {'decay': 0.4, 'max_lag': 3, 'type': 'performance'},

        # Traditional (medium decay)
        'Radio': {'decay': 0.6, 'max_lag': 3, 'type': 'traditional'},
    }

    # Track created features
    adstock_features = []

    # =========================================================================
    # 1. CREATE ADSTOCK FEATURES
    # =========================================================================
    if verbose:
        print("\n[1] Creating Adstock Features:")
        print("-" * 50)

    for channel, config in channel_configs.items():
        if channel in df.columns:
            # Apply adstock transformation
            feature_name = f'{channel}_adstock'

            # Group by product category to apply adstock within each category
            df[feature_name] = df.groupby('product_category')[channel].transform(
                lambda x: apply_adstock(x.values, config['decay'], config['max_lag'])
            )

            adstock_features.append(feature_name)

            if verbose:
                original_sum = df[channel].sum()
                adstock_sum = df[feature_name].sum()
                amplification = (adstock_sum / original_sum - 1) * 100 if original_sum > 0 else 0
                print(f"    {channel:20s}: decay={config['decay']}, lag={config['max_lag']}, "
                      f"amplification={amplification:.1f}%")

    # =========================================================================
    # 2. LOG TRANSFORM FOR MULTIPLICATIVE RELATIONSHIPS
    # =========================================================================
    if verbose:
        print("\n[2] Applying Log Transformations:")
        print("-" * 50)

    log_features = []
    for feature in adstock_features:
        log_feature = f'{feature}_log'
        df[log_feature] = np.log1p(df[feature])  # log(1 + x) to handle zeros
        log_features.append(log_feature)

    if verbose:
        print(f"    Created {len(log_features)} log-transformed features")
        print(f"    Features: {', '.join([f.replace('_adstock_log', '') for f in log_features[:5]])}...")

    # =========================================================================
    # 3. CREATE SHARE OF VOICE (SOV) FEATURES
    # =========================================================================
    if verbose:
        print("\n[3] Creating Share of Voice Features:")
        print("-" * 50)

    # Calculate total marketing spend (use adstocked values)
    marketing_cols = [col for col in adstock_features if col in df.columns]
    df['Total_Marketing_Adstock'] = df[marketing_cols].sum(axis=1)

    sov_features = []
    for channel in marketing_cols:
        sov_feature = f'{channel}_SOV'
        df[sov_feature] = df[channel] / (df['Total_Marketing_Adstock'] + 1e-6)
        sov_features.append(sov_feature)

    if verbose:
        print(f"    Created {len(sov_features)} share of voice features")

    # =========================================================================
    # 4. CREATE MARKETING MIX SEGMENTS
    # =========================================================================
    if verbose:
        print("\n[4] Creating Marketing Mix Segments:")
        print("-" * 50)

    # Above-the-line (ATL) vs Below-the-line (BTL)
    atl_channels = ['TV_adstock', 'Radio_adstock', 'Sponsorship_adstock']
    btl_channels = ['SEM_adstock', 'Affiliates_adstock', 'Online_marketing_adstock', 'Digital_adstock']

    # Create ATL/BTL aggregates
    atl_cols = [col for col in atl_channels if col in df.columns]
    btl_cols = [col for col in btl_channels if col in df.columns]

    if atl_cols:
        df['ATL_spend_adstock'] = df[atl_cols].sum(axis=1)
        df['ATL_spend_adstock_log'] = np.log1p(df['ATL_spend_adstock'])
        if verbose:
            print(f"    ATL channels: {', '.join([c.replace('_adstock', '') for c in atl_cols])}")

    if btl_cols:
        df['BTL_spend_adstock'] = df[btl_cols].sum(axis=1)
        df['BTL_spend_adstock_log'] = np.log1p(df['BTL_spend_adstock'])
        if verbose:
            print(f"    BTL channels: {', '.join([c.replace('_adstock', '') for c in btl_cols])}")

    # ATL/BTL ratio
    df['ATL_BTL_ratio'] = df['ATL_spend_adstock'] / (df['BTL_spend_adstock'] + 1e-6)

    # =========================================================================
    # 5. CREATE LAGGED MARKETING FEATURES
    # =========================================================================
    if verbose:
        print("\n[5] Creating Lagged Marketing Features:")
        print("-" * 50)

    lag_features = []
    for lag in [1, 2, 3]:
        lag_feature = f'Total_Marketing_lag{lag}'
        df[lag_feature] = df.groupby('product_category')['Total_Marketing_Adstock'].shift(lag)
        lag_features.append(lag_feature)

    # Fill NaN values from lagging
    df[lag_features] = df[lag_features].fillna(0)

    if verbose:
        print(f"    Created {len(lag_features)} lagged features")

    # =========================================================================
    # 6. CREATE INTERACTION FEATURES (OPTIONAL)
    # =========================================================================
    if verbose:
        print("\n[6] Creating Interaction Features:")
        print("-" * 50)

    # Price × Marketing interactions (key for MMM)
    if 'Avg_Price' in df.columns and 'Total_Marketing_Adstock' in df.columns:
        df['Price_Marketing_interaction'] = df['Avg_Price'] * df['Total_Marketing_Adstock']
        df['Price_Marketing_interaction_log'] = np.log1p(df['Price_Marketing_interaction'])
        if verbose:
            print("    Created Price × Marketing interaction")

    # Discount × Marketing interaction
    if 'Discount_Pct' in df.columns and 'Total_Marketing_Adstock' in df.columns:
        df['Discount_Marketing_interaction'] = df['Discount_Pct'] * df['Total_Marketing_Adstock']
        if verbose:
            print("    Created Discount × Marketing interaction")

    # =========================================================================
    # 7. CREATE SATURATION INDICATORS
    # =========================================================================
    if verbose:
        print("\n[7] Creating Saturation Indicators:")
        print("-" * 50)

    # Flag high spending periods (for saturation analysis)
    for channel in adstock_features:
        if channel in df.columns:
            # Flag top 10% spending as high
            threshold = df[channel].quantile(0.9)
            df[f'{channel}_high_spend'] = (df[channel] > threshold).astype(int)

    if verbose:
        print(f"    Created high spend indicators for {len(adstock_features)} channels")

    # =========================================================================
    # 8. NORMALIZE MARKETING FEATURES (OPTIONAL BUT RECOMMENDED)
    # =========================================================================
    if verbose:
        print("\n[8] Normalizing Marketing Features:")
        print("-" * 50)

    # Standardize adstock features for better neural network training
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    features_to_scale = adstock_features + log_features

    # Group by category and scale
    for feature in features_to_scale:
        if feature in df.columns:
            scaled_feature = f'{feature}_scaled'
            df[scaled_feature] = df.groupby('product_category')[feature].transform(
                lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
            )

    if verbose:
        print(f"    Scaled {len(features_to_scale)} marketing features")

    # =========================================================================
    # 9. FEATURE SUMMARY
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("MARKETING FEATURES CREATED:")
        print("=" * 80)

        feature_groups = {
            'Adstock Features': adstock_features,
            'Log Features': log_features,
            'Share of Voice': sov_features,
            'Mix Segments': ['ATL_spend_adstock', 'BTL_spend_adstock', 'ATL_BTL_ratio'],
            'Lagged Features': lag_features,
            'Interactions': [c for c in df.columns if 'interaction' in c]
        }

        for group, features in feature_groups.items():
            actual_features = [f for f in features if f in df.columns]
            print(f"\n{group}: {len(actual_features)} features")
            if actual_features and len(actual_features) <= 5:
                print(f"  - {', '.join(actual_features)}")
            elif actual_features:
                print(f"  - {', '.join(actual_features[:5])}...")

    return df


def validate_marketing_features(data: pd.DataFrame) -> Dict:
    """
    Validate that marketing features are ready for Beta-Gamma activation
    """
    validation_results = {}

    print("\n" + "=" * 80)
    print("VALIDATING MARKETING FEATURES FOR BETA-GAMMA")
    print("=" * 80)

    # Check for adstock features
    adstock_features = [col for col in data.columns if 'adstock' in col.lower()]
    validation_results['adstock_count'] = len(adstock_features)

    print(f"\n[1] Adstock Features Found: {len(adstock_features)}")
    for feature in adstock_features[:10]:  # Show first 10
        if '_log' not in feature and '_scaled' not in feature and '_SOV' not in feature:
            # Check value range
            min_val = data[feature].min()
            max_val = data[feature].max()
            print(f"    - {feature:30s}: Range [{min_val:.2f}, {max_val:.2f}]")

    # Check for log transformed features
    log_features = [col for col in data.columns if 'adstock_log' in col]
    validation_results['log_count'] = len(log_features)

    print(f"\n[2] Log-Transformed Features: {len(log_features)}")

    # Check which channels have adstock
    channels_with_adstock = []
    for channel in ['TV', 'Digital', 'SEM', 'Sponsorship', 'Content_Marketing',
                   'Affiliates', 'Radio', 'Online_marketing']:
        if f'{channel}_adstock' in data.columns:
            channels_with_adstock.append(channel)

    validation_results['channels_ready'] = channels_with_adstock

    print(f"\n[3] Channels Ready for Beta-Gamma: {len(channels_with_adstock)}/8")
    print(f"    {', '.join(channels_with_adstock)}")

    # Check feature quality
    print("\n[4] Feature Quality Checks:")
    issues = []

    for feature in adstock_features:
        if '_log' not in feature and '_scaled' not in feature and '_SOV' not in feature:
            # Check for inf or nan
            if data[feature].isnull().any():
                issues.append(f"{feature} has NaN values")
            if np.isinf(data[feature]).any():
                issues.append(f"{feature} has Inf values")

            # Check if all zeros (no marketing)
            if (data[feature] == 0).all():
                issues.append(f"{feature} is all zeros")

    if issues:
        print("    Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("    [OK] All marketing features pass quality checks")

    validation_results['quality_issues'] = issues

    # Final verdict
    print("\n" + "=" * 80)
    print("BETA-GAMMA READINESS:")

    if len(channels_with_adstock) >= 7 and len(issues) == 0:
        print("[SUCCESS] Marketing features are ready for Beta-Gamma activation!")
        print(f"    - {len(channels_with_adstock)} channels with adstock")
        print(f"    - {len(log_features)} log-transformed features")
        print("    - No quality issues detected")
        validation_results['ready'] = True
    else:
        print("[WARNING] Some issues need attention:")
        if len(channels_with_adstock) < 7:
            print(f"    - Only {len(channels_with_adstock)}/8 channels have adstock")
        if issues:
            print(f"    - {len(issues)} quality issues found")
        validation_results['ready'] = False

    print("=" * 80)

    return validation_results


if __name__ == "__main__":
    # Load the merged data from Phase 1
    print("Loading merged data from Phase 1...")
    data = pd.read_csv('data/processed/merged_mmm_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort by date and category for proper adstock calculation
    data = data.sort_values(['product_category', 'Date']).reset_index(drop=True)

    # Create marketing features
    data_with_features = create_marketing_adstock_features(data, verbose=True)

    # Validate features
    validation_results = validate_marketing_features(data_with_features)

    # Save the enhanced dataset
    print("\n[SAVE] Saving enhanced dataset with marketing features...")
    data_with_features.to_csv('data/processed/mmm_data_with_features.csv', index=False)
    print("[OK] Saved to: data/processed/mmm_data_with_features.csv")

    # Create a sample for testing
    sample_data = data_with_features.head(1000)
    sample_data.to_csv('data/processed/sample_mmm_features.csv', index=False)
    print("[OK] Sample saved to: data/processed/sample_mmm_features.csv")

    print("\n" + "=" * 80)
    print("[COMPLETE] PHASE 2 COMPLETE! Marketing features created.")
    print("NEXT: Phase 3 - Fix feature type mapping for Beta-Gamma activation")
    print("=" * 80)