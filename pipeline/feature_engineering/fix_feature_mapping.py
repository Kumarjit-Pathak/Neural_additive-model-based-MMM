"""
Phase 3: Fix Feature Type Mapping for Beta-Gamma Activation
This script ensures proper feature type mapping so Beta-Gamma layers
are activated for marketing channels in the NAM model
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


def map_feature_types_for_mmm(feature_names: List[str], verbose: bool = True) -> List[str]:
    """
    Map feature names to proper NAM architecture types for Marketing Mix Modeling

    For Marketing Mix Models:
    - Marketing channels with adstock -> parametric_beta_gamma (saturation curves)
    - Price features -> monotonic_negative (price elasticity)
    - Discount features -> monotonic_positive
    - Other features -> unconstrained

    Args:
        feature_names: List of feature column names
        verbose: Print detailed mapping information

    Returns:
        List of feature types matching feature_names order
    """

    feature_types = []

    # Counters for summary
    beta_gamma_count = 0
    monotonic_neg_count = 0
    monotonic_pos_count = 0
    unconstrained_count = 0

    if verbose:
        print("=" * 80)
        print("PHASE 3: FEATURE TYPE MAPPING FOR BETA-GAMMA ACTIVATION")
        print("=" * 80)
        print("\n[MAPPING] Processing features...")
        print("-" * 50)

    for i, feat_name in enumerate(feature_names):
        feat_lower = feat_name.lower()

        # =====================================================================
        # 1. MARKETING CHANNELS - Beta-Gamma for saturation modeling
        # =====================================================================
        # Check for marketing channels with adstock or log transformation
        marketing_channels = ['tv', 'digital', 'sem', 'sponsorship', 'content_marketing',
                            'affiliates', 'radio', 'online_marketing', 'atl', 'btl']

        is_marketing = False
        for channel in marketing_channels:
            if channel in feat_lower:
                # Check if it's an adstock or log-transformed feature
                if any(keyword in feat_lower for keyword in ['adstock', '_log', 'spend']):
                    # Skip derivative features like SOV, scaled, high_spend
                    if not any(skip in feat_lower for skip in ['sov', 'scaled', 'high_spend', 'ratio', 'lag']):
                        feature_types.append('parametric_beta_gamma')
                        beta_gamma_count += 1
                        is_marketing = True
                        if verbose and beta_gamma_count <= 10:  # Show first 10
                            print(f"    {i:3d}. {feat_name:40s} -> Beta-Gamma (marketing saturation)")
                        break

        if is_marketing:
            continue

        # =====================================================================
        # 2. PRICE FEATURES - Monotonic negative (price elasticity)
        # =====================================================================
        if any(keyword in feat_lower for keyword in ['price', 'mrp']) and 'cross' not in feat_lower:
            # Exclude interaction features
            if 'interaction' not in feat_lower and 'index' not in feat_lower:
                feature_types.append('monotonic_negative')
                monotonic_neg_count += 1
                if verbose and monotonic_neg_count <= 5:
                    print(f"    {i:3d}. {feat_name:40s} -> Monotonic Negative (price v)")
                continue

        # =====================================================================
        # 3. DISCOUNT FEATURES - Monotonic positive (discount effect)
        # =====================================================================
        if 'discount' in feat_lower:
            # Exclude interaction features
            if 'interaction' not in feat_lower:
                feature_types.append('monotonic_positive')
                monotonic_pos_count += 1
                if verbose and monotonic_pos_count <= 5:
                    print(f"    {i:3d}. {feat_name:40s} -> Monotonic Positive (discount ^)")
                continue

        # =====================================================================
        # 4. SERVICE LEVEL - Monotonic negative (faster delivery is better)
        # =====================================================================
        if 'sla' in feat_lower or 'delivery' in feat_lower:
            feature_types.append('monotonic_negative')
            monotonic_neg_count += 1
            if verbose:
                print(f"    {i:3d}. {feat_name:40s} -> Monotonic Negative (delivery speed)")
            continue

        # =====================================================================
        # 5. ALL OTHER FEATURES - Unconstrained
        # =====================================================================
        feature_types.append('unconstrained')
        unconstrained_count += 1

    # =========================================================================
    # SUMMARY
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("FEATURE MAPPING SUMMARY:")
        print("=" * 80)
        print(f"\n[STATISTICS]")
        print(f"    Total features: {len(feature_names)}")
        print(f"    Beta-Gamma (marketing): {beta_gamma_count}")
        print(f"    Monotonic Negative: {monotonic_neg_count}")
        print(f"    Monotonic Positive: {monotonic_pos_count}")
        print(f"    Unconstrained: {unconstrained_count}")

        print(f"\n[VALIDATION]")
        if beta_gamma_count >= 7:
            print(f"    [SUCCESS] {beta_gamma_count} Beta-Gamma features detected!")
            print(f"    [SUCCESS] Marketing saturation curves will be activated!")
        else:
            print(f"    [WARNING] Only {beta_gamma_count} Beta-Gamma features detected")
            print(f"    [WARNING] Expected at least 7 marketing channels")

        if monotonic_neg_count > 0 and monotonic_pos_count > 0:
            print(f"    [SUCCESS] Price and discount constraints will be enforced")
        else:
            print(f"    [WARNING] Price/discount constraints may be missing")

    return feature_types


def prepare_features_for_nam(data_path: str = 'data/processed/mmm_data_with_features.csv',
                            verbose: bool = True) -> Dict:
    """
    Prepare features and their types for NAM model training
    """

    if verbose:
        print("\n[LOADING] Reading enhanced dataset...")

    # Load data
    data = pd.read_csv(data_path)

    # Identify feature columns (exclude metadata)
    exclude_cols = ['Date', 'product_category', 'product_subcategory']

    # Get numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    # Remove metadata columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    if verbose:
        print(f"[INFO] Found {len(feature_cols)} potential features")

    # Map feature types
    feature_types = map_feature_types_for_mmm(feature_cols, verbose=verbose)

    # Create feature configuration
    feature_config = {
        'feature_names': feature_cols,
        'feature_types': feature_types,
        'n_features': len(feature_cols),
        'beta_gamma_features': [feat for feat, ftype in zip(feature_cols, feature_types)
                               if ftype == 'parametric_beta_gamma'],
        'monotonic_features': [feat for feat, ftype in zip(feature_cols, feature_types)
                              if 'monotonic' in ftype]
    }

    return feature_config


def test_feature_mapping():
    """
    Test the feature mapping with sample features
    """

    print("\n" + "=" * 80)
    print("TESTING FEATURE MAPPING")
    print("=" * 80)

    # Sample feature names that should be in our dataset
    test_features = [
        'GMV',
        'Units',
        'Avg_Price',
        'Discount_Pct',
        'TV_adstock',
        'TV_adstock_log',
        'Digital_adstock',
        'Digital_adstock_log',
        'SEM_adstock_log',
        'Sponsorship_adstock',
        'Content_Marketing_adstock_log',
        'Affiliates_adstock',
        'Radio_adstock',
        'Online_marketing_adstock_log',
        'ATL_spend_adstock',
        'BTL_spend_adstock_log',
        'TV_adstock_SOV',
        'Month_sin',
        'Month_cos',
        'NPS',
        'Category_Total_GMV',
        'Subcategory_GMV_Share',
        'Price_Marketing_interaction'
    ]

    print("\n[TEST] Mapping sample features:")
    feature_types = map_feature_types_for_mmm(test_features, verbose=False)

    # Count types
    type_counts = {}
    for ftype in feature_types:
        type_counts[ftype] = type_counts.get(ftype, 0) + 1

    print("\nTest Results:")
    for ftype, count in type_counts.items():
        print(f"    {ftype}: {count}")

    # Show some specific mappings
    print("\nSample Mappings:")
    important_features = ['TV_adstock', 'Digital_adstock_log', 'Avg_Price',
                         'Discount_Pct', 'NPS', 'Month_sin']
    for feat in important_features:
        if feat in test_features:
            idx = test_features.index(feat)
            print(f"    {feat:30s} -> {feature_types[idx]}")

    # Validate
    beta_gamma_count = type_counts.get('parametric_beta_gamma', 0)
    if beta_gamma_count >= 7:
        print("\n[TEST PASSED] Beta-Gamma activation will work!")
    else:
        print(f"\n[TEST WARNING] Only {beta_gamma_count} Beta-Gamma features")


def save_feature_config(feature_config: Dict, output_path: str = 'configs/feature_config.yaml'):
    """
    Save feature configuration to YAML file
    """
    import yaml
    import os

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to YAML-friendly format
    config = {
        'features': {
            'total_count': feature_config['n_features'],
            'beta_gamma_count': len(feature_config['beta_gamma_features']),
            'monotonic_count': len(feature_config['monotonic_features']),
            'feature_types': {
                name: ftype for name, ftype in
                zip(feature_config['feature_names'][:20], feature_config['feature_types'][:20])
            }
        },
        'marketing_features': feature_config['beta_gamma_features'][:15],  # Save first 15
        'constraint_features': feature_config['monotonic_features'][:10]   # Save first 10
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n[SAVE] Feature configuration saved to: {output_path}")


if __name__ == "__main__":
    # Test mapping first
    test_feature_mapping()

    # Prepare features from actual data
    print("\n" + "=" * 80)
    print("PROCESSING ACTUAL DATASET")
    print("=" * 80)

    feature_config = prepare_features_for_nam(
        data_path='data/processed/mmm_data_with_features.csv',
        verbose=True
    )

    # Save configuration
    save_feature_config(feature_config)

    # Final summary
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE: FEATURE MAPPING FIXED!")
    print("=" * 80)
    print(f"\n[RESULTS]")
    print(f"    Total features prepared: {feature_config['n_features']}")
    print(f"    Beta-Gamma features: {len(feature_config['beta_gamma_features'])}")
    print(f"    Monotonic constraints: {len(feature_config['monotonic_features'])}")

    print(f"\n[KEY BETA-GAMMA FEATURES]")
    for i, feat in enumerate(feature_config['beta_gamma_features'][:10], 1):
        print(f"    {i:2d}. {feat}")

    if len(feature_config['beta_gamma_features']) >= 7:
        print(f"\n[SUCCESS] Ready for Phase 4: Model training with proper MMM architecture!")
        print(f"[SUCCESS] Beta-Gamma layers WILL be activated for marketing saturation!")
    else:
        print(f"\n[WARNING] Feature mapping may need adjustment")

    print("=" * 80)