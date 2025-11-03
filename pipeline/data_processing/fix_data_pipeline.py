"""
Phase 1: Fix Data Pipeline - Load and Merge All Required Data Sources
This script loads firstfile.csv (daily), MediaInvestment.csv, MonthlyNPSscore.csv
and creates a clean hierarchical dataset for MMM
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_all_data(verbose=True):
    """
    Load and merge all data sources for proper MMM
    Uses firstfile.csv for daily granularity
    """

    # =========================================================================
    # 1. LOAD DAILY SALES DATA (firstfile.csv)
    # =========================================================================
    if verbose:
        print("=" * 80)
        print("PHASE 1: LOADING AND MERGING ALL DATA SOURCES")
        print("=" * 80)
        print("\n[DATA] Loading daily sales data from firstfile.csv...")

    try:
        sales_df = pd.read_csv('data/raw/firstfile.csv')

        # Parse date
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])

        # Clean column names
        sales_df.columns = sales_df.columns.str.strip().str.replace(' ', '_')

        if verbose:
            print(f"[OK] Loaded {len(sales_df):,} daily transaction records")
            print(f"[OK] Date range: {sales_df['Date'].min()} to {sales_df['Date'].max()}")
            print(f"[OK] Product categories: {sales_df['product_category'].nunique()}")
            print(f"[OK] Product subcategories: {sales_df['product_subcategory'].nunique()}")

    except Exception as e:
        print(f"[ERROR] Error loading firstfile.csv: {e}")
        return None

    # =========================================================================
    # 2. AGGREGATE DAILY SALES BY PRODUCT HIERARCHY
    # =========================================================================
    if verbose:
        print("\n[INFO] Aggregating by product hierarchy (category & subcategory)...")

    # Define aggregation columns
    agg_dict = {
        'gmv_new': 'sum',  # Total GMV
        'units': 'sum',     # Total units
        'product_mrp': 'mean',  # Average MRP
        'discount': 'mean',  # Average discount
    }

    # Aggregate at category + subcategory level
    daily_sales = sales_df.groupby([
        'Date',
        'product_category',
        'product_subcategory'
    ]).agg(agg_dict).reset_index()

    # Rename columns for clarity
    daily_sales.columns = ['Date', 'product_category', 'product_subcategory',
                           'GMV', 'Units', 'Avg_MRP', 'Avg_Discount']

    # Calculate derived metrics
    daily_sales['Avg_Price'] = daily_sales['GMV'] / (daily_sales['Units'] + 1e-6)
    daily_sales['Discount_Pct'] = daily_sales['Avg_Discount'] / (daily_sales['Avg_MRP'] + 1e-6)

    if verbose:
        print(f"[OK] Aggregated to {len(daily_sales):,} daily category-subcategory records")
        categories = daily_sales['product_category'].unique()
        print(f"[OK] Categories: {', '.join(categories)}")

    # =========================================================================
    # 3. LOAD MARKETING INVESTMENT DATA (monthly)
    # =========================================================================
    if verbose:
        print("\n[INFO] Loading marketing investment data...")

    try:
        media_df = pd.read_csv('data/raw/MediaInvestment.csv')

        # Create date from Year and Month
        media_df['Date'] = pd.to_datetime(media_df[['Year', 'Month']].assign(day=1))

        # Marketing channels (as they appear in the file)
        marketing_columns_raw = ['TV', 'Digital', 'Sponsorship', 'Content Marketing',
                               'Online marketing', ' Affiliates', 'SEM', 'Radio', 'Other']

        # Create clean column mapping
        column_mapping = {
            'TV': 'TV',
            'Digital': 'Digital',
            'Sponsorship': 'Sponsorship',
            'Content Marketing': 'Content_Marketing',
            'Online marketing': 'Online_marketing',
            ' Affiliates': 'Affiliates',  # Note the leading space
            'SEM': 'SEM',
            'Radio': 'Radio',
            'Other': 'Other',
            'Total Investment': 'Total_Investment'
        }

        # Rename columns
        media_df = media_df.rename(columns=column_mapping)

        # Select relevant columns
        marketing_columns_clean = ['TV', 'Digital', 'Sponsorship', 'Content_Marketing',
                                  'Online_marketing', 'Affiliates', 'SEM', 'Radio', 'Other']

        media_clean = media_df[['Date'] + marketing_columns_clean + ['Total_Investment']].copy()

        # Handle missing values (some channels might have NaN)
        media_clean[marketing_columns_clean] = media_clean[marketing_columns_clean].fillna(0)

        if verbose:
            print(f"[OK] Loaded {len(media_clean)} months of marketing data")
            print(f"[OK] Total investment range: ${media_clean['Total_Investment'].min():,.0f} - ${media_clean['Total_Investment'].max():,.0f}")
            print(f"[OK] Marketing channels: {', '.join(marketing_columns_clean)}")

    except Exception as e:
        print(f"[ERROR] Error loading MediaInvestment.csv: {e}")
        return None

    # =========================================================================
    # 4. LOAD NPS SCORES (monthly)
    # =========================================================================
    if verbose:
        print("\n[INFO] Loading NPS scores...")

    try:
        nps_df = pd.read_csv('data/raw/MonthlyNPSscore.csv')
        nps_df['Date'] = pd.to_datetime(nps_df['Date'])
        nps_df = nps_df[['Date', 'NPS']].copy()

        if verbose:
            print(f"[OK] Loaded {len(nps_df)} months of NPS scores")
            print(f"[OK] NPS range: {nps_df['NPS'].min():.1f} - {nps_df['NPS'].max():.1f}")

    except Exception as e:
        print(f"[ERROR] Error loading MonthlyNPSscore.csv: {e}")
        return None

    # =========================================================================
    # 5. MERGE MONTHLY DATA (Marketing + NPS)
    # =========================================================================
    if verbose:
        print("\n[INFO] Merging monthly marketing and NPS data...")

    monthly_data = media_clean.merge(nps_df, on='Date', how='left')

    # =========================================================================
    # 6. INTERPOLATE MONTHLY TO DAILY
    # =========================================================================
    if verbose:
        print("\n[INFO] Interpolating monthly data to daily granularity...")

    # Create daily date range
    date_range = pd.date_range(start=daily_sales['Date'].min(),
                               end=daily_sales['Date'].max(),
                               freq='D')

    daily_template = pd.DataFrame({'Date': date_range})

    # Merge monthly data
    daily_template['Year'] = daily_template['Date'].dt.year
    daily_template['Month'] = daily_template['Date'].dt.month
    monthly_data['Year'] = monthly_data['Date'].dt.year
    monthly_data['Month'] = monthly_data['Date'].dt.month

    # Merge on Year-Month
    daily_marketing = daily_template.merge(
        monthly_data.drop('Date', axis=1),
        on=['Year', 'Month'],
        how='left'
    )

    # Linear interpolation for mid-month transitions
    marketing_cols = marketing_columns_clean + ['Total_Investment', 'NPS']
    daily_marketing[marketing_cols] = daily_marketing[marketing_cols].interpolate(method='linear')

    # Fill any remaining NaNs
    daily_marketing[marketing_cols] = daily_marketing[marketing_cols].fillna(method='ffill').fillna(method='bfill')

    if verbose:
        print(f"[OK] Interpolated to {len(daily_marketing)} daily records")

    # =========================================================================
    # 7. MERGE SALES WITH MARKETING DATA
    # =========================================================================
    if verbose:
        print("\n[INFO] Merging sales with marketing data...")

    # Since marketing is at overall level, we need to merge with each product
    final_data = daily_sales.merge(
        daily_marketing[['Date'] + marketing_cols],
        on='Date',
        how='inner'
    )

    # =========================================================================
    # 8. ADD TEMPORAL FEATURES
    # =========================================================================
    if verbose:
        print("\n[INFO] Adding temporal features...")

    final_data['Month'] = final_data['Date'].dt.month
    final_data['Quarter'] = final_data['Date'].dt.quarter
    final_data['DayOfWeek'] = final_data['Date'].dt.dayofweek
    final_data['WeekOfYear'] = final_data['Date'].dt.isocalendar().week

    # Cyclical encoding for month
    final_data['Month_sin'] = np.sin(2 * np.pi * final_data['Month'] / 12)
    final_data['Month_cos'] = np.cos(2 * np.pi * final_data['Month'] / 12)

    # Time index (days since start)
    final_data['Time_Index'] = (final_data['Date'] - final_data['Date'].min()).dt.days

    # =========================================================================
    # 9. CREATE HIERARCHICAL AGGREGATES
    # =========================================================================
    if verbose:
        print("\n[INFO] Creating hierarchical aggregates...")

    # Category-level totals (for each date)
    category_totals = daily_sales.groupby(['Date', 'product_category']).agg({
        'GMV': 'sum',
        'Units': 'sum'
    }).reset_index()
    category_totals.columns = ['Date', 'product_category', 'Category_Total_GMV', 'Category_Total_Units']

    # Merge back
    final_data = final_data.merge(category_totals, on=['Date', 'product_category'], how='left')

    # Calculate market share within category
    final_data['Subcategory_GMV_Share'] = final_data['GMV'] / (final_data['Category_Total_GMV'] + 1e-6)

    # =========================================================================
    # 10. FINAL CLEANUP AND SORTING
    # =========================================================================
    final_data = final_data.sort_values(['Date', 'product_category', 'product_subcategory']).reset_index(drop=True)

    # Log transform large values
    for col in ['GMV', 'Category_Total_GMV'] + marketing_columns_clean + ['Total_Investment']:
        if col in final_data.columns:
            final_data[f'{col}_log'] = np.log1p(final_data[col])

    if verbose:
        print(f"\n[SUCCESS] FINAL DATASET CREATED:")
        print(f"   Shape: {final_data.shape}")
        print(f"   Date range: {final_data['Date'].min()} to {final_data['Date'].max()}")
        print(f"   Categories: {final_data['product_category'].nunique()}")
        print(f"   Subcategories: {final_data['product_subcategory'].nunique()}")
        print(f"   Total records: {len(final_data):,}")

    return final_data

def show_data_sample(data, n_samples=10):
    """Show sample of the merged data"""

    print("\n" + "=" * 80)
    print("DATA SAMPLE")
    print("=" * 80)

    # Select key columns for display
    display_cols = [
        'Date', 'product_category', 'product_subcategory',
        'GMV', 'Units', 'Avg_Price', 'Discount_Pct',
        'TV', 'Digital', 'SEM', 'NPS',
        'Category_Total_GMV', 'Subcategory_GMV_Share'
    ]

    # Filter columns that exist
    display_cols = [col for col in display_cols if col in data.columns]

    # Show sample
    sample = data[display_cols].head(n_samples)

    print("\n[INFO] First 10 rows of merged data:")
    print(sample.to_string(index=False))

    # Show data summary by category
    print("\n[INFO] Summary by Product Category:")
    category_summary = data.groupby('product_category').agg({
        'GMV': ['sum', 'mean'],
        'Units': ['sum', 'mean'],
        'Date': ['min', 'max', 'count']
    }).round(2)
    print(category_summary)

    # Show marketing investment summary
    marketing_cols = ['TV', 'Digital', 'SEM', 'Sponsorship', 'Content_Marketing',
                     'Affiliates', 'Radio', 'Online_marketing']
    existing_marketing = [col for col in marketing_cols if col in data.columns]

    if existing_marketing:
        print("\n[INFO] Marketing Investment Summary (Daily Interpolated):")
        marketing_summary = data[existing_marketing].describe().round(2)
        print(marketing_summary)

    # Check for hierarchical structure
    print("\n[INFO] Hierarchical Structure:")
    hierarchy = data.groupby(['product_category', 'product_subcategory']).size().reset_index(name='count')
    print(hierarchy.head(15))

    # Feature availability check
    print("\n[SUCCESS] Feature Availability Check:")
    print(f"   Sales features: [OK] (GMV, Units, Price, Discount)")
    print(f"   Marketing features: [OK] ({len(existing_marketing)} channels)")
    print(f"   NPS scores: {'[OK]' if 'NPS' in data.columns else '[X]'}")
    print(f"   Temporal features: [OK] (Month, Quarter, Cyclical)")
    print(f"   Hierarchical features: [OK] (Category totals, Market share)")

    return data

# Main execution
if __name__ == "__main__":
    # Load and merge all data
    merged_data = load_and_merge_all_data(verbose=True)

    if merged_data is not None:
        # Show sample
        show_data_sample(merged_data)

        # Save the merged data
        print("\n[SAVE] Saving merged data...")
        merged_data.to_csv('data/processed/merged_mmm_data.csv', index=False)
        print("[OK] Saved to: data/processed/merged_mmm_data.csv")

        # Create a smaller sample for testing
        sample_data = merged_data.head(1000)
        sample_data.to_csv('data/processed/sample_mmm_data.csv', index=False)
        print("[OK] Sample saved to: data/processed/sample_mmm_data.csv")

        print("\n" + "=" * 80)
        print("[COMPLETE] DATA PIPELINE FIXED! Ready for Phase 2: Marketing Features")
        print("=" * 80)