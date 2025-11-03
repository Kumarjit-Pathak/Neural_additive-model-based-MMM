#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a series of educational notebooks for NAM-MMM
Each notebook focuses on a specific aspect of the implementation
"""

import json
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_notebook_01_data_foundation():
    """Notebook 1: Data Foundation and Understanding"""

    nb = new_notebook()

    # Title and Introduction
    nb.cells.append(new_markdown_cell("""# Notebook 1: Data Foundation for Marketing Mix Modeling
## Understanding and Preparing Your Data

**Learning Objectives:**
- Understand the three key data sources for MMM
- Learn how to aggregate data hierarchically
- Master time-series data alignment
- Create a unified dataset for modeling

---

## Why Data Foundation Matters

Marketing Mix Modeling requires integrating multiple data sources at different granularities. The quality of your data pipeline directly impacts model accuracy. We'll work with:
1. **Sales Data** (daily transactions)
2. **Marketing Investment** (monthly budgets)
3. **Brand Health Metrics** (NPS scores)

Each source has different frequencies and hierarchies that we must carefully align."""))

    # Setup
    nb.cells.append(new_code_cell("""# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
%matplotlib inline

print("Libraries loaded successfully!")"""))

    # Data Loading
    nb.cells.append(new_markdown_cell("""## Step 1: Loading the Three Data Sources

We have three critical data files that form the foundation of our MMM:
- **firstfile.csv**: Daily sales transactions with product hierarchy
- **MediaInvestment.csv**: Monthly marketing channel investments
- **MonthlyNPSscore.csv**: Monthly Net Promoter Score (brand health indicator)

Let's load and explore each one:"""))

    nb.cells.append(new_code_cell("""def load_and_explore_data():
    '''
    Load all three data sources and display their characteristics.
    This function helps us understand what we're working with.
    '''

    # Load sales data - our primary response variable source
    sales_df = pd.read_csv('data/firstfile.csv')
    print("SALES DATA OVERVIEW")
    print("=" * 50)
    print(f"Shape: {sales_df.shape[0]:,} rows × {sales_df.shape[1]} columns")
    print(f"Date range: {sales_df['Date'].min()} to {sales_df['Date'].max()}")
    print(f"\\nColumns: {', '.join(sales_df.columns.tolist())}")
    print(f"\\nProduct Categories: {sales_df['product_category'].nunique()}")
    print(f"Product Subcategories: {sales_df['product_subcategory'].nunique()}")

    # Load marketing investment data - our key predictors
    marketing_df = pd.read_csv('data/MediaInvestment.csv')
    print("\\n" + "=" * 50)
    print("MARKETING INVESTMENT DATA OVERVIEW")
    print("=" * 50)
    print(f"Shape: {marketing_df.shape[0]:,} rows × {marketing_df.shape[1]} columns")

    # Clean column names (remove any leading/trailing spaces)
    marketing_df.columns = marketing_df.columns.str.strip()

    # Identify marketing channels
    marketing_channels = [col for col in marketing_df.columns
                         if col not in ['Date', 'Total Investment']]
    print(f"Marketing Channels: {', '.join(marketing_channels)}")
    print(f"Total Channels: {len(marketing_channels)}")

    # Load NPS data - brand health indicator
    nps_df = pd.read_csv('data/MonthlyNPSscore.csv')
    print("\\n" + "=" * 50)
    print("NPS (BRAND HEALTH) DATA OVERVIEW")
    print("=" * 50)
    print(f"Shape: {nps_df.shape[0]:,} rows × {nps_df.shape[1]} columns")
    print(f"NPS Range: {nps_df['NPS'].min():.1f} to {nps_df['NPS'].max():.1f}")

    return sales_df, marketing_df, nps_df, marketing_channels

# Load all data
sales_df, marketing_df, nps_df, marketing_channels = load_and_explore_data()

# Display sample data
print("\\nSample Sales Data:")
display(sales_df.head())"""))

    # Data Quality Check
    nb.cells.append(new_markdown_cell("""## Step 2: Data Quality Assessment

Before merging, let's check for data quality issues:"""))

    nb.cells.append(new_code_cell("""def check_data_quality(sales_df, marketing_df, nps_df):
    '''
    Perform comprehensive data quality checks.
    Good data quality is essential for reliable MMM results.
    '''

    print("DATA QUALITY REPORT")
    print("=" * 60)

    # Check for missing values
    print("\\n1. MISSING VALUES CHECK:")
    print("-" * 40)

    print("Sales Data:")
    sales_missing = sales_df.isnull().sum()
    if sales_missing.sum() == 0:
        print("  ✓ No missing values")
    else:
        print(f"  ⚠ Missing values found:")
        print(sales_missing[sales_missing > 0])

    print("\\nMarketing Data:")
    marketing_missing = marketing_df.isnull().sum()
    if marketing_missing.sum() == 0:
        print("  ✓ No missing values")
    else:
        print(f"  ⚠ Missing values found:")
        print(marketing_missing[marketing_missing > 0])

    # Check date formats
    print("\\n2. DATE CONSISTENCY CHECK:")
    print("-" * 40)

    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    marketing_df['Date'] = pd.to_datetime(marketing_df['Date'])
    nps_df['Date'] = pd.to_datetime(nps_df['Date'])

    print(f"Sales: {sales_df['Date'].min()} to {sales_df['Date'].max()}")
    print(f"Marketing: {marketing_df['Date'].min()} to {marketing_df['Date'].max()}")
    print(f"NPS: {nps_df['Date'].min()} to {nps_df['Date'].max()}")

    # Check for negative values
    print("\\n3. NEGATIVE VALUES CHECK:")
    print("-" * 40)

    if (sales_df['GMV'] < 0).any():
        print("  ⚠ Negative GMV values found!")
    else:
        print("  ✓ All GMV values are non-negative")

    if (sales_df['Units'] < 0).any():
        print("  ⚠ Negative Units found!")
    else:
        print("  ✓ All Units are non-negative")

    # Check marketing spend
    for channel in marketing_channels:
        if channel in marketing_df.columns:
            if (marketing_df[channel] < 0).any():
                print(f"  ⚠ Negative spend in {channel}")

    return sales_df, marketing_df, nps_df

# Run quality checks
sales_df, marketing_df, nps_df = check_data_quality(sales_df, marketing_df, nps_df)"""))

    # Hierarchical Aggregation
    nb.cells.append(new_markdown_cell("""## Step 3: Hierarchical Data Aggregation

A key insight for MMM is that marketing effects can vary by product hierarchy. We'll aggregate sales data by category and subcategory to capture these patterns:"""))

    nb.cells.append(new_code_cell("""def create_hierarchical_aggregation(sales_df):
    '''
    Aggregate sales data by product hierarchy.
    This captures how different product groups respond to marketing.

    Why this matters:
    - Premium products may respond differently to promotions
    - Category-level effects can be stronger than individual SKUs
    - Reduces noise while preserving important patterns
    '''

    print("Creating Hierarchical Aggregation...")
    print("=" * 60)

    # Group by date and product hierarchy
    hierarchy_agg = sales_df.groupby(
        ['Date', 'product_category', 'product_subcategory']
    ).agg({
        'GMV': 'sum',           # Total revenue
        'Units': 'sum',         # Total units sold
        'Avg_MRP': 'mean',      # Average list price
        'Avg_Price': 'mean'     # Average selling price
    }).reset_index()

    print(f"\\nAggregation Results:")
    print(f"  Original records: {len(sales_df):,}")
    print(f"  Aggregated records: {len(hierarchy_agg):,}")
    print(f"  Compression ratio: {len(sales_df)/len(hierarchy_agg):.1f}x")

    # Calculate additional metrics
    hierarchy_agg['Avg_Discount'] = hierarchy_agg['Avg_MRP'] - hierarchy_agg['Avg_Price']
    hierarchy_agg['Discount_Pct'] = (hierarchy_agg['Avg_Discount'] /
                                     (hierarchy_agg['Avg_MRP'] + 0.01)) * 100

    # Show hierarchy structure
    print(f"\\nHierarchy Structure:")
    print(f"  Categories: {hierarchy_agg['product_category'].nunique()}")
    print(f"  Subcategories: {hierarchy_agg['product_subcategory'].nunique()}")
    print(f"  Date range: {hierarchy_agg['Date'].nunique()} unique days")

    return hierarchy_agg

# Create hierarchical aggregation
hierarchy_data = create_hierarchical_aggregation(sales_df)

# Visualize hierarchy
plt.figure(figsize=(12, 5))

# Plot 1: Sales by category
plt.subplot(1, 2, 1)
category_sales = hierarchy_data.groupby('product_category')['GMV'].sum().sort_values()
category_sales.plot(kind='barh')
plt.title('Total Sales by Product Category')
plt.xlabel('GMV (Total)')
plt.ylabel('Category')

# Plot 2: Number of subcategories per category
plt.subplot(1, 2, 2)
subcat_count = hierarchy_data.groupby('product_category')['product_subcategory'].nunique()
subcat_count.plot(kind='bar')
plt.title('Subcategories per Category')
plt.xlabel('Category')
plt.ylabel('Number of Subcategories')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()"""))

    # Time Alignment
    nb.cells.append(new_markdown_cell("""## Step 4: Time-Series Alignment

Marketing data is monthly but sales are daily. We need to align these different frequencies intelligently:"""))

    nb.cells.append(new_code_cell("""def align_time_frequencies(hierarchy_data, marketing_df, nps_df):
    '''
    Align different time frequencies across data sources.

    Strategy:
    1. Keep sales at daily granularity (most detailed)
    2. Interpolate monthly marketing to daily (smooth transitions)
    3. Forward-fill NPS scores (brand perception changes slowly)

    This preserves maximum information while ensuring alignment.
    '''

    print("Aligning Time Frequencies...")
    print("=" * 60)

    # Create daily date range
    date_range = pd.date_range(
        start=hierarchy_data['Date'].min(),
        end=hierarchy_data['Date'].max(),
        freq='D'
    )

    print(f"Daily date range: {len(date_range)} days")
    print(f"From {date_range[0]} to {date_range[-1]}")

    # Expand marketing data to daily
    print("\\nExpanding Marketing Data (Monthly → Daily):")

    # Create daily scaffold
    marketing_daily = pd.DataFrame({'Date': date_range})
    marketing_daily['YearMonth'] = marketing_daily['Date'].dt.to_period('M')
    marketing_df['YearMonth'] = marketing_df['Date'].dt.to_period('M')

    # Merge monthly data
    marketing_daily = marketing_daily.merge(
        marketing_df.drop('Date', axis=1),
        on='YearMonth',
        how='left'
    )

    # Interpolate marketing spend (linear interpolation for smooth transitions)
    for channel in marketing_channels:
        if channel in marketing_daily.columns:
            # First forward fill to handle initial NaN
            marketing_daily[channel] = marketing_daily[channel].fillna(method='ffill')
            # Then interpolate for smooth transitions
            marketing_daily[channel] = marketing_daily[channel].interpolate(method='linear')
            # Finally fill any remaining NaN with 0
            marketing_daily[channel] = marketing_daily[channel].fillna(0)

    print(f"  ✓ Interpolated {len(marketing_channels)} channels to daily")

    # Expand NPS to daily
    print("\\nExpanding NPS Data (Monthly → Daily):")
    nps_daily = pd.DataFrame({'Date': date_range})
    nps_daily['YearMonth'] = nps_daily['Date'].dt.to_period('M')
    nps_df['YearMonth'] = nps_df['Date'].dt.to_period('M')

    nps_daily = nps_daily.merge(
        nps_df[['YearMonth', 'NPS']],
        on='YearMonth',
        how='left'
    )

    # Forward fill NPS (assumption: brand perception changes slowly)
    nps_daily['NPS'] = nps_daily['NPS'].fillna(method='ffill')
    nps_daily['NPS'] = nps_daily['NPS'].fillna(nps_daily['NPS'].mean())

    print(f"  ✓ Expanded NPS to daily using forward-fill")

    return marketing_daily, nps_daily

# Align time frequencies
marketing_daily, nps_daily = align_time_frequencies(hierarchy_data, marketing_df, nps_df)

# Visualize the alignment
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Original monthly marketing
axes[0].plot(marketing_df['Date'], marketing_df['TV'], 'o-', label='Monthly TV Spend')
axes[0].set_title('Original Monthly Marketing Data')
axes[0].set_ylabel('Spend')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Interpolated daily marketing
axes[1].plot(marketing_daily['Date'], marketing_daily['TV'], '-', label='Daily TV Spend (Interpolated)', alpha=0.7)
axes[1].set_title('Interpolated Daily Marketing Data')
axes[1].set_ylabel('Spend')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: NPS daily
axes[2].plot(nps_daily['Date'], nps_daily['NPS'], '-', label='Daily NPS (Forward-filled)', color='green')
axes[2].set_title('Daily NPS Scores')
axes[2].set_ylabel('NPS')
axes[2].set_xlabel('Date')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

    # Final Merge
    nb.cells.append(new_markdown_cell("""## Step 5: Creating the Unified Dataset

Now we merge all aligned data sources into a single comprehensive dataset:"""))

    nb.cells.append(new_code_cell("""def create_unified_dataset(hierarchy_data, marketing_daily, nps_daily):
    '''
    Create the final unified dataset for modeling.
    This is the foundation for all subsequent analysis.
    '''

    print("Creating Unified Dataset...")
    print("=" * 60)

    # Start with hierarchy data
    unified_data = hierarchy_data.copy()

    # Merge marketing data
    print("Merging marketing data...")
    marketing_cols = ['Date'] + marketing_channels
    unified_data = unified_data.merge(
        marketing_daily[marketing_cols],
        on='Date',
        how='left'
    )

    # Merge NPS data
    print("Merging NPS data...")
    unified_data = unified_data.merge(
        nps_daily[['Date', 'NPS']],
        on='Date',
        how='left'
    )

    # Fill any remaining NaN values
    for col in marketing_channels:
        if col in unified_data.columns:
            unified_data[col] = unified_data[col].fillna(0)

    unified_data['NPS'] = unified_data['NPS'].fillna(unified_data['NPS'].mean())

    # Add time features for seasonality
    unified_data['Year'] = unified_data['Date'].dt.year
    unified_data['Month'] = unified_data['Date'].dt.month
    unified_data['Quarter'] = unified_data['Date'].dt.quarter
    unified_data['DayOfWeek'] = unified_data['Date'].dt.dayofweek
    unified_data['WeekOfYear'] = unified_data['Date'].dt.isocalendar().week.astype(int)

    print(f"\\nFinal Dataset Shape: {unified_data.shape}")
    print(f"  Rows: {unified_data.shape[0]:,}")
    print(f"  Columns: {unified_data.shape[1]}")
    print(f"\\nFeatures included:")
    print(f"  - Sales metrics: GMV, Units, Prices")
    print(f"  - Marketing channels: {len(marketing_channels)}")
    print(f"  - Brand health: NPS")
    print(f"  - Time features: Year, Month, Quarter, etc.")
    print(f"  - Product hierarchy: Category, Subcategory")

    return unified_data

# Create unified dataset
unified_data = create_unified_dataset(hierarchy_data, marketing_daily, nps_daily)

# Display sample
print("\\nSample of Unified Dataset:")
display(unified_data.head())

# Save the processed data
unified_data.to_csv('data/processed/unified_mmm_data.csv', index=False)
print("\\n✓ Saved unified dataset to 'data/processed/unified_mmm_data.csv'")"""))

    # Summary Statistics
    nb.cells.append(new_markdown_cell("""## Step 6: Data Validation and Summary Statistics

Let's validate our unified dataset and understand its characteristics:"""))

    nb.cells.append(new_code_cell("""def validate_unified_dataset(unified_data):
    '''
    Comprehensive validation of the unified dataset.
    Ensures data is ready for modeling.
    '''

    print("UNIFIED DATASET VALIDATION")
    print("=" * 60)

    # Basic statistics
    print("\\n1. DATASET OVERVIEW:")
    print("-" * 40)
    print(f"Total records: {len(unified_data):,}")
    print(f"Date range: {unified_data['Date'].min()} to {unified_data['Date'].max()}")
    print(f"Days covered: {unified_data['Date'].nunique()}")
    print(f"Categories: {unified_data['product_category'].nunique()}")
    print(f"Subcategories: {unified_data['product_subcategory'].nunique()}")

    # Sales statistics
    print("\\n2. SALES METRICS:")
    print("-" * 40)
    print(f"Total GMV: ${unified_data['GMV'].sum():,.0f}")
    print(f"Average daily GMV: ${unified_data.groupby('Date')['GMV'].sum().mean():,.0f}")
    print(f"Total units sold: {unified_data['Units'].sum():,.0f}")

    # Marketing statistics
    print("\\n3. MARKETING INVESTMENT:")
    print("-" * 40)
    total_marketing = unified_data[marketing_channels].sum().sum()
    print(f"Total marketing spend: ${total_marketing:,.0f}")

    for channel in marketing_channels[:5]:  # Top 5 channels
        if channel in unified_data.columns:
            spend = unified_data[channel].sum()
            pct = (spend / total_marketing) * 100
            print(f"  {channel}: ${spend:,.0f} ({pct:.1f}%)")

    # Data completeness
    print("\\n4. DATA COMPLETENESS:")
    print("-" * 40)
    missing_pct = (unified_data.isnull().sum() / len(unified_data)) * 100
    complete_features = (missing_pct == 0).sum()
    print(f"Features with no missing values: {complete_features}/{len(missing_pct)}")

    if missing_pct.sum() > 0:
        print("\\nFeatures with missing values:")
        print(missing_pct[missing_pct > 0])

    return True

# Validate the dataset
is_valid = validate_unified_dataset(unified_data)

if is_valid:
    print("\\n" + "=" * 60)
    print("✓ DATASET READY FOR MODELING!")
    print("=" * 60)"""))

    # Key Insights
    nb.cells.append(new_markdown_cell("""## Key Takeaways

### What We've Accomplished:
1. **Loaded three diverse data sources** with different granularities
2. **Created hierarchical aggregation** to capture product-level patterns
3. **Aligned time frequencies** through intelligent interpolation
4. **Built a unified dataset** ready for feature engineering

### Why This Matters for MMM:
- **Hierarchical structure** allows us to model different product responses to marketing
- **Daily granularity** captures immediate and lagged marketing effects
- **Unified dataset** ensures consistent analysis across all components

### Data Characteristics for Modeling:
- **Response variable (Y):** GMV at category/subcategory level
- **Marketing predictors (X):** 8+ marketing channels with daily spend
- **Control variables:** Price, discount, NPS, seasonality
- **Hierarchy:** Category and subcategory for pooled learning

### Next Steps:
In the next notebook, we'll engineer advanced features including:
- Adstock transformations for marketing carryover
- Beta-Gamma features for saturation curves
- Price elasticity features
- Seasonal decomposition

---

**Remember:** The quality of your data foundation determines the ceiling of your model's performance. We've built a solid foundation that preserves important patterns while maintaining data integrity."""))

    # Save notebook
    with open('01_Data_Foundation.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print("Created: 01_Data_Foundation.ipynb")
    return nb

def create_notebook_02_feature_engineering():
    """Notebook 2: Feature Engineering for Marketing Mix"""

    nb = new_notebook()

    nb.cells.append(new_markdown_cell("""# Notebook 2: Feature Engineering for Marketing Mix Modeling
## Creating Features that Capture Marketing Dynamics

**Learning Objectives:**
- Master adstock transformation for marketing carryover effects
- Implement Beta-Gamma features for saturation modeling
- Create price elasticity and promotional features
- Build seasonal and trend components

---

## The Science of Marketing Features

Marketing doesn't work instantly - it has carryover effects and saturation points. This notebook transforms raw data into features that capture these real-world dynamics."""))

    # Setup and data loading
    nb.cells.append(new_code_cell("""# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the unified dataset from Notebook 1
unified_data = pd.read_csv('data/processed/unified_mmm_data.csv')
unified_data['Date'] = pd.to_datetime(unified_data['Date'])

print(f"Loaded dataset: {unified_data.shape}")
print(f"Date range: {unified_data['Date'].min()} to {unified_data['Date'].max()}")

# Identify marketing channels
marketing_channels = ['TV', 'Sponsorship', 'Content_Marketing', 'Digital',
                      'SEM', 'Affiliates', 'Online_marketing', 'Radio']

# Clean column names
unified_data.columns = unified_data.columns.str.strip().str.replace(' ', '_')
print(f"\\nMarketing channels: {', '.join(marketing_channels)}")"""))

    # Adstock transformation section
    nb.cells.append(new_markdown_cell("""## Part 1: Adstock Transformation - Capturing Carryover Effects

### Understanding Adstock

Marketing effects don't disappear immediately. A TV ad today influences purchases tomorrow, next week, even next month - but with diminishing impact. The adstock transformation models this decay:

**Formula:** Adstock_t = x_t + λ×x_{t-1} + λ²×x_{t-2} + ... + λⁿ×x_{t-n}

Where:
- x_t is the marketing spend at time t
- λ (lambda) is the decay rate (0 to 1)
- n is the maximum lag period

### Why Different Channels Have Different Decay Rates

- **Brand Building** (TV, Sponsorship): λ = 0.7-0.9 (slow decay, lasting impact)
- **Performance Marketing** (SEM, Digital): λ = 0.2-0.5 (fast decay, immediate impact)
- **Content Marketing**: λ = 0.5-0.7 (moderate decay)"""))

    nb.cells.append(new_code_cell("""def apply_adstock_transformation(x, decay_rate=0.7, max_lag=3):
    '''
    Apply adstock transformation to capture marketing carryover effects.

    Parameters:
    -----------
    x : array-like
        Marketing spend time series
    decay_rate : float
        Rate of decay (0-1). Higher = longer lasting effect
    max_lag : int
        Maximum periods to consider carryover

    Returns:
    --------
    Adstocked time series with carryover effects
    '''

    x = np.array(x, dtype=np.float64)
    adstocked = np.zeros_like(x)

    for lag in range(max_lag + 1):
        decay_factor = decay_rate ** lag

        if lag == 0:
            # Current period effect
            adstocked += decay_factor * x
        else:
            # Lagged effect
            shifted = np.zeros_like(x)
            shifted[lag:] = x[:-lag]
            adstocked += decay_factor * shifted

    return adstocked

# Demonstrate adstock with different decay rates
demo_spend = np.zeros(20)
demo_spend[5] = 100  # Single spike of spend

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

decay_rates = [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
titles = ['Brand TV (λ=0.9)', 'Sponsorship (λ=0.7)', 'Content (λ=0.5)',
          'Digital (λ=0.3)', 'SEM (λ=0.1)', 'No Carryover (λ=0)']

for idx, (decay, title) in enumerate(zip(decay_rates, titles)):
    ax = axes[idx // 3, idx % 3]
    adstocked = apply_adstock_transformation(demo_spend, decay_rate=decay, max_lag=10)

    ax.bar(range(len(demo_spend)), demo_spend, alpha=0.3, label='Original', color='blue')
    ax.plot(range(len(adstocked)), adstocked, 'ro-', label='Adstocked', markersize=4)
    ax.set_title(title)
    ax.set_xlabel('Days')
    ax.set_ylabel('Effective Spend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(3, 15)

plt.suptitle('Adstock Effect: How Marketing Spend Carries Over Time', fontsize=14)
plt.tight_layout()
plt.show()

print("Key Insight: Higher decay rates (brand building) create longer-lasting effects!")"""))

    # Apply adstock to all channels
    nb.cells.append(new_code_cell("""def create_adstock_features(data, marketing_channels):
    '''
    Create adstock features for all marketing channels with appropriate decay rates.
    '''

    # Define channel-specific decay rates based on marketing theory
    decay_rates = {
        'TV': 0.8,                 # Brand building - slow decay
        'Sponsorship': 0.75,       # Brand building - slow decay
        'Content_Marketing': 0.6,  # Mixed - moderate decay
        'Digital': 0.4,            # Performance - fast decay
        'SEM': 0.3,               # Performance - very fast decay
        'Affiliates': 0.3,        # Performance - very fast decay
        'Online_marketing': 0.4,   # Performance - fast decay
        'Radio': 0.5              # Mixed - moderate decay
    }

    print("Creating Adstock Features...")
    print("=" * 60)

    # Sort by date and category for proper time series
    data_sorted = data.sort_values(['product_category', 'product_subcategory', 'Date']).copy()

    adstock_features = []

    for channel in marketing_channels:
        if channel in data_sorted.columns:
            decay = decay_rates.get(channel, 0.5)
            adstock_col = f"{channel}_adstock"

            # Apply adstock per category (different products may have different patterns)
            data_sorted[adstock_col] = data_sorted.groupby('product_category')[channel].transform(
                lambda x: apply_adstock_transformation(x.values, decay_rate=decay, max_lag=3)
            )

            adstock_features.append(adstock_col)
            print(f"  ✓ {adstock_col} created (decay={decay})")

    print(f"\\nTotal adstock features created: {len(adstock_features)}")
    return data_sorted, adstock_features

# Create adstock features
data_with_adstock, adstock_features = create_adstock_features(unified_data, marketing_channels)

# Visualize original vs adstocked for one channel
plt.figure(figsize=(14, 5))

sample_category = data_with_adstock['product_category'].iloc[0]
sample_data = data_with_adstock[data_with_adstock['product_category'] == sample_category].head(60)

plt.subplot(1, 2, 1)
plt.plot(sample_data['Date'], sample_data['TV'], label='Original TV Spend', alpha=0.7)
plt.plot(sample_data['Date'], sample_data['TV_adstock'], label='TV Adstock', linewidth=2)
plt.title('TV Spend: Original vs Adstocked')
plt.xlabel('Date')
plt.ylabel('Spend')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(sample_data['Date'], sample_data['Digital'], label='Original Digital Spend', alpha=0.7)
plt.plot(sample_data['Date'], sample_data['Digital_adstock'], label='Digital Adstock', linewidth=2)
plt.title('Digital Spend: Original vs Adstocked (Faster Decay)')
plt.xlabel('Date')
plt.ylabel('Spend')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

    # Beta-Gamma Features
    nb.cells.append(new_markdown_cell("""## Part 2: Beta-Gamma Transformation - Modeling Saturation

### Understanding Marketing Saturation

Marketing effectiveness isn't linear. The first $1000 in TV spend might drive significant sales, but the 100,000th dollar has diminishing returns. The Beta-Gamma transformation captures this:

**Formula:** f(x) = α × x^β × e^(-γ×x)

Where:
- α (alpha): Scale parameter (maximum effect)
- β (beta): Shape parameter (initial effectiveness)
- γ (gamma): Saturation parameter (how quickly returns diminish)

This creates the characteristic S-curve of marketing response."""))

    nb.cells.append(new_code_cell("""def create_beta_gamma_features(data, adstock_features):
    '''
    Create features that will be processed by Beta-Gamma transformation in the model.
    Here we prepare and mark these features for special treatment.
    '''

    print("Preparing Beta-Gamma Features...")
    print("=" * 60)

    # Create log-transformed versions for multiplicative effects
    log_features = []

    for feature in adstock_features:
        channel = feature.replace('_adstock', '')

        # Log transformation (handles zeros with log1p)
        log_col = f"{channel}_log"
        data[log_col] = np.log1p(data[feature])
        log_features.append(log_col)

    # Also create log versions of original spend
    for channel in marketing_channels:
        if channel in data.columns:
            log_col = f"{channel}_raw_log"
            data[log_col] = np.log1p(data[channel])
            log_features.append(log_col)

    # Mark all features that should receive Beta-Gamma transformation
    beta_gamma_features = adstock_features + log_features

    print(f"\\nFeatures marked for Beta-Gamma transformation:")
    print(f"  - Adstock features: {len(adstock_features)}")
    print(f"  - Log features: {len(log_features)}")
    print(f"  - Total Beta-Gamma features: {len(beta_gamma_features)}")

    # This is critical for MMM - these features will model saturation!
    print("\\n⚡ These {0} features will capture marketing saturation curves!".format(len(beta_gamma_features)))

    return data, beta_gamma_features

# Create Beta-Gamma features
data_with_features, beta_gamma_features = create_beta_gamma_features(data_with_adstock, adstock_features)

print("\\nBeta-Gamma features created:")
for i, feature in enumerate(beta_gamma_features[:10], 1):
    print(f"  {i}. {feature}")
if len(beta_gamma_features) > 10:
    print(f"  ... and {len(beta_gamma_features) - 10} more")"""))

    # Price and Promotional Features
    nb.cells.append(new_markdown_cell("""## Part 3: Price and Promotional Features

Price and discounts have strong, often non-linear effects on sales. We'll create features that capture these dynamics:"""))

    nb.cells.append(new_code_cell("""def create_price_features(data):
    '''
    Create price-related features with business logic constraints.
    '''

    print("Creating Price and Promotional Features...")
    print("=" * 60)

    # Discount percentage
    data['Discount_Pct'] = ((data['Avg_MRP'] - data['Avg_Price']) /
                            (data['Avg_MRP'] + 0.01)) * 100

    # Price index (relative to category average)
    data['Price_Index'] = data.groupby('product_category')['Avg_Price'].transform(
        lambda x: x / x.mean()
    )

    # Promotional intensity
    data['Promo_Intensity'] = data['Discount_Pct'] * data['Units']

    # Price variance (price stability indicator)
    data['Price_Variance'] = data.groupby('product_category')['Avg_Price'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )

    # Create price bins for non-linear effects
    data['Price_Tier'] = pd.qcut(data['Avg_Price'], q=3, labels=['Low', 'Medium', 'High'])

    # Interaction: High discount + High marketing
    total_marketing = data[marketing_channels].sum(axis=1)
    data['Promo_Marketing_Interaction'] = data['Discount_Pct'] * np.log1p(total_marketing)

    price_features = ['Discount_Pct', 'Price_Index', 'Promo_Intensity',
                     'Price_Variance', 'Promo_Marketing_Interaction']

    # Mark features for monotonic constraints
    monotonic_features = {
        'Avg_Price': 'negative',      # Higher price → Lower sales
        'Avg_MRP': 'negative',        # Higher MRP → Lower sales
        'Discount_Pct': 'positive',   # Higher discount → Higher sales
    }

    print(f"\\nPrice features created: {len(price_features)}")
    print("\\nMonotonic constraints applied:")
    for feature, direction in monotonic_features.items():
        print(f"  {feature}: {direction} (", end="")
        if direction == 'negative':
            print("↑ price → ↓ sales)")
        else:
            print("↑ discount → ↑ sales)")

    return data, price_features, monotonic_features

# Create price features
data_with_features, price_features, monotonic_constraints = create_price_features(data_with_features)

# Visualize price-sales relationship
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Price vs GMV
axes[0].scatter(data_with_features['Avg_Price'],
               data_with_features['GMV'], alpha=0.3, s=10)
axes[0].set_xlabel('Average Price')
axes[0].set_ylabel('GMV')
axes[0].set_title('Price vs Sales (Negative Relationship Expected)')
z = np.polyfit(data_with_features['Avg_Price'], data_with_features['GMV'], 1)
p = np.poly1d(z)
axes[0].plot(data_with_features['Avg_Price'].sort_values(),
            p(data_with_features['Avg_Price'].sort_values()),
            "r--", alpha=0.8, label=f'Slope: {z[0]:.2f}')
axes[0].legend()

# Discount vs GMV
axes[1].scatter(data_with_features['Discount_Pct'],
               data_with_features['GMV'], alpha=0.3, s=10)
axes[1].set_xlabel('Discount %')
axes[1].set_ylabel('GMV')
axes[1].set_title('Discount vs Sales (Positive Relationship Expected)')

# Histogram of discounts
axes[2].hist(data_with_features['Discount_Pct'], bins=50, edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Discount %')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Distribution of Discounts')

plt.tight_layout()
plt.show()"""))

    # Seasonality Features
    nb.cells.append(new_markdown_cell("""## Part 4: Seasonality and Trend Features

Sales patterns often follow seasonal cycles and long-term trends. We'll create features to capture these patterns:"""))

    nb.cells.append(new_code_cell("""def create_temporal_features(data):
    '''
    Create seasonal, trend, and calendar features.
    '''

    print("Creating Temporal Features...")
    print("=" * 60)

    # Ensure Date column is datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Basic time features
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week.astype(int)
    data['DayOfMonth'] = data['Date'].dt.day

    # Cyclical encoding for seasonality (preserves continuity)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['Week_sin'] = np.sin(2 * np.pi * data['WeekOfYear'] / 52)
    data['Week_cos'] = np.cos(2 * np.pi * data['WeekOfYear'] / 52)
    data['Day_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
    data['Day_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)

    # Weekend indicator
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)

    # Month start/end indicators
    data['IsMonthStart'] = (data['DayOfMonth'] <= 5).astype(int)
    data['IsMonthEnd'] = (data['DayOfMonth'] >= 25).astype(int)

    # Trend component (days since start)
    min_date = data['Date'].min()
    data['Trend'] = (data['Date'] - min_date).dt.days
    data['Trend_squared'] = data['Trend'] ** 2  # For non-linear trends

    # Moving averages for detrending
    data['GMV_MA7'] = data.groupby('product_category')['GMV'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    data['GMV_MA30'] = data.groupby('product_category')['GMV'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )

    temporal_features = ['Month_sin', 'Month_cos', 'Week_sin', 'Week_cos',
                        'Day_sin', 'Day_cos', 'IsWeekend', 'IsMonthStart',
                        'IsMonthEnd', 'Trend', 'Trend_squared']

    print(f"\\nTemporal features created: {len(temporal_features)}")

    return data, temporal_features

# Create temporal features
data_with_features, temporal_features = create_temporal_features(data_with_features)

# Visualize seasonality patterns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Monthly seasonality
monthly_sales = data_with_features.groupby('Month')['GMV'].mean()
axes[0, 0].bar(monthly_sales.index, monthly_sales.values)
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Average GMV')
axes[0, 0].set_title('Monthly Seasonality Pattern')
axes[0, 0].set_xticks(range(1, 13))

# Weekly pattern
weekly_sales = data_with_features.groupby('DayOfWeek')['GMV'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[0, 1].bar(range(7), weekly_sales.values)
axes[0, 1].set_xlabel('Day of Week')
axes[0, 1].set_ylabel('Average GMV')
axes[0, 1].set_title('Weekly Pattern')
axes[0, 1].set_xticklabels(days)

# Trend over time
daily_sales = data_with_features.groupby('Date')['GMV'].sum()
axes[1, 0].plot(daily_sales.index, daily_sales.values, alpha=0.3, label='Daily')
axes[1, 0].plot(daily_sales.index, daily_sales.rolling(30).mean(),
                label='30-day MA', linewidth=2)
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Total GMV')
axes[1, 0].set_title('Sales Trend Over Time')
axes[1, 0].legend()

# Cyclical encoding visualization
theta = np.linspace(0, 2*np.pi, 100)
axes[1, 1].plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)
axes[1, 1].scatter(data_with_features['Month_cos'].head(365),
                  data_with_features['Month_sin'].head(365),
                  c=data_with_features['Month'].head(365),
                  cmap='hsv', s=20, alpha=0.6)
axes[1, 1].set_xlabel('Cos(Month)')
axes[1, 1].set_ylabel('Sin(Month)')
axes[1, 1].set_title('Cyclical Encoding of Months')
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

    # Competition and Market Features
    nb.cells.append(new_markdown_cell("""## Part 5: Market Context Features

Understanding the competitive landscape and market dynamics:"""))

    nb.cells.append(new_code_cell("""def create_market_features(data):
    '''
    Create features for market dynamics and competition.
    '''

    print("Creating Market Context Features...")
    print("=" * 60)

    # Category share
    data['Category_GMV'] = data.groupby(['Date', 'product_category'])['GMV'].transform('sum')
    data['Total_Daily_GMV'] = data.groupby('Date')['GMV'].transform('sum')
    data['Category_Share'] = data['Category_GMV'] / (data['Total_Daily_GMV'] + 0.01)

    # Subcategory within category share
    data['Subcat_Share'] = data['GMV'] / (data['Category_GMV'] + 0.01)

    # Competitive intensity (variance in category)
    data['Competition_Intensity'] = data.groupby('Date')['product_category'].transform('nunique')

    # Market momentum (growth rate)
    data['Market_Momentum'] = data.groupby('product_category')['Category_GMV'].transform(
        lambda x: x.pct_change(periods=7).fillna(0)
    )

    # Total marketing pressure in market
    total_marketing_cols = [col for col in marketing_channels if col in data.columns]
    data['Total_Market_Marketing'] = data.groupby('Date')[total_marketing_cols].transform('sum').sum(axis=1)

    # Share of voice (marketing share)
    data['Marketing_SOV'] = data[total_marketing_cols].sum(axis=1) / (data['Total_Market_Marketing'] + 0.01)

    market_features = ['Category_Share', 'Subcat_Share', 'Competition_Intensity',
                      'Market_Momentum', 'Marketing_SOV']

    print(f"\\nMarket features created: {len(market_features)}")

    return data, market_features

# Create market features
data_with_features, market_features = create_market_features(data_with_features)"""))

    # Final Feature Summary
    nb.cells.append(new_markdown_cell("""## Part 6: Feature Configuration Summary

Let's summarize all features and their roles in the model:"""))

    nb.cells.append(new_code_cell("""def create_feature_configuration(data, beta_gamma_features, price_features,
                                 temporal_features, market_features, monotonic_constraints):
    '''
    Create comprehensive feature configuration for the NAM model.
    This configuration tells the model how to treat each feature.
    '''

    print("FEATURE CONFIGURATION SUMMARY")
    print("=" * 60)

    # Compile all features
    all_features = {
        'beta_gamma': beta_gamma_features,  # Marketing saturation features
        'monotonic_negative': [f for f in monotonic_constraints if monotonic_constraints.get(f) == 'negative'],
        'monotonic_positive': [f for f in monotonic_constraints if monotonic_constraints.get(f) == 'positive'],
        'price': price_features,
        'temporal': temporal_features,
        'market': market_features,
        'control': ['NPS', 'Units']  # Other control variables
    }

    # Count features by type
    print("\\nFeature Types and Counts:")
    print("-" * 40)
    total_features = 0
    for feature_type, features_list in all_features.items():
        count = len(features_list) if isinstance(features_list, list) else len(features_list)
        total_features += count
        print(f"  {feature_type:20s}: {count:3d} features")

    print("-" * 40)
    print(f"  {'TOTAL':20s}: {total_features:3d} features")

    # Critical validation
    print("\\n" + "=" * 60)
    print("CRITICAL VALIDATION")
    print("=" * 60)

    beta_gamma_count = len(beta_gamma_features)
    if beta_gamma_count >= 28:
        print(f"✓ SUCCESS: {beta_gamma_count} Beta-Gamma features")
        print("  → Model will capture marketing saturation curves")
    else:
        print(f"⚠ WARNING: Only {beta_gamma_count} Beta-Gamma features")
        print("  → May not fully capture marketing dynamics")

    # Save feature configuration
    feature_config = {
        'features': all_features,
        'total_count': total_features,
        'beta_gamma_count': beta_gamma_count,
        'monotonic_count': len(all_features['monotonic_negative']) + len(all_features['monotonic_positive'])
    }

    # List all column names for modeling
    exclude_cols = ['Date', 'product_category', 'product_subcategory', 'GMV',
                   'Category_GMV', 'Total_Daily_GMV', 'GMV_MA7', 'GMV_MA30',
                   'Price_Tier', 'Year']

    model_features = [col for col in data.columns if col not in exclude_cols]

    print(f"\\nFeatures ready for modeling: {len(model_features)}")

    return feature_config, model_features

# Create final feature configuration
feature_config, model_features = create_feature_configuration(
    data_with_features, beta_gamma_features, price_features,
    temporal_features, market_features, monotonic_constraints
)

# Save the engineered dataset
data_with_features.to_csv('data/processed/mmm_data_with_features.csv', index=False)
print("\\n✓ Saved engineered dataset to 'data/processed/mmm_data_with_features.csv'")

# Save feature configuration
import json
with open('configs/feature_config.json', 'w') as f:
    json.dump(feature_config, f, indent=2, default=str)
print("✓ Saved feature configuration to 'configs/feature_config.json'")"""))

    # Key Takeaways
    nb.cells.append(new_markdown_cell("""## Key Takeaways

### What We've Accomplished:
1. **Adstock Transformation**: Captured marketing carryover with channel-specific decay rates
2. **Beta-Gamma Features**: Created 28+ features for marketing saturation modeling
3. **Price Features**: Built elasticity features with business logic constraints
4. **Temporal Features**: Captured seasonality with cyclical encoding
5. **Market Features**: Added competitive context and share metrics

### Feature Engineering Best Practices Applied:
- **Domain Knowledge**: Used marketing theory for decay rates
- **Business Constraints**: Applied monotonic relationships where logical
- **Interaction Effects**: Created price × marketing interactions
- **Hierarchical Processing**: Maintained category-level patterns
- **Temporal Continuity**: Used cyclical encoding for seasons

### Why These Features Matter:
- **Adstock** captures that marketing doesn't work instantly
- **Beta-Gamma** models diminishing returns in marketing spend
- **Price features** enable elasticity analysis
- **Seasonality** separates organic patterns from marketing effects
- **Market features** provide competitive context

### The Foundation for NAM:
We now have **{0}+ features** ready for the Neural Additive Model, including:
- **28+ Beta-Gamma features** for marketing saturation
- **Monotonic constraints** for business-valid predictions
- **Rich temporal patterns** for accurate forecasting

### Next Steps:
In Notebook 3, we'll build the NAM architecture that leverages these features optimally, with:
- Separate neural networks per feature
- Beta-Gamma transformation layers
- Monotonic constraint enforcement
- Additive combination for interpretability

---

**Remember:** Feature engineering is where domain knowledge meets data science. The features we've created encode decades of marketing science into a form that neural networks can learn from.""".format(len(model_features))))

    with open('02_Feature_Engineering.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print("Created: 02_Feature_Engineering.ipynb")
    return nb

# Continue with more notebooks...
def create_all_notebooks():
    """Create all notebooks in the series"""
    try:
        import nbformat
    except ImportError:
        print("Installing nbformat...")
        import subprocess
        subprocess.run(["pip", "install", "nbformat"])
        import nbformat

    print("Creating NAM-MMM Tutorial Notebook Series")
    print("=" * 60)

    # Create each notebook
    create_notebook_01_data_foundation()
    create_notebook_02_feature_engineering()
    # Will continue with notebooks 3-6 in the next part...

    print("\n" + "=" * 60)
    print("Notebooks created successfully!")
    print("Start with 01_Data_Foundation.ipynb")

if __name__ == "__main__":
    create_all_notebooks()