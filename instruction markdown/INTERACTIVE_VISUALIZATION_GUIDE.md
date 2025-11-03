# 🎨 Interactive NAM Visualization System

**NEW ADDITION - Built on top of existing system without modifying core code**

## 🎯 What's Been Added

### 1. Interactive Plotly Dashboards
**File:** `src/visualization/interactive_dashboard.py` (NEW)

**Features:**
- Interactive elasticity curves (zoom, pan, hover)
- NAM decomposition waterfall charts
- Time series with contribution stacking
- Product-level prediction analysis
- Feature contribution heatmaps

### 2. NAM Decomposition Analyzer
**Breaks down predictions into:**
- **Baseline Sales**: Inherent product demand
- **Price/Discount Effects**: Impact of pricing strategies
- **Marketing Investment**: TV, Digital, SEM contributions
- **Temporal Effects**: Seasonality, day-of-week patterns
- **Brand Health**: NPS, brand equity effects
- **Other Factors**: Special sales, external factors

### 3. Multi-Granularity Elasticity
**Shows elasticity curves at:**
- Aggregate level (all products)
- Product level (individual SKUs)
- Daily granularity
- Weekly aggregates
- Monthly trends

---

## 🚀 How to Generate Interactive Visualizations

### Quick Start:

```bash
# 1. Ensure Plotly is installed
pip install plotly kaleido

# 2. Run the main daily pipeline (if not already done)
python main_daily.py

# 3. Generate interactive dashboards
python -m src.visualization.interactive_dashboard
```

**Output:** 5 interactive HTML files in `outputs/figures/`

---

## 📊 Generated Interactive Visualizations

### 1. Elasticity Curves (Interactive)
**File:** `elasticity_interactive_aggregate.html`

**Features:**
- Zoom into specific price ranges
- Hover to see exact GMV contribution
- Click legend to show/hide features
- Interactive sliders for granularity

**Shows:**
- Current price point (red star)
- Optimal price point (gold diamond)
- Elasticity coefficient
- GMV curve shape

### 2. Decomposition Waterfall
**File:** `decomposition_waterfall.html`

**Shows:**
```
Total Prediction =
  Baseline
  + Price/Discount Effects
  + Marketing Investment
  + Temporal Effects
  + Brand Health
  + Other
  = Final GMV
```

**Interactive:**
- Hover to see exact contribution values
- Click categories to isolate
- Export as PNG for presentations

### 3. Time Series Decomposition
**File:** `decomposition_time_series.html`

**Shows:**
- Stacked area chart over time
- Each layer = one contribution category
- Actual vs predicted overlay
- Interactive date range selection

**Use Case:**
- "How much did TV contribute on May 15th?"
- "What was the discount impact in week 20?"
- "Show me baseline vs total sales"

### 4. Product Predictions Interactive
**File:** `product_predictions_interactive.html`

**Features:**
- Top panel: Actual vs predicted timeline
- Bottom panel: Error bars by date
- Hover for exact values
- Zoom into specific date ranges
- Pan across time

**Business Value:**
- Identify prediction accuracy by period
- See which days have high/low errors
- Understand seasonal patterns

### 5. Contribution Heatmap
**File:** `contribution_heatmap.html`

**Shows:**
- Rows: Features
- Columns: Dates
- Color: Contribution magnitude
  - Red: Positive contribution
  - Blue: Negative contribution
  - White: Neutral

**Use Case:**
- Which features matter most on specific dates?
- Temporal patterns in feature importance
- Identify key drivers by period

---

## 🔬 NAM Decomposition Explained

### How It Works:

NAM's additive structure allows us to decompose predictions:

```
y_pred = f₁(x₁) + f₂(x₂) + ... + f₉(x₉)
         ↓        ↓              ↓
      Price   Marketing    Temporal
```

**Each feature's contribution is independently extractable!**

### Business Categories:

**1. Baseline Sales**
- Product-specific GMV features
- Inherent demand
- **Use:** Understand base performance

**2. Price/Discount Effects**
- Price_CE, Discount features
- Elasticity impact
- **Use:** Optimize pricing strategy

**3. Marketing Investment**
- TV_adstock_log, Digital_adstock_log, SEM_adstock_log
- Investment response
- **Use:** Budget allocation, ROI optimization

**4. Temporal Effects**
- month_sin, month_cos, time_index
- Seasonality, trends
- **Use:** Forecast adjustments, planning

**5. Brand Health**
- NPS features
- Brand equity
- **Use:** Long-term strategy

**6. Special Factors**
- is_special_sale
- One-time events
- **Use:** Promotional planning

---

## 💡 Example Use Cases

### Use Case 1: Price Optimization
```python
# Open: elasticity_interactive_aggregate.html
# 1. Find Price_CE curve
# 2. See current price (red star)
# 3. Find optimal price (gold diamond)
# 4. Calculate revenue impact
```

**Business Decision:**
"If we increase price by X%, GMV changes by Y%"

### Use Case 2: Marketing ROI
```python
# Open: decomposition_waterfall.html
# 1. See marketing contribution
# 2. Compare to investment amount
# 3. Calculate ROI = contribution / investment
```

**Business Decision:**
"TV contributes $X to GMV for $Y investment = Z% ROI"

### Use Case 3: Trend Analysis
```python
# Open: decomposition_time_series.html
# 1. Zoom into May 2016
# 2. See stacked contributions
# 3. Identify which factor drove spike
```

**Business Decision:**
"The May spike was 60% due to discounts, 30% marketing, 10% baseline"

### Use Case 4: Product Performance
```python
# Open: product_predictions_interactive.html
# 1. Hover over specific dates
# 2. See actual vs predicted
# 3. Identify error patterns
```

**Business Decision:**
"Model underpredict on weekends - adjust forecasts accordingly"

### Use Case 5: Feature Importance Over Time
```python
# Open: contribution_heatmap.html
# 1. See which features matter when
# 2. Identify temporal patterns
# 3. Focus on high-impact features
```

**Business Decision:**
"Digital marketing most effective in early month, TV in late month"

---

## 🎨 Customization Options

### Change Granularity:

```python
from src.visualization.interactive_dashboard import InteractiveNAMVisualizer

viz = InteractiveNAMVisualizer()

# Aggregate level
viz.plot_elasticity_curves_interactive('aggregate')

# Daily level
viz.plot_elasticity_curves_interactive('daily')

# Weekly level
viz.plot_elasticity_curves_interactive('weekly')

# Monthly level
viz.plot_elasticity_curves_interactive('monthly')
```

### Custom Decomposition:

```python
# Get specific feature contributions
analyzer = NAMDecompositionAnalyzer(model, data, scalers, features)
contributions = analyzer.get_feature_contributions(X_test)

# Access specific feature
tv_contribution = contributions['TV_adstock_log']
print(f"TV contributed: {tv_contribution.mean():.2f} average")
```

---

## 📋 Installation & Setup

### Install Plotly:
```bash
pip install plotly kaleido
```

### Generate All Dashboards:
```bash
python -m src.visualization.interactive_dashboard
```

**Or import and use:**
```python
from src.visualization.interactive_dashboard import generate_all_interactive_visualizations

generate_all_interactive_visualizations()
```

---

## 🎯 What Makes This Special

### vs Static Matplotlib Charts:

| Feature | Matplotlib | Plotly |
|---------|------------|--------|
| **Zoom** | Limited | ✓ Unlimited |
| **Hover** | None | ✓ Detailed tooltips |
| **Export** | PNG only | ✓ PNG, SVG, HTML |
| **Sharing** | Image file | ✓ Interactive HTML |
| **Exploration** | Static | ✓ Dynamic filtering |
| **Business-Ready** | Basic | ✓ Professional |

### Key Advantages:

✅ **Interactive exploration** - Zoom, pan, hover for insights
✅ **Business presentations** - Professional interactive dashboards
✅ **Self-service analysis** - Stakeholders explore themselves
✅ **Export flexibility** - HTML, PNG, SVG formats
✅ **No code changes** - Built as add-on to existing system

---

## 🔍 Technical Details

### Module Structure:

```
src/visualization/
└── interactive_dashboard.py (NEW)
    ├── NAMDecompositionAnalyzer
    │   ├── get_feature_contributions()
    │   └── decompose_by_category()
    └── InteractiveNAMVisualizer
        ├── plot_decomposition_waterfall()
        ├── plot_time_series_decomposition()
        ├── plot_elasticity_curves_interactive()
        ├── plot_product_predictions()
        ├── plot_contribution_heatmap()
        └── create_elasticity_dashboard()
```

### Data Flow:

```
Trained NAM Model (outputs/elasticity_data.pkl)
    ↓
NAMDecompositionAnalyzer
    ↓
Extract feature contributions
    ↓
Categorize by business type
    ↓
InteractiveNAMVisualizer
    ↓
Generate Plotly HTML dashboards
    ↓
5 interactive HTML files
```

---

## ✨ Summary

**What You Now Have:**

✅ **5 Interactive HTML Dashboards**
- Elasticity curves with optimal points
- Decomposition waterfall
- Time series with stacked contributions
- Product predictions with error analysis
- Contribution heatmap

✅ **NAM Decomposition**
- Breaks predictions into 6 business categories
- Shows contribution of each factor
- Time-varying analysis

✅ **Multi-Granularity**
- Aggregate, product, daily, weekly, monthly views
- Flexible analysis levels

✅ **Production-Ready**
- Share HTML files with stakeholders
- Interactive exploration without coding
- Export charts for presentations

**All built as NEW modules - your existing system remains untouched!** ✓

---

## 🚀 Quick Command Reference

```bash
# Generate all interactive visualizations
python -m src.visualization.interactive_dashboard

# View in browser
start outputs/figures/*.html

# Or specific chart
start outputs/figures/elasticity_interactive_aggregate.html
```

**Your NAM system now has enterprise-grade interactive visualization capabilities!** 🎉
