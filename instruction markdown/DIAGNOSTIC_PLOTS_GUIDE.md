# NAM Model Diagnostic Plots Guide

## Overview
This guide explains the comprehensive diagnostic plots generated after training the Neural Additive Model for 200 epochs. These visualizations provide deep insights into model performance, marketing effectiveness, and pricing dynamics.

---

## 1. Training Diagnostics (200 Epochs)

### Plot: `training_diagnostics_200epochs.png`

**Contains 4 subplots:**

#### a) Training vs Validation Loss
- **Purpose:** Monitor learning progression and convergence
- **What to Look For:**
  - Both curves should decrease over time
  - Gap between curves indicates overfitting if too large
  - Convergence point shows optimal training duration

#### b) Mean Absolute Error Over Time
- **Purpose:** Track prediction accuracy improvement
- **Interpretation:**
  - Lower MAE = better predictions
  - Plateau indicates model has learned maximum from data

#### c) Overfitting Detection (Loss Ratio)
- **Purpose:** Identify when model starts memorizing rather than learning
- **Key Insight:**
  - Ratio > 1 indicates overfitting
  - Ratio ≈ 1 indicates good generalization
  - Ratio < 1 may indicate underfitting

#### d) Learning Progress (Gradient)
- **Purpose:** Visualize rate of improvement
- **Interpretation:**
  - Negative values = model improving
  - Near zero = learning has plateaued
  - Positive = model getting worse (rare)

---

## 2. Error Metrics Analysis

### Plots: `error_metrics_test.png` & `error_metrics_validation.png`

**Key Metrics Displayed:**

#### a) MAPE (Mean Absolute Percentage Error)
- **Target:** < 15%
- **Business Meaning:** Average prediction error as percentage of actual value
- **Example:** MAPE of 20% means predictions are off by 20% on average

#### b) SMAPE (Symmetric MAPE)
- **Advantage:** Treats over and under-predictions equally
- **Range:** 0-100%
- **Better for:** Comparing models with different scales

#### c) Actual vs Predicted Scatter
- **Perfect Model:** All points on diagonal line
- **Interpretation:**
  - Points above line = model underestimates
  - Points below line = model overestimates
  - Spread indicates uncertainty

#### d) Residual Plots
- **Purpose:** Identify systematic errors
- **Good Pattern:** Random scatter around zero
- **Problems to Spot:**
  - Funnel shape = heteroscedasticity
  - Curves = non-linear relationships missed

#### e) Q-Q Plot
- **Purpose:** Check if residuals are normally distributed
- **Ideal:** Points follow straight diagonal line
- **Deviations:** Indicate non-normal errors

---

## 3. Price Elasticity Curves

### Plot: `price_elasticity_curves.png`

**What It Shows:**
- How sales (GMV) respond to price changes
- Elasticity coefficient at different price points

**Key Insights:**

#### Elasticity Interpretation:
- **Elasticity = -1:** 1% price increase → 1% sales decrease (unit elastic)
- **Elasticity < -1:** Highly elastic (price sensitive)
- **Elasticity > -1:** Inelastic (less price sensitive)

#### Business Applications:
1. **Optimal Pricing:** Find price point maximizing revenue
2. **Promotion Planning:** Identify products responsive to discounts
3. **Category Strategy:** Compare price sensitivity across categories

**Example Reading:**
- If elasticity = -2.5 at current price:
  - 10% price increase → 25% sales decrease
  - Revenue impact: 1.1 × 0.75 = 0.825 (17.5% revenue loss)

---

## 4. Marketing Investment Saturation Curves

### Plot: `marketing_saturation_curves.png`

**What It Shows:**
- Diminishing returns for each marketing channel
- Optimal investment levels before saturation

**Components per Channel:**

#### a) Incremental GMV Curve (Blue)
- **Shape:** Should be concave (diminishing returns)
- **Interpretation:**
  - Steep initially = high effectiveness at low spend
  - Flattening = approaching saturation
  - Plateau = maximum achievable impact

#### b) Marginal ROI Curve (Red)
- **Definition:** Return per additional dollar spent
- **Key Threshold:** ROI = 1.0 (breakeven)
- **Decision Rule:**
  - ROI > 1: Increase spending
  - ROI < 1: Decrease or reallocate
  - ROI = 1: Optimal point

#### c) Saturation Point (Green Line)
- **Definition:** Investment level where ROI drops below 1
- **Business Insight:** Maximum efficient spend level

**Channel Comparison:**
- **High Saturation Point:** Can absorb more investment
- **Low Saturation Point:** Limited capacity, reallocate excess
- **Steep Curve:** High impact channel
- **Gradual Curve:** Steady but modest returns

---

## 5. Feature Importance Analysis

### Plot: `feature_importance.png`

**What It Shows:**
- Top 20 most influential features for predictions
- Variance contribution of each feature

**Business Applications:**

#### Marketing Insights:
- Which channels drive most sales
- Relative importance of different campaigns
- ATL vs BTL effectiveness

#### Pricing Insights:
- Price vs discount importance
- Competitive pricing impact

#### Other Factors:
- Seasonality importance
- Brand health (NPS) impact
- Category/subcategory effects

---

## How to Use These Plots for Decision Making

### 1. Budget Allocation
Use saturation curves to:
- Identify channels below saturation (increase spend)
- Find oversaturated channels (decrease spend)
- Calculate optimal budget distribution

### 2. Pricing Strategy
Use elasticity curves to:
- Set optimal regular prices
- Plan discount depths
- Predict revenue impact of price changes

### 3. Model Validation
Use error metrics to:
- Confirm model accuracy (MAPE < 15%)
- Identify prediction biases
- Validate business assumptions

### 4. Performance Monitoring
Use training diagnostics to:
- Ensure model convergence
- Detect overfitting early
- Optimize training duration

---

## Success Metrics Interpretation

### Current Performance (After 200 Epochs)

| Metric | Current | Target | Status | Business Impact |
|--------|---------|--------|--------|-----------------|
| **R²** | ~0.70 | 0.85 | Approaching | Model explains 70% of sales variation |
| **MAPE** | ~20% | <15% | Needs Work | Predictions ±20% accurate on average |
| **Val Loss** | ~0.005 | <0.01 | Good | Model generalizes well |

### What the Numbers Mean for Business

**R² = 0.70:**
- Model captures most important factors
- 30% variation due to unmeasured factors
- Sufficient for strategic decisions
- May need enhancement for tactical execution

**MAPE = 20%:**
- Budget planning: Add 20% buffer
- ROI calculations: Consider ±20% uncertainty
- Scenario planning: Run sensitivity analysis

---

## Recommendations Based on Diagnostics

### If Saturation Curves Show:
- **Quick saturation:** Diversify marketing mix
- **No saturation:** Test higher spend levels
- **Negative returns:** Reduce or eliminate channel

### If Elasticity Curves Show:
- **High elasticity:** Focus on value proposition
- **Low elasticity:** Opportunity for price increases
- **Non-linear:** Implement dynamic pricing

### If Error Patterns Show:
- **Systematic bias:** Add missing features
- **High variance:** Increase regularization
- **Temporal patterns:** Implement time-series components

---

## Next Steps

1. **Review all plots** to understand model behavior
2. **Compare channels** using saturation analysis
3. **Validate elasticities** against business knowledge
4. **Plan experiments** to test model predictions
5. **Monitor performance** in production deployment

---

*These diagnostic plots transform the "black box" NAM model into transparent, actionable business insights.*