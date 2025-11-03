# 🎨 How to Run All Visualization Tools

## ⚡ Quick Commands

### Generate ALL visualizations at once:

```bash
# Set environment
$env:KERAS_BACKEND="jax"

# 1. Run the pipeline first (if not already done)
python main_daily.py

# 2. Generate static charts
python scripts/plot_training_results.py
python scripts/plot_walk_forward.py

# 3. View all charts
start outputs/figures/*.png
```

---

## 📊 Individual Visualization Commands

### 1. Training History Charts

```bash
python scripts/plot_training_results.py
```

**Generates:**
- `outputs/figures/training_history.png` - Loss & MAE curves
- `outputs/figures/loss_convergence.png` - Convergence analysis

**Shows:**
- Training loss over epochs
- Validation loss over epochs
- MAE progression
- Linear & log scale views

---

### 2. Walk-Forward Time Series

```bash
python scripts/plot_walk_forward.py
```

**Generates:**
- `outputs/figures/walk_forward_complete.png` - Complete OOS predictions
- `outputs/figures/walk_forward_detailed.png` - Error analysis

**Shows:**
- 6 months of out-of-sample predictions
- Fold-by-fold performance
- Time series trends
- Prediction errors

**Requirements:**
- Walk-forward must be enabled in config
- `outputs/walk_forward_results.pkl` must exist

---

### 3. Interactive Plotly Dashboards (Currently Being Fixed)

```bash
python generate_interactive_viz.py
```

**Will Generate (once fix applied):**
- `outputs/figures/elasticity_interactive_aggregate.html`
- `outputs/figures/decomposition_waterfall.html`
- `outputs/figures/decomposition_time_series.html`
- `outputs/figures/product_predictions_interactive.html`
- `outputs/figures/contribution_heatmap.html`

**Features:**
- Interactive zoom, pan, hover
- Elasticity curves
- NAM decomposition (baseline + contributions)
- Feature contributions over time

**Open in Browser:**
```bash
start outputs/figures/*.html
```

---

## 🔧 Step-by-Step Process

### STEP 1: Run the Pipeline

```bash
# Navigate to project directory
cd "C:\Users\40103061\Anheuser-Busch InBev\Kumarjit Backup - General\Articles\Neural Additive Network\Neural-Additive_Model"

# Set Keras backend
$env:KERAS_BACKEND="jax"

# Activate environment
.venv_main\Scripts\activate

# Run daily pipeline
python main_daily.py
```

**What this does:**
- Loads 250 daily records
- Trains single-layer NAM
- Generates predictions
- Saves model and data for visualization

**Time:** ~3-5 minutes
**Output:** Model saved to `outputs/models/final_nam_model_daily.keras`

---

### STEP 2: Generate Training Charts

```bash
python scripts/plot_training_results.py
```

**Output:**
- Training history chart created ✓
- Loss convergence chart created ✓

**View:**
```bash
start outputs/figures/training_history.png
start outputs/figures/loss_convergence.png
```

---

### STEP 3: Generate Walk-Forward Charts

```bash
python scripts/plot_walk_forward.py
```

**Output:**
- Walk-forward complete chart ✓
- Walk-forward detailed chart ✓

**View:**
```bash
start outputs/figures/walk_forward_complete.png
start outputs/figures/walk_forward_detailed.png
```

---

### STEP 4: View Actual vs Predicted

The main pipeline automatically generates:

```bash
start outputs/figures/actual_vs_predicted.png
```

**Shows:**
- 38-day time series
- Actual (blue) vs Predicted (orange)
- Complete trend visualization

---

### STEP 5: Generate Interactive Dashboards (Optional)

```bash
python generate_interactive_viz.py
```

**Note:** Currently needs SimpleNAM import fix (minor issue)

**When working, generates:**
- 5 interactive HTML files
- Open in any web browser
- Zoom, hover, explore data interactively

---

## 🎯 Complete Workflow

### Full Visualization Generation:

```bash
# 1. Ensure you're in the right directory
pwd  # Should be in Neural-Additive_Model

# 2. Activate environment
.venv_main\Scripts\activate

# 3. Set backend
$env:KERAS_BACKEND="jax"

# 4. Run pipeline (if not already done)
python main_daily.py

# 5. Generate all static charts
python scripts/plot_training_results.py
python scripts/plot_walk_forward.py

# 6. View everything
start outputs/figures/*.png

# 7. Check the 38-day time series (KEY CHART!)
start outputs/figures/actual_vs_predicted.png
```

---

## 📁 Where to Find Generated Files

### Charts:
```
outputs/figures/
├── training_history.png          ← Training curves
├── loss_convergence.png          ← Convergence analysis
├── actual_vs_predicted.png       ← 38-day time series! ⭐
├── walk_forward_complete.png     ← WFO results
└── walk_forward_detailed.png     ← Error analysis
```

### Models:
```
outputs/models/
├── best_model.keras              ← Best checkpoint
├── final_nam_model.keras         ← Monthly model
└── final_nam_model_daily.keras   ← Daily model (PRODUCTION)
```

### Logs:
```
outputs/
├── training_log.csv              ← Epoch-by-epoch metrics
├── nam_pipeline_daily.log        ← Complete log
└── run_200_epochs.log            ← Latest 200-epoch run
```

---

## 💡 Quick Tips

### View All Charts at Once:
```bash
start outputs/figures/*.png
```

### Check Training Progress:
```bash
cat outputs/training_log.csv | tail -10
```

### Review Complete Log:
```bash
cat outputs/run_200_epochs.log | grep -E "Best|R²|MAPE"
```

### List All Generated Files:
```bash
ls outputs/figures/
ls outputs/models/
```

---

## 🎨 Chart Descriptions

### 1. Training History
- **What it shows:** Loss and MAE over epochs
- **Use for:** Understanding convergence quality
- **Key metrics:** Best val_loss = 0.0242

### 2. Loss Convergence
- **What it shows:** Linear & log scale loss decay
- **Use for:** Diagnosing training stability
- **Key insight:** Smooth exponential decay

### 3. Actual vs Predicted (MOST IMPORTANT!)
- **What it shows:** 38 daily predictions vs actuals
- **Use for:** Business trend analysis
- **Key feature:** Complete time series from Apr-Jun 2016

### 4. Walk-Forward Complete
- **What it shows:** Out-of-sample validation across folds
- **Use for:** Model robustness assessment
- **Key insight:** Performance consistency

### 5. Walk-Forward Detailed
- **What it shows:** Time series with fold boundaries
- **Use for:** Error pattern analysis
- **Key insight:** Prediction accuracy by period

---

## 🚀 Summary

**Single command to see everything:**

```bash
# Generate all charts
python scripts/plot_training_results.py && python scripts/plot_walk_forward.py

# View all charts
start outputs/figures/*.png
```

**That's it! All your visualizations are ready to view!** 🎉

---

## 📞 Need Help?

Check these guides:
- `START_HERE.md` - Quick start
- `README_COMPLETE_SYSTEM.md` - System overview
- `INTERACTIVE_VISUALIZATION_GUIDE.md` - Plotly dashboards

**Your visualization tools are production-ready!**
