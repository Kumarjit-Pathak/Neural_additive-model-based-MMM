# 🤖 Multi-Agent System - Complete Run Guide

Complete guide to running the NAM multi-agent system with live progress tracking.

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Quick Start](#quick-start)
3. [Step-by-Step Execution](#step-by-step-execution)
4. [Monitoring Agent Progress](#monitoring-agent-progress)
5. [Understanding Agent Roles](#understanding-agent-roles)
6. [Advanced Options](#advanced-options)

---

## 🎯 System Overview

**6 Specialized Agents:**
```
Agent 1 (Data Engineer)      → Load, clean, engineer features
Agent 2 (Model Architect)    → Build NAM architecture
Agent 3 (Training Specialist) → Train model + walk-forward validation
Agent 4 (Evaluation Engineer) → Evaluate performance, visualize
Agent 5 (Business Analyst)    → ROI, budget optimization tools
Agent 6 (Test Automation)     → Run tests, ensure quality
```

**Current Architecture:**
- **Single-layer NAM**: 52 features × [16 hidden] × 1 output = 2,548 params
- **Explainability**: High (coefficient-level interpretation possible)
- **Training**: 10 epochs (test mode) or 300 epochs (full mode)

---

## 🚀 Quick Start

### Option 1: Simple Run (No Live Tracking)

```powershell
# Set Keras backend
$env:KERAS_BACKEND="jax"

# Activate environment
.venv_main\Scripts\activate

# Run complete pipeline
python main.py
```

**Result**: All agents execute sequentially, no progress tracking.

---

### Option 2: Run with Live Agent Monitoring (Recommended)

**Terminal 1 - Run Pipeline:**
```powershell
# Set backend
$env:KERAS_BACKEND="jax"

# Activate environment
.venv_main\Scripts\activate

# Run pipeline with enhanced tracking
python main_with_tracking.py
```

**Terminal 2 - Monitor Progress:**
```powershell
# Watch agent status in real-time
python scripts/monitor_agents.py
```

---

## 📊 Step-by-Step Execution

### Step 1: Configure Training

Edit `configs/training_config.yaml`:

```yaml
training:
  max_epochs: 100  # 10 for quick test, 300 for full training

walk_forward:
  enabled: true   # true for robustness testing, false for quick run
```

### Step 2: Check Initial Status

```powershell
python scripts/check_agent_status.py
```

**Expected Output:**
```
======================================================================
NAM Project - Agent Status Report
======================================================================

Agent 01: DATA
  [------------------------------] 0% (0/0)

Agent 02: MODEL
  [------------------------------] 0% (0/0)

... (all agents at 0%)
======================================================================
```

### Step 3: Run Pipeline

```powershell
$env:KERAS_BACKEND="jax"
python main.py
```

**What Happens:**
```
[1/8] Loading configurations...       ← System setup
[2/8] Agent 1: Data processing...     ← Agent 1 active
[3/8] Agent 2: Building model...      ← Agent 2 active
[4/8] Agent 3: Training...            ← Agent 3 active
[5/8] Agent 3: Walk-forward...        ← Agent 3 continued
[6/8] Agent 4: Evaluation...          ← Agent 4 active
[7/8] Agent 5: Business tools...      ← Agent 5 active
[8/8] Agent 6: Testing...             ← Agent 6 info
```

### Step 4: Monitor Progress (During Execution)

**While pipeline runs, in another terminal:**

```powershell
# Check status every 5 seconds
while ($true) {
    cls
    python scripts/check_agent_status.py
    Start-Sleep -Seconds 5
}
```

### Step 5: Review Results

After completion:

```powershell
# View final metrics
cat outputs/nam_pipeline.log | Select-String "R²|MAPE|Complete"

# View visualizations
start outputs/figures/actual_vs_predicted.png

# Check model
ls outputs/models/
```

---

## 🔍 Monitoring Agent Progress

### Live Status Monitoring

Create `scripts/monitor_agents.py`:

```python
#!/usr/bin/env python3
"""Live agent monitoring"""
import time
import os
from pathlib import Path
import json

def monitor():
    """Monitor agents in real-time"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')

        print("="*70)
        print("NAM Multi-Agent System - LIVE STATUS")
        print("="*70)
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print()

        # Check if pipeline is running
        if Path('outputs/nam_pipeline.log').exists():
            with open('outputs/nam_pipeline.log', 'r') as f:
                lines = f.readlines()

            # Extract current stage
            for line in reversed(lines[-50:]):
                if '[' in line and '/' in line:
                    print(f"Current Stage: {line.split('|')[-1].strip()}")
                    break

        # Show agent status
        os.system('python scripts/check_agent_status.py')

        print("\nPress Ctrl+C to stop monitoring")
        time.sleep(5)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
```

**Usage:**
```powershell
python scripts/monitor_agents.py
```

---

## 👥 Understanding Agent Roles

### Agent 1: Data Engineer
**Responsibility**: Data preparation
**Tasks**:
- Load data from `data/raw/`
- Handle missing values (forward fill, interpolation)
- Detect and treat outliers (IQR method)
- Engineer features (adstock, temporal, interactions)
- Scale features (StandardScaler)
- Split data (train/val/test)

**Output**: Preprocessed data ready for modeling

**Progress Indicators**:
```
✓ Data loaded (12 records)
✓ Missing values handled
✓ Outliers treated
✓ Features engineered (50 columns)
✓ Data scaled
✓ Train/val/test split complete
```

---

### Agent 2: Model Architect
**Responsibility**: Build NAM architecture
**Tasks**:
- Create SimpleNAM with single-layer networks
- Configure feature types (unconstrained, monotonic, parametric)
- Build model with correct input dimensions
- Verify architecture

**Output**: Compiled Keras model

**Progress Indicators**:
```
✓ Model initialized (52 features)
✓ Feature networks created
✓ Model compiled
✓ Architecture verified
```

---

### Agent 3: Training Specialist
**Responsibility**: Model training & validation
**Tasks**:
- Train model with callbacks (early stopping, checkpointing)
- Monitor training metrics (loss, MAE, MSE)
- Run walk-forward validation (if enabled)
- Save best model

**Output**: Trained model + validation metrics

**Progress Indicators**:
```
✓ Training started
  Epoch 1/100 - loss: xxx
  Epoch 2/100 - loss: xxx
  ...
✓ Training complete
✓ Walk-forward validation started
  Fold 1/6 complete
  Fold 2/6 complete
  ...
✓ Validation complete
```

---

### Agent 4: Evaluation Engineer
**Responsibility**: Performance assessment
**Tasks**:
- Calculate metrics (R², MAPE, RMSE)
- Generate visualizations
- Analyze residuals
- Create diagnostic plots

**Output**: Metrics + visualizations

**Progress Indicators**:
```
✓ Test predictions generated
✓ Metrics calculated (R²: 0.xx, MAPE: xx%)
✓ Actual vs Predicted plot saved
✓ Residual diagnostics saved
```

---

### Agent 5: Business Analyst
**Responsibility**: Business intelligence tools
**Tasks**:
- Budget optimization (available as module)
- ROI simulation (available as module)
- Elasticity analysis (available as module)
- Scenario planning (available as module)

**Output**: Ready-to-use business tools

**Progress Indicators**:
```
✓ Budget optimizer ready
✓ ROI simulator ready
✓ Elasticity analyzer ready
✓ Scenario planner ready
```

---

### Agent 6: Test Automation
**Responsibility**: Quality assurance
**Tasks**:
- Unit tests for all modules
- Integration tests for pipeline
- Verify model constraints
- Check output quality

**Output**: Test results

**Progress Indicators**:
```
Run: pytest
  tests/unit/test_models.py ............. PASSED
  tests/unit/test_data.py ............... PASSED
  tests/integration/test_pipeline.py ... PASSED
```

---

## ⚙️ Advanced Options

### Full Production Run

```yaml
# configs/training_config.yaml
training:
  max_epochs: 300
  early_stopping:
    patience: 15

walk_forward:
  enabled: true
  initial_train_size: 6

hyperparameter_tuning:
  enabled: false  # Enable for optimal params
```

```powershell
$env:KERAS_BACKEND="jax"
python main.py
```

**Expected Time**: 30-60 minutes

---

### Parallel Monitoring Setup

**Terminal 1 - Pipeline:**
```powershell
$env:KERAS_BACKEND="jax"
python main.py 2>&1 | Tee-Object -FilePath "run_log.txt"
```

**Terminal 2 - Live Status:**
```powershell
python scripts/monitor_agents.py
```

**Terminal 3 - Log Watching:**
```powershell
Get-Content outputs/nam_pipeline.log -Wait -Tail 20
```

---

### Hyperparameter Tuning

Enable Optuna-based hyperparameter search:

```yaml
# configs/training_config.yaml
hyperparameter_tuning:
  enabled: true
  n_trials: 50
  timeout: 7200  # 2 hours
```

**Run:**
```powershell
python scripts/hyperparameter_tuning.py
```

---

## 📁 Output Files & Agent Artifacts

After completion, check:

```
outputs/
├── models/
│   ├── final_nam_model.keras       ← Agent 3 output
│   └── best_model.keras             ← Agent 3 checkpoint
├── figures/
│   ├── actual_vs_predicted.png      ← Agent 4 output
│   ├── walk_forward_results.png     ← Agent 3/4 output
│   └── residual_diagnostics.png     ← Agent 4 output
├── nam_pipeline.log                 ← All agents log
└── training_log.csv                 ← Agent 3 training history

.agents/
├── agent_01_data/
│   ├── progress.json                ← Agent 1 status (if tracked)
│   └── tasks.md                     ← Agent 1 task list
├── agent_02_model/
│   ├── progress.json                ← Agent 2 status
│   └── tasks.md
... (all 6 agents)
```

---

## 🔧 Troubleshooting

### Agent Status Shows 0%

**Cause**: Current implementation doesn't write progress.json files

**Solution**: The pipeline still runs correctly. To see progress:
1. Monitor `outputs/nam_pipeline.log`
2. Watch console output
3. Or implement progress.json writing in main.py

### Pipeline Hangs at Training

**Cause**: Long training epochs

**Solution**:
```yaml
# Reduce epochs for testing
max_epochs: 10
```

### Memory Issues

**Cause**: Large batch size or model

**Solution**:
```yaml
batch_size: 16  # Reduce from 32
```

---

## 🎯 Success Criteria Checklist

After running, verify:

- [ ] Pipeline completed without errors (exit code 0)
- [ ] All 8 stages executed
- [ ] Model saved to `outputs/models/`
- [ ] Visualizations created in `outputs/figures/`
- [ ] Log file complete in `outputs/nam_pipeline.log`
- [ ] Test metrics reasonable (R² > 0.5 for good performance)
- [ ] Walk-forward validation stable (if enabled)

---

## 📞 Quick Reference Commands

```powershell
# Run pipeline
$env:KERAS_BACKEND="jax"; python main.py

# Check status
python scripts/check_agent_status.py

# View logs
Get-Content outputs/nam_pipeline.log -Tail 50

# Run tests
pytest

# View results
start outputs/figures/actual_vs_predicted.png
```

---

**Ready to run the complete multi-agent system!** 🚀
