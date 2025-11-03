# Neural Additive Model for Marketing Mix Modeling

Complete implementation of Hierarchically Regularized Constrained Neural Additive Models (HR-NAM) for quantifying marketing effects on sales revenue using **agent-based parallel development**.

## 📖 Documentation

**Main Implementation Guide:**
👉 **[NAM_Comprehensive_Implementation_Plan.md](./NAM_Comprehensive_Implementation_Plan.md)**
- Complete technical specifications and implementations

**Agent-Based Development:**
👉 **[Agent_Based_Development_Proposal.md](./Agent_Based_Development_Proposal.md)**
- 6-agent parallel development architecture
- Complete setup and orchestration guide

**Supporting Documentation:**
- [data-details.md](./data-details.md) - Dataset documentation
- [github.md](./github.md) - Git workflow reference
- [WFO.md](./WFO.md) - Walk-Forward Optimization methodology

---

## 🎯 Project Objective

Build a production-ready Neural Additive Model to:
- ✅ Predict monthly GMV/Revenue (R² > 0.85)
- ✅ Estimate economically valid elasticities (price, marketing, brand)
- ✅ Enable business decisions (budget optimization, ROI forecasting)
- ✅ Maintain interpretability with transparent feature contributions

---

## 🚀 Quick Start with uv

### Prerequisites
- Python 3.9+
- Git

### 1. Install uv (Ultra-fast Package Manager)

```bash
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### 2. Clone Repository (if applicable)

```bash
git clone <repository-url>
cd Neural-Additive_Model
```

### 3. Generate Lock Files (First Time Setup)

```bash
# Generate lock files for reproducibility
python scripts/generate_lock_files.py

# This creates .lock files for all requirements
```

### 4. Setup All Agent Environments (Parallel - Very Fast!)

```bash
# Setup all 6 agent environments in parallel using uv
python scripts/setup_agent_environment.py --all

# This creates:
# - .venv_agent_01 (Data Engineer)
# - .venv_agent_02 (Model Architect)
# - .venv_agent_03 (Training Specialist)
# - .venv_agent_04 (Evaluation Engineer)
# - .venv_agent_05 (Business Tools)
# - .venv_agent_06 (Test Automation)
```

### 5. Verify Keras 3 Installation

```bash
# Activate any agent environment
# Windows:
.venv_agent_02\Scripts\activate
# Linux/macOS:
# source .venv_agent_02/bin/activate

# Check Keras 3 and backend
python -c "import keras; print(f'Keras: {keras.__version__}, Backend: {keras.backend.backend()}')"

# Expected output: Keras: 3.x.x, Backend: jax
```

### 6. Place Data Files

Place your CSV files in `data/raw/`:
- Sales.csv
- firstfile.csv
- Secondfile.csv
- MediaInvestment.csv
- MonthlyNPSscore.csv
- ProductList.csv
- SpecialSale.csv

---

## 📊 Agent-Based Development Workflow

### Agent Structure

```
A0 - Orchestrator      → Coordinates all agents
A1 - Data Engineer     → Data pipeline & features
A2 - Model Architect   → Keras 3 NAM architecture
A3 - Training Specialist → Training & walk-forward
A4 - Evaluation Engineer → Metrics & visualization
A5 - Business Tools    → Budget optimizer, ROI sim
A6 - Test Automation   → Continuous testing
```

### Development Phases

**Week 1**: Setup & Foundation (A1, A2, A6 in parallel)
**Week 2**: Data Processing & Model Implementation (A1, A2, A3, A6 in parallel)
**Week 3**: Training & Hierarchical NAM (A2, A3, A4, A6 in parallel)
**Week 4**: Walk-Forward & Evaluation (A3, A4, A5, A6 in parallel)
**Week 5**: Business Tools & Reporting (A4, A5, A6 in parallel)
**Week 6-7**: Finalization & Validation

### Agent Configurations

Each agent has:
- Configuration: `.agents/agent_XX_<name>/config.yaml`
- Task list: `.agents/agent_XX_<name>/tasks.md`
- Progress tracking: `.agents/agent_XX_<name>/progress.json`
- Dedicated virtual environment: `.venv_agent_XX`

---

## 📦 Project Structure

```
NAM_Project/
├── .agents/           # Agent configurations
├── data/              # Data pipeline (Agent 1)
├── src/
│   ├── data/          # A1: Data processing
│   ├── models/        # A2: Keras 3 models
│   ├── training/      # A3: Training loops
│   ├── evaluation/    # A4: Metrics & viz
│   ├── optimization/  # A5: Business tools
│   └── utils/         # Shared utilities
├── tests/             # A6: All tests
├── configs/           # YAML configurations
├── scripts/           # Setup & orchestration
├── requirements/      # uv requirements & lock files
└── outputs/           # Models, figures, reports
```

---

## 🛠️ Key Features

- ✅ **Keras 3** with JAX backend for fast computation
- ✅ **uv package manager** for 10-100x faster installs
- ✅ **Constrained neural networks** (monotonicity, concavity)
- ✅ **Hierarchical brand-SKU structure**
- ✅ **Walk-Forward Optimization** for time-series validation
- ✅ **Multiple training strategies** (joint, coordinate descent)
- ✅ **Business optimization tools** (budget optimizer, scenario planner)
- ✅ **Complete testing framework** (95%+ coverage target)

---

## 📝 Development Commands

### uv Commands

```bash
# Install dependencies (fast!)
uv pip install -r requirements/base.txt

# Install from lock file (exact versions)
uv pip sync requirements/base.lock

# Add new package
uv pip install <package-name>

# Update lock file
uv pip compile requirements/base.txt -o requirements/base.lock
```

### Agent Management

```bash
# Setup specific agent environment
python scripts/setup_agent_environment.py --agent data --id 01

# Setup all agents in parallel
python scripts/setup_agent_environment.py --all
```

### Testing

```bash
# Run all tests
pytest

# Run specific agent tests
pytest tests/unit/test_models.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run parallel tests for all agents
python scripts/parallel_test.py
```

---

## 📈 Success Metrics

- [ ] All 6 agents complete tasks within 6-7 weeks
- [ ] 95%+ test coverage across all modules
- [ ] Walk-Forward validation R² > 0.75
- [ ] All constraints satisfied (100% monotonicity compliance)
- [ ] Business tools functional and validated
- [ ] Ready for deployment decision

---

## 🔧 Troubleshooting

### uv not found
```bash
# Install uv
# Windows:
irm https://astral.sh/uv/install.ps1 | iex
# Linux/macOS:
# curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Keras backend issues
```bash
# Set JAX backend explicitly
export KERAS_BACKEND=jax  # Linux/macOS
# Or in Python:
# import os
# os.environ['KERAS_BACKEND'] = 'jax'
```

### Import errors
```bash
# Make sure you're in the right environment
# Windows:
.venv_agent_XX\Scripts\activate
# Linux/macOS:
# source .venv_agent_XX/bin/activate

# Reinstall dependencies
uv pip sync requirements/agent_XX_<name>.lock
```

---

## 📚 Resources

### Package Manager
- **uv Documentation**: https://docs.astral.sh/uv/

### ML Framework
- **Keras 3 Documentation**: https://keras.io/api/
- **JAX Documentation**: https://jax.readthedocs.io/

### Project Documentation
- See `docs/` folder for detailed guides
- See `.agents/` folder for agent-specific configurations

---

## 📝 Project Status

**Version**: 0.1.0 (Initial Setup)
**Status**: Ready for Development
**Last Updated**: 2025-10-29
**Framework**: Keras 3 with JAX backend
**Package Manager**: uv

---

## 👥 Agent Contact Points

- **Agent 1 (Data)**: Data processing questions → `.agents/agent_01_data/`
- **Agent 2 (Model)**: Architecture questions → `.agents/agent_02_model/`
- **Agent 3 (Training)**: Training questions → `.agents/agent_03_training/`
- **Agent 4 (Evaluation)**: Metrics/viz questions → `.agents/agent_04_evaluation/`
- **Agent 5 (Business)**: Optimization questions → `.agents/agent_05_business/`
- **Agent 6 (Testing)**: Test questions → `.agents/agent_06_testing/`

---

For complete implementation details, refer to **NAM_Comprehensive_Implementation_Plan.md**
