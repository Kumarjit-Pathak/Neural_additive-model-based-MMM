# 🤖 Agent-Based Development Proposal for NAM Project
## Using Claude Code SDK with Keras 3 for Parallel Development & Testing

## 📋 Executive Summary

This proposal outlines an **agent-based development architecture** using Claude Code SDK to build the Neural Additive Model (NAM) for Marketing Mix Modeling with **Keras 3**. The approach enables:

- ✅ **Parallel development** of independent components by specialized agents
- ✅ **Continuous testing** with dedicated test agents
- ✅ **Faster iteration** through concurrent workstreams
- ✅ **Quality assurance** with automated code review
- ✅ **Focus on model quality** before deployment decisions

**Timeline**: 6-7 weeks with parallel agent execution (deployment excluded)
**Team**: 6 specialized Claude Code agents + 1 orchestrator agent
**Framework**: Keras 3 (multi-backend: JAX/TensorFlow/PyTorch)
**Package Manager**: uv (10-100x faster than pip, with lock files for reproducibility)

---

## 🎯 1. Agent Architecture Overview

### 1.1 Core Development Agents

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                        │
│              (Coordinates all other agents)                  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
    ┌───▼───┐           ┌─────▼─────┐        ┌─────▼─────┐
    │ AGENT │           │   AGENT   │        │   AGENT   │
    │   1   │           │     2     │        │     3     │
    │ DATA  │           │   MODEL   │        │ TRAINING  │
    └───────┘           └───────────┘        └───────────┘
        │                     │                     │
    ┌───▼───┐           ┌─────▼─────┐        ┌─────▼─────┐
    │ AGENT │           │   AGENT   │        │   AGENT   │
    │   4   │           │     5     │        │     6     │
    │  EVAL │           │ BUSINESS  │        │   TEST    │
    └───────┘           └───────────┘        └───────────┘
```

### 1.2 Agent Specifications

| Agent ID | Name | Responsibility | Can Run in Parallel With | Tech Focus |
|----------|------|----------------|--------------------------|------------|
| **A0** | Orchestrator | Coordinates agents, manages dependencies | All | Python, Async |
| **A1** | Data Engineer | Data pipeline, feature engineering | A2, A6 | Pandas, NumPy, Dask |
| **A2** | Model Architect | NAM architecture, constraints | A1, A3, A6 | **Keras 3**, NumPy |
| **A3** | Training Specialist | Training loops, optimization | A2, A4, A6 | **Keras 3**, Optuna, MLflow |
| **A4** | Evaluation Engineer | Metrics, validation, WFO | A3, A5, A6 | Scikit-learn, Matplotlib |
| **A5** | Business Tools | Budget optimizer, ROI simulator | A4, A6 | SciPy, Optimization |
| **A6** | Test Automation | Unit tests, integration tests | All | Pytest, Coverage |

**Note**: Deployment (Agent 6 in original) is **excluded** - will be handled separately after model validation.

---

## 🗂️ 2. Folder Structure for Agent-Based Development

### 2.1 Complete Directory Structure

```
NAM_Project/
├── .agents/                           # Agent configurations and states
│   ├── orchestrator/
│   │   ├── config.yaml
│   │   ├── workflow.yaml
│   │   └── state.json
│   ├── agent_01_data/
│   │   ├── config.yaml
│   │   ├── tasks.md
│   │   └── progress.json
│   ├── agent_02_model/
│   ├── agent_03_training/
│   ├── agent_04_evaluation/
│   ├── agent_05_business/
│   └── agent_06_testing/
│
├── docs/
│   ├── NAM_Comprehensive_Implementation_Plan.md  # Master plan
│   ├── agent_handoffs.md                          # Inter-agent communication
│   └── keras3_architecture_notes.md               # Keras 3 specific notes
│
├── data/
│   ├── raw/                           # A1: Raw CSVs
│   ├── processed/                     # A1: Cleaned data
│   ├── features/                      # A1: Engineered features
│   └── splits/                        # A1: Train/val/test
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                          # A1: Data Agent
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── data_validation.py
│   │   └── adstock.py
│   │
│   ├── models/                        # A2: Model Agent
│   │   ├── __init__.py
│   │   ├── constrained_layers.py     # Keras 3 custom layers
│   │   ├── hierarchical_nam.py       # Keras 3 Model subclass
│   │   ├── baseline_models.py
│   │   └── model_utils.py
│   │
│   ├── training/                      # A3: Training Agent
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── loss_functions.py         # Keras 3 Loss classes
│   │   ├── callbacks.py              # Keras 3 Callbacks
│   │   ├── walk_forward.py
│   │   ├── coordinate_descent.py
│   │   └── hyperparameter_tuning.py
│   │
│   ├── evaluation/                    # A4: Evaluation Agent
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   ├── diagnostics.py
│   │   └── model_comparison.py
│   │
│   ├── optimization/                  # A5: Business Agent
│   │   ├── __init__.py
│   │   ├── budget_optimizer.py
│   │   ├── scenario_planner.py
│   │   ├── roi_simulator.py
│   │   └── elasticity_analyzer.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       └── serialization.py
│
├── tests/                             # A6: Test Agent
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_data_preprocessing.py
│   │   ├── test_models.py
│   │   ├── test_training.py
│   │   ├── test_evaluation.py
│   │   └── test_optimization.py
│   ├── integration/
│   │   ├── test_end_to_end.py
│   │   ├── test_walk_forward.py
│   │   └── test_model_pipeline.py
│   └── fixtures/
│       ├── sample_data.py
│       └── mock_models.py
│
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── data_config.yaml
│
├── notebooks/                         # Shared exploration
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Experiments.ipynb
│   └── 04_Results_Analysis.ipynb
│
├── scripts/
│   ├── orchestrator.py                # Main orchestration script
│   ├── run_agent.py                   # Individual agent runner
│   └── parallel_test.py               # Parallel testing
│
├── experiments/                       # MLflow tracking
│   └── mlruns/
│
├── outputs/
│   ├── models/                        # Saved Keras models (.keras)
│   ├── figures/
│   └── reports/
│
├── requirements/                      # Agent-specific requirements (uv)
│   ├── base.txt                       # Base requirements
│   ├── base.lock                      # uv lock file for reproducibility
│   ├── agent_01_data.txt
│   ├── agent_01_data.lock
│   ├── agent_02_model.txt
│   ├── agent_02_model.lock
│   ├── agent_03_training.txt
│   ├── agent_03_training.lock
│   ├── agent_04_evaluation.txt
│   ├── agent_04_evaluation.lock
│   ├── agent_05_business.txt
│   ├── agent_05_business.lock
│   ├── agent_06_testing.txt
│   └── agent_06_testing.lock
│
├── README.md
├── pyproject.toml
├── uv.lock                            # Project-level uv lock file
└── .gitignore
```

---

## 🛠️ 3. Tech Stack by Component

### 3.1 Core ML Stack (Updated for Keras 3)

| Component | Technology | Reason | Agent |
|-----------|------------|--------|-------|
| **Deep Learning** | **Keras 3.0+** | Unified API, multi-backend (JAX/TF/Torch), easy constraints | A2, A3 |
| **Backend** | JAX (recommended) | Fast, functional, easy auto-diff | A2, A3 |
| **Data Processing** | Pandas 2.0+ | Standard, efficient | A1 |
| **Numerical** | NumPy 1.24+ | Fast array operations | A1, A2, A3 |
| **Parallel Data** | Dask (optional) | Large dataset processing | A1 |
| **Optimization** | SciPy 1.11+ | Budget optimization | A5 |
| **ML Utilities** | Scikit-learn 1.3+ | Baseline models, metrics | A2, A4 |

### 3.2 Experiment Tracking & Tuning

| Component | Technology | Reason | Agent |
|-----------|------------|--------|-------|
| **Experiment Tracking** | MLflow 2.7+ | Industry standard, Keras integration | A3, A4 |
| **Hyperparameter Tuning** | Optuna 3.3+ | Keras-compatible, efficient | A3 |
| **Visualization** | Matplotlib 3.7+, Seaborn 0.12+, Plotly 5.14+ | Comprehensive plotting | A4 |
| **Keras Tuner** | Keras Tuner 1.4+ | Native Keras hyperparameter tuning | A3 |

### 3.3 Testing & Quality

| Component | Technology | Reason | Agent |
|-----------|------------|--------|-------|
| **Testing Framework** | Pytest 7.4+ | Standard, powerful | A6 |
| **Coverage** | pytest-cov 4.1+ | Code coverage | A6 |
| **Mocking** | pytest-mock 3.11+ | Test isolation | A6 |
| **Property Testing** | Hypothesis 6.87+ | Automated test generation | A6 |
| **Type Checking** | MyPy 1.5+ | Static type checking | All |
| **Linting** | Ruff 0.0.290+ | Fast linter | All |
| **Formatting** | Black 23.9+ | Code formatting | All |

### 3.4 Agent Orchestration

| Component | Technology | Reason | Agent |
|-----------|------------|--------|-------|
| **Async Framework** | asyncio (built-in) | Parallel agent execution | A0 |
| **Message Broker** | Redis 5.0+ (optional) | Inter-agent communication | A0 |
| **Configuration** | Hydra 1.3+ | Complex config management | A0 |

### 3.5 Development Tools

| Component | Technology | Reason |
|-----------|------------|--------|
| **Version Control** | Git + GitHub | Standard |
| **CI/CD** | GitHub Actions | Integrated, free |
| **Pre-commit Hooks** | pre-commit 3.4+ | Code quality gates |
| **Documentation** | Sphinx 7.2+ | Auto-generate docs |
| **Notebooks** | JupyterLab 4.0+ | Exploration |

---

## 🤖 4. Detailed Agent Specifications with Keras 3

### 4.1 Agent 0: Orchestrator

**Role**: Coordinates all agents, manages dependencies, monitors progress

**Configuration** (.agents/orchestrator/workflow.yaml):
```yaml
workflow:
  phases:
    - name: setup
      parallel: true
      agents:
        - data: setup_pipeline
        - model: design_architecture
        - testing: setup_framework

    - name: data_processing
      parallel: false
      agents:
        - data: process_all_data

    - name: implementation
      parallel: true
      agents:
        - model: implement_keras_constraints
        - training: setup_training
        - testing: write_unit_tests

    - name: training
      parallel: false
      agents:
        - training: train_baseline
        - training: train_nam_keras
        - training: walk_forward_validation

    - name: evaluation
      parallel: true
      agents:
        - evaluation: evaluate_models
        - evaluation: generate_visualizations
        - testing: integration_tests

    - name: business_tools
      parallel: true
      agents:
        - business: budget_optimizer
        - business: scenario_planner
        - business: roi_simulator

dependencies:
  model: [data]
  training: [data, model]
  evaluation: [training]
  business: [training, evaluation]
  testing: []  # Can run anytime
```

---

### 4.2 Agent 1: Data Engineer

**Role**: Data pipeline, feature engineering, data validation

**Tasks** (.agents/agent_01_data/tasks.md):
```markdown
# Data Engineer Agent Tasks

## Phase 1: Setup (Week 1)
- [ ] Set up data loading pipeline
- [ ] Implement data validation framework
- [ ] Create data quality checks

## Phase 2: Processing (Week 1-2)
- [ ] Implement missing value handling
- [ ] Outlier detection and treatment
- [ ] Multi-collinearity analysis

## Phase 3: Feature Engineering (Week 2)
- [ ] Price features (avg_price, price_index, discount)
- [ ] Marketing features with adstock transformations
- [ ] Temporal features (seasonality, trends)
- [ ] Brand health features (NPS, lags)
- [ ] Promotional features

## Phase 4: Finalization (Week 2-3)
- [ ] Time-series aware train/val/test split
- [ ] Save processed data and scalers
- [ ] Generate data documentation
```

**Implementation Template**:
```python
# src/data/feature_engineering.py
import numpy as np
import pandas as pd

class FeatureEngineer:
    """
    Feature engineering for NAM
    Agent 1 (Data Engineer) owns this module
    """
    def __init__(self, config):
        self.config = config

    def create_marketing_features(self, data):
        """Marketing features with adstock transformation"""
        channels = ['TV', 'Digital', 'SEM', 'Radio', 'Sponsorship']

        for channel in channels:
            # Adstock transformation (carryover effect)
            data[f'{channel}_adstock'] = self.apply_adstock(
                data[channel],
                decay_rate=0.7,
                max_lag=3
            )

            # Share of voice
            data[f'{channel}_share'] = data[channel] / (data['total_investment'] + 1e-6)

            # Lagged features
            for lag in [1, 2]:
                data[f'{channel}_lag{lag}'] = data[channel].shift(lag)

        return data

    def apply_adstock(self, x, decay_rate=0.5, max_lag=4):
        """
        Adstock transformation to model carryover effects

        Args:
            x: Time series of marketing spend
            decay_rate: How much effect decays each period (0-1)
            max_lag: Number of periods to consider

        Returns:
            Adstocked series
        """
        adstocked = np.zeros_like(x)

        for lag in range(max_lag):
            decay = decay_rate ** lag
            shifted = np.roll(x, lag)
            if lag > 0:
                shifted[:lag] = 0  # No carryover before start
            adstocked += decay * shifted

        return adstocked
```

---

### 4.3 Agent 2: Model Architect (Keras 3)

**Role**: Design and implement NAM architecture with Keras 3

**Tasks**:
```markdown
# Model Architect Agent Tasks

## Phase 1: Keras 3 Setup (Week 1)
- [ ] Install and configure Keras 3 with JAX backend
- [ ] Design constrained layer interfaces
- [ ] Test Keras 3 subclassing pattern

## Phase 2: Constrained Layers (Week 2)
- [ ] Implement MonotonicPositiveLayer (Keras Layer)
- [ ] Implement MonotonicNegativeLayer (Keras Layer)
- [ ] Implement BetaGammaLayer (parametric investment function)
- [ ] Implement HillLayer (alternative investment function)

## Phase 3: Hierarchical Structure (Week 2-3)
- [ ] Design Brand-SKU hierarchical architecture (Keras Model)
- [ ] Implement HierarchicalNAM class
- [ ] Feature network factory
- [ ] Model initialization strategy

## Phase 4: Baseline Models (Week 3)
- [ ] Linear regression baseline (Keras Sequential)
- [ ] Ridge regression (Keras with L2)
- [ ] Simple NN baseline
```

**Implementation Template with Keras 3**:

```python
# src/models/constrained_layers.py
import keras
from keras import ops
from keras import layers

class MonotonicPositiveLayer(layers.Layer):
    """
    Monotonic increasing neural network layer
    Ensures positive weights for monotonicity

    Agent 2 (Model Architect) owns this module
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Initialize weights
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='kernel',
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias',
            trainable=True
        )

    def call(self, inputs):
        # Apply softplus to weights to ensure positivity
        positive_weights = ops.softplus(self.kernel)

        # Linear transformation with positive weights
        output = ops.matmul(inputs, positive_weights) + self.bias

        return output

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class MonotonicNegativeLayer(layers.Layer):
    """
    Monotonic decreasing neural network layer
    Ensures negative weights for decreasing monotonicity

    Agent 2 (Model Architect) owns this module
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='kernel',
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias',
            trainable=True
        )

    def call(self, inputs):
        # Apply negative softplus to ensure negative weights
        negative_weights = -ops.softplus(self.kernel)

        output = ops.matmul(inputs, negative_weights) + self.bias

        return output

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class BetaGammaLayer(layers.Layer):
    """
    Parametric Beta-Gamma function for investment response
    f(x) = a * x^alpha * exp(-beta * x)

    Models diminishing returns with concave shape
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Learnable parameters
        self.a = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(1.0),
            name='scale',
            trainable=True
        )
        self.alpha = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(0.5),
            name='alpha',
            trainable=True,
            constraint=lambda x: ops.clip(x, 0.1, 1.0)  # Constrain alpha
        )
        self.beta = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(0.1),
            name='beta',
            trainable=True,
            constraint=lambda x: ops.clip(x, 0.01, 0.5)  # Constrain beta
        )

    def call(self, inputs):
        # Ensure positive parameters
        a_pos = ops.softplus(self.a)
        alpha_pos = ops.softplus(self.alpha)
        beta_pos = ops.softplus(self.beta)

        # Beta-Gamma transformation
        # f(x) = a * x^alpha * exp(-beta * x)
        powered = ops.power(inputs + 1e-6, alpha_pos)
        decayed = ops.exp(-beta_pos * inputs)

        return a_pos * powered * decayed

    def get_elasticity(self, inputs):
        """Compute elasticity: d(log y) / d(log x)"""
        # Elasticity = alpha - beta * x
        alpha_pos = ops.softplus(self.alpha)
        beta_pos = ops.softplus(self.beta)

        return alpha_pos - beta_pos * inputs


# src/models/hierarchical_nam.py
class HierarchicalNAM(keras.Model):
    """
    Hierarchical Neural Additive Model using Keras 3

    Architecture:
    - Brand-level subnetworks (shared patterns)
    - SKU-level subnetworks (product-specific)
    - Hierarchical combination with learnable weight

    Agent 2 (Model Architect) owns this module
    """
    def __init__(self, feature_configs, brand_ids, sku_to_brand_mapping, hier_weight=0.7, **kwargs):
        super().__init__(**kwargs)

        self.feature_configs = feature_configs
        self.brand_ids = brand_ids
        self.sku_to_brand = sku_to_brand_mapping
        self.hier_weight_val = hier_weight

        # Build brand-level networks
        self.brand_networks = {}
        for brand in brand_ids:
            self.brand_networks[brand] = self._create_feature_networks(
                feature_configs,
                name_prefix=f'brand_{brand}'
            )

        # Build SKU-level networks
        self.sku_networks = {}
        for sku in sku_to_brand_mapping.keys():
            self.sku_networks[sku] = self._create_feature_networks(
                feature_configs,
                name_prefix=f'sku_{sku}'
            )

        # Hierarchical weight (learnable)
        self.hier_weight = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(hier_weight),
            name='hierarchical_weight',
            trainable=True,
            constraint=lambda x: ops.clip(x, 0.0, 1.0)
        )

    def _create_feature_networks(self, feature_configs, name_prefix):
        """Factory for creating feature-specific networks"""
        networks = {}

        for feat_name, config in feature_configs.items():
            feat_type = config['type']

            if feat_type == 'monotonic_positive':
                # Monotonic increasing network
                network = keras.Sequential([
                    layers.Input(shape=(1,)),
                    MonotonicPositiveLayer(config['hidden_dims'][0], name=f'{name_prefix}_{feat_name}_1'),
                    layers.ReLU(),
                    MonotonicPositiveLayer(config['hidden_dims'][1], name=f'{name_prefix}_{feat_name}_2'),
                    layers.ReLU(),
                    MonotonicPositiveLayer(1, name=f'{name_prefix}_{feat_name}_out')
                ], name=f'{name_prefix}_{feat_name}')

            elif feat_type == 'monotonic_negative':
                # Monotonic decreasing network (for price)
                network = keras.Sequential([
                    layers.Input(shape=(1,)),
                    MonotonicNegativeLayer(config['hidden_dims'][0], name=f'{name_prefix}_{feat_name}_1'),
                    layers.ReLU(),
                    MonotonicNegativeLayer(config['hidden_dims'][1], name=f'{name_prefix}_{feat_name}_2'),
                    layers.ReLU(),
                    MonotonicNegativeLayer(1, name=f'{name_prefix}_{feat_name}_out')
                ], name=f'{name_prefix}_{feat_name}')

            elif feat_type == 'parametric_investment':
                # Parametric Beta-Gamma function
                network = keras.Sequential([
                    layers.Input(shape=(1,)),
                    BetaGammaLayer(name=f'{name_prefix}_{feat_name}_betagamma')
                ], name=f'{name_prefix}_{feat_name}')

            else:
                # Unconstrained network
                network = keras.Sequential([
                    layers.Input(shape=(1,)),
                    layers.Dense(config['hidden_dims'][0], activation='relu',
                                name=f'{name_prefix}_{feat_name}_1'),
                    layers.Dense(config['hidden_dims'][1], activation='relu',
                                name=f'{name_prefix}_{feat_name}_2'),
                    layers.Dense(1, name=f'{name_prefix}_{feat_name}_out')
                ], name=f'{name_prefix}_{feat_name}')

            networks[feat_name] = network

        return networks

    def call(self, inputs, training=None):
        """
        Forward pass

        Args:
            inputs: Dict with keys:
                - features: Dict of feature tensors {feat_name: tensor}
                - brand_id: Brand identifier
                - sku_id: SKU identifier

        Returns:
            prediction: Final GMV prediction
        """
        features = inputs['features']
        brand_id = inputs['brand_id']
        sku_id = inputs['sku_id']

        # Brand-level prediction
        brand_output = 0.0
        for feat_name, feat_value in features.items():
            brand_contrib = self.brand_networks[brand_id][feat_name](feat_value, training=training)
            brand_output = brand_output + brand_contrib

        # SKU-level prediction
        sku_output = 0.0
        for feat_name, feat_value in features.items():
            sku_contrib = self.sku_networks[sku_id][feat_name](feat_value, training=training)
            sku_output = sku_output + sku_contrib

        # Hierarchical combination
        final_output = self.hier_weight * brand_output + (1.0 - self.hier_weight) * sku_output

        return final_output

    def get_feature_contribution(self, inputs, brand_id, sku_id):
        """Get contribution of each feature for interpretability"""
        features = inputs['features']

        contributions = {}
        for feat_name, feat_value in features.items():
            brand_contrib = self.brand_networks[brand_id][feat_name](feat_value, training=False)
            sku_contrib = self.sku_networks[sku_id][feat_name](feat_value, training=False)

            total_contrib = self.hier_weight * brand_contrib + (1.0 - self.hier_weight) * sku_contrib
            contributions[feat_name] = total_contrib

        return contributions

    def get_config(self):
        """For model serialization"""
        config = super().get_config()
        config.update({
            'feature_configs': self.feature_configs,
            'brand_ids': self.brand_ids,
            'sku_to_brand_mapping': self.sku_to_brand,
            'hier_weight': float(self.hier_weight_val)
        })
        return config
```

---

### 4.4 Agent 3: Training Specialist (Keras 3)

**Role**: Training loops, optimization strategies with Keras 3

**Tasks**:
```markdown
# Training Specialist Agent Tasks

## Phase 1: Training Infrastructure (Week 2-3)
- [ ] Implement Keras 3 training loop
- [ ] Custom loss function with constraints (Keras Loss)
- [ ] Keras callbacks (EarlyStopping, ModelCheckpoint, LearningRateScheduler)
- [ ] MLflow integration with Keras

## Phase 2: Optimization Strategies (Week 3)
- [ ] Standard Keras fit() with custom training step
- [ ] Coordinate descent trainer
- [ ] Custom training loop with GradientTape

## Phase 3: Walk-Forward Validation (Week 3-4)
- [ ] WalkForwardSplitter
- [ ] WalkForwardNAMTrainer (Keras models)
- [ ] Ensemble from multiple folds

## Phase 4: Hyperparameter Tuning (Week 4)
- [ ] Keras Tuner integration
- [ ] Optuna with Keras
```

**Implementation with Keras 3**:

```python
# src/training/loss_functions.py
import keras
from keras import ops

class NAMLoss(keras.losses.Loss):
    """
    Custom loss function for NAM with all components

    Agent 3 (Training Specialist) owns this module
    """
    def __init__(self,
                 lambda_constraint=0.5,
                 lambda_hierarchical=0.3,
                 lambda_smooth=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.lambda_constraint = lambda_constraint
        self.lambda_hierarchical = lambda_hierarchical
        self.lambda_smooth = lambda_smooth

    def call(self, y_true, y_pred):
        """
        Compute total loss

        Args:
            y_true: True GMV values
            y_pred: Predicted GMV values

        Returns:
            Total loss (scalar)
        """
        # 1. Prediction loss (MSE)
        loss_fit = ops.mean(ops.square(y_true - y_pred))

        # 2. Constraint loss (computed externally, passed via add_loss)
        # This is handled in the training loop

        # Total loss
        total_loss = loss_fit

        return total_loss


# src/training/trainer.py
class NAMTrainer:
    """
    Trainer for NAM using Keras 3

    Agent 3 (Training Specialist) owns this module
    """
    def __init__(self, model, train_data, val_data, config):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config

        # Setup optimizer
        learning_rate = config['learning_rate']
        self.optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # Setup loss
        self.loss_fn = NAMLoss(
            lambda_constraint=config['lambda_constraint'],
            lambda_hierarchical=config['lambda_hierarchical'],
            lambda_smooth=config['lambda_smooth']
        )

        # Setup metrics
        self.metrics = {
            'mse': keras.metrics.MeanSquaredError(),
            'mae': keras.metrics.MeanAbsoluteError()
        }

        # Compile model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=list(self.metrics.values())
        )

        # Callbacks
        self.callbacks = self._create_callbacks()

    def _create_callbacks(self):
        """Create Keras callbacks"""
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),

            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath='outputs/models/best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),

            # Learning rate scheduler
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),

            # CSV logger
            keras.callbacks.CSVLogger('outputs/training_log.csv'),

            # TensorBoard
            keras.callbacks.TensorBoard(
                log_dir='outputs/logs',
                histogram_freq=1
            )
        ]

        return callbacks

    def train(self):
        """
        Train the model using Keras fit()
        """
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=self.config['max_epochs'],
            batch_size=self.config['batch_size'],
            callbacks=self.callbacks,
            verbose=1
        )

        return history

    def custom_training_loop(self):
        """
        Custom training loop for advanced constraint handling
        """
        train_dataset = self.prepare_dataset(self.train_data)
        val_dataset = self.prepare_dataset(self.val_data)

        best_val_loss = float('inf')

        for epoch in range(self.config['max_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['max_epochs']}")

            # Training
            train_loss = self.train_step(train_dataset)

            # Validation
            val_loss = self.validate_step(val_dataset)

            # Log metrics
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save('outputs/models/best_model.keras')
                print("✓ Model saved")

            # Early stopping check
            # (implement early stopping logic)

        return self.model

    def train_step(self, dataset):
        """Single training step with GradientTape"""
        total_loss = 0.0
        num_batches = 0

        for batch in dataset:
            with keras.ops.GradientTape() as tape:
                # Forward pass
                predictions = self.model(batch['inputs'], training=True)

                # Compute loss
                loss = self.loss_fn(batch['targets'], predictions)

                # Add constraint losses
                loss += self.compute_constraint_loss(self.model, batch)

            # Compute gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)

            # Apply gradients
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            total_loss += loss
            num_batches += 1

        return total_loss / num_batches

    def compute_constraint_loss(self, model, batch):
        """Compute constraint violation penalties"""
        constraint_loss = 0.0

        # Check monotonicity violations
        # (implement based on model architecture)

        # Check hierarchical consistency
        # (implement brand-SKU distance penalty)

        return constraint_loss


# src/training/walk_forward.py
class WalkForwardNAMTrainer:
    """
    Walk-forward optimization for Keras NAM

    Agent 3 (Training Specialist) owns this module
    """
    def __init__(self, model_fn, model_config, training_config):
        """
        Args:
            model_fn: Function that creates a fresh Keras model
            model_config: Model configuration dict
            training_config: Training configuration dict
        """
        self.model_fn = model_fn
        self.model_config = model_config
        self.training_config = training_config
        self.fold_results = []

    def run_walk_forward(self, data):
        """Execute walk-forward optimization"""
        from .walk_forward_splitter import WalkForwardSplitter

        splitter = WalkForwardSplitter(
            initial_train_size=6,
            test_size=1,
            window_type='expanding'
        )

        all_oos_predictions = []
        all_oos_actuals = []

        for fold_idx, (train_data, test_data, fold_info) in enumerate(splitter.split(data)):
            print(f"\nFold {fold_idx}: {fold_info['test_period']}")

            # Create fresh model
            model = self.model_fn(self.model_config)

            # Train
            trainer = NAMTrainer(model, train_data, None, self.training_config)
            trainer.train()

            # Evaluate on OOS test
            test_pred = model.predict(test_data)
            test_actual = test_data['GMV'].values

            # Metrics
            from sklearn.metrics import r2_score, mean_absolute_percentage_error

            fold_metrics = {
                'r2': r2_score(test_actual, test_pred),
                'mape': mean_absolute_percentage_error(test_actual, test_pred) * 100
            }

            self.fold_results.append({
                'fold': fold_idx,
                'test_period': fold_info['test_period'],
                'metrics': fold_metrics
            })

            all_oos_predictions.extend(test_pred.flatten())
            all_oos_actuals.extend(test_actual)

            print(f"  R²: {fold_metrics['r2']:.3f}, MAPE: {fold_metrics['mape']:.2f}%")

        # Aggregate
        overall_r2 = r2_score(all_oos_actuals, all_oos_predictions)

        print("\n" + "="*60)
        print("Walk-Forward Optimization Results")
        print("="*60)
        print(f"Overall OOS R²: {overall_r2:.3f}")

        return self.fold_results, overall_r2
```

---

### 4.5 Agent 4: Evaluation Engineer

**Role**: Metrics, visualization, model comparison

**Tasks**:
```markdown
# Evaluation Engineer Agent Tasks

## Phase 1: Metrics Framework (Week 3-4)
- [ ] Implement regression metrics
- [ ] Elasticity computation from Keras models
- [ ] Confidence intervals (bootstrap)

## Phase 2: Visualization (Week 4)
- [ ] Feature response curves (Keras model.predict)
- [ ] Hierarchical brand-SKU plots
- [ ] Actual vs predicted plots
- [ ] Residual analysis plots

## Phase 3: Model Comparison (Week 4-5)
- [ ] Compare Keras NAM vs baselines
- [ ] Walk-forward results visualization
- [ ] Constraint satisfaction reports

## Phase 4: Reporting (Week 5)
- [ ] Automated report generation
- [ ] Executive summary
```

**Implementation**:
```python
# src/evaluation/visualization.py
import numpy as np
import matplotlib.pyplot as plt

class NAMVisualizer:
    """
    Visualization for Keras NAM
    Agent 4 (Evaluation Engineer) owns this module
    """
    def __init__(self, model, data, output_dir='outputs/figures/'):
        self.model = model
        self.data = data
        self.output_dir = output_dir

    def plot_feature_response_curves(self, feature_ranges, brand_id, sku_id):
        """Plot shape function for each feature"""
        n_features = len(feature_ranges)
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for idx, (feat_name, (feat_min, feat_max)) in enumerate(feature_ranges.items()):
            ax = axes[idx]

            # Generate input range
            x_range = np.linspace(feat_min, feat_max, 100)

            # Prepare input for Keras model
            feature_input = {feat_name: x_range.reshape(-1, 1)}
            model_input = {
                'features': feature_input,
                'brand_id': brand_id,
                'sku_id': sku_id
            }

            # Get contributions
            contributions = self.model.get_feature_contribution(
                model_input,
                brand_id,
                sku_id
            )

            y_response = contributions[feat_name].numpy().flatten()

            # Plot
            ax.plot(x_range, y_response, linewidth=2, label=feat_name)
            ax.set_xlabel(feat_name)
            ax.set_ylabel('Contribution to GMV')
            ax.set_title(f'{feat_name} Response Curve')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}feature_response_curves.png', dpi=300)
        plt.close()
```

---

### 4.6 Agent 5: Business Tools Developer

**Role**: Budget optimizer, ROI simulator, scenario planner

**Tasks**: (Same as before - uses trained Keras model for predictions)

**Implementation**:
```python
# src/optimization/budget_optimizer.py
from scipy.optimize import minimize
import numpy as np

class BudgetOptimizer:
    """
    Budget optimizer using trained Keras NAM
    Agent 5 (Business Tools) owns this module
    """
    def __init__(self, keras_model, brand_id, sku_id):
        self.model = keras_model
        self.brand_id = brand_id
        self.sku_id = sku_id

    def optimize_allocation(self, total_budget, current_features):
        """
        Find optimal budget allocation

        Args:
            total_budget: Total marketing budget
            current_features: Dict of other feature values (price, NPS, etc.)

        Returns:
            Optimal allocation dict
        """
        channels = ['TV', 'Digital', 'SEM', 'Radio', 'Sponsorship']
        n_channels = len(channels)

        # Initial guess (equal split)
        x0 = np.ones(n_channels) * (total_budget / n_channels)

        # Objective: Maximize GMV (minimize negative GMV)
        def objective(x):
            # Prepare features with marketing spend
            features = current_features.copy()
            for i, channel in enumerate(channels):
                features[f'{channel}_adstock'] = x[i]

            # Predict GMV using Keras model
            model_input = {
                'features': {k: np.array([[v]]) for k, v in features.items()},
                'brand_id': self.brand_id,
                'sku_id': self.sku_id
            }

            gmv = self.model(model_input, training=False).numpy()[0, 0]

            return -gmv  # Negative for maximization

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: x.sum() - total_budget},
            {'type': 'ineq', 'fun': lambda x: x - 0.10 * total_budget / n_channels},
            {'type': 'ineq', 'fun': lambda x: 0.50 * total_budget - x}
        ]

        bounds = [(0, total_budget) for _ in range(n_channels)]

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_allocation = dict(zip(channels, result.x))
        predicted_gmv = -result.fun

        return {
            'allocation': optimal_allocation,
            'predicted_gmv': predicted_gmv,
            'optimization_success': result.success
        }
```

---

### 4.7 Agent 6: Test Automation

**Role**: Unit tests, integration tests for Keras models

**Tasks**: (Same structure as before)

**Implementation with Keras 3**:
```python
# tests/unit/test_models.py
import pytest
import keras
import numpy as np
from src.models.constrained_layers import MonotonicPositiveLayer, MonotonicNegativeLayer
from src.models.hierarchical_nam import HierarchicalNAM

def test_monotonic_positive_layer():
    """Test monotonic increasing constraint"""
    layer = MonotonicPositiveLayer(units=16)

    # Build layer
    layer.build((None, 1))

    # Sorted inputs
    x = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
    outputs = layer(x).numpy()

    # Check monotonicity
    diffs = outputs[1:] - outputs[:-1]
    violations = (diffs < 0).sum()

    assert violations == 0, f"Monotonicity violated {violations} times"

def test_monotonic_negative_layer():
    """Test monotonic decreasing constraint"""
    layer = MonotonicNegativeLayer(units=16)
    layer.build((None, 1))

    x = np.linspace(0, 10, 100).reshape(-1, 1).astype(np.float32)
    outputs = layer(x).numpy()

    diffs = outputs[1:] - outputs[:-1]
    violations = (diffs > 0).sum()

    assert violations == 0, f"Decreasing monotonicity violated {violations} times"

def test_hierarchical_nam_forward():
    """Test HierarchicalNAM forward pass"""
    feature_configs = {
        'price': {'type': 'monotonic_negative', 'hidden_dims': [32, 16]},
        'investment': {'type': 'parametric_investment', 'hidden_dims': [32, 16]}
    }

    model = HierarchicalNAM(
        feature_configs=feature_configs,
        brand_ids=['Brand1'],
        sku_to_brand_mapping={'SKU1': 'Brand1'}
    )

    # Test input
    inputs = {
        'features': {
            'price': np.random.randn(10, 1).astype(np.float32),
            'investment': np.random.randn(10, 1).astype(np.float32)
        },
        'brand_id': 'Brand1',
        'sku_id': 'SKU1'
    }

    output = model(inputs, training=False)

    assert output.shape == (10, 1)

def test_model_save_load():
    """Test Keras model serialization"""
    feature_configs = {
        'price': {'type': 'monotonic_negative', 'hidden_dims': [32, 16]}
    }

    model = HierarchicalNAM(
        feature_configs=feature_configs,
        brand_ids=['Brand1'],
        sku_to_brand_mapping={'SKU1': 'Brand1'}
    )

    # Save
    model.save('outputs/models/test_model.keras')

    # Load
    loaded_model = keras.models.load_model(
        'outputs/models/test_model.keras',
        custom_objects={
            'HierarchicalNAM': HierarchicalNAM,
            'MonotonicPositiveLayer': MonotonicPositiveLayer,
            'MonotonicNegativeLayer': MonotonicNegativeLayer
        }
    )

    assert loaded_model is not None
```

---

## 🔄 5. Parallel Development Workflow (Updated)

### 5.1 Week-by-Week Agent Activity

```
Week 1: Setup & Foundation
├── A0: Initialize project, setup Keras 3 environment
├── A1: Data loading, validation framework ✓ (Parallel)
├── A2: Keras 3 setup, design architecture ✓ (Parallel)
├── A6: Test framework setup ✓ (Parallel)
└── [3 agents running in parallel]

Week 2: Data Processing & Model Implementation
├── A1: Feature engineering, adstock ✓ (Parallel)
├── A2: Implement Keras constrained layers ✓ (Parallel)
├── A3: Setup training with Keras ✓ (Parallel)
├── A6: Unit tests for data & models ✓ (Parallel)
└── [4 agents running in parallel]

Week 3: Training & Hierarchical Structure
├── A2: Hierarchical NAM (Keras Model) ✓ (Parallel)
├── A3: Training loop, Keras callbacks ✓ (Parallel)
├── A4: Evaluation framework setup ✓ (Parallel)
├── A6: Training tests ✓ (Parallel)
└── [4 agents running in parallel]

Week 4: Walk-Forward & Evaluation
├── A3: Walk-forward validation with Keras
├── A4: Metrics, visualization ✓ (Parallel with A3)
├── A5: Business tools - budget optimizer ✓ (Parallel)
├── A6: Integration tests ✓ (Parallel)
└── [4 agents running in parallel]

Week 5: Business Tools & Reporting
├── A4: Reporting framework ✓ (Parallel)
├── A5: ROI simulator, scenario planner ✓ (Parallel)
├── A6: Business tools tests ✓ (Parallel)
└── [3 agents running in parallel]

Week 6-7: Finalization & Validation
├── A3: Hyperparameter tuning (Keras Tuner)
├── A4: Final reports and visualizations
├── A5: Final business tool validation
├── A6: Complete test coverage
├── A0: Final validation and quality checks
└── [All agents coordinate]
```

### 5.2 Agent Synchronization Points (Updated)

```yaml
sync_points:
  - name: "Keras 3 Environment Ready"
    week: 1
    prerequisite: A0 completes setup
    unblocks: [A2_model_architecture]

  - name: "Data Ready"
    week: 2
    prerequisite: A1 completes data processing
    unblocks: [A3_training]

  - name: "Model Architecture Complete"
    week: 3
    prerequisite: A2 completes Keras models
    unblocks: [A3_training_execution]

  - name: "Training Complete"
    week: 4
    prerequisite: A3 completes training & WFO
    unblocks: [A4_full_evaluation, A5_business_tools]

  - name: "Ready for Deployment Decision"
    week: 7
    prerequisite: All agents complete
    decision: User reviews results before deployment
```

---

## 🛠️ 6. Package Management with uv

### 6.1 Why uv for This Project?

**uv** is an extremely fast Python package manager written in Rust, providing:

- ✅ **10-100x faster** than pip (critical for 6 parallel agents)
- ✅ **Lock files** for perfect reproducibility across agents
- ✅ **Better dependency resolution** (handles complex Keras 3 + JAX dependencies)
- ✅ **Drop-in pip replacement** (same requirements.txt format)
- ✅ **Built-in virtual environment** management
- ✅ **Cross-platform** (Windows, Linux, macOS)

**Installation**:
```bash
# On Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip (if needed)
pip install uv

# Verify installation
uv --version
```

### 6.2 Base Requirements

**File Structure with uv**:
```
requirements/
├── base.txt          # Human-readable requirements
├── base.lock         # uv-generated lock file (exact versions)
├── agent_01_data.txt
├── agent_01_data.lock
└── ...
```

```txt
# requirements/base.txt
# Core ML with Keras 3
keras>=3.0.0
jax[cpu]>=0.4.20        # JAX backend for Keras (CPU version)
# jax[cuda12]>=0.4.20   # Uncomment for GPU support

# Data processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Optimization
scipy>=1.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Experiment tracking
mlflow>=2.7.0

# Configuration
pyyaml>=6.0
hydra-core>=1.3.0

# Utilities
tqdm>=4.65.0
loguru>=0.7.0
```

### 6.2 Agent-Specific Requirements

```txt
# requirements/agent_02_model.txt
keras>=3.0.0
jax[cpu]>=0.4.20
numpy>=1.24.0
```

```txt
# requirements/agent_03_training.txt
keras>=3.0.0
jax[cpu]>=0.4.20
optuna>=3.3.0
keras-tuner>=1.4.0
mlflow>=2.7.0
```

```txt
# requirements/agent_06_testing.txt
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
hypothesis>=6.87.0
```

### 6.3 Using uv for Package Management

**Generate Lock Files** (do this once, commit to git):
```bash
# Generate lock file from requirements
cd requirements/
uv pip compile base.txt -o base.lock
uv pip compile agent_01_data.txt -o agent_01_data.lock
uv pip compile agent_02_model.txt -o agent_02_model.lock
uv pip compile agent_03_training.txt -o agent_03_training.lock
uv pip compile agent_04_evaluation.txt -o agent_04_evaluation.lock
uv pip compile agent_05_business.txt -o agent_05_business.lock
uv pip compile agent_06_testing.txt -o agent_06_testing.lock
```

**Install Dependencies** (each agent):
```bash
# Create virtual environment with uv
uv venv .venv_agent_01

# Activate environment
# Windows:
.venv_agent_01\Scripts\activate
# Linux/macOS:
source .venv_agent_01/bin/activate

# Install from lock file (exact versions, reproducible)
uv pip sync requirements/agent_01_data.lock

# Or install from requirements.txt (if lock file not needed)
uv pip install -r requirements/agent_01_data.txt
```

**Update Dependencies**:
```bash
# Add new package
echo "new-package>=1.0.0" >> requirements/base.txt

# Regenerate lock file
uv pip compile requirements/base.txt -o requirements/base.lock

# Sync environment
uv pip sync requirements/base.lock
```

**Advantages for Agent Development**:
1. **Fast parallel setup**: All 6 agents can install dependencies simultaneously
2. **Reproducible**: Lock files ensure exact same versions across all agents
3. **No conflicts**: Better resolver handles Keras 3 + JAX + MLflow complex deps
4. **CI/CD friendly**: Much faster GitHub Actions runs

### 6.4 Setup Scripts with uv

**Agent Environment Setup Script**:
```python
# scripts/setup_agent_environment.py
"""
Setup individual agent environments using uv
Much faster than pip-based setup
"""
import subprocess
import sys
from pathlib import Path

def setup_agent_environment(agent_name, agent_id):
    """
    Setup environment for a specific agent using uv

    Args:
        agent_name: Name of agent (e.g., 'data', 'model')
        agent_id: Agent ID (e.g., '01', '02')
    """
    print(f"Setting up environment for Agent {agent_id}: {agent_name}")

    # Create virtual environment with uv (very fast!)
    venv_path = f".venv_agent_{agent_id}"
    print(f"Creating virtual environment: {venv_path}")

    result = subprocess.run(['uv', 'venv', venv_path], check=True)

    # Determine activation script
    if sys.platform == 'win32':
        activate_script = f"{venv_path}\\Scripts\\activate"
        python_exe = f"{venv_path}\\Scripts\\python.exe"
    else:
        activate_script = f"source {venv_path}/bin/activate"
        python_exe = f"{venv_path}/bin/python"

    print(f"Virtual environment created. Activate with: {activate_script}")

    # Install dependencies from lock file (reproducible)
    requirements_lock = f"requirements/agent_{agent_id}_{agent_name}.lock"

    if Path(requirements_lock).exists():
        print(f"Installing from lock file: {requirements_lock}")
        subprocess.run([
            'uv', 'pip', 'install',
            '--python', python_exe,
            '-r', requirements_lock
        ], check=True)
    else:
        # Fallback to requirements.txt
        requirements_txt = f"requirements/agent_{agent_id}_{agent_name}.txt"
        print(f"Installing from requirements: {requirements_txt}")
        subprocess.run([
            'uv', 'pip', 'install',
            '--python', python_exe,
            '-r', requirements_txt
        ], check=True)

    print(f"✓ Agent {agent_id} ({agent_name}) environment ready!")
    return venv_path

def setup_all_agents():
    """Setup environments for all 6 agents in parallel"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    agents = [
        ('data', '01'),
        ('model', '02'),
        ('training', '03'),
        ('evaluation', '04'),
        ('business', '05'),
        ('testing', '06')
    ]

    print("Setting up all agent environments in parallel using uv...")
    print("This is much faster than sequential pip install!")

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(setup_agent_environment, name, id): (name, id)
            for name, id in agents
        }

        for future in as_completed(futures):
            name, id = futures[future]
            try:
                venv_path = future.result()
                print(f"✓ Agent {id} ({name}) completed")
            except Exception as e:
                print(f"✗ Agent {id} ({name}) failed: {e}")

    print("\n✓ All agent environments ready!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Setup agent environments with uv')
    parser.add_argument('--agent', help='Specific agent name')
    parser.add_argument('--id', help='Agent ID')
    parser.add_argument('--all', action='store_true', help='Setup all agents')

    args = parser.parse_args()

    if args.all:
        setup_all_agents()
    elif args.agent and args.id:
        setup_agent_environment(args.agent, args.id)
    else:
        print("Usage: python setup_agent_environment.py --all")
        print("   or: python setup_agent_environment.py --agent data --id 01")
```

**Generate All Lock Files Script**:
```python
# scripts/generate_lock_files.py
"""
Generate uv lock files for all requirements
Run this once, then commit lock files to git
"""
import subprocess
from pathlib import Path

def generate_lock_files():
    """Generate lock files for all requirements"""
    requirements_dir = Path("requirements")

    # Find all .txt requirement files
    requirement_files = list(requirements_dir.glob("*.txt"))

    print(f"Found {len(requirement_files)} requirement files")
    print("Generating lock files with uv...\n")

    for req_file in requirement_files:
        lock_file = req_file.with_suffix('.lock')

        print(f"Compiling: {req_file.name} → {lock_file.name}")

        try:
            subprocess.run([
                'uv', 'pip', 'compile',
                str(req_file),
                '-o', str(lock_file)
            ], check=True)
            print(f"✓ Generated {lock_file.name}\n")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to generate {lock_file.name}: {e}\n")

    print("✓ All lock files generated!")
    print("\nCommit these files to git:")
    print("  git add requirements/*.lock")
    print("  git commit -m 'Add uv lock files for reproducibility'")

if __name__ == "__main__":
    generate_lock_files()
```

**Usage**:
```bash
# First time setup - generate all lock files
python scripts/generate_lock_files.py

# Setup all agent environments in parallel (very fast with uv!)
python scripts/setup_agent_environment.py --all

# Setup specific agent
python scripts/setup_agent_environment.py --agent data --id 01
```

### 6.5 CI/CD with uv (GitHub Actions)

**Example GitHub Actions workflow**:
```yaml
# .github/workflows/test_agents.yml
name: Test All Agents

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        agent: [data, model, training, evaluation, business, testing]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Set up Python with uv
        run: |
          uv venv .venv
          source .venv/bin/activate
          echo "VIRTUAL_ENV=$VIRTUAL_ENV" >> $GITHUB_ENV
          echo "$VIRTUAL_ENV/bin" >> $GITHUB_PATH

      - name: Install dependencies with uv (from lock file)
        run: |
          uv pip sync requirements/agent_0${{ matrix.agent }}_*.lock

      - name: Run tests for ${{ matrix.agent }}
        run: |
          pytest tests/unit/test_${{ matrix.agent }}.py -v --cov

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

**Benefits in CI/CD**:
- **5-10x faster** GitHub Actions runs
- **Consistent environments** via lock files
- **Parallel agent testing** works great with uv
- **Lower costs** (less compute time)

---

## 📊 7. Resource Requirements (Updated)

### 7.1 Compute Resources

```yaml
agent_resources:
  A0_orchestrator:
    cpu: 1 core
    memory: 2GB

  A1_data:
    cpu: 4 cores
    memory: 16GB
    runtime: Week 1-2

  A2_model:
    cpu: 2 cores
    memory: 8GB
    runtime: Week 1-3

  A3_training:
    cpu: 8 cores (or GPU recommended)
    memory: 16GB
    gpu: Optional (JAX can use GPU/TPU)
    runtime: Week 2-5

  A4_evaluation:
    cpu: 4 cores
    memory: 16GB
    runtime: Week 3-6

  A5_business:
    cpu: 4 cores
    memory: 8GB
    runtime: Week 4-6

  A6_testing:
    cpu: 4 cores
    memory: 8GB
    runtime: Continuous (Week 1-7)

total_peak_resources:
  cpu: 27 cores
  memory: 74GB
  gpu: Optional (JAX backend can leverage GPU/TPU)
```

---

## 🎯 8. Summary & Next Steps

### 8.1 Key Changes from Original Proposal

**✅ Framework**: PyTorch → **Keras 3**
- Multi-backend support (JAX/TensorFlow/PyTorch)
- Cleaner API for custom layers
- Better for rapid prototyping
- Native support for constraints via custom layers

**✅ Agent Count**: 7 → **6 development agents**
- Removed: Agent 6 (Deployment Engineer)
- Focus: Model development, validation, and business tools
- Deployment: Will be handled separately after validation

**✅ Timeline**: 8-10 weeks → **6-7 weeks**
- Faster with Keras 3's higher-level API
- No deployment work in this phase

### 8.2 Benefits of Keras 3

1. **Simpler constraint implementation**: Custom layers are cleaner than PyTorch
2. **Multi-backend**: Can switch between JAX/TF/Torch without code changes
3. **Better serialization**: `.keras` format is more robust
4. **Native callbacks**: Built-in EarlyStopping, ModelCheckpoint, etc.
5. **Easier for prototyping**: Higher-level API, less boilerplate

### 8.3 Next Steps

**Week 0 (Setup)**:
```bash
# 1. Install uv (if not already installed)
# Windows (PowerShell):
irm https://astral.sh/uv/install.ps1 | iex
# Linux/macOS:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify uv installation
uv --version

# 2. Create project structure
python scripts/setup_project_structure.py

# 3. Create virtual environment with uv
uv venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 4. Install base requirements with uv
uv pip install -r requirements/base.txt

# Or use lock file for exact reproducibility
uv pip sync requirements/base.lock

# 5. Generate all lock files (first time setup)
cd requirements/
uv pip compile base.txt -o base.lock
uv pip compile agent_01_data.txt -o agent_01_data.lock
uv pip compile agent_02_model.txt -o agent_02_model.lock
uv pip compile agent_03_training.txt -o agent_03_training.lock
uv pip compile agent_04_evaluation.txt -o agent_04_evaluation.lock
uv pip compile agent_05_business.txt -o agent_05_business.lock
uv pip compile agent_06_testing.txt -o agent_06_testing.lock
cd ..

# 6. Initialize agent configurations
python scripts/init_agents.py

# 7. Verify Keras 3 setup
python -c "import keras; print(f'Keras: {keras.__version__}, Backend: {keras.backend.backend()}')"
```

**Week 1 (Kickoff)**:
```bash
# Start orchestrator
python scripts/orchestrator.py

# Monitor progress
streamlit run scripts/agent_dashboard.py
```

### 8.4 Success Metrics

- [ ] All 6 agents complete tasks within 6-7 weeks
- [ ] 95%+ test coverage
- [ ] Walk-Forward validation R² > 0.75
- [ ] All constraints satisfied (100% monotonicity)
- [ ] Business tools functional and validated
- [ ] **Ready for deployment decision** by user

---

## 📄 Appendix: Quick Reference

### Tech Stack Summary (Updated)
- **Package Manager**: **uv** (10-100x faster than pip)
- **ML Framework**: **Keras 3** (JAX backend recommended)
- **Data**: Pandas, NumPy
- **Optimization**: SciPy, Optuna, Keras Tuner
- **Tracking**: MLflow
- **Testing**: Pytest, Coverage, Hypothesis
- **Orchestration**: AsyncIO, Hydra

### Key Keras 3 Resources
- **Official Docs**: https://keras.io/api/
- **Custom Layers**: https://keras.io/guides/making_new_layers_and_models_via_subclassing/
- **Custom Training**: https://keras.io/guides/customizing_what_happens_in_fit/
- **Multi-backend**: https://keras.io/keras_3/

### Key uv Commands
```bash
# Install uv
# Windows:
irm https://astral.sh/uv/install.ps1 | iex
# Linux/macOS:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment
uv venv .venv

# Install dependencies (fast!)
uv pip install -r requirements/base.txt

# Install from lock file (reproducible)
uv pip sync requirements/base.lock

# Generate lock file
uv pip compile requirements/base.txt -o requirements/base.lock

# Add new package
uv pip install package-name
```

### Key Project Commands
```bash
# Run orchestrator
python scripts/orchestrator.py

# Run individual agent
python scripts/run_agent.py --agent model --task implement_keras_constraints

# Run parallel tests
python scripts/parallel_test.py

# Train model
python scripts/train_nam.py --config configs/training_config.yaml

# Evaluate model
python scripts/evaluate_model.py --model outputs/models/best_model.keras
```

### Key uv Resources
- **Official Docs**: https://docs.astral.sh/uv/
- **GitHub**: https://github.com/astral-sh/uv
- **Migration from pip**: https://docs.astral.sh/uv/pip/

---

**Document Version**: 3.0 (Keras 3 + uv Package Manager)
**Created**: 2025-10-28
**Status**: Proposal - Ready for Review
**Estimated Timeline**: 6-7 weeks with parallel agent execution
**Framework**: Keras 3 with JAX backend
**Package Manager**: uv (for fast, reproducible dependency management)
