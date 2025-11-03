"""
Streamlit Dashboard for NAM Visualization - FIXED VERSION
All 5 issues from streamliterror.md resolved

Run with: streamlit run streamlit_app.py
(Make sure to activate .venv_main first!)
"""
import os
os.environ['KERAS_BACKEND'] = 'jax'

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

# Import custom classes FIRST (before any keras model loading)
from src.models.simple_nam import SimpleNAM
from src.training.loss_functions import NAMLoss

# Now import other modules
from src.data.data_loader import DataLoader
from src.data.data_preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.training.trainer import NAMTrainer
from src.evaluation.advanced_metrics import compute_all_metrics

# Page config
st.set_page_config(
    page_title="NAM Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🎯 Neural Additive Model - Interactive Dashboard")
st.markdown("**200-Epoch Production Results with Daily Data**")
st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Navigation")
show_section = st.sidebar.selectbox(
    "Select View:",
    ["📊 Overview & Results", "📈 Training Analysis", "🎯 Predictions & Trends",
     "🔬 Elasticity Curves", "📉 NAM Decomposition", "📋 Metrics from Last Run"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**System Status:**
- ✅ 200-epoch training complete
- ✅ Best val_loss: 0.0242
- ✅ Daily data: 250 records
- ✅ Test: 38-day trends
""")

# Load data
@st.cache_data
def load_data():
    """Load daily data"""
    loader = DataLoader('data/raw')
    data = loader.load_daily_sales()

    preprocessor = DataPreprocessor({})
    data = preprocessor.handle_missing_values(data)
    engineer = FeatureEngineer({})
    data = engineer.engineer_all_features(data)
    data_scaled, scalers = preprocessor.scale_features(data)

    return data_scaled, scalers

try:
    data_scaled, scalers = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Could not load data: {e}")
    data_loaded = False

if data_loaded:
    # Split
    train_size = int(len(data_scaled) * 0.70)
    val_size = int(len(data_scaled) * 0.15)
    train_data = data_scaled.iloc[:train_size]
    val_data = data_scaled.iloc[train_size:train_size+val_size]
    test_data = data_scaled.iloc[train_size+val_size:]

#--------------------------------------------------
# OVERVIEW SECTION - FIXED ISSUE #1
#--------------------------------------------------
if show_section == "📊 Overview & Results":
    st.header("📊 System Overview")

    if not data_loaded:
        st.warning("Data not loaded. Check data/raw/ directory.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(data_scaled):,}")
        with col2:
            st.metric("Train Samples", f"{len(train_data)}")
        with col3:
            st.metric("Test Samples", f"{len(test_data)}")
        with col4:
            feat_count = len([c for c in data_scaled.columns if c.endswith('_log')]) + 2
            st.metric("Features", f"{feat_count}")

        st.markdown("---")
        st.subheader("📈 Data Timeline")

        # FIX ISSUE #1: Use index instead of Date operations
        fig = go.Figure()

        # Plot total GMV over time using index
        if 'total_gmv_log' in data_scaled.columns:
            y_values = data_scaled['total_gmv_log'].values
        else:
            numeric_cols = data_scaled.select_dtypes(include=[np.number]).columns
            y_values = data_scaled[numeric_cols[0]].values if len(numeric_cols) > 0 else np.zeros(len(data_scaled))

        fig.add_trace(go.Scatter(
            x=list(range(len(data_scaled))),
            y=y_values,
            mode='lines',
            name='GMV (scaled)',
            line=dict(color='#2E86AB', width=2)
        ))

        # Mark splits with vertical lines
        fig.add_vline(x=train_size, line_dash="dash", line_color="green",
                      annotation_text="Train End", annotation_position="top")
        fig.add_vline(x=train_size+val_size, line_dash="dash", line_color="orange",
                      annotation_text="Val End", annotation_position="top")

        fig.update_layout(
            title="Complete Data Timeline (250 Daily Records)",
            xaxis_title="Day Index",
            yaxis_title="GMV (Scaled)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"✓ Daily data provides {len(data_scaled)} records (20x more than monthly!)")

#--------------------------------------------------
# TRAINING ANALYSIS SECTION
#--------------------------------------------------
elif show_section == "📈 Training Analysis":
    st.header("📈 Training Analysis - 200 Epoch Run")

    # Load training log
    try:
        training_log = pd.read_csv('outputs/training_log.csv')

        st.subheader("Loss & MAE Convergence")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training & Validation Loss', 'Mean Absolute Error'),
            horizontal_spacing=0.15
        )

        # Loss curves
        fig.add_trace(go.Scatter(
            x=training_log['epoch'], y=training_log['loss'],
            name='Train Loss',
            line=dict(color='#2E86AB', width=3)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=training_log['epoch'], y=training_log['val_loss'],
            name='Val Loss',
            line=dict(color='#A23B72', width=3)
        ), row=1, col=1)

        # MAE curves
        fig.add_trace(go.Scatter(
            x=training_log['epoch'], y=training_log['mae'],
            name='Train MAE',
            line=dict(color='#2E86AB', width=3),
            showlegend=False
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=training_log['epoch'], y=training_log['val_mae'],
            name='Val MAE',
            line=dict(color='#A23B72', width=3),
            showlegend=False
        ), row=1, col=2)

        fig.update_layout(height=500, showlegend=True)
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # Key metrics
        best_epoch = training_log['val_loss'].idxmin()
        best_val_loss = training_log['val_loss'].min()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Epoch", f"{best_epoch}")
        with col2:
            st.metric("Best Val Loss", f"{best_val_loss:.4f}")
        with col3:
            st.metric("Total Epochs", f"{len(training_log)}")

        st.success(f"✓ Model converged at epoch {best_epoch} with val_loss = {best_val_loss:.4f}")

    except FileNotFoundError:
        st.warning("Training log not found. Run `python main_daily.py` first.")

#--------------------------------------------------
# PREDICTIONS SECTION - FIXED ISSUE #2
#--------------------------------------------------
elif show_section == "🎯 Predictions & Trends":
    st.header("🎯 38-Day Prediction Trends")

    if not data_loaded:
        st.warning("Data not loaded.")
    else:
        # FIX ISSUE #2: Load model with custom classes already imported
        try:
            import keras

            # Load the trained model
            model_path = 'outputs/models/final_nam_model_daily.keras'

            if Path(model_path).exists():
                with st.spinner("Loading model..."):
                    # Custom classes already imported at top of file
                    model = keras.models.load_model(model_path)

                    X_test, y_test = NAMTrainer.prepare_data_for_keras(test_data)
                    predictions = model.predict(X_test, verbose=0).flatten()

                    st.success(f"✓ Model loaded! Generated predictions for {len(predictions)} test days")

                    # Create interactive time series
                    test_indices = list(range(len(predictions)))

                    fig = go.Figure()

                    # Actual line
                    fig.add_trace(go.Scatter(
                        x=test_indices,
                        y=y_test,
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='#2E86AB', width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>Actual</b><br>Day: %{x}<br>GMV: %{y:.3f}<extra></extra>'
                    ))

                    # Predicted line
                    fig.add_trace(go.Scatter(
                        x=test_indices,
                        y=predictions,
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color='#A23B72', width=2, dash='dash'),
                        marker=dict(size=6, symbol='square'),
                        hovertemplate='<b>Predicted</b><br>Day: %{x}<br>GMV: %{y:.3f}<extra></extra>'
                    ))

                    fig.update_layout(
                        title=f'Test Period: 38-Day Time Series (Apr-Jun 2016)',
                        xaxis_title='Day Index',
                        yaxis_title='GMV (Scaled)',
                        height=600,
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Calculate metrics
                    from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
                    from src.evaluation.advanced_metrics import symmetric_mape, weighted_mape

                    r2 = r2_score(y_test, predictions)
                    mape = mean_absolute_percentage_error(y_test, predictions) * 100
                    smape = symmetric_mape(y_test, predictions)
                    wmape = weighted_mape(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("R²", f"{r2:.4f}")
                    with col2:
                        st.metric("MAPE", f"{mape:.2f}%")
                    with col3:
                        st.metric("sMAPE", f"{smape:.2f}%")
                    with col4:
                        st.metric("wMAPE", f"{wmape:.2f}%")
                    with col5:
                        st.metric("RMSE", f"{rmse:.4f}")

                    if r2 > 0.3:
                        st.success(f"✓ R² = {r2:.3f} - Good predictive power!")

            else:
                st.warning(f"Model not found at {model_path}")
                st.info("Run `python main_daily.py` to train and save the model first.")

        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.code(str(e))

#--------------------------------------------------
# ELASTICITY SECTION - FIXED ISSUE #3
#--------------------------------------------------
elif show_section == "🔬 Elasticity Curves":
    st.header("🔬 Interactive Elasticity Curves")

    st.markdown("""
    **Business Question:** \"How does GMV change if I adjust price/investment?\"

    **NAM Answer:** Plot the learned curve for each feature!
    """)

    if not data_loaded:
        st.warning("Data not loaded.")
    else:
        # FIX ISSUE #3: Actually generate and display elasticity curves
        try:
            import keras
            model_path = 'outputs/models/final_nam_model_daily.keras'

            if Path(model_path).exists():
                with st.spinner("Loading model and generating elasticity curves..."):
                    model = keras.models.load_model(model_path)

                    # Prepare baseline
                    X_all, y_all = NAMTrainer.prepare_data_for_keras(data_scaled)
                    X_baseline = np.median(X_all, axis=0, keepdims=True)

                    # Get feature names
                    numeric_cols = data_scaled.select_dtypes(include=[np.number]).columns.tolist()
                    target_cols = ['total_gmv_log', 'total_gmv']
                    feature_names = [col for col in numeric_cols if col not in target_cols]

                    st.success(f"✓ Generating elasticity curves for {len(feature_names)} features")

                    # Create subplot grid for all features
                    n_features = len(feature_names)
                    n_cols = 3
                    n_rows = (n_features + n_cols - 1) // n_cols

                    fig = make_subplots(
                        rows=n_rows,
                        cols=n_cols,
                        subplot_titles=feature_names,
                        vertical_spacing=0.12,
                        horizontal_spacing=0.1
                    )

                    for idx, feat_name in enumerate(feature_names):
                        row = idx // n_cols + 1
                        col = idx % n_cols + 1

                        # Vary feature from -3 to +3
                        feature_range = np.linspace(-3, 3, 50)
                        contributions = []

                        for value in feature_range:
                            X_test = X_baseline.copy()
                            X_test[:, idx] = value
                            pred = model.predict(X_test, verbose=0).flatten()[0]
                            contributions.append(pred)

                        # Plot curve
                        fig.add_trace(
                            go.Scatter(
                                x=feature_range,
                                y=contributions,
                                mode='lines',
                                name=feat_name,
                                line=dict(width=3),
                                showlegend=False,
                                hovertemplate=f'<b>{feat_name}</b><br>Value: %{{x:.2f}}<br>GMV: %{{y:.3f}}<extra></extra>'
                            ),
                            row=row, col=col
                        )

                        # Mark current value
                        current_val = X_baseline[0, idx]
                        current_contrib = contributions[np.argmin(np.abs(feature_range - current_val))]
                        fig.add_trace(
                            go.Scatter(
                                x=[current_val],
                                y=[current_contrib],
                                mode='markers',
                                marker=dict(size=10, color='red', symbol='star'),
                                showlegend=False,
                                hovertemplate='<b>Current</b><extra></extra>'
                            ),
                            row=row, col=col
                        )

                        # Mark optimal
                        optimal_idx = np.argmax(contributions)
                        fig.add_trace(
                            go.Scatter(
                                x=[feature_range[optimal_idx]],
                                y=[contributions[optimal_idx]],
                                mode='markers',
                                marker=dict(size=10, color='gold', symbol='diamond'),
                                showlegend=False,
                                hovertemplate='<b>Optimal</b><extra></extra>'
                            ),
                            row=row, col=col
                        )

                    fig.update_layout(
                        title_text="Elasticity Curves for All Features",
                        height=400 * n_rows,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.info("🔴 Red star = Current value | 🟡 Gold diamond = Optimal value for max GMV")

            else:
                st.warning("Model not found. Run `python main_daily.py` first.")

        except Exception as e:
            st.error(f"Error: {e}")
            st.code(str(e))

#--------------------------------------------------
# DECOMPOSITION SECTION - FIXED ISSUE #4
#--------------------------------------------------
elif show_section == "📉 NAM Decomposition":
    st.header("📉 NAM Decomposition Analysis")

    st.markdown("""
    **Break down predictions into business drivers:**
    ```
    Total GMV = Baseline + Price/Discount + Marketing + Temporal + Brand + Other
    ```
    """)

    if not data_loaded:
        st.warning("Data not loaded.")
    else:
        # FIX ISSUE #4: Actually implement decomposition
        try:
            import keras
            model_path = 'outputs/models/final_nam_model_daily.keras'

            if Path(model_path).exists():
                with st.spinner("Loading model and computing decomposition..."):
                    model = keras.models.load_model(model_path)

                    X_test, y_test = NAMTrainer.prepare_data_for_keras(test_data)
                    predictions = model.predict(X_test, verbose=0).flatten()

                    # Get feature names
                    numeric_cols = data_scaled.select_dtypes(include=[np.number]).columns.tolist()
                    target_cols = ['total_gmv_log', 'total_gmv']
                    feature_names = [col for col in numeric_cols if col not in target_cols]

                    # Categorize features into business groups
                    categories = {
                        'Price/Discount': [],
                        'Marketing': [],
                        'Temporal': [],
                        'Brand': [],
                        'Other': []
                    }

                    for i, feat_name in enumerate(feature_names):
                        # Get contribution of this feature
                        X_single = np.median(X_test, axis=0, keepdims=True).repeat(len(X_test), axis=0)
                        X_single[:, i] = X_test[:, i]

                        X_zero = np.median(X_test, axis=0, keepdims=True).repeat(len(X_test), axis=0)

                        contrib = (model.predict(X_single, verbose=0).flatten() -
                                   model.predict(X_zero, verbose=0).flatten())

                        # Categorize
                        if any(x in feat_name.lower() for x in ['price', 'discount', 'mrp']):
                            categories['Price/Discount'].append(contrib.mean())
                        elif any(x in feat_name.lower() for x in ['adstock', 'investment']):
                            categories['Marketing'].append(contrib.mean())
                        elif any(x in feat_name.lower() for x in ['month', 'time']):
                            categories['Temporal'].append(contrib.mean())
                        elif 'nps' in feat_name.lower():
                            categories['Brand'].append(contrib.mean())
                        else:
                            categories['Other'].append(contrib.mean())

                    # Sum contributions
                    decomp_values = {cat: sum(vals) if vals else 0 for cat, vals in categories.items()}

                    # Waterfall chart
                    cats = list(decomp_values.keys())
                    vals = list(decomp_values.values())

                    fig = go.Figure(go.Waterfall(
                        orientation="v",
                        measure=["relative"] * len(cats) + ["total"],
                        x=cats + ['Total'],
                        y=vals + [predictions.mean()],
                        text=[f"{v:.3f}" for v in vals] + [f"{predictions.mean():.3f}"],
                        textposition="outside",
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))

                    fig.update_layout(
                        title="Average Contribution Breakdown (Waterfall)",
                        xaxis_title="Category",
                        yaxis_title="GMV Contribution (Scaled)",
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Show breakdown table
                    st.subheader("📊 Contribution Summary")
                    decomp_df = pd.DataFrame({
                        'Category': cats,
                        'Avg Contribution': [f"{v:.4f}" for v in vals],
                        'Percentage': [f"{abs(v)/sum(abs(x) for x in vals)*100:.1f}%" for v in vals]
                    })
                    st.table(decomp_df)

            else:
                st.warning("Model not found.")

        except Exception as e:
            st.error(f"Error: {e}")

#--------------------------------------------------
# METRICS SECTION - FIXED ISSUE #5
#--------------------------------------------------
elif show_section == "📋 Metrics from Last Run":
    st.header("📋 Actual Results from 200-Epoch Run")

    st.markdown("**These are the REAL metrics from your latest training run:**")

    # FIX ISSUE #5: Load actual metrics from last run
    if not data_loaded:
        st.warning("Data not loaded.")
    else:
        try:
            import keras
            model_path = 'outputs/models/final_nam_model_daily.keras'

            if Path(model_path).exists():
                with st.spinner("Computing metrics from last run..."):
                    model = keras.models.load_model(model_path)

                    X_test, y_test = NAMTrainer.prepare_data_for_keras(test_data)
                    X_train, y_train = NAMTrainer.prepare_data_for_keras(train_data)

                    predictions = model.predict(X_test, verbose=0).flatten()

                    # Compute all metrics
                    metrics = compute_all_metrics(y_test, predictions, y_train)

                    st.subheader("📊 Complete Metrics Report")

                    # Accuracy metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score", f"{metrics['r2']:.4f}")
                    with col2:
                        st.metric("MAE", f"{metrics['mae']:.4f}")
                    with col3:
                        st.metric("RMSE", f"{metrics['rmse']:.4f}")

                    st.markdown("---")

                    # Percentage errors
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAPE", f"{metrics['mape']:.2f}%")
                    with col2:
                        st.metric("Weighted MAPE", f"{metrics['wmape']:.2f}%")
                    with col3:
                        st.metric("Symmetric MAPE", f"{metrics['smape']:.2f}%")

                    st.markdown("---")

                    # Scaled errors
                    col1, col2 = st.columns(2)
                    with col1:
                        if metrics['mase']:
                            st.metric("MASE (vs naive)", f"{metrics['mase']:.4f}",
                                      delta="Better than naive" if metrics['mase'] < 1 else "Worse than naive")
                    with col2:
                        st.metric("Bias %", f"{metrics['bias_pct']:.2f}%",
                                  delta="Over-forecast" if metrics['bias_pct'] > 0 else "Under-forecast")

                    st.markdown("---")

                    # Show detailed table
                    st.subheader("📋 All Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': ['R²', 'MAE', 'RMSE', 'MAPE', 'wMAPE', 'sMAPE',
                                   'MASE', 'RMSSE', 'Bias', 'Bias %', 'Relative RMSE %', 'Relative MAE %'],
                        'Value': [
                            f"{metrics['r2']:.4f}",
                            f"{metrics['mae']:.4f}",
                            f"{metrics['rmse']:.4f}",
                            f"{metrics['mape']:.2f}%",
                            f"{metrics['wmape']:.2f}%",
                            f"{metrics['smape']:.2f}%",
                            f"{metrics['mase']:.4f}" if metrics['mase'] else "N/A",
                            f"{metrics['rmsse']:.4f}" if metrics['rmsse'] else "N/A",
                            f"{metrics['bias']:.4f}",
                            f"{metrics['bias_pct']:.2f}%",
                            f"{metrics['relative_rmse']:.2f}%",
                            f"{metrics['relative_mae']:.2f}%"
                        ]
                    })
                    st.table(metrics_df)

                    # Training info
                    st.markdown("---")
                    st.subheader("🎯 Training Configuration")
                    st.info(f"""
                    **Data:** 250 daily records (Aug 2015 - Jun 2016)
                    **Split:** 175 train / 37 val / 38 test
                    **Architecture:** Single-layer NAM with 441 parameters
                    **Best Val Loss:** 0.0242 (from 200-epoch run)
                    **Test Samples:** {len(y_test)} days
                    """)

            else:
                st.warning("Model not found. Run python main_daily.py first.")

        except Exception as e:
            st.error(f"Error: {e}")
            st.code(str(e))

# Footer
st.markdown("---")
st.markdown("""
### 📚 Quick Links

**Documentation:**
- `START_HERE.md` - Complete system guide
- `VISUALIZATION_TOOLS_GUIDE.md` - All 3 visualization options
- `HOW_TO_RUN_VISUALIZATIONS.md` - Detailed instructions

**Alternative Tools:**
- **Jupyter Notebook:** `NAM_Educational_Tutorial.ipynb` (Best for students!)
- **Static Charts:** `start outputs\\figures\\*.png` (Quick view)

**Run Commands:**
```bash
# Train the model
python main_daily.py

# Launch this dashboard
streamlit run streamlit_app.py

# Or use notebook (recommended for education)
jupyter notebook NAM_Educational_Tutorial.ipynb
```

**All 5 issues fixed! Your NAM dashboard is ready!** 🎉
""")
