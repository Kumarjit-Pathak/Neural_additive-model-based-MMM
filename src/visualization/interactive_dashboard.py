"""
Interactive Plotly-based visualizations for NAM analysis
Creates interactive dashboards for elasticity curves, decomposition, and product-level insights

NEW MODULE - Built on top of existing system without modifying core code
"""
import os
os.environ['KERAS_BACKEND'] = 'jax'

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple


class NAMDecompositionAnalyzer:
    """
    Decompose NAM predictions into feature contributions
    Shows: baseline + discount + investment + temporal + other effects
    """

    def __init__(self, model, data_scaled, scalers, feature_names):
        self.model = model
        self.data_scaled = data_scaled
        self.scalers = scalers
        self.feature_names = feature_names

    def get_feature_contributions(self, X):
        """
        Extract contribution of each feature using NAM's additive structure

        Args:
            X: Input features (scaled)

        Returns:
            dict: {feature_name: contribution_array}
        """
        contributions = {}

        # Get contributions from each feature network
        if hasattr(self.model, 'get_feature_contributions'):
            # Use model's built-in method
            contribs = self.model.get_feature_contributions(X)
            for i, contrib in enumerate(contribs):
                if i < len(self.feature_names):
                    contributions[self.feature_names[i]] = contrib.flatten()
        else:
            # Manual extraction: pass one feature at a time
            for i in range(X.shape[1]):
                # Create input with only this feature (others at median)
                X_single = np.median(X, axis=0, keepdims=True).repeat(len(X), axis=0)
                X_single[:, i] = X[:, i]

                # Get prediction (this is baseline + this feature's effect)
                pred = self.model.predict(X_single, verbose=0).flatten()

                # Subtract baseline (all features at median)
                X_baseline = np.median(X, axis=0, keepdims=True).repeat(len(X), axis=0)
                baseline = self.model.predict(X_baseline, verbose=0).flatten()

                contrib = pred - baseline
                if i < len(self.feature_names):
                    contributions[self.feature_names[i]] = contrib

        return contributions

    def decompose_by_category(self, X, feature_names):
        """
        Group feature contributions by business categories

        Returns:
            dict: {category: total_contribution}
        """
        contributions = self.get_feature_contributions(X)

        # Categorize features
        categories = {
            'baseline': [],
            'price_discount': [],
            'marketing_investment': [],
            'temporal': [],
            'brand': [],
            'other': []
        }

        for feat_name in feature_names:
            if feat_name in contributions:
                contrib = contributions[feat_name]

                # Categorize
                if any(x in feat_name.lower() for x in ['price', 'discount', 'mrp']):
                    categories['price_discount'].append(contrib)
                elif any(x in feat_name.lower() for x in ['tv', 'digital', 'sem', 'radio', 'investment', 'adstock']):
                    categories['marketing_investment'].append(contrib)
                elif any(x in feat_name.lower() for x in ['month', 'time', 'day', 'week', 'season']):
                    categories['temporal'].append(contrib)
                elif any(x in feat_name.lower() for x in ['nps', 'brand', 'health']):
                    categories['brand'].append(contrib)
                elif 'gmv_log' in feat_name.lower() and 'total' not in feat_name.lower():
                    categories['baseline'].append(contrib)
                else:
                    categories['other'].append(contrib)

        # Sum contributions per category
        decomposition = {}
        for category, contribs in categories.items():
            if contribs:
                decomposition[category] = np.sum(contribs, axis=0)
            else:
                decomposition[category] = np.zeros(len(X))

        return decomposition


class InteractiveNAMVisualizer:
    """
    Create interactive Plotly visualizations for NAM model analysis
    """

    def __init__(self, model_path='outputs/elasticity_data.pkl'):
        """Load model and data"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.data_scaled = data['data_scaled']
        self.scalers = data['scalers']

        # Get feature names
        numeric_cols = self.data_scaled.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = ['total_gmv_log', 'total_gmv', 'GMV_log', 'GMV']
        self.feature_names = [col for col in numeric_cols if col not in target_cols]

        self.analyzer = NAMDecompositionAnalyzer(
            self.model, self.data_scaled, self.scalers, self.feature_names
        )

    def plot_decomposition_waterfall(self, test_data, predictions, dates=None):
        """
        Create waterfall chart showing contribution breakdown

        Shows: Total Prediction = Baseline + Discounts + Investments + Temporal + Other
        """
        print("\n=== Creating Decomposition Waterfall Chart ===")

        # Prepare test data
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
        from src.training.trainer import NAMTrainer

        X_test, y_test = NAMTrainer.prepare_data_for_keras(test_data)

        # Get decomposition
        decomposition = self.analyzer.decompose_by_category(X_test, self.feature_names)

        # Average contribution per category
        avg_decomp = {cat: np.mean(contrib) for cat, contrib in decomposition.items()}

        # Create waterfall
        categories = ['Baseline', 'Price/Discount', 'Marketing', 'Temporal', 'Brand', 'Other', 'Total']
        values = [
            avg_decomp.get('baseline', 0),
            avg_decomp.get('price_discount', 0),
            avg_decomp.get('marketing_investment', 0),
            avg_decomp.get('temporal', 0),
            avg_decomp.get('brand', 0),
            avg_decomp.get('other', 0),
            np.mean(predictions)
        ]

        fig = go.Figure(go.Waterfall(
            name="GMV Decomposition",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "relative", "relative", "total"],
            x=categories,
            y=values,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig.update_layout(
            title="NAM Decomposition: Average Contribution by Category",
            xaxis_title="Category",
            yaxis_title="GMV Contribution (Scaled)",
            height=600,
            showlegend=False
        )

        output_path = 'outputs/figures/decomposition_waterfall.html'
        fig.write_html(output_path)
        print(f"Saved: {output_path}")

        return fig

    def plot_time_series_decomposition(self, test_data, predictions, dates=None):
        """
        Interactive time series showing stacked contributions over time
        """
        print("\n=== Creating Time Series Decomposition ===")

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
        from src.training.trainer import NAMTrainer

        X_test, y_test = NAMTrainer.prepare_data_for_keras(test_data)

        # Get decomposition
        decomposition = self.analyzer.decompose_by_category(X_test, self.feature_names)

        # Prepare dates
        if dates is None:
            dates = np.arange(len(predictions))

        # Create stacked area chart
        fig = go.Figure()

        # Add each category as stacked area
        cumulative = np.zeros(len(predictions))

        colors = {
            'baseline': '#1f77b4',
            'price_discount': '#ff7f0e',
            'marketing_investment': '#2ca02c',
            'temporal': '#d62728',
            'brand': '#9467bd',
            'other': '#8c564b'
        }

        for category, contrib in decomposition.items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative + contrib,
                fill='tonexty',
                name=category.replace('_', ' ').title(),
                line=dict(color=colors.get(category, '#gray')),
                stackgroup='one'
            ))
            cumulative += contrib

        # Add actual line
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_test,
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=3, dash='dash'),
            marker=dict(size=8)
        ))

        # Add predicted line
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted (Total)',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="NAM Time Series Decomposition: Contribution by Category",
            xaxis_title="Date",
            yaxis_title="GMV Contribution",
            height=700,
            hovermode='x unified'
        )

        output_path = 'outputs/figures/decomposition_time_series.html'
        fig.write_html(output_path)
        print(f"Saved: {output_path}")

        return fig

    def plot_elasticity_curves_interactive(self, granularity='aggregate'):
        """
        Interactive elasticity curves with sliders

        Args:
            granularity: 'aggregate', 'daily', 'weekly', 'monthly'
        """
        print(f"\n=== Creating Interactive Elasticity Curves ({granularity}) ===")

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
        from src.training.trainer import NAMTrainer

        X_all, y_all = NAMTrainer.prepare_data_for_keras(self.data_scaled)
        X_baseline = np.median(X_all, axis=0, keepdims=True)

        # Find price-related features
        price_features = []
        for i, feat_name in enumerate(self.feature_names):
            if any(x in feat_name.lower() for x in ['price', 'mrp', 'discount']):
                price_features.append((i, feat_name))

        if not price_features:
            print("No price features found. Using all features.")
            price_features = [(i, self.feature_names[i]) for i in range(min(len(self.feature_names), 5))]

        # Create subplot grid
        n_features = len(price_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[name for _, name in price_features],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        for idx, (feat_idx, feat_name) in enumerate(price_features):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            # Vary feature from -3 to +3 (scaled range)
            feature_range = np.linspace(-3, 3, 100)
            contributions = []

            for value in feature_range:
                X_test = X_baseline.copy()
                X_test[:, feat_idx] = value
                pred = self.model.predict(X_test, verbose=0).flatten()[0]
                contributions.append(pred)

            contributions = np.array(contributions)

            # Plot curve
            fig.add_trace(
                go.Scatter(
                    x=feature_range,
                    y=contributions,
                    mode='lines',
                    name=feat_name,
                    line=dict(width=3),
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )

            # Mark current value
            current_val = X_baseline[0, feat_idx]
            current_contrib = contributions[np.argmin(np.abs(feature_range - current_val))]

            fig.add_trace(
                go.Scatter(
                    x=[current_val],
                    y=[current_contrib],
                    mode='markers',
                    name='Current',
                    marker=dict(size=12, color='red', symbol='star'),
                    showlegend=(idx == 0)
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
                    name='Optimal',
                    marker=dict(size=12, color='gold', symbol='diamond'),
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )

        fig.update_layout(
            title_text=f"Interactive Elasticity Curves ({granularity.title()} Level)",
            height=300 * n_rows,
            showlegend=True
        )

        fig.update_xaxes(title_text="Feature Value (Scaled)")
        fig.update_yaxes(title_text="GMV Contribution")

        output_path = f'outputs/figures/elasticity_interactive_{granularity}.html'
        fig.write_html(output_path)
        print(f"Saved: {output_path}")

        return fig

    def plot_product_predictions(self, test_data, predictions, dates=None):
        """
        Interactive plot showing actual vs predicted by product over time
        """
        print("\n=== Creating Interactive Product Predictions ===")

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
        from src.training.trainer import NAMTrainer

        X_test, y_test = NAMTrainer.prepare_data_for_keras(test_data)

        if dates is None:
            dates = test_data['Date'].values if 'Date' in test_data.columns else np.arange(len(predictions))

        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Actual vs Predicted Over Time', 'Prediction Error'),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )

        # Plot 1: Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=y_test,
                mode='lines+markers',
                name='Actual',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8),
                hovertemplate='<b>Actual</b><br>Date: %{x}<br>GMV: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predictions,
                mode='lines+markers',
                name='Predicted',
                line=dict(color='#A23B72', width=2, dash='dash'),
                marker=dict(size=6, symbol='square'),
                hovertemplate='<b>Predicted</b><br>Date: %{x}<br>GMV: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Plot 2: Error bars
        errors = y_test - predictions
        error_pct = (errors / y_test) * 100

        colors = ['green' if e < 0 else 'red' for e in errors]

        fig.add_trace(
            go.Bar(
                x=dates,
                y=error_pct,
                name='Error %',
                marker_color=colors,
                hovertemplate='<b>Error</b><br>Date: %{x}<br>Error: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

        fig.update_layout(
            title_text="Daily Predictions: Actual vs Predicted with Error Analysis",
            height=900,
            hovermode='x unified'
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="GMV (Scaled)", row=1, col=1)
        fig.update_yaxes(title_text="Error (%)", row=2, col=1)

        output_path = 'outputs/figures/product_predictions_interactive.html'
        fig.write_html(output_path)
        print(f"Saved: {output_path}")

        return fig

    def plot_contribution_heatmap(self, test_data):
        """
        Heatmap showing feature contributions across time
        """
        print("\n=== Creating Contribution Heatmap ===")

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
        from src.training.trainer import NAMTrainer

        X_test, y_test = NAMTrainer.prepare_data_for_keras(test_data)

        # Get individual feature contributions
        contributions = self.analyzer.get_feature_contributions(X_test)

        # Create contribution matrix
        contrib_matrix = []
        feature_labels = []

        for feat_name, contrib in contributions.items():
            if len(contrib) > 0:
                contrib_matrix.append(contrib)
                feature_labels.append(feat_name)

        contrib_matrix = np.array(contrib_matrix)

        # Create heatmap
        dates = test_data['Date'].values if 'Date' in test_data.columns else np.arange(contrib_matrix.shape[1])

        fig = go.Figure(data=go.Heatmap(
            z=contrib_matrix,
            x=dates,
            y=feature_labels,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='Feature: %{y}<br>Date: %{x}<br>Contribution: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title="Feature Contributions Heatmap Across Time",
            xaxis_title="Date",
            yaxis_title="Feature",
            height=400 + 30 * len(feature_labels)
        )

        output_path = 'outputs/figures/contribution_heatmap.html'
        fig.write_html(output_path)
        print(f"Saved: {output_path}")

        return fig

    def create_elasticity_dashboard(self):
        """
        Complete interactive dashboard with all elasticity visualizations
        """
        print("\n=== Creating Complete Elasticity Dashboard ===")

        # This would combine all visualizations into a single dashboard
        # For now, generate individual components

        print("Dashboard components:")
        print("  1. Elasticity curves (multiple granularities)")
        print("  2. Decomposition analysis")
        print("  3. Product-level predictions")
        print("  4. Contribution heatmap")

        print("\nAll components will be interactive Plotly HTML files")
        print("Open in browser for full interactivity!")


def generate_all_interactive_visualizations(model_path='outputs/elasticity_data.pkl',
                                            test_data_path=None):
    """
    Generate all interactive Plotly visualizations

    Args:
        model_path: Path to saved model and data
        test_data_path: Optional path to test data
    """
    print("="*70)
    print("GENERATING INTERACTIVE PLOTLY VISUALIZATIONS")
    print("="*70)

    # Initialize visualizer
    try:
        visualizer = InteractiveNAMVisualizer(model_path)
    except FileNotFoundError:
        print(f"ERROR: {model_path} not found.")
        print("Run main_daily.py first to generate elasticity_data.pkl")
        return

    # Get test data
    if test_data_path:
        test_data = pd.read_pickle(test_data_path)
    else:
        # Load from data pipeline
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        from src.data.data_loader import DataLoader
        from src.data.data_preprocessing import DataPreprocessor
        from src.data.feature_engineering import FeatureEngineer

        loader = DataLoader('data/raw')
        data = loader.load_daily_sales()
        preprocessor = DataPreprocessor({})
        data = preprocessor.handle_missing_values(data)
        engineer = FeatureEngineer({})
        data = engineer.engineer_all_features(data)
        data_scaled, _ = preprocessor.scale_features(data)

        # Use last 38 days as test
        test_data = data_scaled.iloc[-38:]

    # Get predictions
    from src.training.trainer import NAMTrainer
    X_test, y_test = NAMTrainer.prepare_data_for_keras(test_data)
    predictions = visualizer.model.predict(X_test, verbose=0).flatten()

    # Generate visualizations
    print("\n1. Elasticity Curves (Aggregate)...")
    visualizer.plot_elasticity_curves_interactive('aggregate')

    print("\n2. Decomposition Waterfall...")
    visualizer.plot_decomposition_waterfall(test_data, predictions)

    print("\n3. Time Series Decomposition...")
    visualizer.plot_decomposition_waterfall(test_data, predictions)

    print("\n4. Product Predictions...")
    visualizer.plot_product_predictions(test_data, predictions)

    print("\n5. Contribution Heatmap...")
    visualizer.plot_contribution_heatmap(test_data)

    print("\n" + "="*70)
    print("ALL INTERACTIVE VISUALIZATIONS GENERATED")
    print("="*70)
    print("\nGenerated HTML files (open in browser):")
    print("  1. outputs/figures/elasticity_interactive_aggregate.html")
    print("  2. outputs/figures/decomposition_waterfall.html")
    print("  3. outputs/figures/decomposition_time_series.html")
    print("  4. outputs/figures/product_predictions_interactive.html")
    print("  5. outputs/figures/contribution_heatmap.html")
    print("\nAll files are interactive! Hover, zoom, pan, and explore!")


if __name__ == "__main__":
    generate_all_interactive_visualizations()
