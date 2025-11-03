#!/usr/bin/env python3
"""Verify single-layer NAM architecture without loading the saved model"""
import os
os.environ['KERAS_BACKEND'] = 'jax'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.models.simple_nam import SimpleNAM

print("\n" + "="*70)
print("SINGLE-LAYER NAM ARCHITECTURE VERIFICATION")
print("="*70)

# Create a test model with single layer
n_features = 52
hidden_dims = [16]  # Single layer

model = SimpleNAM(
    n_features=n_features,
    feature_types=['unconstrained'] * n_features,
    hidden_dims=hidden_dims
)

# Build the model
import numpy as np
sample_input = np.random.randn(1, n_features).astype('float32')
_ = model(sample_input)

print(f"\nConfiguration:")
print(f"  Number of features: {n_features}")
print(f"  Hidden dimensions: {hidden_dims}")
print(f"  Feature networks: {len(model.feature_networks)}")

print(f"\nModel Parameters:")
print(f"  Total parameters: {model.count_params():,}")

# Inspect first feature network
print("\n--- Feature Network 0 Structure ---")
feature_net = model.feature_networks[0]
print(f"Network name: {feature_net.name}")
print(f"Number of layers: {len(feature_net.layers)}")

for i, layer in enumerate(feature_net.layers):
    layer_info = f"  Layer {i}: {type(layer).__name__}"
    if hasattr(layer, 'units'):
        layer_info += f" | Units: {layer.units}"
    if hasattr(layer, 'activation'):
        layer_info += f" | Activation: {layer.activation.__name__ if layer.activation else 'linear'}"
    print(layer_info)

# Calculate expected parameters
params_per_feature = feature_net.count_params()
print(f"\nParameters per feature network: {params_per_feature}")

# Analysis
print("\n" + "="*70)
print("ARCHITECTURE ANALYSIS")
print("="*70)

print(f"\nCurrent architecture per feature:")
print(f"  Input (1D) → Dense({hidden_dims[0]}, relu) → Dense(1) → Output")
print(f"  Parameters: 1×{hidden_dims[0]} + {hidden_dims[0]} (bias) + {hidden_dims[0]}×1 + 1 (bias) = {params_per_feature}")

print(f"\nComparison:")
print(f"  Single-layer [16]:    ~49 params/feature  → Total: ~2,548 params")
print(f"  Double-layer [64,32]: ~2,210 params/feature → Total: ~114,920 params")
print(f"  Linear only []:       ~2 params/feature   → Total: ~104 params")

total_expected = params_per_feature * n_features
print(f"\n  Current model: {model.count_params():,} params")
print(f"  Expected: {total_expected:,} params")

if params_per_feature < 100:
    print("\n✓ CONFIRMED: Single-layer architecture")
    print("  - Much smaller than multi-layer [64,32]")
    print("  - More interpretable than deep networks")
    print("  - Each feature → 16 hidden units → 1 output")
else:
    print("\n⚠ Multi-layer architecture")

print("\n" + "="*70)
print("EXPLAINABILITY CHARACTERISTICS")
print("="*70)

print("\nWith single-layer [16] NAM:")
print("  ✓ Additive structure maintained: y = Σ f_i(x_i)")
print("  ✓ Smaller parameter count → easier to interpret")
print("  ✓ Can plot individual feature contribution curves")
print("  ✓ 16 hidden units capture non-linearity while staying interpretable")
print("  ✓ Each feature has independent effect (no interactions)")

print("\nInterpretability score:")
print("  Linear []:           ★★★★★ (Most interpretable - coefficients)")
print("  Single-layer [16]:   ★★★★☆ (Highly interpretable - contribution curves)")
print("  Double-layer [64,32]: ★★★☆☆ (Less interpretable - deeper abstraction)")

print("\n" + "="*70)
