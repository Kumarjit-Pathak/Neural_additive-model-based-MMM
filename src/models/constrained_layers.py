"""
Constrained neural network layers for Keras 3
Agent 2: Model Architect
"""
import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
from keras import ops, layers


class MonotonicPositiveLayer(layers.Layer):
    """
    Monotonic increasing layer - ensures output increases with input
    Used for: cross-price elasticity, distribution
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
        super().build(input_shape)

    def call(self, inputs):
        # Apply softplus to ensure positive weights
        positive_weights = ops.softplus(self.kernel)
        output = ops.matmul(inputs, positive_weights) + self.bias
        return output

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class MonotonicNegativeLayer(layers.Layer):
    """
    Monotonic decreasing layer - ensures output decreases with input
    Used for: own-price elasticity
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
        super().build(input_shape)

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
    Used for: marketing investment effects
    """

    def __init__(self, init_a=1.0, init_alpha=0.5, init_beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.init_a = init_a
        self.init_alpha = init_alpha
        self.init_beta = init_beta

    def build(self, input_shape):
        self.a = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(self.init_a),
            name='scale',
            trainable=True
        )
        self.alpha = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(self.init_alpha),
            name='alpha',
            trainable=True
        )
        self.beta = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(self.init_beta),
            name='beta',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # Ensure positive parameters
        a_pos = ops.softplus(self.a)
        alpha_pos = ops.softplus(self.alpha)
        beta_pos = ops.softplus(self.beta)

        # Beta-Gamma transformation: a * x^alpha * exp(-beta * x)
        powered = ops.power(inputs + 1e-6, alpha_pos)
        decayed = ops.exp(-beta_pos * inputs)

        return a_pos * powered * decayed

    def get_config(self):
        config = super().get_config()
        config.update({
            'init_a': self.init_a,
            'init_alpha': self.init_alpha,
            'init_beta': self.init_beta
        })
        return config
