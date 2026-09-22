import numpy as np

from mlp.activations import ACTIVATIONS
from mlp.errors import ConfigurationError
from mlp.initializers import DEFAULT_INITIALIZER, INITIALIZERS
from mlp.registry import get_from_registry


class DenseLayer:
    def __init__(self, units: int, activation: str,
                 weights_initializer: str = DEFAULT_INITIALIZER):
        if not isinstance(units, int) or units <= 0:
            raise ConfigurationError(
                "expected a positive integer for 'units', "
                f"received {units!r}")
        self.weights = np.empty((0))
        self.bias = np.empty((0))
        self.units = units

        # Resolved once, here: the layer holds an immutable Activation,
        # so nothing downstream ever has to instantiate anything.
        self.activation = get_from_registry(
            ACTIVATIONS, activation, "activation")
        self.weights_initializer = get_from_registry(
            INITIALIZERS, weights_initializer, "weights initializer")

    def init_weightBias(self, input_size: int, rng=None):
        if not isinstance(input_size, int) or input_size <= 0:
            raise ConfigurationError(
                "expected a positive integer for 'input_size', "
                f"received {input_size!r}")
        if rng is None:
            rng = np.random.default_rng()

        self.weights = self.weights_initializer(self.units, input_size, rng)
        self.bias = np.zeros((1, self.units))

    def forward(self, x):
        self.input = x
        self.z = self.input @ self.weights.T + self.bias
        self.a = self.activation.forward(self.z)
        return self.a

    def backward(self, dA, from_loss=False):
        if from_loss is True:
            dZ = dA
        else:
            dZ = self.activation.backward(self.z, dA)
        m = self.input.shape[0]
        self.dW = (dZ.T @ self.input) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = dZ @ self.weights
        return dA_prev

    def update(self, alpha):
        # alpha c'est le learning rate.

        self.weights -= alpha * self.dW
        self.bias -= alpha * self.db

    def __str__(self):
        return f"units={self.units}, activation={self.activation.name}"
