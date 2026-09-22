"""Activation functions, as pure functions gathered in a registry.

An activation exposes two pure functions:

``forward(z) -> a``
    the activated output;
``backward(z, dA) -> dZ``
    the gradient of the loss with respect to ``z``.

``backward`` takes the incoming gradient ``dA`` instead of only returning
the local derivative, because softmax cannot be differentiated
element-wise: each of its outputs depends on every neuron of the layer,
so the chain rule goes through a Jacobian matrix. Passing ``dA`` in lets
every activation return ``dZ`` directly.

Nothing here is a class to instantiate: a layer resolves its activation
once, in its constructor, and the resulting :class:`Activation` is
immutable and shareable.
"""

from dataclasses import dataclass
from typing import Callable, Final, Mapping, TypeAlias

import numpy as np

from mlp.types import FloatArray

ForwardFn: TypeAlias = Callable[[FloatArray], FloatArray]
BackwardFn: TypeAlias = Callable[[FloatArray, FloatArray], FloatArray]

LEAKY_RELU_SLOPE: Final[float] = 0.01


def sigmoid(z: FloatArray) -> FloatArray:
    """Return the logistic function of `z`, element-wise.

    Computed branch by branch on the sign of `z` so that ``exp`` never
    overflows on large magnitudes.
    """
    exp_neg = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + exp_neg), exp_neg / (1.0 + exp_neg))


def sigmoid_backward(z: FloatArray, dA: FloatArray) -> FloatArray:
    """Return dZ for a sigmoid activation, given dA."""
    a = sigmoid(z)
    return dA * a * (1.0 - a)


def relu(z: FloatArray) -> FloatArray:
    """Return ``max(0, z)``, element-wise."""
    return np.maximum(0.0, z)


def relu_backward(z: FloatArray, dA: FloatArray) -> FloatArray:
    """Return dZ for a ReLU activation, given dA."""
    return dA * np.where(z <= 0.0, 0.0, 1.0)


def leaky_relu(z: FloatArray) -> FloatArray:
    """Return `z` with the negative part scaled by LEAKY_RELU_SLOPE."""
    return np.where(z <= 0.0, LEAKY_RELU_SLOPE * z, z)


def leaky_relu_backward(z: FloatArray, dA: FloatArray) -> FloatArray:
    """Return dZ for a leaky ReLU activation, given dA."""
    return dA * np.where(z <= 0.0, LEAKY_RELU_SLOPE, 1.0)


def softmax(z: FloatArray) -> FloatArray:
    """Return the row-wise softmax of `z`.

    The maximum of each row is subtracted first, which leaves the result
    unchanged and keeps ``exp`` away from overflow.
    """
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def softmax_backward(z: FloatArray, dA: FloatArray) -> FloatArray:
    """Return dZ for a softmax activation, given dA.

    Jacobian-vector product, computed without ever building the
    Jacobian: ``dZ_i = a_i * (dA_i - sum_k(dA_k * a_k))``.

    When softmax is the output layer and the loss is the categorical
    cross-entropy, the derivative with respect to `z` simplifies to
    ``y_hat - y``; that shortcut lives in :mod:`mlp.losses`. This
    function is what makes softmax usable outside of that pairing, as a
    hidden layer or with another loss.
    """
    a = softmax(z)
    return a * (dA - np.sum(dA * a, axis=1, keepdims=True))


@dataclass(frozen=True)
class Activation:
    """An activation function and its gradient, under a canonical name."""

    name: str
    forward: ForwardFn
    backward: BackwardFn


SIGMOID: Final = Activation("sigmoid", sigmoid, sigmoid_backward)
RELU: Final = Activation("relu", relu, relu_backward)
LEAKY_RELU: Final = Activation("leaky_relu", leaky_relu, leaky_relu_backward)
SOFTMAX: Final = Activation("softmax", softmax, softmax_backward)

ACTIVATIONS: Final[Mapping[str, Activation]] = {
    "sigmoid": SIGMOID,
    "relu": RELU,
    "leaky_relu": LEAKY_RELU,
    "softmax": SOFTMAX,
    # Backward compatible aliases: the CamelCase names used so far.
    "Sigmoid": SIGMOID,
    "ReLU": RELU,
    "LeakyReLU": LEAKY_RELU,
    "Softmax": SOFTMAX,
}

__all__ = [
    "Activation",
    "ForwardFn",
    "BackwardFn",
    "LEAKY_RELU_SLOPE",
    "sigmoid",
    "sigmoid_backward",
    "relu",
    "relu_backward",
    "leaky_relu",
    "leaky_relu_backward",
    "softmax",
    "softmax_backward",
    "SIGMOID",
    "RELU",
    "LEAKY_RELU",
    "SOFTMAX",
    "ACTIVATIONS",
]
