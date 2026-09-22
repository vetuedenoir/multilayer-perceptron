"""Loss functions, as pure functions gathered in a registry.

A loss exposes ``forward(y, y_hat) -> float`` and
``backward(y, y_hat) -> dA``, the gradient of the loss with respect to
the output of the network.

Each loss also names the output activation it fuses with. When the last
layer uses that activation, the gradient with respect to the
pre-activation ``z`` collapses to ``y_hat - y``: one subtraction instead
of a division by ``y_hat * (1 - y_hat)`` followed by the derivative of
the activation. :func:`resolve_output_grad` picks the right pair.
"""

from dataclasses import dataclass
from typing import Callable, Final, Mapping, TypeAlias

import numpy as np

from mlp.types import FloatArray

LossFn: TypeAlias = Callable[[FloatArray, FloatArray], float]
GradFn: TypeAlias = Callable[[FloatArray, FloatArray], FloatArray]

EPS: Final[float] = 1e-15


def binary_crossentropy(y: FloatArray, y_hat: FloatArray) -> float:
    """Return the mean binary cross-entropy between `y` and `y_hat`."""
    clipped = np.clip(y_hat, EPS, 1.0 - EPS)
    terms = y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)
    return float(-np.mean(terms))


def binary_crossentropy_grad(y: FloatArray, y_hat: FloatArray) -> FloatArray:
    """Return the gradient of the binary cross-entropy w.r.t. `y_hat`."""
    clipped = np.clip(y_hat, EPS, 1.0 - EPS)
    return (clipped - y) / (clipped * (1.0 - clipped))


def categorical_crossentropy(y: FloatArray, y_hat: FloatArray) -> float:
    """Return the mean categorical cross-entropy between `y` and `y_hat`.

    `y` is expected one-hot encoded, `y_hat` a row-wise probability
    distribution over the same classes.
    """
    clipped = np.clip(y_hat, EPS, 1.0 - EPS)
    return float(-np.mean(np.sum(y * np.log(clipped), axis=1)))


def categorical_crossentropy_grad(
    y: FloatArray,
    y_hat: FloatArray,
) -> FloatArray:
    """Return the gradient of the categorical cross-entropy w.r.t. y_hat."""
    clipped = np.clip(y_hat, EPS, 1.0 - EPS)
    return -y / clipped


def fused_grad(y: FloatArray, y_hat: FloatArray) -> FloatArray:
    """Return dZ directly, for a loss fused with its output activation.

    Valid for sigmoid + binary cross-entropy and for softmax +
    categorical cross-entropy: in both cases the chain rule simplifies
    to this subtraction.
    """
    return y_hat - y


@dataclass(frozen=True)
class Loss:
    """A loss, its gradient, and the activation it fuses with."""

    name: str
    forward: LossFn
    backward: GradFn
    fused_output_activation: str | None


BINARY_CROSSENTROPY: Final = Loss(
    name="binaryCrossentropy",
    forward=binary_crossentropy,
    backward=binary_crossentropy_grad,
    fused_output_activation="sigmoid",
)
CATEGORICAL_CROSSENTROPY: Final = Loss(
    name="categoricalCrossentropy",
    forward=categorical_crossentropy,
    backward=categorical_crossentropy_grad,
    fused_output_activation="softmax",
)

LOSSES: Final[Mapping[str, Loss]] = {
    "binaryCrossentropy": BINARY_CROSSENTROPY,
    "categoricalCrossentropy": CATEGORICAL_CROSSENTROPY,
    # Backward compatible aliases: the CamelCase names used so far.
    "BinaryCrossentropy": BINARY_CROSSENTROPY,
    "CategoricalCrossentropy": CATEGORICAL_CROSSENTROPY,
}


def resolve_output_grad(
    loss: Loss,
    last_activation: str,
) -> tuple[GradFn, bool]:
    """Return the gradient function to apply after the output layer.

    The boolean tells whether the returned function already yields dZ
    (fused path, the output layer must skip its own backward) rather
    than dA.
    """
    if loss.fused_output_activation == last_activation:
        return fused_grad, True
    return loss.backward, False


__all__ = [
    "Loss",
    "LossFn",
    "GradFn",
    "EPS",
    "binary_crossentropy",
    "binary_crossentropy_grad",
    "categorical_crossentropy",
    "categorical_crossentropy_grad",
    "fused_grad",
    "BINARY_CROSSENTROPY",
    "CATEGORICAL_CROSSENTROPY",
    "LOSSES",
    "resolve_output_grad",
]
