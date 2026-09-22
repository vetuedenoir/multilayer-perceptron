"""Gradient checks for the losses, fused paths included."""

import numpy as np
import pytest
from conftest import numerical_gradient, relative_error

from mlp.activations import (
    sigmoid,
    sigmoid_backward,
    softmax,
    softmax_backward,
)
from mlp.losses import (
    BINARY_CROSSENTROPY,
    CATEGORICAL_CROSSENTROPY,
    LOSSES,
    Loss,
    binary_crossentropy,
    binary_crossentropy_grad,
    categorical_crossentropy,
    categorical_crossentropy_grad,
    fused_grad,
    resolve_output_grad,
)
from mlp.types import FloatArray

TOLERANCE = 1e-7
SAMPLES = 6
CLASSES = 3


def binary_targets(seed: int = 0) -> tuple[FloatArray, FloatArray]:
    """Return (y, z) for a binary problem, z being pre-activations."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(SAMPLES, 1)).astype(np.float64)
    z = rng.normal(size=(SAMPLES, 1))
    return y, z


def categorical_targets(seed: int = 0) -> tuple[FloatArray, FloatArray]:
    """Return (y, z) for a multiclass problem, y being one-hot."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, CLASSES, size=SAMPLES)
    y = np.eye(CLASSES)[labels]
    z = rng.normal(size=(SAMPLES, CLASSES))
    return y, z


def test_binary_crossentropy_grad_matches_finite_differences() -> None:
    """The analytic gradient w.r.t. y_hat is m times the numerical one.

    The forward averages over the m samples while the backward does
    not: the division by m is applied once, by the layer.
    """
    y, z = binary_targets()
    y_hat = sigmoid(z)

    analytic = binary_crossentropy_grad(y, y_hat)
    numeric = numerical_gradient(lambda p: binary_crossentropy(y, p), y_hat)

    assert relative_error(analytic, numeric * SAMPLES) < TOLERANCE


def test_categorical_crossentropy_grad_matches_finite_differences() -> None:
    """Same check for the categorical cross-entropy."""
    y, z = categorical_targets()
    y_hat = softmax(z)

    analytic = categorical_crossentropy_grad(y, y_hat)
    numeric = numerical_gradient(
        lambda p: categorical_crossentropy(y, p), y_hat)

    assert relative_error(analytic, numeric * SAMPLES) < TOLERANCE


def test_fused_sigmoid_path_matches_finite_differences() -> None:
    """y_hat - y is the gradient of the BCE w.r.t. z, sigmoid output."""
    y, z = binary_targets()

    analytic = fused_grad(y, sigmoid(z))
    numeric = numerical_gradient(
        lambda values: binary_crossentropy(y, sigmoid(values)), z)

    assert relative_error(analytic, numeric * SAMPLES) < TOLERANCE


def test_fused_softmax_path_matches_finite_differences() -> None:
    """y_hat - y is the gradient of the CCE w.r.t. z, softmax output."""
    y, z = categorical_targets()

    analytic = fused_grad(y, softmax(z))
    numeric = numerical_gradient(
        lambda values: categorical_crossentropy(y, softmax(values)), z)

    assert relative_error(analytic, numeric * SAMPLES) < TOLERANCE


def test_fused_sigmoid_equals_the_generic_path() -> None:
    """The shortcut and the two-step chain rule agree, BCE + sigmoid."""
    y, z = binary_targets()
    y_hat = sigmoid(z)

    generic = sigmoid_backward(z, binary_crossentropy_grad(y, y_hat))

    assert relative_error(fused_grad(y, y_hat), generic) < TOLERANCE


def test_fused_softmax_equals_the_generic_path() -> None:
    """The shortcut and the two-step chain rule agree, CCE + softmax."""
    y, z = categorical_targets()
    y_hat = softmax(z)

    generic = softmax_backward(z, categorical_crossentropy_grad(y, y_hat))

    assert relative_error(fused_grad(y, y_hat), generic) < TOLERANCE


def test_losses_are_positive_and_minimal_on_a_perfect_prediction() -> None:
    """A loss is non-negative and drops when the prediction improves."""
    y, _ = categorical_targets()
    perfect = np.clip(y, 1e-9, 1.0)
    uniform = np.full_like(y, 1.0 / CLASSES)

    assert categorical_crossentropy(y, perfect) >= 0.0
    assert categorical_crossentropy(y, perfect) < categorical_crossentropy(
        y, uniform)


def test_binary_crossentropy_is_finite_on_saturated_predictions() -> None:
    """Clipping at EPS keeps log(0) out of the result."""
    y = np.array([[1.0], [0.0]])
    y_hat = np.array([[0.0], [1.0]])

    assert np.isfinite(binary_crossentropy(y, y_hat))


@pytest.mark.parametrize(
    ("loss", "activation", "expected_fused"),
    [
        (BINARY_CROSSENTROPY, "sigmoid", True),
        (BINARY_CROSSENTROPY, "relu", False),
        (CATEGORICAL_CROSSENTROPY, "softmax", True),
        (CATEGORICAL_CROSSENTROPY, "sigmoid", False),
    ],
)
def test_resolve_output_grad_picks_the_fused_path(
    loss: Loss,
    activation: str,
    expected_fused: bool,
) -> None:
    """The shortcut is used only when the output activation matches."""
    grad_fn, fused = resolve_output_grad(loss, activation)

    assert fused is expected_fused
    assert (grad_fn is fused_grad) is expected_fused


def test_registry_aliases_point_to_the_same_loss() -> None:
    """The CamelCase keys resolve to the canonical entries."""
    assert LOSSES["BinaryCrossentropy"] is LOSSES["binaryCrossentropy"]
    assert LOSSES["CategoricalCrossentropy"] is LOSSES[
        "categoricalCrossentropy"]
