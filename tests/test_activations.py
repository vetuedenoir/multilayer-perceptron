"""Gradient checks and basic properties of the activation functions."""

import numpy as np
import pytest
from conftest import away_from_zero, numerical_gradient, relative_error

from mlp.activations import (
    ACTIVATIONS,
    LEAKY_RELU_SLOPE,
    Activation,
    leaky_relu,
    relu,
    sigmoid,
    softmax,
)
from mlp.types import FloatArray

TOLERANCE = 1e-7
SHAPE = (7, 4)

ELEMENTWISE = ["sigmoid", "relu", "leaky_relu"]
ALL_NAMES = ELEMENTWISE + ["softmax"]


def sample_z(seed: int = 0) -> FloatArray:
    """Return a reproducible pre-activation matrix, never touching 0."""
    rng = np.random.default_rng(seed)
    return away_from_zero(rng.normal(size=SHAPE))


@pytest.mark.parametrize("name", ALL_NAMES)
def test_backward_matches_finite_differences(name: str) -> None:
    """The returned dZ must match a finite-difference estimate."""
    activation = ACTIVATIONS[name]
    z = sample_z()
    rng = np.random.default_rng(1)
    weights: FloatArray = rng.normal(size=SHAPE)

    def scalar(z_values: FloatArray) -> float:
        return float(np.sum(weights * activation.forward(z_values)))

    analytic = activation.backward(z, weights)
    numeric = numerical_gradient(scalar, z)

    assert relative_error(analytic, numeric) < TOLERANCE


def test_sigmoid_range_and_symmetry() -> None:
    """Sigmoid stays in (0, 1), maps 0 to 0.5 and is antisymmetric."""
    z = sample_z()
    a = sigmoid(z)

    assert np.all(a > 0.0) and np.all(a < 1.0)
    assert sigmoid(np.zeros((1, 1)))[0, 0] == pytest.approx(0.5)
    assert np.allclose(sigmoid(-z), 1.0 - a)


def test_sigmoid_does_not_overflow_on_large_inputs() -> None:
    """Large magnitudes saturate instead of raising a warning."""
    z = np.array([[-800.0, 0.0, 800.0]])

    with np.errstate(over="raise"):
        a = sigmoid(z)

    assert np.allclose(a, [[0.0, 0.5, 1.0]])


def test_relu_and_leaky_relu_shapes() -> None:
    """The negative part is clamped by ReLU, scaled by leaky ReLU."""
    z = np.array([[-2.0, 3.0]])

    assert np.allclose(relu(z), [[0.0, 3.0]])
    assert np.allclose(leaky_relu(z), [[-2.0 * LEAKY_RELU_SLOPE, 3.0]])


def test_softmax_rows_are_distributions() -> None:
    """Each row is positive and sums to one."""
    a = softmax(sample_z())

    assert np.all(a > 0.0)
    assert np.allclose(np.sum(a, axis=1), 1.0)


def test_softmax_is_shift_invariant() -> None:
    """Adding a constant per row leaves the result unchanged."""
    z = sample_z()
    shift = np.array([[1000.0]])

    with np.errstate(over="raise"):
        shifted = softmax(z + shift)

    assert np.allclose(shifted, softmax(z))


def test_registry_aliases_point_to_the_same_activation() -> None:
    """The CamelCase keys resolve to the canonical entries."""
    assert ACTIVATIONS["ReLU"] is ACTIVATIONS["relu"]
    assert ACTIVATIONS["LeakyReLU"] is ACTIVATIONS["leaky_relu"]
    assert ACTIVATIONS["Sigmoid"] is ACTIVATIONS["sigmoid"]
    assert ACTIVATIONS["Softmax"] is ACTIVATIONS["softmax"]


@pytest.mark.parametrize("name", ALL_NAMES)
def test_registry_entries_are_named_and_frozen(name: str) -> None:
    """An Activation carries its canonical name and cannot be mutated."""
    activation = ACTIVATIONS[name]

    assert isinstance(activation, Activation)
    assert activation.name == name
    with pytest.raises(AttributeError):
        activation.name = "other"  # type: ignore[misc]
