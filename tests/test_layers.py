"""Tests of the dense layer and of its pure functions."""

import numpy as np
import pytest
from conftest import numerical_gradient, relative_error

from mlp.errors import ConfigurationError, NotBuiltError, ShapeError
from mlp.layers import DenseLayer, Layer, dense_backward, dense_forward
from mlp.types import FloatArray

TOLERANCE = 1e-7
SAMPLES = 5
INPUTS = 4
UNITS = 3


def dense_problem(
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return (W, b, x, G), G being the upstream gradient dL/dz * m."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(UNITS, INPUTS))
    b = rng.normal(size=(1, UNITS))
    x = rng.normal(size=(SAMPLES, INPUTS))
    G = rng.normal(size=(SAMPLES, UNITS))
    return W, b, x, G


def test_dense_forward_shape_and_value() -> None:
    """The pre-activation is x @ W.T + b, of shape (m, units)."""
    W, b, x, _ = dense_problem()
    z = dense_forward(W, b, x)
    assert z.shape == (SAMPLES, UNITS)
    np.testing.assert_allclose(z[0], W @ x[0] + b[0])


@pytest.mark.parametrize("wrt", ["W", "b", "x"])
def test_dense_backward_matches_finite_differences(wrt: str) -> None:
    """The gradients dW, db and dA_prev are those of sum(z * G) / m.

    dA_prev is not divided by m (the division is applied once, on the
    parameter gradients), so it is compared to m times the numerical
    gradient.
    """
    W, b, x, G = dense_problem()
    dW, db, dA_prev = dense_backward(W, x, G)

    def loss(W: FloatArray, b: FloatArray, x: FloatArray) -> float:
        return float(np.sum(dense_forward(W, b, x) * G)) / SAMPLES

    if wrt == "W":
        analytic = dW
        numeric = numerical_gradient(lambda p: loss(p, b, x), W)
    elif wrt == "b":
        analytic = db
        numeric = numerical_gradient(lambda p: loss(W, p, x), b)
    else:
        analytic = dA_prev
        numeric = SAMPLES * numerical_gradient(lambda p: loss(W, b, p), x)
    assert relative_error(analytic, numeric) < TOLERANCE


def test_dense_layer_implements_the_protocol() -> None:
    """A DenseLayer is recognised as a Layer."""
    assert isinstance(DenseLayer(2, "relu"), Layer)


@pytest.mark.parametrize("units", [0, -3, 2.5, True])
def test_invalid_units_raise(units: int) -> None:
    """Units must be a positive integer."""
    with pytest.raises(ConfigurationError, match="units"):
        DenseLayer(units, "relu")


def test_unknown_activation_raises() -> None:
    """The error lists the valid activations."""
    with pytest.raises(ConfigurationError, match="'relu'"):
        DenseLayer(2, "gelu")


def test_forward_before_build_raises() -> None:
    """A layer without weights cannot compute anything."""
    with pytest.raises(NotBuiltError):
        DenseLayer(2, "relu").forward(np.zeros((1, 3)))


def test_forward_with_wrong_input_size_raises() -> None:
    """The number of columns of x must match the input size."""
    layer = DenseLayer(2, "relu")
    layer.build(3, np.random.default_rng(0))
    with pytest.raises(ShapeError, match=r"\(m, 3\)"):
        layer.forward(np.zeros((1, 4)))


def test_build_is_seeded() -> None:
    """Two builds with the same seed draw the same weights."""
    a = DenseLayer(4, "relu", "heUniform")
    b = DenseLayer(4, "relu", "heUniform")
    a.build(3, np.random.default_rng(7))
    b.build(3, np.random.default_rng(7))
    np.testing.assert_array_equal(a.weights, b.weights)
    assert a.bias.shape == (1, 4)


def test_set_params_checks_shapes() -> None:
    """Weights of the wrong shape are rejected, the right ones build."""
    layer = DenseLayer(2, "relu")
    with pytest.raises(ShapeError):
        layer.set_params([np.zeros((3, 4)), np.zeros((1, 2))])
    with pytest.raises(ShapeError):
        layer.set_params([np.zeros((2, 4)), np.zeros((1, 3))])
    layer.set_params([np.ones((2, 4)), np.zeros((1, 2))])
    assert layer.built
    with pytest.raises(ShapeError, match=r"\(2, 4\)"):
        layer.set_params([np.ones((2, 5)), np.zeros((1, 2))])


def test_get_config() -> None:
    """The config holds canonical activation names."""
    config = DenseLayer(3, "ReLU", "heUniform").get_config()
    assert config == {
        "type": "dense",
        "units": 3,
        "activation": "relu",
        "initializer": "heUniform",
    }
