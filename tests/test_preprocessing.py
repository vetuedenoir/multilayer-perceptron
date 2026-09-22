"""Tests of the scalers and of the one-hot encoding."""

import numpy as np
import pytest

from mlp.errors import ConfigurationError, ModelFileError, ShapeError
from mlp.preprocessing import (
    Scaler,
    fit_minmax,
    fit_scaler,
    fit_standard,
    one_hot,
    transform,
)


def test_standard_uses_the_training_set_only() -> None:
    """Fitted on train, the scaler does not see the validation values."""
    train = np.array([[0.0, 10.0], [2.0, 30.0]])
    valid = np.array([[100.0, -50.0]])
    scaler = fit_standard(train)
    np.testing.assert_allclose(scaler.a, [1.0, 20.0])
    np.testing.assert_allclose(scaler.b, [1.0, 10.0])
    np.testing.assert_allclose(transform(scaler, train),
                               [[-1.0, -1.0], [1.0, 1.0]])
    np.testing.assert_allclose(transform(scaler, valid), [[99.0, -7.0]])


def test_minmax_maps_train_onto_unit_interval() -> None:
    """The minimum goes to 0 and the maximum to 1."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 3))
    scaled = transform(fit_minmax(x), x)
    np.testing.assert_allclose(scaled.min(axis=0), 0.0)
    np.testing.assert_allclose(scaled.max(axis=0), 1.0)


@pytest.mark.parametrize("kind", ["standard", "minmax"])
def test_constant_column(kind: str) -> None:
    """A zero spread is replaced by 1: no NaN, the column maps to 0."""
    x = np.array([[1.0, 5.0], [3.0, 5.0], [2.0, 5.0]])
    scaler = fit_scaler(kind, x)
    scaled = transform(scaler, x)
    assert np.all(np.isfinite(scaled))
    np.testing.assert_array_equal(scaled[:, 1], 0.0)


def test_round_trip_on_new_data() -> None:
    """A scaler rebuilt from its dict transforms new data identically."""
    rng = np.random.default_rng(1)
    scaler = fit_standard(rng.normal(3.0, 2.0, size=(50, 4)))
    rebuilt = Scaler.from_dict(scaler.to_dict())
    new = rng.normal(size=(7, 4))
    np.testing.assert_array_equal(transform(rebuilt, new),
                                  transform(scaler, new))
    assert rebuilt.kind == "standard"


def test_transform_rejects_wrong_width() -> None:
    """The number of features is the one seen at fit time."""
    scaler = fit_standard(np.ones((3, 2)))
    with pytest.raises(ShapeError, match=r"\(m, 2\)"):
        transform(scaler, np.ones((3, 3)))


def test_fit_rejects_empty_or_1d_input() -> None:
    """There is nothing to fit on an empty or flat array."""
    with pytest.raises(ShapeError):
        fit_standard(np.empty((0, 3)))
    with pytest.raises(ShapeError):
        fit_minmax(np.ones(3))


def test_unknown_scaler_kind() -> None:
    """The registry lists the valid kinds."""
    with pytest.raises(ConfigurationError, match="standard"):
        fit_scaler("robust", np.ones((2, 2)))


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "standard", "a": [0.0]},
        {"kind": "standard", "a": [0.0, 1.0], "b": [1.0]},
        {"kind": "standard", "a": [0.0], "b": [0.0]},
        {"kind": "unknown", "a": [0.0], "b": [1.0]},
        {"kind": "standard", "a": ["x"], "b": [1.0]},
    ],
)
def test_from_dict_rejects_invalid_data(data: dict[str, object]) -> None:
    """Every malformed scaler is a model file problem."""
    with pytest.raises(ModelFileError):
        Scaler.from_dict(data)


def test_one_hot() -> None:
    """Each row has a single 1, at the column of its label."""
    np.testing.assert_array_equal(
        one_hot(np.array([0, 1, 1, 0]), 2),
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
    )
    assert one_hot(np.array([], dtype=np.int64), 3).shape == (0, 3)


def test_one_hot_rejects_invalid_input() -> None:
    """Out of range labels, 2-D labels or a single class are refused."""
    with pytest.raises(ShapeError, match=r"\[0, 2\)"):
        one_hot(np.array([0, 2]), 2)
    with pytest.raises(ShapeError):
        one_hot(np.array([[0, 1]]), 2)
    with pytest.raises(ConfigurationError):
        one_hot(np.array([0, 0]), 1)
